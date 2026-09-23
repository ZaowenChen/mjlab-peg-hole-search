#!/usr/bin/env python3
"""Compare fresh CPU/GPU forward evaluations at identical probe states."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import mujoco
from mjlab.envs import ManagerBasedRlEnv
from mjlab_contact_prep.config import ContactConfig
from mjlab_contact_prep.probe import ProbeConfig
from mjlab_contact_prep.environment import make_env
from mjlab_contact_prep.backend import backend_metadata

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--plane',action='store_true')
    p.add_argument('--collision-model',choices=['legacy','partitioned'],default='partitioned')
    p.add_argument('--replay',type=Path,help='saved states.npz from prior identical robot/body topology')
    p.add_argument('--ccd-iterations',type=int,default=50)
    p.add_argument('--ccd-tolerance',type=float)
    p.add_argument('--broadphase-filter',type=int)
    p.add_argument('--margin-gap',type=float)
    p.add_argument('--primitive-peg',action='store_true',help='diagnostic cylinder only; not a production asset edit')
    p.add_argument('--gjk-minval',type=float,help='process-local diagnostic for tiny convex contact degeneracy')
    p.add_argument('--gjk-min-dist',type=float,help='process-local EPA initial face squared-distance threshold')
    p.add_argument('--physics-backend',choices=['stock','contact-fix'],default='contact-fix')
    a=p.parse_args()
    if a.physics_backend!='stock' and (a.gjk_minval is not None or a.gjk_min_dist is not None):
        p.error('GJK scalar experiments require --physics-backend stock')
    a.output.mkdir(parents=True,exist_ok=False)
    if a.gjk_minval is not None:
        from mujoco_warp._src import collision_gjk
        collision_gjk.MINVAL=a.gjk_minval
    if a.gjk_min_dist is not None:
        from mujoco_warp._src import collision_gjk
        collision_gjk.MIN_DIST=a.gjk_min_dist
    cases=[dict(id='5mm_xp',dx_mm=5,dy_mm=0),dict(id='5mm_ym',dx_mm=0,dy_mm=-5),dict(id='20mm_xm',dx_mm=-20,dy_mm=0)]
    cfg=make_env(cases,ContactConfig(),plane=a.plane,seconds=10,
        probe=ProbeConfig(amplitude_deg=.05,frequency=.25,ramp_time=1),collision_model=a.collision_model,physics_backend=a.physics_backend)
    cfg.sim.mujoco.ccd_iterations=a.ccd_iterations
    if a.primitive_peg:
        original=cfg.scene.entities['robot'].spec_fn
        def primitive_spec():
            spec=original()
            for g in spec.geoms:
                if g.meshname.endswith('pa2') and g.contype:
                    g.type=mujoco.mjtGeom.mjGEOM_CYLINDER;g.meshname=''
                    g.size[:]=(.0174495,.025,0);g.pos[:]=(.0176245,.01762511,-.025)
            return spec
        cfg.scene.entities['robot'].spec_fn=primitive_spec
    env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0')
    rows=[];saved=[]
    names=['qpos','qvel','act','ctrl','qacc_warmstart','mocap_pos','mocap_quat','qfrc_applied','xfrc_applied']
    try:
        env.reset();term=env.action_manager.get_term('contact');m=env.sim.mj_model;cpu=mujoco.MjData(m)
        if a.broadphase_filter is not None:
            env.sim.wp_model.opt.broadphase_filter=a.broadphase_filter
            env.sim.use_cuda_graph=False
        if a.margin_gap is not None:
            for n in ['geom_margin','geom_gap']:
                values=getattr(env.sim.wp_model,n).numpy()
                ids=np.array([g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name.startswith('hole/')])
                getattr(m,n)[ids]=a.margin_gap
                values[:,ids]=a.margin_gap
                getattr(env.sim.wp_model,n).assign(values)
        if a.ccd_tolerance is not None:
            m.opt.ccd_tolerance=a.ccd_tolerance
            env.sim.wp_model.opt.ccd_tolerance.assign(np.array([a.ccd_tolerance],dtype=np.float32))
        replay=None
        if a.replay:
            with np.load(a.replay) as data:replay={n:data[n] for n in names}
        for k in range(len(replay['qpos']) if replay is not None else 250):
            if replay is None:env.step(torch.zeros(3,2,device=env.device))
            else:
                for n in names:
                    value=torch.as_tensor(replay[n][k],device=env.device)
                    assert value.shape==getattr(env.sim.data,n).shape,(n,value.shape,getattr(env.sim.data,n).shape)
                    getattr(env.sim.data,n)[:]=value
            if replay is not None or (k>=100 and k%10==0):
                # Recompute both backends at q(t), avoiding pre-integration
                # sensors from mj_step being compared against post-step qpos.
                env.sim.forward();term.read_wrench()
                state={n:getattr(env.sim.data,n).cpu().numpy().copy() for n in names}
                saved.append(state)
                gpu=term.wrench.cpu().numpy().copy()
                for i,case in enumerate(cases):
                    for n in names:getattr(cpu,n)[:]=state[n][i]
                    mujoco.mj_forward(m,cpu)
                    sr=cpu.site_xmat[term.sensor_id].reshape(3,3)
                    sp=cpu.site_xpos[term.sensor_id];tp=cpu.site_xpos[term.geometry_id]
                    raw=np.r_[cpu.sensordata[term.force_adr:term.force_adr+3],cpu.sensordata[term.torque_adr:term.torque_adr+3]]
                    grav=sr.T@(m.opt.gravity*term.payload_mass)
                    f=-raw[:3]-grav
                    moment=-raw[3:]-np.cross(sr.T@(cpu.subtree_com[term.payload_id]-sp),grav)
                    moment-=np.cross(sr.T@(tp-sp),f)
                    rotation=np.diag([1,-1,-1])@sr
                    wrench=np.r_[rotation@f,rotation@moment];wrench[2]*=-1
                    co=env.sim.wp_data.contact;nc=int(env.sim.wp_data.nacon.numpy()[0]);world=co.worldid.numpy()[:nc]
                    rows.append(dict(case=case['id'],time_s=(k*10+101 if replay is not None else k+1)*env.step_dt,gpu_wrench=gpu[i].tolist(),cpu_wrench=wrench.tolist(),cpu_ncon=cpu.ncon,gpu_ncon=int(np.sum(world==i)),geom_position_max_difference=float(np.max(np.abs(env.sim.data.geom_xpos[i].cpu().numpy()-cpu.geom_xpos))),cpu_geoms=[[m.geom(int(g)).name for g in c.geom] for c in cpu.contact[:cpu.ncon]],cpu_distances=[float(c.dist) for c in cpu.contact[:cpu.ncon]]))
        np.savez_compressed(a.output/'states.npz',**{n:np.stack([s[n] for s in saved]) for n in names})
        delta=np.array([np.array(r['gpu_wrench'])-r['cpu_wrench'] for r in rows])
        result=dict(physics_backend=backend_metadata(a.physics_backend),primitive_peg=a.primitive_peg,gjk_minval=a.gjk_minval,gjk_min_dist=a.gjk_min_dist,backend_source=__import__("mujoco_warp").__file__,plane=a.plane,collision_model=a.collision_model,ccd_iterations=a.ccd_iterations,ccd_tolerance=float(m.opt.ccd_tolerance),margin_gap=a.margin_gap,broadphase_filter=a.broadphase_filter,naconmax=env.sim.wp_data.naconmax,naccdmax=env.sim.wp_data.naccdmax,ncollision=int(env.sim.wp_data.ncollision.numpy()[0]),replay=str(a.replay) if a.replay else None,cases=cases,samples=len(rows),mean_abs_difference=np.abs(delta).mean(0).tolist(),max_abs_difference=np.abs(delta).max(0).tolist(),rows=rows)
        (a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
    finally:env.close()

if __name__=='__main__':main()
