"""Compare CPU force at the exact pre-step state for the 20 mm / 135 deg transient."""
import json
from pathlib import Path
import numpy as np
import torch
import mujoco
from mjlab.envs import ManagerBasedRlEnv
from mjlab_contact_prep.environment import make_env
from mjlab_contact_prep.config import ContactConfig
from mjlab_contact_prep.probe import ProbeConfig
from mjlab_contact_prep.backend import backend_metadata

root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'evaluation/repaired_probe_matrix/manifest.json').read_text())
cases=manifest['cases'];world=next(i for i,c in enumerate(cases) if c['id']=='20mm_135deg')
env=ManagerBasedRlEnv(cfg=make_env(cases,ContactConfig(),probe=ProbeConfig(amplitude_deg=0)),device='cuda:0')
rows=[];counter=0;bad_states=[]
try:
    env.reset();term=env.action_manager.get_term('contact');m=env.sim.mj_model;cpu=mujoco.MjData(m)
    original=env.sim.step
    names=['qpos','qvel','act','ctrl','qacc_warmstart','mocap_pos','mocap_quat','qfrc_applied','xfrc_applied']
    def step(*args,**kwargs):
        global counter
        watch=1000<=counter<1700
        if watch:
            for n in names:getattr(cpu,n)[:]=getattr(env.sim.data,n)[world].cpu().numpy()
        original(*args,**kwargs)
        if watch:
            term.read_wrench();gpu=float(term.wrench[world,2]);mujoco.mj_forward(m,cpu)
            sr=cpu.site_xmat[term.sensor_id].reshape(3,3)
            grav=sr.T@(m.opt.gravity*term.payload_mass)
            f=-cpu.sensordata[term.force_adr:term.force_adr+3]-grav
            fz=float(-((np.diag([1,-1,-1])@sr)@f)[2])
            co=env.sim.wp_data.contact;nc=int(env.sim.wp_data.nacon.numpy()[0]);w=co.worldid.numpy()[:nc];ids=np.flatnonzero(w==world)
            rows.append(dict(time_s=counter*.002,gpu_fz=gpu,cpu_fz=fz,cpu_ncon=cpu.ncon,gpu_ncon=len(ids),
                cpu_dist=[float(c.dist) for c in cpu.contact[:cpu.ncon]],gpu_dist=co.dist.numpy()[ids].tolist(),
                cpu_geoms=[[m.geom(int(g)).name for g in c.geom] for c in cpu.contact[:cpu.ncon]],
                gpu_geoms=[[m.geom(int(g)).name for g in pair] for pair in co.geom.numpy()[ids]],
                cpu_frames=[c.frame.reshape(3,3).tolist() for c in cpu.contact[:cpu.ncon]],gpu_frames=co.frame.numpy()[ids].tolist(),
                cpu_pos=[c.pos.tolist() for c in cpu.contact[:cpu.ncon]],gpu_pos=co.pos.numpy()[ids].tolist()))
            if abs(gpu-fz)>5:bad_states.append({n:getattr(cpu,n).copy() for n in names})
        counter+=1
    env.sim.step=step
    for k in range(90):env.step(torch.zeros(len(cases),2,device=env.device))
    result=dict(case=cases[world],backend=backend_metadata('contact-fix'),samples=len(rows),
        max_gpu_fz=max(r['gpu_fz'] for r in rows),max_cpu_fz=max(r['cpu_fz'] for r in rows),
        mean_abs_fz_difference=float(np.mean([abs(r['gpu_fz']-r['cpu_fz']) for r in rows])),rows=rows)
    if bad_states:np.savez_compressed(root/'evaluation/repaired_peak_states.npz',**{n:np.stack([s[n] for s in bad_states])[:,None] for n in names})
    (root/'evaluation/repaired_peak_diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k not in ('rows','backend')})
finally:env.close()
