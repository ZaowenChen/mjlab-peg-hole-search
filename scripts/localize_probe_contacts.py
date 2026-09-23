"""Identify exact mesh/geom pairs at frozen audit states; no controller changes."""
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import mujoco
from mjlab.envs import ManagerBasedRlEnv
from mjlab_contact_prep.environment import make_env
from mjlab_contact_prep.config import ContactConfig
from mjlab_contact_prep.probe import ProbeConfig

p=argparse.ArgumentParser();p.add_argument('audit',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--collision-model',choices=['legacy','partitioned'],default='legacy');p.add_argument('--physics-backend',choices=['stock','contact-fix'],default='stock');a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
report=json.loads((a.audit/'results.json').read_text())
with np.load(a.audit/'states.npz') as z:states={k:z[k] for k in z.files}
env=ManagerBasedRlEnv(cfg=make_env(report['cases'],ContactConfig(),plane=report['plane'],probe=ProbeConfig(),collision_model=a.collision_model,physics_backend=a.physics_backend),device='cuda:0')
try:
    env.reset();m=env.sim.mj_model;cpu=mujoco.MjData(m);rows=[]
    catalog=[]
    for g in range(m.ngeom):
        mesh=int(m.geom_dataid[g]) if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH else -1
        b=int(m.geom_bodyid[g])
        catalog.append(dict(id=g,name=m.geom(g).name,body_id=b,body=m.body(b).name,mesh_id=mesh,
            mesh=m.mesh(mesh).name if mesh>=0 else '',type=int(m.geom_type[g]),contype=int(m.geom_contype[g]),conaffinity=int(m.geom_conaffinity[g]),
            condim=int(m.geom_condim[g]),friction=m.geom_friction[g].tolist(),pos=m.geom_pos[g].tolist(),quat=m.geom_quat[g].tolist()))
    hole=m.site('hole/hole_mouth_site').id
    for k in range(len(states['qpos'])):
        for n,v in states.items():getattr(env.sim.data,n)[:]=torch.as_tensor(v[k],device=env.device)
        env.sim.forward()
        co=env.sim.wp_data.contact;n=int(env.sim.wp_data.nacon.numpy()[0])
        world=co.worldid.numpy()[:n];geom=co.geom.numpy()[:n];dist=co.dist.numpy()[:n]
        pos=co.pos.numpy()[:n];frame=co.frame.numpy()[:n]
        for i,case in enumerate(report['cases']):
            for key,v in states.items():getattr(cpu,key)[:]=v[k,i]
            mujoco.mj_forward(m,cpu)
            hr=cpu.site_xmat[hole].reshape(3,3);hp=cpu.site_xpos[hole]
            def record(pair,position,distance,normal):
                return dict(ids=[int(j) for j in pair],meshes=[catalog[int(j)]['mesh'] for j in pair],
                    bodies=[catalog[int(j)]['body'] for j in pair],hole_position_mm=((position-hp)@hr*1000).tolist(),
                    distance_um=float(distance*1e6),hole_normal=(normal@hr).tolist())
            gpu=[record(geom[j],pos[j],dist[j],frame[j,0]) for j in np.flatnonzero(world==i)]
            cp=[]
            for j in range(cpu.ncon):
                c=cpu.contact[j];r=record(c.geom,c.pos,c.dist,c.frame.reshape(3,3)[0])
                f=np.zeros(6);mujoco.mj_contactForce(m,cpu,j,f);r['force_local']=f.tolist();cp.append(r)
            rows.append(dict(snapshot=k,case=case['id'],cpu=cp,gpu=gpu))
    (a.output/'catalog.json').write_text(json.dumps(catalog,indent=2)+'\n')
    (a.output/'contacts.json').write_text(json.dumps(rows,indent=2)+'\n')
    summary=[]
    for case in report['cases']:
        subset=[r for r in rows if r['case']==case['id']]
        for backend in ['cpu','gpu']:
            count=Counter(tuple(c['ids']) for r in subset for c in r[backend])
            entry=dict(case=case['id'],backend=backend,pairs=[dict(ids=list(pair),meshes=[catalog[j]['mesh'] for j in pair],total_contacts=v,
                snapshots=sum(any(tuple(c['ids'])==pair for c in r[backend]) for r in subset)) for pair,v in count.most_common()])
            summary.append(entry)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
finally:env.close()
