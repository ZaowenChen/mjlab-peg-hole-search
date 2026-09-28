"""Replay saved audit states and compare named CPU/GPU collision pairs."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import mujoco
from mjlab.envs import ManagerBasedRlEnv
from mjlab_contact_prep.environment import make_env
from mjlab_contact_prep.config import ContactConfig
from mjlab_contact_prep.probe import ProbeConfig

p=argparse.ArgumentParser();p.add_argument('audit',type=Path);a=p.parse_args()
report=json.loads((a.audit/'results.json').read_text());states=np.load(a.audit/'states.npz')
env=ManagerBasedRlEnv(cfg=make_env(report['cases'],ContactConfig(),plane=report['plane'],probe=ProbeConfig()),device='cuda:0')
try:
    env.reset();m=env.sim.mj_model;cpu=mujoco.MjData(m);rows=[]
    name=lambda pair:[m.geom(int(j)).name for j in pair]
    for k in range(len(states['qpos'])):
        for n in states.files:getattr(env.sim.data,n)[:]=torch.as_tensor(states[n][k],device=env.device)
        env.sim.forward()
        contact=env.sim.wp_data.contact;n=int(env.sim.wp_data.nacon.numpy()[0])
        world=contact.worldid.numpy()[:n];geom=contact.geom.numpy()[:n];dist=contact.dist.numpy()[:n]
        for i,case in enumerate(report['cases']):
            for key in states.files:getattr(cpu,key)[:]=states[key][k,i]
            mujoco.mj_forward(m,cpu)
            rows.append(dict(snapshot=k,case=case['id'],gpu_pairs=[name(x) for x in geom[world==i]],gpu_dist=dist[world==i].tolist(),
                cpu_pairs=[name(x) for x in cpu.contact.geom[:cpu.ncon]],cpu_dist=cpu.contact.dist[:cpu.ncon].tolist()))
    (a.audit/'contacts.json').write_text(json.dumps(rows,indent=2)+'\n')
    for case in report['cases']:
        r=[x for x in rows if x['case']==case['id']]
        print(case['id'],'GPU count',sorted(set(len(x['gpu_pairs']) for x in r)),'CPU count',sorted(set(len(x['cpu_pairs']) for x in r)))
finally:env.close()
