"""Isolate translation sensitivity of GPU collision detection at frozen states."""
import json
from pathlib import Path
import numpy as np
import torch
import mujoco_warp
from mjlab.envs import ManagerBasedRlEnv
from mjlab_contact_prep.environment import make_env
from mjlab_contact_prep.config import ContactConfig
from mjlab_contact_prep.probe import ProbeConfig

root=Path(__file__).resolve().parents[1];audit=root/'evaluation/probe_cpu_gpu_hole'
cases=json.loads((audit/'results.json').read_text())['cases']
env=ManagerBasedRlEnv(cfg=make_env(cases,ContactConfig(),probe=ProbeConfig(),collision_model='partitioned'),device='cuda:0')
try:
    env.reset();states=np.load(audit/'states.npz')
    for name in states.files:getattr(env.sim.data,name)[:]=torch.as_tensor(states[name][0],device=env.device)
    env.sim.forward();positions=env.sim.data.geom_xpos.clone();rows=[]
    for shift in [(0,0,0),(-.866,0,-.1),(-.8,0,0)]:
        env.sim.data.geom_xpos[:]=positions+torch.tensor(shift,device=env.device)
        mujoco_warp.collision(env.sim.wp_model,env.sim.wp_data)
        n=int(env.sim.wp_data.nacon.numpy()[0]);world=env.sim.wp_data.contact.worldid.numpy()[:n]
        rows.append(dict(shift=shift,counts=[int((world==i).sum()) for i in range(3)]))
    (root/'evaluation/partition_origin_check.json').write_text(json.dumps(rows,indent=2)+'\n');print(rows)
finally:env.close()
