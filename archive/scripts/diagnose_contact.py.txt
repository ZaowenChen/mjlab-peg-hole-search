"""Compare CPU/GPU forward dynamics at identical saved contact states."""
import json
import argparse
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
import mujoco
from mjlab.envs import ManagerBasedRlEnv
from mjlab_contact_prep.config import ContactConfig
from mjlab_contact_prep.environment import make_env

parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
out=args.output;out.mkdir(exist_ok=False)
cfg=ContactConfig(**json.loads(args.config.read_text())['contact'])
cases=[dict(id='2mm_xm',dx_mm=-2.,dy_mm=0.,ry_deg=0.)]
env=ManagerBasedRlEnv(cfg=make_env(cases,cfg,seconds=7),device='cuda:0')
try:
 env.reset();term=env.action_manager.get_term('contact');term.record=True
 names=['qpos','qvel','act','ctrl','qacc_warmstart','mocap_pos','mocap_quat','qfrc_applied','xfrc_applied','sensordata','qacc']
 samples={n:[] for n in names}
 gpu_contacts=[]
 for k in range(175):
  env.step(torch.zeros(1,2,device=env.device))
  if k>=150:
   for n in names:samples[n].append(getattr(env.sim.data,n)[0].clone())
   n=int(env.sim.wp_data.nacon.numpy()[0]);co=env.sim.wp_data.contact
   gpu_contacts.append(dict(ncon=n,geom=co.geom.numpy()[:n].tolist(),dist=co.dist.numpy()[:n].tolist()))
 saved={n:torch.stack(v).cpu().numpy() for n,v in samples.items()}
 np.savez_compressed(out/'states.npz',**saved)
 m=env.sim.mj_model;d=mujoco.MjData(m);rows=[]
 for i in range(25):
  for n in names[:-2]:getattr(d,n)[:]=saved[n][i]
  mujoco.mj_forward(m,d)
  a=term.force_adr
  rows.append(dict(gpu_sensor_F=saved['sensordata'][i,a:a+3].tolist(),cpu_sensor_F=d.sensordata[a:a+3].tolist(),cpu_ncon=d.ncon,cpu_geom=d.contact.geom[:d.ncon].tolist(),cpu_dist=d.contact.dist[:d.ncon].tolist(),gpu_contact=gpu_contacts[i],qacc_difference_max=float(np.max(np.abs(d.qacc-saved['qacc'][i])))))
 (out/'runtime.json').write_text(json.dumps(dict(enableflags=int(m.opt.enableflags),geom_names=[m.geom(i).name for i in range(m.ngeom)]),indent=2))
 (out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
 print(json.dumps(rows,indent=2))
finally:env.close()
