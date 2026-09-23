"""Matched fixed-action diagnostic, separate from PPO evaluation."""
import json
from pathlib import Path
import numpy as np
import torch
from mjlab_contact_prep.distribution_env import DistributionEnv
from mjlab_contact_prep.execution import COLUMNS
out=Path('evaluation/gpu_direction_response_v2');out.mkdir(exist_ok=False)
cases=[dict(id=f'r{r}_a{a}',radius_mm=r,angle_deg=a,dx_mm=0.,dy_mm=r if a==90 else -r) for r in [1.2,5.] for a in [90,270]]
(out/'bank').mkdir();env=DistributionEnv(cases,out/'bank');results=[]
try:
 for amp in [0.,.05]:
  env.set_arm(amp,7,training=False);t=env.term;t.record=True;t.records.clear();t.probe_records.clear()
  # Fixed control-frame Y actions: world Y has the opposite sign.
  actions=torch.tensor([[0.,-.6],[0.,.6],[0.,-.6],[0.,.6]],device=env.device)
  with torch.no_grad():
   for _ in range(75):env.env.step(actions)
  trace=torch.stack(t.records).cpu().numpy();np.savez_compressed(out/f'amp_{amp:g}.npz',trace=trace,columns=COLUMNS)
  for i,c in enumerate(cases):
   d={k:trace[:,i,j] for j,k in enumerate(COLUMNS)}
   results.append(dict(**c,amplitude_deg=amp,prep_eligible=env.preparation[i]['eligible'],action_y=-.6 if c['angle_deg']==90 else .6,
      measured_dy_mm=float((d['tip_y'][-1]-d['tip_y'][0])*1000),measured_dx_mm=float((d['tip_x'][-1]-d['tip_x'][0])*1000),
      ready_fraction=float(d['ready'].mean()),mean_Fz=float(d['Fz'].mean()),peak_Fz=float(d['Fz'].max()),fault_reason=int(d['reason'][-1])))
  t.record=False;t.records.clear();t.probe_records.clear()
 (out/'results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
finally:env.close()
