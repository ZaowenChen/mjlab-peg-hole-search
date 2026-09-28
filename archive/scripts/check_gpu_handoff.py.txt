"""One bounded GPU-only handoff check: static hold, XY pulses, then rocking."""
import argparse,hashlib,json,math,time
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab_contact_prep.config import ContactConfig
from mjlab_contact_prep.environment import make_env
from mjlab_contact_prep.execution import COLUMNS
from mjlab_contact_prep.probe import ProbeConfig,PROBE_COLUMNS
from mjlab_contact_prep.metrics import assess
from mjlab_contact_prep.backend import backend_metadata

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
root=Path(__file__).resolve().parents[1];cfg=ContactConfig()
cases=[dict(id=f'{r}mm_{angle}deg',dx_mm=r*math.cos(math.radians(angle)),dy_mm=r*math.sin(math.radians(angle))) for r in [1,2,5] for angle in [0,90,180,270]]
manifest=dict(cases=cases,contact=asdict(cfg),backend=backend_metadata('contact-fix'),collision_model='partitioned',
 protocol='0..6s static; X+ 6..7, Y+ 8..9, X- 14..15, Y- 15..16; otherwise zero XY. Rocking arm enables 0.05deg/0.25Hz at 10s.',
 criteria='Static: existing sustained 2s criteria by 6s. Motion: signed displacement relative to matched hold >0.03mm for each positive pulse; no fault; <=5% samples below 5N after 6s. Rocking: finite, no fault, same contact-loss limit. No requirement for instantaneous 20N during motion.',
 source_sha256={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for folder in ['src','scripts','assets'] for f in (root/folder).rglob('*') if f.is_file() and '__pycache__' not in f.parts})
(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
traces={};rows=[]
for arm in ['hold','xy','xy_probe']:
 env=ManagerBasedRlEnv(cfg=make_env(cases,cfg,probe=ProbeConfig(amplitude_deg=0)),device='cuda:0')
 try:
  env.reset();term=env.action_manager.get_term('contact');term.record=True
  for k in range(450):
   t=k*env.step_dt;action=torch.zeros(len(cases),2,device=env.device)
   if arm!='hold':
    if 6<=t<7:action[:,0]=1
    if 8<=t<9:action[:,1]=1
    if 14<=t<15:action[:,0]=-1
    if 15<=t<16:action[:,1]=-1
   if arm=='xy_probe' and k==250:term.probe.cfg=replace(term.probe.cfg,amplitude_deg=.05)
   env.step(action)
   if (k+1)%100==0:print('PROGRESS',arm,round((k+1)*env.step_dt,2),flush=True)
  tr=torch.stack(term.records).cpu().numpy();pr=torch.stack(term.probe_records).cpu().numpy();traces[arm]=tr
  np.savez_compressed(a.output/f'{arm}.npz',samples=tr,columns=np.array(COLUMNS),probe=pr,probe_columns=np.array(PROBE_COLUMNS))
  for i,case in enumerate(cases):
   d={n:tr[:,i,j] for j,n in enumerate(COLUMNS)};static=tr[d['time']<6,i];moving=d['time']>=6
   def displacement(start,end,axis):
    pos=tr[:,i,COLUMNS.index('tip_'+axis)];j=np.argmin(abs(d['time']-start));l=np.argmin(abs(d['time']-end));value=pos[l]-pos[j]
    ref=traces['hold'][:,i,COLUMNS.index('tip_'+axis)];return float((value-(ref[l]-ref[j]))*1000)
   x=displacement(6,7,'x');y=-displacement(8,9,'y') # nominal frame Y points to world -Y
   stable=assess(static,COLUMNS,cfg);loss=float((d['Fz'][moving]<5).mean());fault=int(d['reason'].max())
   row=dict(arm=arm,case=case,static_sustained_2s=stable['sustained_2s'],static_stable_s=stable['stable_contact_s'],
       static_first_touch_s=stable['first_touch_s'],fault=fault,peak_force_N=float(d['Fz'].max()),motion_loss_fraction=loss,
       x_response_mm=x,y_response_mm=y,passed=bool(stable['sustained_2s'] and not fault and loss<=.05 and (arm=='hold' or min(x,y)>.03)))
   rows.append(row)
  (a.output/'results.json').write_text(json.dumps(dict(rows=rows),indent=2)+'\n');print('RESULT',arm,sum(x['passed'] for x in rows if x['arm']==arm),len(cases),flush=True)
 finally:env.close()
passed=[r for r in [1,2,5] if all(x['passed'] for x in rows if x['case']['id'].startswith(f'{r}mm_'))]
eligible=[case for case in cases if all(x['passed'] for x in rows if x['case']['id']==case['id'])]
smallest=min((math.hypot(c['dx_mm'],c['dy_mm']) for c in eligible),default=None)
selected=[c for c in eligible if abs(math.hypot(c['dx_mm'],c['dy_mm'])-smallest)<1e-6] if smallest else []
(a.output/'decision.json').write_text(json.dumps(dict(passed_radii_mm=passed,eligible_cases=eligible,ppo_radius_mm=smallest,ppo_cases=selected,scope='Only selected tested directions; a partially passed radius is not full-direction coverage.'),indent=2)+'\n')
print('DECISION',passed,flush=True)
