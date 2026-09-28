"""Read-only decomposition of saved rollout timing and force-band failures."""
import json
from pathlib import Path
import numpy as np
from mjlab_contact_prep.metrics import rolling_mean
r=Path(__file__).resolve().parents[1];out=[]
for run in ['final_offsets','final_repeat2','final_repeat3','large_offsets_r1','large_offsets_r2','large_offsets_r3']:
 z=np.load(r/'evaluation'/run/'trace.npz');a=z['samples'];cols=list(z['columns']);results=json.loads((r/'evaluation'/run/'results.json').read_text())['rows']
 for j,row in enumerate(results):
  d={k:a[:,j,i].astype(float) for i,k in enumerate(cols)};t=d['time'];final=t>=10;first=lambda b:float(t[np.flatnonzero(b)[0]]) if b.any() else None
  x={'run':run,'case':row['case']['id'],'touch_s':row['first_touch_s'],'target20_s':first(d['target']>=19.999),'ready_s':row['first_ready_s'],'stable_s':row['stable_contact_s'],'short_stable_s':row['first_250ms_stable_s'],'held_to_end':row['stable_through_end'],'mean_raw_final2s':float(d['Fz'][final].mean()),'attitude_max_deg':row['max_attitude_drift_deg']}
  for key in ['Fz','filtered_force']:
   f=d[key][final];err=f-20;n=125
   x[key]={'mean':float(f.mean()),'rmse':float(np.sqrt(np.mean(err**2))),'in_band':float(np.mean(np.abs(err)<=4)),'low_force_fraction':float(np.mean(f<5)),'mean_bad_windows':float(np.mean(np.abs(rolling_mean(f,n)-20)>2)),'rmse_bad_windows':float(np.mean(rolling_mean(err**2,n)>16)),'band_bad_windows':float(np.mean(rolling_mean(np.abs(err)<=4,n)<.85-1e-10)),'loss_bad_windows':float(np.mean(rolling_mean(f<5,n)>.05+1e-10))}
  out.append(x)
(r/'evaluation/readiness_diagnosis.json').write_text(json.dumps(out,indent=2)+'\n')
for x in out:
 if x['run'] in ['final_offsets','large_offsets_r1']:
  print(x['case'],'touch/target/ready/stable',*[None if x[k] is None else round(x[k],3) for k in ['touch_s','target20_s','ready_s','stable_s']], 'raw', {k:round(x['Fz'][k],3) for k in ['mean','rmse','in_band','low_force_fraction','mean_bad_windows','rmse_bad_windows','band_bad_windows']},'filtered',{k:round(x['filtered_force'][k],3) for k in ['rmse','in_band','band_bad_windows']})
print('ready<2',sum(x['ready_s'] is not None and x['ready_s']<2 for x in out),'/',len(out))
print('stable<2',sum(x['stable_s'] is not None and x['stable_s']<2 for x in out),'/',len(out))
