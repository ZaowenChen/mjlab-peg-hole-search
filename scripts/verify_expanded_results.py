import json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];p=root/'evaluation/gpu_expanded_ab_v3'
rs=json.loads((p/'results.json').read_text());assert len(rs)==6
m=json.loads((p/'manifest.json').read_text());prep=json.loads((p/'test_bank/preparation.json').read_text())
assert len(m['train_cases'])==96 and len(m['test_cases'])==72
assert len({(r['seed'],r['amplitude_deg']) for r in rs})==6
for r in rs:
 path=p/f"seed_{r['seed']}_amp_{r['amplitude_deg']:g}"
 with np.load(path/'test_trace.npz') as d:
  cols=list(d['columns']);trace=d['trace']
  assert len(r['rows'])==72 and all(x['outcome']!='pending' for x in r['rows'])
  assert r['successes']==sum(x['success'] for x in r['rows'])
  for i,x in enumerate(r['rows']):
   assert x['id']==prep[i]['id'] and x['prep_eligible']==prep[i]['eligible']
   if not x['prep_eligible']:assert not x['success'];continue
   t=trace[:x['terminal_samples'],i]
   assert len(t)>0 and np.isfinite(t).all()
   assert x['search_reason']==int(t[-1,cols.index('reason')])
   if x['success']:
    assert (t[:,cols.index('reason')]==0).all()
    assert x['final_error_mm']<=.15 and x['depth_mm']>=.1 and 0<x['success_s']<=40
 print('VERIFIED',r['seed'],r['amplitude_deg'],flush=True)
(p/'validation.json').write_text(json.dumps(dict(passed=True,runs=6,cases_per_run=72,checks=['all predetermined cases retained','same held-out preparation eligibility across arms','finite pre-terminal traces','fault reasons agree with traces','terminal capture geometry','success counts']),indent=2)+'\n')
