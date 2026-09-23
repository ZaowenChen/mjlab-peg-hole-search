"""Inference only, independently prepared held-out positions; no cached resets."""
import copy, hashlib, json, math, time
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from rsl_rl.runners import OnPolicyRunner
from mjlab_contact_prep.search_env import SearchEnv
from mjlab_contact_prep.environment import make_env
from mjlab_contact_prep.config import ContactConfig
from mjlab_contact_prep.probe import ProbeConfig
from mjlab_contact_prep.execution import COLUMNS
from mjlab_contact_prep.metrics import assess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evaluation/gpu_unseen_holes_v1'
def save(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def integrity():
    frozen=json.loads((ROOT/'reference/gpu_contact_candidate_v1.json').read_text())
    bad=[n for n,h in frozen['files'].items() if sha(ROOT/n)!=h]
    assert not bad,bad
    return {str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'src/mjlab_contact_prep/search_env.py',*[ROOT/f'evaluation/gpu_ppo_pilot_v2/amp_{a}/policy.pt' for a in ['0','0.05']]]}

cases=[]
for center in [90,270]:
    for offset in [-20,-10,0,10,20]:
        for radius in [.8,1.,1.2]:
            if offset==0 and radius==1.:continue
            angle=center+offset
            cases.append(dict(id=f'r{radius:g}_a{angle}',center_deg=center,angle_deg=angle,radius_mm=radius,
                dx_mm=radius*math.cos(math.radians(angle)),dy_mm=radius*math.sin(math.radians(angle))))
assert len(cases)==28
before=integrity();OUT.mkdir(exist_ok=False)
save(OUT/'manifest.json',dict(cases=cases,hashes=before,script_sha256=sha(Path(__file__)),
    preparation='Each arm starts a fresh physical environment; 6s descent with zero XY and zero rocking. No training cache loaded. Entry: no fault and online ready at 6s, identical to training. Original strict 2s stability reported separately, not used to discard cases.',
    evaluation='Deterministic checkpoint inference only; original observation/history and 0.2s decisions; at most 12s. No autoreset. Capture: radial <=0.15mm and depth >=0.1mm for 0.2s, checked every 0.04s. Shallow capture, not full insertion.',
    overload='One terminal overload episode when latched reason is axial/radial/moment load (1/2/3). Prep and search counted separately. Post-terminal trajectories excluded.',
    denominator='Overall success / all 28, conditional success / preparation eligible; paired comparison on common eligible also reported. Success times exclude 6s preparation; failed episodes have null success times.',seed=7))
results=[]
for amp in [0.,.05]:
    torch.manual_seed(7);start=time.monotonic();out=OUT/f'amp_{amp:g}';out.mkdir()
    e=SearchEnv.__new__(SearchEnv) # Reuse exact trained observation code, never its contact-bank constructor/reset.
    e.device='cuda:0';e.num_envs=len(cases);e.num_actions=2;e.max_episode_length=60;e.cfg={}
    e.env=ManagerBasedRlEnv(cfg=make_env(cases,ContactConfig(),probe=ProbeConfig(amplitude_deg=0)),device=e.device)
    try:
        e.env.reset();e.term=e.env.action_manager.get_term('contact');t=e.term;t.record=True
        zeros=torch.zeros(e.num_envs,2,device=e.device)
        with torch.no_grad():
            for _ in range(150):e.env.step(zeros)
        prep=torch.stack(t.records).cpu().numpy();t.records.clear();t.probe_records.clear()
        eligible=((t.controller.reason==0)&t.controller.ready).clone()
        rows=[]
        for i,c in enumerate(cases):
            reason=int(t.controller.reason[i]);stats=assess(prep[:,i],COLUMNS,ContactConfig())
            rows.append(dict(**c,prep_eligible=bool(eligible[i]),prep_reason=reason,prep_stats=stats,
                prep_overload=reason in [1,2,3],outcome='pending' if eligible[i] else 'contact_preparation_failed',
                success=False,success_search_s=None,success_total_s=None,search_overload=False,search_reason=None))
        np.savez_compressed(out/'preparation.npz',trace=prep,columns=COLUMNS)
        save(out/'preparation.json',rows)
        t.probe.cfg=replace(t.probe.cfg,amplitude_deg=amp)
        e.env.sim.forward();t.read_wrench()
        e.origin=e.env.sim.data.site_xpos[:,t.geometry_id].clone()
        e.last_action=zeros.clone();e.history=torch.zeros(e.num_envs,32,20,device=e.device)
        e.episode_length_buf=torch.zeros(e.num_envs,dtype=torch.long,device=e.device)
        e.history[:,-1]=e._frame()
        cfg=json.loads((ROOT/f'evaluation/gpu_ppo_pilot_v2/amp_{amp:g}/config.json').read_text())['ppo']
        runner=OnPolicyRunner(e,copy.deepcopy(cfg),None,device=e.device)
        runner.load(str(ROOT/f'evaluation/gpu_ppo_pilot_v2/amp_{amp:g}/policy.pt'),map_location=e.device)
        policy=runner.get_inference_policy();active=eligible.clone();dwell=torch.zeros(e.num_envs,device=e.device)
        action_log=[]
        with torch.inference_mode():
            for step in range(1,61):
                if not active.any():break
                actions=policy(e.get_observations()).clamp(-1,1);actions[~active]=0
                action_log.append(actions.cpu().numpy().copy())
                for _ in range(5):
                    e.env.step(actions);distance,depth=e.metrics()
                    inside=(distance<=.00015)&(depth>=.0001)&(t.controller.reason==0)
                    dwell=torch.where(inside,dwell+e.env.step_dt,0)
                e.last_action[:]=actions;e.episode_length_buf+=1
                e.history[:]=torch.roll(e.history,-1,1);e.history[:,-1]=e._frame()
                fault=t.controller.reason!=0;success=(dwell>=.2-1e-6)&~fault
                done=active&(success|fault|(step==60))
                for i in torch.nonzero(done).flatten().tolist():
                    r=rows[i];reason=int(t.controller.reason[i]);ok=bool(success[i])
                    r.update(outcome='success' if ok else ('search_fault' if reason else 'search_timeout'),success=ok,
                        success_search_s=round(step*.2,2) if ok else None,success_total_s=round(6+step*.2,2) if ok else None,
                        search_overload=reason in [1,2,3],search_reason=reason,terminal_search_s=round(step*.2,2),
                        final_error_mm=float(distance[i]*1000),depth_mm=float(depth[i]*1000),terminal_trace_samples=step*100)
                active &= ~done
        trace=torch.stack(t.records).cpu().numpy() if t.records else np.empty((0,e.num_envs,len(COLUMNS)))
        np.savez_compressed(out/'search.npz',trace=trace,columns=COLUMNS,actions=np.array(action_log))
        for i,r in enumerate(rows):
            if r['prep_eligible']:
                segment=trace[:r['terminal_trace_samples'],i]
                r['search_peak_axial_N']=float(np.abs(segment[:,COLUMNS.index('Fz')]).max())
        times=[r['success_search_s'] for r in rows if r['success']]
        n=sum(r['prep_eligible'] for r in rows);s=sum(r['success'] for r in rows)
        result=dict(amplitude_deg=amp,total=len(rows),prep_eligible=n,prep_failed=len(rows)-n,
            prep_strict_2s=sum(r['prep_stats']['sustained_2s'] for r in rows),successes=s,overall_success_rate=s/len(rows),
            conditional_success_rate=s/n if n else None,prep_overloads=sum(r['prep_overload'] for r in rows),
            search_overloads=sum(r['search_overload'] for r in rows),success_search_s=dict(min=min(times),median=float(np.median(times)),max=max(times)) if times else None,
            wall_s=time.monotonic()-start,rows=rows)
        save(out/'results.json',result);results.append(result);save(OUT/'results.json',results)
        print('UNSEEN_RESULT', {k:v for k,v in result.items() if k!='rows'},flush=True)
    finally:e.close()
after=integrity();assert before==after
save(OUT/'integrity.json',dict(unchanged=True,hashes=after))
common=[i for i in range(len(cases)) if all(r['rows'][i]['prep_eligible'] for r in results)]
save(OUT/'paired.json',dict(common_eligible=len(common),case_ids=[cases[i]['id'] for i in common],
    successes={str(r['amplitude_deg']):sum(r['rows'][i]['success'] for i in common) for r in results}))
