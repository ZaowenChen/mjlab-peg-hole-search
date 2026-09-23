"""Predeclared three-seed equal-budget rocking experiment, fixed contact physics."""
import copy,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from mjlab_contact_prep.distribution_env import DistributionEnv
from mjlab_contact_prep.execution import COLUMNS
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'evaluation/gpu_expanded_ab_v3'
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check_frozen():
    f=json.loads((ROOT/'reference/gpu_contact_candidate_v1.json').read_text())
    assert all(digest(ROOT/n)==h for n,h in f['files'].items())
def cases(angles,radii):
    return [dict(id=f'r{r:g}_a{a}',radius_mm=r,angle_deg=a,dx_mm=r*math.cos(math.radians(a)),dy_mm=r*math.sin(math.radians(a))) for a in angles for r in radii]
train=cases([c+d for c in [90,270] for d in [-35,-25,-15,-5,5,15,25,35]],[.6,.8,1.,1.2,2.,5.])
test=cases([c+d for c in [90,270] for d in [-30,-18,-6,6,18,30]],[.7,.9,1.1,1.3,2.,5.])
assert not ({(x['radius_mm'],x['angle_deg']) for x in train}&{(x['radius_mm'],x['angle_deg']) for x in test})
check_frozen();OUT.mkdir(exist_ok=False)
source={str(p.relative_to(ROOT)):digest(p) for folder in ['src','scripts'] for p in (ROOT/folder).rglob('*.py')}
seeds=[7,17,27];amps=[0.,.05];iterations=48
save(OUT/'manifest.json',dict(train_cases=train,test_cases=test,seeds=seeds,amplitudes_deg=amps,iterations=iterations,
    environments=len(train),steps_per_iteration=32,transitions_per_run=len(train)*32*iterations,source_sha256=source,
    initialization='Fresh matched-seed actor/critic, not warm-started; only rocking amplitude differs.',
    preparation='6s, original no-fault plus online-ready entry; strict 2s reported separately. One shared fresh training bank for all arms; invalid training cases explicitly excluded from both arms. Test bank generated independently from test positions AFTER all training, identical paired snapshots for all checkpoints; never training caches.',
    evaluation='Final fixed-budget checkpoint only; no best-checkpoint selection. All 48 test positions retained. New angles at radii 0.7/0.9/1.1/1.3/2/5mm; report small and large radius strata and each direction separately. Shallow capture at <=0.15mm error and >=0.1mm depth for 0.2s, max40s; no fault.',
    comparison='Primary: paired per-seed overall success rates. Secondary: overloads and mean capped completion time (40s for failures, excluding preparation); successful-only times descriptive. Three seeds give limited uncertainty; no definitive superiority based on pooled episodes alone. No C force-ablation arm this round.'))
base=json.loads((ROOT/'evaluation/gpu_ppo_pilot_v2/amp_0/config.json').read_text())['ppo']
training=[];(OUT/'training_bank').mkdir();env=DistributionEnv(train,OUT/'training_bank')
try:
    assert len(env.valid_ids)>0
    for seed in seeds:
        for amp in amps:
            name=f'seed_{seed}_amp_{amp:g}';out=OUT/name;out.mkdir()
            env.set_arm(amp,seed,training=True)
            cfg=copy.deepcopy(base);cfg.update(seed=seed,max_iterations=iterations,save_interval=16,run_name=name)
            save(out/'config.json',dict(ppo=cfg,amplitude_deg=amp,seed=seed,train_eligible=len(env.valid_ids)))
            save(OUT/'status.json',dict(stage='training',run=name,completed_runs=len(training),total_runs=6))
            runner=OnPolicyRunner(env,copy.deepcopy(cfg),str(out/'logs'),device=env.device)
            initial={k:v.detach().cpu().clone() for k,v in runner.alg.actor.state_dict().items()}
            if amp==0.:reference_initial=initial
            else:assert all(torch.equal(v,reference_initial[k]) for k,v in initial.items()),'Unmatched actor initialization'
            start=time.monotonic();runner.learn(iterations,init_at_random_ep_len=False)
            runner.save(str(out/'policy.pt'),infos={'seed':seed,'amplitude_deg':amp,'iterations':iterations})
            row=dict(seed=seed,amplitude_deg=amp,wall_s=time.monotonic()-start,episodes=len(env.completed),checkpoint_sha256=digest(out/'policy.pt'))
            save(out/'training_episodes.json',env.completed);training.append(row);save(OUT/'training.json',training)
            print('EXPANDED_TRAIN_DONE',row,flush=True)
            del runner
finally:env.close()

# Independent held-out preparation shared across policies for a matched comparison.
(OUT/'test_bank').mkdir();env=DistributionEnv(test,OUT/'test_bank');results=[]
try:
    for run in training:
        seed=run['seed'];amp=run['amplitude_deg'];out=OUT/f'seed_{seed}_amp_{amp:g}'
        env.set_arm(amp,seed,training=False);cfg=json.loads((out/'config.json').read_text())['ppo']
        runner=OnPolicyRunner(env,copy.deepcopy(cfg),None,device=env.device)
        runner.load(str(out/'policy.pt'),map_location=env.device);policy=runner.get_inference_policy()
        save(OUT/'status.json',dict(stage='testing',seed=seed,amplitude=amp,completed_tests=len(results)))
        active=env.eligible.clone();dwell=torch.zeros(env.num_envs,device=env.device);rows=[]
        for c,p in zip(test,env.preparation):
            rows.append(dict(**c,prep_eligible=p['eligible'],prep_reason=p['reason'],prep_overload=p['overload'],success=False,
                outcome='pending' if p['eligible'] else 'contact_preparation_failed',success_s=None,search_overload=False,
                capped_completion_s=40.,terminal_samples=0))
        t=env.term;t.record=True;t.records.clear();t.probe_records.clear();actions_log=[]
        with torch.inference_mode():
            for step in range(1,201):
                if not active.any():break
                a=policy(env.get_observations()).clamp(-1,1);a[~active]=0;actions_log.append(a.cpu().numpy().copy())
                for _ in range(5):
                    env.env.step(a);dist,depth=env.metrics()
                    inside=(dist<=.00015)&(depth>=.0001)&(t.controller.reason==0)
                    dwell=torch.where(inside,dwell+env.env.step_dt,0)
                env.last_action[:]=a;env.history[:]=torch.roll(env.history,-1,1);env.history[:,-1]=env._frame()
                fault=t.controller.reason!=0;success=(dwell>=.2-1e-6)&~fault;done=active&(success|fault|(step==200))
                for i in torch.nonzero(done).flatten().tolist():
                    reason=int(t.controller.reason[i]);ok=bool(success[i]);elapsed=round(step*.2,2)
                    rows[i].update(success=ok,outcome='success' if ok else ('search_fault' if reason else 'search_timeout'),
                        success_s=elapsed if ok else None,capped_completion_s=elapsed if ok else 40.,search_reason=reason,
                        search_overload=reason in [1,2,3],final_error_mm=float(dist[i]*1000),depth_mm=float(depth[i]*1000),terminal_samples=step*100)
                active &= ~done
        trace=torch.stack(t.records).cpu().numpy();np.savez_compressed(out/'test_trace.npz',trace=trace,columns=COLUMNS,actions=np.array(actions_log))
        t.records.clear();t.probe_records.clear();t.record=False
        times=[r['success_s'] for r in rows if r['success']];eligible=sum(r['prep_eligible'] for r in rows);count=sum(r['success'] for r in rows)
        result=dict(seed=seed,amplitude_deg=amp,total=len(rows),prep_failed=len(rows)-eligible,successes=count,
            overall_success_rate=count/len(rows),conditional_success_rate=count/eligible if eligible else None,
            search_overloads=sum(r['search_overload'] for r in rows),prep_overloads=sum(r['prep_overload'] for r in rows),
            mean_capped_completion_s=float(np.mean([r['capped_completion_s'] for r in rows])),
            median_success_s=float(np.median(times)) if times else None,rows=rows)
        results.append(result);save(out/'test_results.json',result);save(OUT/'results.json',results)
        print('EXPANDED_TEST_DONE',{k:v for k,v in result.items() if k!='rows'},flush=True)
finally:env.close()
check_frozen();assert all(digest(ROOT/n)==h for n,h in source.items())
save(OUT/'integrity.json',dict(frozen_physics_unchanged=True,experiment_sources_unchanged=True))
save(OUT/'status.json',dict(stage='complete',training_runs=6,test_runs=6))
