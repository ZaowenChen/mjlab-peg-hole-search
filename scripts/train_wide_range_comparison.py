"""Predeclared three-seed equal-budget rocking experiment, fixed contact physics."""
import argparse,copy,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from mjlab_contact_prep.distribution_env import DistributionEnv
from mjlab_contact_prep.execution import COLUMNS
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--max-radius-mm',type=int,choices=[10,20],default=10)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--prepare-only',action='store_true',help='Validate independent train/test contact states without training')
a=p.parse_args();OUT=a.output.resolve()
plan=json.loads((ROOT/f'configs/search_range_{a.max_radius_mm}mm.json').read_text())
TRAIN_TIMEOUT=float(plan['training_search_timeout_s'])
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check_frozen():
    f=json.loads((ROOT/'reference/gpu_contact_candidate_v1.json').read_text())
    assert all(digest(ROOT/n)==h for n,h in f['files'].items())
def cases(angles,radii):
    return [dict(id=f'r{r:g}_a{a}',radius_mm=r,angle_deg=a,dx_mm=r*math.cos(math.radians(a)),dy_mm=r*math.sin(math.radians(a))) for a in angles for r in radii]
train=cases(plan['train_angles_deg'],plan['train_radii_mm'])
test=cases(plan['test_angles_deg'],plan['test_radii_mm'])
assert not ({(x['radius_mm'],x['angle_deg']) for x in train}&{(x['radius_mm'],x['angle_deg']) for x in test})
check_frozen();OUT.mkdir(exist_ok=False)
source={str(p.relative_to(ROOT)):digest(p) for folder in ['src','scripts'] for p in (ROOT/folder).rglob('*.py')}
seeds=plan['seeds'];amps=plan['amplitudes_deg'];iterations=plan['iterations']
save(OUT/'manifest.json',dict(train_cases=train,test_cases=test,seeds=seeds,amplitudes_deg=amps,iterations=iterations,
    environments=len(train),steps_per_iteration=32,transitions_per_run=len(train)*32*iterations,source_sha256=source,plan=plan,mode='prepare_only' if a.prepare_only else 'train_and_test',
    initialization='Fresh matched-seed actor/critic, not warm-started; only rocking amplitude differs.',
    preparation='6s, original no-fault plus online-ready entry; strict 2s reported separately. One shared fresh training bank for all arms; invalid training cases explicitly excluded from both arms. Test bank generated independently from test positions AFTER all training, identical paired snapshots for all checkpoints; never training caches.',
    evaluation=f'All {len(test)} predeclared test positions retained; original shallow-capture threshold unchanged. Training timeout {TRAIN_TIMEOUT}s; test case timeout 8 * max(training maximum radius, case radius) seconds. Report interpolation and extrapolation separately.',
    comparison='Primary: paired per-seed overall success rates. Secondary: overloads and mean case-budget-normalized completion time (1 for failures, excluding preparation); also raw capped times by radius; successful-only times descriptive. Three seeds give limited uncertainty; no definitive superiority based on pooled episodes alone. No C force-ablation arm this round.'))
if a.prepare_only:
    summaries={}
    for split,items in [('training_bank',train),('test_bank',test)]:
        folder=OUT/split;folder.mkdir();env=DistributionEnv(items,folder)
        try:
            summaries[split]=dict(total=len(items),eligible=int(env.eligible.sum()),failed=int((~env.eligible).sum()),
                strict_2s=sum(x['stats']['sustained_2s'] for x in env.preparation),by_radius={str(rad):dict(total=sum(x['radius_mm']==rad for x in env.preparation),eligible=sum(x['eligible'] for x in env.preparation if x['radius_mm']==rad)) for rad in sorted({x['radius_mm'] for x in items})})
            save(OUT/'preparation_summary.json',summaries)
        finally:env.close()
    check_frozen();save(OUT/'status.json',dict(stage='preparation_complete',training_launched=False));print(summaries,flush=True)
    raise SystemExit(0)
base=json.loads((ROOT/'evaluation/gpu_ppo_pilot_v2/amp_0/config.json').read_text())['ppo']
training=[];(OUT/'training_bank').mkdir();env=DistributionEnv(train,OUT/'training_bank')
env.max_episode_length=round(TRAIN_TIMEOUT/env.dt)
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
        limits=torch.tensor([8*max(a.max_radius_mm,c['radius_mm']) for c in test],device=env.device)
        active=env.eligible.clone();dwell=torch.zeros(env.num_envs,device=env.device);rows=[]
        for c,p in zip(test,env.preparation):
            rows.append(dict(**c,prep_eligible=p['eligible'],prep_reason=p['reason'],prep_overload=p['overload'],success=False,
                outcome='pending' if p['eligible'] else 'contact_preparation_failed',success_s=None,search_overload=False,within_training_radius=c['radius_mm']<=a.max_radius_mm,
                timeout_s=8*max(a.max_radius_mm,c['radius_mm']),capped_completion_s=8*max(a.max_radius_mm,c['radius_mm']),normalized_completion=1.,terminal_samples=0))
        t=env.term;t.record=True;t.records.clear();t.probe_records.clear();actions_log=[]
        with torch.inference_mode():
            for step in range(1,int(limits.max().item()/env.dt)+1):
                if not active.any():break
                actions=policy(env.get_observations()).clamp(-1,1);actions[~active]=0;actions_log.append(actions.cpu().numpy().copy())
                for _ in range(5):
                    env.env.step(actions);dist,depth=env.metrics()
                    inside=(dist<=.00015)&(depth>=.0001)&(t.controller.reason==0)
                    dwell=torch.where(inside,dwell+env.env.step_dt,0)
                env.last_action[:]=actions;env.history[:]=torch.roll(env.history,-1,1);env.history[:,-1]=env._frame()
                fault=t.controller.reason!=0;success=(dwell>=.2-1e-6)&~fault;done=active&(success|fault|(step*env.dt>=limits-1e-5))
                for i in torch.nonzero(done).flatten().tolist():
                    reason=int(t.controller.reason[i]);ok=bool(success[i]);elapsed=round(step*.2,2)
                    rows[i].update(success=ok,outcome='success' if ok else ('search_fault' if reason else 'search_timeout'),
                        success_s=elapsed if ok else None,capped_completion_s=elapsed if ok else rows[i]['timeout_s'],normalized_completion=elapsed/rows[i]['timeout_s'] if ok else 1.,search_reason=reason,
                        search_overload=reason in [1,2,3],final_error_mm=float(dist[i]*1000),depth_mm=float(depth[i]*1000),terminal_samples=step*100)
                active &= ~done
        trace=torch.stack(t.records).cpu().numpy();np.savez_compressed(out/'test_trace.npz',trace=trace,columns=COLUMNS,actions=np.array(actions_log))
        t.records.clear();t.probe_records.clear();t.record=False
        times=[r['success_s'] for r in rows if r['success']];eligible=sum(r['prep_eligible'] for r in rows);count=sum(r['success'] for r in rows)
        result=dict(seed=seed,amplitude_deg=amp,total=len(rows),prep_failed=len(rows)-eligible,successes=count,
            overall_success_rate=count/len(rows),conditional_success_rate=count/eligible if eligible else None,
            search_overloads=sum(r['search_overload'] for r in rows),prep_overloads=sum(r['prep_overload'] for r in rows),
            mean_capped_completion_s=float(np.mean([r['capped_completion_s'] for r in rows])),
            median_success_s=float(np.median(times)) if times else None,mean_normalized_completion=float(np.mean([r['normalized_completion'] for r in rows])),by_scope={name:dict(total=sum(r['within_training_radius']==inside for r in rows),successes=sum(r['success'] for r in rows if r['within_training_radius']==inside)) for name,inside in [('within_training_radius',True),('radius_extrapolation',False)]},rows=rows)
        results.append(result);save(out/'test_results.json',result);save(OUT/'results.json',results)
        print('EXPANDED_TEST_DONE',{k:v for k,v in result.items() if k!='rows'},flush=True)
finally:env.close()
check_frozen();assert all(digest(ROOT/n)==h for n,h in source.items())
save(OUT/'integrity.json',dict(frozen_physics_unchanged=True,experiment_sources_unchanged=True))
save(OUT/'status.json',dict(stage='complete',training_runs=6,test_runs=6))
