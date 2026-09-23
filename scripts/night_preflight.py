"""Necessary prelaunch checks. No smoke samples enter the formal bank or budget."""
import copy,json,time,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from mjlab_contact_prep.night_run.environment import BankEnv,prepare_bank,save_json
from mjlab_contact_prep.night_run.jobs import random_cases,make_plan,train_job,evaluate_job,report,case,ROOT
out=ROOT/'evaluation/night_fullcircle_5mm_v1';check=out/'preflight';check.mkdir(exist_ok=True)
mini=random_cases(96,2026092299,'smoke')
bank=prepare_bank(mini,check/'bank',96)
# Reset fidelity, observation ownership, untouched-world state, and physical counter restoration.
e=BankEnv(bank,8,0,7)
try:
 with torch.no_grad():
  snapshot=e.snapshot();obs=e.get_observations()['actor'].clone()
  e.step(torch.full((8,2),.2,device=e.device));e.restore_snapshot(snapshot)
  assert torch.equal(obs,e.get_observations()['actor'])
  for k in ['qpos','qvel','act','ctrl']:assert torch.equal(getattr(e.env.sim.data,k).cpu(),snapshot['physics'][k])
  for k,v in snapshot['controller'].items():assert torch.equal(getattr(e.term.controller,k).cpu(),v)
  e.step(torch.zeros(8,2,device=e.device));before=e.snapshot();e.reset(torch.tensor([0],device=e.device))
  for k in ['qpos','qvel','act','ctrl']:assert torch.equal(getattr(e.env.sim.data,k)[1:].cpu(),before['physics'][k][1:])
  assert torch.equal(e.history[1:].cpu(),before['buffers']['history'][1:])
  assert int(e.inside_samples[0])==0 and int(e.ticks[0])==0
 save_json(check/'reset_check.json',dict(passed=True,checks=['snapshot observation identity','physical/controller state restored','partial reset preserves other worlds','dwell counter reset']))
finally:e.close()
# A real successful trajectory from existing weights must pass the new independent 2ms audit.
pilot_cases=[case(.8,angle,i,i%3,'timing_audit') for i,angle in enumerate([90.,270.])]
pilot_bank=prepare_bank(pilot_cases,check/'timing_bank',2)
pilot=evaluate_job(check,7,0,bank=pilot_bank,checkpoint=ROOT/'evaluation/gpu_expanded_ab_v3/seed_7_amp_0/policy.pt',tag='timing_audit')
assert pilot['successes']>0 and pilot['audit_passed'],'No real successful trajectory exercised corrected timer'
save_json(check/'timing_check.json',dict(passed=True,successes=pilot['successes'],total=pilot['total']))
# Benchmark both arms; complete worker processes isolate Warp/CUDA allocations.
for n in [128,256]:
 for amp in [0.,.05]:
  cmd=[sys.executable,'-u',str(ROOT/'scripts/night_experiment.py'),'worker','--output',str(out),'--stage','benchmark','--envs',str(n),'--amp',str(amp),'--bank',str(bank)]
  with (check/f'benchmark_{n}_{amp:g}.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
# Short exact-order queue: all A seeds, then all B seeds; saved checkpoints loaded for evaluation.
smoke=check/'queue_smoke';smoke.mkdir(exist_ok=True);plan=json.loads((out/'plan.json').read_text())
plan.update(num_envs=8,iterations=2,samples_per_seed=64,train_cases=mini,test_cases=random_cases(8,2026092298,'smoke_test'));save_json(smoke/'plan.json',plan)
order=[]
for amp in [0.,.05]:
 for seed in [7,17,27]:
  result=train_job(smoke,seed,amp,bank=bank,n=8,iterations=2,steps=4)
  order.append(dict(amplitude=amp,seed=seed,transitions=result['transitions']))
# Snapshot restart/load round-trip exercised with full saved payload, without modifying final smoke models.
from rsl_rl.runners import OnPolicyRunner
from mjlab_contact_prep.night_run.jobs import ppo_config
env=BankEnv(bank,8,0,7)
try:
 runner=OnPolicyRunner(env,copy.deepcopy(ppo_config(7,2,4)),None,device=env.device)
 blob=torch.load(smoke/'amp_0/seed_7/policy.pt',weights_only=False,map_location='cpu')
 runner.alg.load(blob,None,True);env.restore_snapshot(blob['environment'])
 assert torch.equal(env.get_observations()['actor'].cpu(),blob['environment']['buffers']['history'].flatten(1))
 with torch.no_grad():env.step(runner.get_inference_policy()(env.get_observations()))
finally:env.close()
# Same independently prepared tiny held-out set for each smoke model; a short timeout tests termination/report paths.
smoke_cases=random_cases(8,2026092298,'smoke_test');testbank=prepare_bank(smoke_cases,smoke/'test_bank',8)
for amp in [0.,.05]:
 for seed in [7,17,27]:evaluate_job(smoke,seed,amp,bank=testbank,horizon=4)
assert report(smoke)
save_json(check/'queue_check.json',dict(passed=True,order=order,save_load_resume_state=True,evaluation_audit=True,report=True,formal_samples=0))
save_json(check/'status.json',dict(status='passed',formal_training_started=False))
print('PREFLIGHT_PASSED',flush=True)
