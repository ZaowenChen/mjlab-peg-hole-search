"""Bounded RSL-RL PPO comparison on admitted GPU handoff cases only."""
import argparse,copy,hashlib,json,math,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from mjlab.rl import RslRlOnPolicyRunnerCfg
from rsl_rl.runners import OnPolicyRunner
from mjlab_contact_prep.search_env import SearchEnv

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--handoff',type=Path,default=Path('evaluation/gpu_handoff_frozen_v1'));p.add_argument('--iterations',type=int,default=16);p.add_argument('--num-envs',type=int,default=32);p.add_argument('--amplitudes',type=float,nargs='+',default=[0,.05]);p.add_argument('--seed',type=int,default=7);a=p.parse_args()
if a.iterations<1 or a.num_envs<4:p.error('positive iterations and >=4 envs required')
decision=json.loads((a.handoff/'decision.json').read_text());cases=decision['ppo_cases'];radius=decision['ppo_radius_mm']
if not cases:raise RuntimeError('No admitted GPU cases: training blocked by handoff results.')
angles=[math.degrees(math.atan2(c['dy_mm'],c['dx_mm'])) for c in cases]
root=Path(__file__).resolve().parents[1];freeze=json.loads((root/'reference/gpu_contact_candidate_v1.json').read_text())
for name,sha in freeze['files'].items():
 if hashlib.sha256((root/name).read_bytes()).hexdigest()!=sha:raise RuntimeError(f'Frozen contact implementation changed: {name}')
a.output.mkdir(parents=True,exist_ok=False)
def save(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def evaluate(env,policy):
 env.completed=[];env.reset(torch.arange(env.num_envs,device=env.device),randomize=False);first={}
 with torch.inference_mode():
  for _ in range(env.max_episode_length):
   obs=env.get_observations();actions=torch.zeros(env.num_envs,2,device=env.device) if policy is None else policy(obs)
   env.step(actions)
   for row in env.completed:first.setdefault(row['world_id'],row)
   if len(first)==env.num_envs:break
 rows=list(first.values())
 assert len(rows)==env.num_envs,'evaluation missed terminal results'
 return dict(episodes=len(rows),successes=sum(x['success'] for x in rows),faults=sum(x['fault'] for x in rows),mean_final_error_mm=float(np.mean([x['final_error_mm'] for x in rows])),rows=rows)
summary=[]
for amp in a.amplitudes:
 out=a.output/f'amp_{amp:g}';out.mkdir();save(out/'status.json',dict(status='preparing_contact_bank'))
 env=SearchEnv(a.num_envs,radius,amp,a.seed,angles_deg=angles)
 try:
  cfg=asdict(RslRlOnPolicyRunnerCfg());cfg.update(seed=a.seed,num_steps_per_env=32,max_iterations=a.iterations,save_interval=8,logger='tensorboard',experiment_name='gpu_search_pilot',run_name=f'amp_{amp:g}')
  for key in ['actor','critic']:
   cfg[key].update(hidden_dims=[128,128],obs_normalization=True)
   for n in ['cnn_cfg','rnn_type','rnn_hidden_dim','rnn_num_layers']:cfg[key].pop(n,None)
  cfg['actor']['distribution_cfg']['init_std']=.5;cfg['critic'].pop('distribution_cfg',None)
  cfg['algorithm'].update(num_learning_epochs=4,num_mini_batches=4,learning_rate=3e-4,gamma=.995,rnd_cfg=None,symmetry_cfg=None)
  hashes={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for folder in ['src','scripts'] for f in (root/folder).rglob('*') if f.is_file() and '__pycache__' not in f.parts}
  save(out/'config.json',dict(environment=env.cfg,ppo=cfg,source_sha256=hashes,candidate=freeze['archive_sha256'],admission_cases=cases,scope='Same admitted two-direction task; pilot, not all-angle or full-insertion validation'))
  runner=OnPolicyRunner(env,copy.deepcopy(cfg),str(out/'logs'),device=env.device)
  save(out/'status.json',dict(status='baseline_evaluation'))
  zero=evaluate(env,None);initial=evaluate(env,runner.get_inference_policy())
  save(out/'before.json',dict(zero=zero,initial_policy=initial))
  env.completed=[];env.reset(torch.arange(env.num_envs,device=env.device),randomize=False);torch.manual_seed(a.seed)
  save(out/'status.json',dict(status='training',iterations=a.iterations))
  start=time.monotonic();runner.learn(a.iterations,init_at_random_ep_len=False);wall=time.monotonic()-start
  runner.save(str(out/'policy.pt'),infos={'scope':env.cfg,'candidate':freeze['archive_sha256']})
  training=list(env.completed);result=evaluate(env,runner.get_inference_policy())
  save(out/'training_episodes.json',dict(rows=training))
  row=dict(amplitude_deg=amp,iterations=a.iterations,transitions=a.num_envs*32*a.iterations,training_wall_s=wall,zero=zero,initial_policy=initial,trained_policy=result)
  save(out/'results.json',row);summary.append(row);save(out/'status.json',dict(status='complete',learned_success_is_not_guaranteed=True));save(a.output/'results.json',dict(rows=summary))
  print('PILOT_RESULT',amp,'zero',zero['successes'],'initial',initial['successes'],'trained',result['successes'],'/',result['episodes'],flush=True)
 finally:env.close()
