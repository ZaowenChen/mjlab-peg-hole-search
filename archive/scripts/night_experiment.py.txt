"""Sequential background queue for the explicitly requested full-circle 0.5–5mm A/B experiment."""
import argparse,json,os,sys,time,subprocess,traceback
from pathlib import Path
import numpy as np
import torch
from mjlab_contact_prep.night_run.environment import BankEnv,prepare_bank,save_json
from mjlab_contact_prep.night_run.jobs import make_plan,train_job,evaluate_job,report,ppo_config,verify_frozen

def worker(a):
    out=a.output.resolve();plan=json.loads((out/'plan.json').read_text());verify_frozen(plan)
    if a.stage=='bank':
        bank=prepare_bank(plan['train_cases'],out/'training_bank',plan.get('num_envs',256))
        summary=json.loads((out/'training_bank/summary.json').read_text())
        if not plan.get('smoke',False) and any(x['eligible']==0 for x in summary['by_sector']):raise RuntimeError('Full-circle coverage lost: an entire radius-band/30-degree sector has no usable contact. No training launched.')
    elif a.stage=='train':train_job(out,a.seed,a.amp,steps=plan.get('steps_per_update',32))
    elif a.stage=='test-bank':prepare_bank(plan['test_cases'],out/'test_bank',plan['num_envs'])
    elif a.stage=='test':evaluate_job(out,a.seed,a.amp,horizon=plan.get('evaluation_horizon_steps',200))
    elif a.stage=='report':
        if not report(out):raise RuntimeError('Formal comparison incomplete or audit failed')
    elif a.stage=='benchmark':
        from rsl_rl.runners import OnPolicyRunner
        import copy
        env=BankEnv(a.bank,a.envs,a.amp,7);cfg=ppo_config(7,3);runner=OnPolicyRunner(env,copy.deepcopy(cfg),None,device=env.device)
        elapsed=[];orig=runner.logger.log
        def timing(*args,**kw):elapsed.append(kw['collect_time']+kw['learn_time']);orig(*args,**kw)
        runner.logger.log=timing
        try:
            runner.learn(3,init_at_random_ep_len=False)
            used=float(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()[0])
            result=dict(num_envs=a.envs,amplitude=a.amp,seconds_per_update=float(np.mean(elapsed[1:])),decisions_per_second=a.envs*32/float(np.mean(elapsed[1:])),gpu_memory_used_MiB=used,finite=True)
            save_json(out/f'benchmark_{a.envs}_{a.amp:g}.json',result)
        finally:env.close()

def queue(a):
    out=a.output.resolve();plan=json.loads((out/'plan.json').read_text());verify_frozen(plan)
    # File lock prevents accidental concurrent GPU queues for this experiment.
    import fcntl
    lock=(out/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock.write(str(os.getpid()));lock.flush();save_json(out/'queue_pid.json',dict(pid=os.getpid(),started_at=time.time()))
    schedule=[('bank',None)]+[('train',x) for x in plan['queue_order']]+[('test-bank',None)]+[('test',x) for x in plan['queue_order']]+[('report',None)]
    failed=[];stages=[]
    for index,(stage,job) in enumerate(schedule):
        if stage in ['test-bank','test'] and any(x['stage'] in ['bank','train'] for x in failed):
            stages.append(dict(stage=stage,job=job,status='skipped_incomplete_training'));continue
        if stage=='bank' and failed:break
        label=stage if job is None else f"{stage}_amp_{job['amplitude']:g}_seed_{job['seed']}"
        successful=False
        for attempt in range(3):
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'worker','--output',str(out),'--stage',stage]
            if job:command+=['--amp',str(job['amplitude']),'--seed',str(job['seed'])]
            save_json(out/'queue_status.json',dict(status='running',stage=stage,job=job,attempt=attempt+1,stage_index=index,total_stages=len(schedule),failed=failed,updated_at=time.time()))
            logpath=out/'queue_logs'/f'{label}_attempt_{attempt+1}.log';logpath.parent.mkdir(exist_ok=True)
            warnings=0
            with logpath.open('w',buffering=1) as log:
                child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                save_json(out/'active_worker.json',dict(pid=child.pid,stage=stage,job=job,log=str(logpath)))
                for line in child.stdout:
                    log.write(line)
                    if 'EPA' in line and ('Warning' in line or 'overflow' in line):
                        warnings+=1
                        with (out/'warnings.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),stage=stage,job=job,attempt=attempt+1,line=line.strip()))+'\n')
                code=child.wait()
            stages.append(dict(stage=stage,job=job,attempt=attempt+1,returncode=code,EPA_warning_lines=warnings,log=str(logpath)))
            save_json(out/'stage_history.json',stages)
            if code==0:successful=True;break
            if code==42:break # Numerical failures are never retried automatically.
            time.sleep(2)
        if not successful:
            failed.append(dict(stage=stage,job=job,code=code))
            if stage in ['bank','test-bank']:break
    if failed:report(out)
    save_json(out/'queue_status.json',dict(status='complete' if not failed else 'incomplete',failed=failed,updated_at=time.time(),report=str(out/'REPORT.md')))
    verify_frozen(plan)
    save_json(out/'integrity.json',dict(frozen_physics_unchanged=True,frozen_experiment_unchanged=True))
    return 0 if not failed else 1

p=argparse.ArgumentParser();p.add_argument('mode',choices=['init','worker','queue']);p.add_argument('--output',type=Path,required=True)
p.add_argument('--stage',choices=['bank','train','test-bank','test','report','benchmark']);p.add_argument('--seed',type=int,default=7);p.add_argument('--amp',type=float,default=0)
p.add_argument('--bank',type=Path);p.add_argument('--envs',type=int,default=256);a=p.parse_args()
try:
    if a.mode=='init':make_plan(a.output)
    elif a.mode=='worker':worker(a)
    else:sys.exit(queue(a))
except Exception as exc:
    traceback.print_exc()
    numerical=isinstance(exc,FloatingPointError) or any(w in str(exc).lower() for w in ['nan','non-finite','nonfinite'])
    sys.exit(42 if numerical else 1)
