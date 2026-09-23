import copy, hashlib, json, os, random, time, math
from pathlib import Path
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from .environment import BankEnv,prepare_bank,save_json,AUDIT_COLUMNS
ROOT=Path(__file__).resolve().parents[3]

def case(radius,angle,index,bucket,kind):
    return dict(id=f'{kind}_{index:05d}',radius_mm=float(radius),angle_deg=float(angle),dx_mm=float(radius*math.cos(math.radians(angle))),dy_mm=float(radius*math.sin(math.radians(angle))),bucket=bucket,kind=kind)

def random_cases(n,seed,kind):
    rng=np.random.default_rng(seed);bands=[(.5,1.5),(1.5,3.),(3.,5.)];out=[]
    for i in range(n):
        b=i%3;out.append(case(rng.uniform(*bands[b]),rng.uniform(0,360),i,b,kind))
    return out

def make_plan(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    train=random_cases(4096,2026092201,'train')
    test=[case(r,360*i/32,j*32+i,0 if r<=1.5 else (1 if r<=3 else 2),'grid') for j,r in enumerate([1.,2.,5.]) for i in range(32)]
    test+=random_cases(128,2026092302,'heldout_random')
    assert not ({(x['dx_mm'],x['dy_mm']) for x in train}&{(x['dx_mm'],x['dy_mm']) for x in test})
    plan=dict(train_cases=train,test_cases=test,seeds=[7,17,27],amplitudes=[0.,.05],deadline='2026-09-23T08:00:00+08:00',
      horizon_steps=200,preparation_s=6,steps_per_update=32,target_samples=2097152,fallback_samples=1572864,
      queue_order=[dict(amplitude=amp,seed=seed) for amp in [0.,.05] for seed in [7,17,27]],
      safeguards=dict(retries=2,nonfinite='save diagnostics, stop affected task without retry',EPA='log stage and count only; no automatic physics changes'),
      success='Physics-sampled: radial <=0.15mm, depth >=0.1mm, no fault; 101 consecutive 2ms samples span 0.2s. Same training and evaluation implementation; independent trace audit.',
      sampling='Uniform within each radial band, uniform full-circle angle. Select each eligible radius band with probability 1/3; log every rejected preparation. Never exclude test failures.',
      test_selection='Final equal-budget checkpoints only. Test bank constructed after all training, never reused from training.',
      physics_frozen=json.loads((ROOT/'reference/gpu_contact_candidate_v1.json').read_text())['files'])
    save_json(out/'plan.json',plan);return plan

def verify_frozen(plan):
    for n,h in plan['physics_frozen'].items():
        if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=h:raise RuntimeError('Frozen physics changed: '+n)
    for n,h in plan.get('experiment_source_sha256',{}).items():
        if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=h:raise RuntimeError('Frozen experiment source changed: '+n)

def ppo_config(seed,iterations,steps=32):
    cfg=json.loads((ROOT/'evaluation/gpu_ppo_pilot_v2/amp_0/config.json').read_text())['ppo']
    cfg.update(seed=seed,max_iterations=iterations,num_steps_per_env=steps,save_interval=1,run_name=f'formal_seed_{seed}',check_for_nan=True)
    # Retain the already validated PPO settings, including lam=0.95 and gamma=0.995.
    return cfg

def model_hash(actor):
    h=hashlib.sha256()
    for k,v in actor.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def train_job(out,seed,amp,*,bank=None,n=None,iterations=None,steps=32,resume=True):
    out=Path(out);plan=json.loads((out/'plan.json').read_text());verify_frozen(plan)
    folder=out/f'amp_{amp:g}'/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():return json.loads((folder/'complete.json').read_text())
    n=n or plan['num_envs'];iterations=iterations or plan['iterations'];bank=bank or out/'training_bank/bank.pt'
    torch.manual_seed(seed);env=BankEnv(bank,n,amp,seed);cfg=ppo_config(seed,iterations,steps)
    runner=OnPolicyRunner(env,copy.deepcopy(cfg),str(folder/'logs'),device=env.device)
    initial_hash=model_hash(runner.alg.actor);save_json(folder/'initialization.json',dict(seed=seed,amplitude=amp,actor_sha256=initial_hash))
    counterpart=out/'amp_0'/f'seed_{seed}'/'initialization.json'
    if amp and counterpart.exists():assert json.loads(counterpart.read_text())['actor_sha256']==initial_hash,'Initial actor mismatch'
    save_json(folder/'config.json',dict(ppo=cfg,num_envs=n,iterations=iterations,transitions=n*steps*iterations))
    wall_start=time.monotonic();last_progress={};checkpoint_restored=False
    def checkpoint(path,infos=None):
        if (runner.current_learning_iteration+1)%16 and runner.current_learning_iteration+1!=iterations:return
        payload=runner.alg.save();payload.update(iter=runner.current_learning_iteration,next_iteration=runner.current_learning_iteration+1,
            infos=infos,adaptive_learning_rate=runner.alg.learning_rate,environment=env.snapshot(),rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate())
        dest=folder/f'checkpoint_{payload["next_iteration"]:04d}.pt';tmp=dest.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(dest)
        save_json(folder/'checkpoint.json',dict(path=str(dest),next_iteration=payload['next_iteration']))
    runner.save=checkpoint
    try:
        if resume:
            for f in sorted(folder.glob('checkpoint_*.pt'),reverse=True):
                try:blob=torch.load(f,weights_only=False,map_location='cpu')
                except Exception:continue
                runner.alg.load(blob,None,True);runner.alg.learning_rate=blob.get('adaptive_learning_rate',blob['optimizer_state_dict']['param_groups'][0]['lr']);runner.current_learning_iteration=blob['next_iteration']
                env.restore_snapshot(blob['environment']);torch.set_rng_state(blob['rng_cpu']);torch.cuda.set_rng_state_all(blob['rng_cuda'])
                np.random.set_state(blob['numpy_rng']);random.setstate(blob['python_rng']);checkpoint_restored=True;break
        original_log=runner.logger.log
        def progress(*args,**kw):
            original_log(*args,**kw)
            finished=int(kw['it'])+1;duration=float(kw['collect_time']+kw['learn_time'])
            state=dict(stage='training',amplitude=amp,seed=seed,updates=finished,total_updates=iterations,
              samples=finished*n*steps,total_samples=iterations*n*steps,iteration_s=duration,
              model_remaining_s=(iterations-finished)*duration,updated_at=time.time(),resumed=checkpoint_restored,
              terminated_episodes=len(env.completed),fault_episodes=sum(x['reason']!=0 for x in env.completed))
            state['queue_remaining_training_s']=state['model_remaining_s']+sum(plan.get('benchmark_seconds_per_update',{}).get(str(j['amplitude']),duration)*iterations for j in plan['queue_order'] if (j['amplitude']!=amp or j['seed']!=seed) and not (out/f"amp_{j['amplitude']:g}"/f"seed_{j['seed']}"/'complete.json').exists())
            last_progress.update(state);save_json(folder/'progress.json',state);save_json(out/'progress.json',state)
        runner.logger.log=progress
        if runner.current_learning_iteration<iterations:runner.learn(iterations-runner.current_learning_iteration,init_at_random_ep_len=False)
        # Latest checkpoint includes optimizer, normalization, all reset/history state and RNGs.
        latest=json.loads((folder/'checkpoint.json').read_text())
        assert latest['next_iteration']==iterations
        final=folder/'policy.pt'
        if final.exists():final.unlink()
        os.link(latest['path'],final)
        save_json(folder/'episodes.json',env.completed)
        result=dict(seed=seed,amplitude=amp,iterations=iterations,num_envs=n,transitions=iterations*n*steps,
          wall_s_this_attempt=time.monotonic()-wall_start,resumed=checkpoint_restored,final_checkpoint=str(final),episodes=len(env.completed),
          faults=sum(x['reason']!=0 for x in env.completed),fault_reasons={str(i):sum(x['reason']==i for x in env.completed) for i in range(1,10)},initial_actor_sha256=initial_hash)
        save_json(folder/'complete.json',result);return result
    except Exception as exc:
        numerical=isinstance(exc,FloatingPointError) or any(w in str(exc).lower() for w in ['nan','non-finite','nonfinite'])
        torch.save(env.snapshot(),folder/'failure_state.pt');save_json(folder/'failure.json',dict(error=repr(exc),numerical=numerical,progress=last_progress))
        raise
    finally:env.close()

def evaluate_job(out,seed,amp,*,bank=None,checkpoint=None,horizon=200,tag=None):
    out=Path(out);folder=out/f'amp_{amp:g}'/f'seed_{seed}';destination=folder/(tag or 'evaluation');destination.mkdir(parents=True,exist_ok=True)
    bank=bank or out/'test_bank/bank.pt';raw=torch.load(bank,weights_only=False,map_location='cpu');count=len(raw['rows'])
    env=BankEnv(bank,count,amp,seed,autoreset=False,audit=True,horizon=horizon)
    cfg=ppo_config(seed,1);runner=OnPolicyRunner(env,copy.deepcopy(cfg),None,device=env.device)
    runner.load(str(checkpoint or folder/'policy.pt'),map_location=env.device);policy=runner.get_inference_policy()
    active=torch.tensor([r['eligible'] for r in env.rows],device=env.device);rows=[]
    for r in env.rows:rows.append(dict(**r,prep_eligible=r['eligible'],success=False,outcome='pending' if r['eligible'] else 'preparation_failed',search_s=None,total_s=None if r['eligible'] else 6.,terminal_samples=0,search_reason=None,capped_time_s=horizon*.2))
    try:
        with torch.inference_mode():
            for step in range(1,horizon+1):
                if not active.any():break
                actions=policy(env.get_observations()).clamp(-1,1);actions[~active]=0
                _,_,done,_=env.step(actions);dist,depth=env.metrics()
                for i in torch.nonzero(active&done.bool()).flatten().tolist():
                    success=bool(env.last_success[i]);reason=int(env.term.controller.reason[i]);elapsed=round(step*.2,3)
                    rows[i].update(success=success,outcome='success' if success else ('fault' if reason else 'timeout'),search_s=elapsed,total_s=6+elapsed,
                      terminal_samples=int(env.ticks[i]),search_reason=reason,capped_time_s=elapsed if success else horizon*.2,
                      final_error_mm=float(dist[i]*1000),final_depth_mm=float(depth[i]*1000))
                active &= ~done.bool()
        trace=torch.stack(env.audit_records).cpu().numpy() if env.audit_records else np.empty((0,count,len(AUDIT_COLUMNS)))
        np.savez_compressed(destination/'trajectory.npz',trace=trace,columns=AUDIT_COLUMNS)
        # Independent reconstruction from saved geometry/reason; does not use the online sample counter.
        reloaded=np.load(destination/'trajectory.npz');t=reloaded['trace'];columns=list(reloaded['columns']);audits=[]
        for i,row in enumerate(rows):
            if not row['prep_eligible']:continue
            x=t[:row['terminal_samples'],i];assert np.isfinite(x).all()
            mask=(x[:,columns.index('radial_m')]<=.00015)&(x[:,columns.index('depth_m')]>=.0001)&(x[:,columns.index('reason')]==0)
            bad=np.flatnonzero(~mask);tail=len(mask)-(int(bad[-1])+1 if len(bad) else 0)
            duration=max(0,tail-1)*.002
            expected=duration>=.2-1e-9 and int(x[-1,columns.index('reason')])==0
            assert row['success']==expected,(row['id'],row['success'],duration)
            assert int(x[-1,columns.index('reason')])==row['search_reason']
            if len(x)>1:assert np.allclose(np.diff(x[:,0]),.002,atol=4e-6)
            row['audited_final_dwell_s']=duration;audits.append(True)
        result=dict(seed=seed,amplitude=amp,total=count,eligible=sum(x['prep_eligible'] for x in rows),successes=sum(x['success'] for x in rows),
          preparation_failures=sum(not x['prep_eligible'] for x in rows),faults=sum(x['outcome']=='fault' for x in rows),
          mean_capped_time_s=float(np.mean([x['capped_time_s'] for x in rows])),mean_observed_total_s=float(np.mean([x['total_s'] for x in rows])),audit_passed=all(audits),rows=rows)
        save_json(destination/'results.json',result);return result
    finally:env.close()

def report(out):
    out=Path(out);plan=json.loads((out/'plan.json').read_text());runs=[]
    for job in plan['queue_order']:
        f=out/f"amp_{job['amplitude']:g}"/f"seed_{job['seed']}"/'evaluation/results.json'
        if f.exists():runs.append(json.loads(f.read_text()))
    complete=len(runs)==6 and all(x['audit_passed'] and x['total']==len(plan['test_cases']) and {r['id'] for r in x['rows']}=={r['id'] for r in plan['test_cases']} for x in runs)
    for j in plan['queue_order']:
        completion=out/f"amp_{j['amplitude']:g}"/f"seed_{j['seed']}"/'complete.json'
        complete=complete and completion.exists() and json.loads(completion.read_text())['transitions']==plan['samples_per_seed']
    lines=['# 全方向 0.5–5 mm 正式 A/B 实验','',f"状态：{'全部完成并通过轨迹核对' if complete else '未完整完成；不得据此宣称完成正式对照'}。",'',f"配置：{plan.get('num_envs')} 个环境，每种子 {plan.get('iterations')} 次更新，{plan.get('samples_per_seed')} 个决策样本；每组种子 7/17/27。",'', '成功指连续 0.2 秒浅入孔捕获，不是完整插入。准备失败计入总分母。','', '| 种子 | 摆动 ° | 成功／全部 | 准备失败 | 寻孔故障 | 失败计40s的平均完成时间 |','|---:|---:|---:|---:|---:|---:|']
    for r in runs:lines.append(f"| {r['seed']} | {r['amplitude']} | {r['successes']}/{r['total']} | {r['preparation_failures']} | {r['faults']} | {r['mean_capped_time_s']:.2f} s |")
    lines+=['','## 固定半径测试（每模型32个全圆方向）','','| 半径 mm | 无摆动 | 有摆动 |','|---:|---:|---:|']
    for rad in [1,2,5]:
        cells=[]
        for amp in [0.,.05]:
            rr=[x for r in runs if r['amplitude']==amp for x in r['rows'] if x['kind']=='grid' and x['radius_mm']==rad]
            cells.append(f"{sum(x['success'] for x in rr)}/{len(rr)}")
        lines.append(f'| {rad} | {cells[0]} | {cells[1]} |')
    lines+=['','## 故障原因与实际总时间','','| 种子/摆动 | 原因计数（控制器编码） | 成功时总耗时中位数（含6s准备） |','|---|---|---:|']
    for r in runs:
        reasons={str(k):sum(x['search_reason']==k for x in r['rows']) for k in range(1,10)}
        times=[x['total_s'] for x in r['rows'] if x['success']]
        lines.append(f"| {r['seed']}/{r['amplitude']} | {reasons} | {np.median(times) if times else None} |")
    lines+=['','## 判断','','']
    if complete:
        diffs=[]
        for seed in plan['seeds']:
            a=next(x for x in runs if x['seed']==seed and x['amplitude']==0);b=next(x for x in runs if x['seed']==seed and x['amplitude']==.05)
            diffs.append(b['successes']-a['successes']);lines.append(f"- 种子{seed}：有摆动比无摆动成功数变化 {diffs[-1]:+d}，平均完成时间变化 {b['mean_capped_time_s']-a['mean_capped_time_s']:+.2f} s。")
        lines+=['','结论：'+('三个种子成功数都提高，但仍需结合故障和耗时判断；不代表所有摆动参数有效。' if all(x>0 for x in diffs) else '本轮未显示跨三个种子一致的成功率优势。')]
    lines+=['','仅三个训练种子，不能把重复测试孔位当作大量独立训练实验。警告见 warnings.jsonl，执行状态见 queue_status.json；学习曲线见各模型 logs 和 progress.json，训练回合及故障原因见 episodes.json。','']
    (out/'REPORT.md').write_text('\n'.join(lines));save_json(out/'summary.json',dict(complete=complete,runs=runs))
    # Read existing TensorBoard logs and export curves without altering model selection.
    try:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for j in plan['queue_order']:
            folder=out/f"amp_{j['amplitude']:g}"/f"seed_{j['seed']}"/'logs';ea=EventAccumulator(str(folder));ea.Reload()
            tags=ea.Tags()['scalars']
            for ax,ends in zip(axes,[['mean_reward','total_reward'],['success_rate']]):
                tag=next((t for t in tags if any(t.endswith(v) for v in ends)),None)
                if tag:
                    vals=ea.Scalars(tag);ax.plot([v.step for v in vals],[v.value for v in vals],label=f"{j['amplitude']}deg seed{j['seed']}");ax.set_title(tag)
        for ax in axes:ax.set_xlabel('PPO update');ax.legend(fontsize=6)
        fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=140);plt.close(fig)
    except Exception as exc:save_json(out/'curve_export_error.json',dict(error=repr(exc)))
    return complete
