"""Paired speed-only evaluation of frozen PPO weights; no training."""
import argparse, copy, hashlib, json, math, time
from pathlib import Path
from dataclasses import replace, asdict
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from mjlab_contact_prep.night_run.environment import BankEnv, prepare_bank, save_json, AUDIT_COLUMNS
from mjlab_contact_prep.night_run.jobs import ppo_config, verify_frozen
from mjlab_contact_prep.config import ContactConfig

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'evaluation/night_fullcircle_5mm_v1'
EXTRA=['tip_x_m','tip_y_m','tip_z_m','requested_vx','requested_vy','requested_vz','joint_command_vx','joint_command_vy','joint_command_vz','action_x','action_y','accel_limited','energy_limited']
COLS=AUDIT_COLUMNS+EXTRA

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def cases(pilot):
    rs=[1,5] if pilot else [.5,1,2,5]
    n=4 if pilot else 16
    # Pilot directions are separate from formal grid.
    offset=0 if pilot else 5.625
    return [dict(id=f'{"pilot" if pilot else "speed"}_{r:g}_{j:02}',radius_mm=r,angle_deg=offset+j*360/n,
      dx_mm=r*math.cos(math.radians(offset+j*360/n)),dy_mm=r*math.sin(math.radians(offset+j*360/n)),
      bucket=0 if r<=1.5 else (1 if r<=3 else 2),kind='speed_grid') for r in rs for j in range(n)]

def prep_audit(out):
    rows=json.loads((out/'bank/preparation.json').read_text())
    a=np.load(out/'bank/batch_0000/preparation.npz');t=a['trace'];c=list(a['columns'])
    for i,row in enumerate(rows):
        x=t[:,i];rad=np.hypot(x[:,c.index('hole_dx')],x[:,c.index('hole_dy')]);depth=-x[:,c.index('hole_dz')]
        valid=(rad<=.00015)&(depth>=.0001)&(x[:,c.index('reason')]==0)
        bad=np.flatnonzero(~valid);tail=len(valid)-(bad[-1]+1 if len(bad) else 0)
        count=0;first=None
        for j,v in enumerate(valid):
            count=count+1 if v else 0
            if count>=101 and first is None:first=float(x[j,c.index('time')])
        row.update(prep_captured=bool(tail>=101),prep_capture_dwell_s=max(0,int(tail)-1)*.002,
          prep_capture_first_s=first,prep_class='captured' if tail>=101 else ('ready' if row['eligible'] else 'failed'))
    save_json(out/'preparation_audit.json',rows)
    return rows

def summary_rows(rows):
    success=[r for r in rows if r['success']]
    return dict(total=len(rows),successes=len(success),ppo_successes=sum(r['outcome']=='success' for r in rows),
      prepared_captures=sum(r['outcome']=='preparation_captured' for r in rows),preparation_failures=sum(r['outcome']=='preparation_failed' for r in rows),
      search_eligible=sum(r['prep_class']=='ready' for r in rows),faults=sum(r['outcome']=='fault' for r in rows),timeouts=sum(r['outcome']=='timeout' for r in rows),
      fault_reasons={str(k):sum(r.get('search_reason')==k for r in rows) for k in range(1,10)},
      mean_success_search_s=float(np.mean([r['search_s'] for r in success])) if success else None,
      mean_capped_search_s=float(np.mean([r['capped_search_s'] for r in rows])),
      mean_capped_total_s=float(np.mean([r['capped_total_s'] for r in rows])))

def run(out,pilot):
    out.mkdir(parents=True,exist_ok=False)
    original=json.loads((BASE/'plan.json').read_text());verify_frozen(original)
    plan=dict(pilot=pilot,cases=cases(pilot),seeds=[7] if pilot else [7,17,27],speeds_mm_s=[.2,.5,1.],horizon_s=40,
       preparation_s=6,source=str(Path(__file__).resolve()),source_sha256=sha(__file__),base=str(BASE),
       checkpoints={str(s):dict(path=str(BASE/f'amp_0/seed_{s}/policy.pt'),sha256=sha(BASE/f'amp_0/seed_{s}/policy.pt')) for s in ([7] if pilot else [7,17,27])},
       success='radial<=0.15mm and depth>=0.1mm and no fault for 101 consecutive 2ms samples',
       preparation_captured='End-of-preparation capture dwell >=0.2s; counted in pipeline success, not PPO success; same in all arms',
       timing='6s preparation. Search failures capped at40s; pipeline failures capped at46s. Prepared captures have search0 and pipeline6s.',
       contact=asdict(ContactConfig()),frozen_physics=original['physics_frozen'],frozen_experiment=original['experiment_source_sha256'])
    save_json(out/'plan.json',plan)
    start=time.monotonic();print('PREPARE',len(plan['cases']),flush=True)
    bank=prepare_bank(plan['cases'],out/'bank',batch=len(plan['cases']))
    rows0=prep_audit(out);print('PREPARATION', {c:sum(r['prep_class']==c for r in rows0) for c in ['ready','captured','failed']},flush=True)
    save_json(out/'progress.json',dict(stage='evaluate',prepared=len(rows0),elapsed_s=time.monotonic()-start))
    env=BankEnv(bank,len(rows0),0,7,autoreset=False,audit=False,horizon=200)
    runner=OnPolicyRunner(env,copy.deepcopy(ppo_config(7,1)),None,device=env.device)
    records=[]
    original_apply=env.term.apply_actions
    def audit_action():
        original_apply()
        t=env.term;d=env.env.sim.data;rad,depth=env.metrics()
        records.append(torch.cat((((env.ticks.float()-1)*.002)[:,None],rad[:,None],depth[:,None],t.controller.reason[:,None],t.wrench,
          t.controller.ready[:,None],t.probe.phase[:,None],t.controller.recoveries[:,None],d.site_xpos[:,t.geometry_id],
          t.controller.last_xy,t.executed_twist[:,:3],t.xy_action,t.accel_limited[:,None],t.energy_limited[:,None]),-1).detach().clone())
    env.term.apply_actions=audit_action
    ids=torch.arange(env.num_envs,device=env.device)
    runs=[]
    try:
      for seed in plan['seeds']:
        runner.load(plan['checkpoints'][str(seed)]['path'],map_location=env.device);policy=runner.get_inference_policy()
        for speed in plan['speeds_mm_s']:
          dest=out/f'seed_{seed}/speed_{speed:g}';dest.mkdir(parents=True)
          cfg=replace(ContactConfig(),xy_speed=speed*.001)
          env.term.cfg.contact=cfg;env.term.controller.cfg=cfg
          env.restore(ids,ids);env.audit_records.clear();records.clear()
          # Verify exact restoration of every saved physics/controller/term/probe tensor.
          reset_ok=True
          for group,obj in [('physics',env.env.sim.data),('controller',env.term.controller),('term',env.term),('probe',env.term.probe)]:
            for key,value in env.pool[group].items():
              equal=torch.equal(getattr(obj,key),value)
              # Forward recomputes derived qacc_warmstart on some backends; retain explicit audit.
              if not equal:raise AssertionError(('restore mismatch',group,key))
          active=torch.tensor([r['prep_class']=='ready' for r in rows0],device=env.device)
          rows=[dict(**r,seed=seed,speed_mm_s=speed,success=r['prep_class']=='captured',outcome='preparation_captured' if r['prep_class']=='captured' else ('preparation_failed' if r['prep_class']=='failed' else 'pending'),
            search_s=0. if r['prep_class']!='ready' else None,total_s=6. if r['prep_class']!='ready' else None,search_reason=None,terminal_samples=0,
            capped_search_s=0. if r['prep_class']=='captured' else 40.,capped_total_s=6. if r['prep_class']=='captured' else 46.) for r in rows0]
          blocks=[];wall=time.monotonic()
          with torch.inference_mode():
            for step in range(1,201):
              if not active.any():break
              actions=policy(env.get_observations()).clamp(-1,1);actions[~active]=0
              _,_,done,_=env.step(actions)
              blocks.append(torch.stack(records).cpu().numpy());records.clear()
              rad,dep=env.metrics()
              for i in torch.nonzero(active&done.bool()).flatten().tolist():
                suc=bool(env.last_success[i]);reason=int(env.term.controller.reason[i]);elapsed=round(step*.2,3)
                rows[i].update(success=suc,outcome='success' if suc else ('fault' if reason else 'timeout'),search_s=elapsed,total_s=6+elapsed,search_reason=reason,
                  terminal_samples=int(env.ticks[i]),capped_search_s=elapsed if suc else 40.,capped_total_s=6+elapsed if suc else 46.,
                  final_error_mm=float(rad[i]*1000),final_depth_mm=float(dep[i]*1000))
              active &= ~done.bool()
              if step%25==0:
                save_json(out/'progress.json',dict(stage='evaluate',seed=seed,speed_mm_s=speed,step=step,total_steps=200,active=int(active.sum()),completed_runs=len(runs),elapsed_s=time.monotonic()-start,run_elapsed_s=time.monotonic()-wall))
                print('PROGRESS',seed,speed,step,int(active.sum()),round(time.monotonic()-wall,1),flush=True)
          trace=np.concatenate(blocks,axis=0) if blocks else np.empty((0,len(rows),len(COLS)),dtype=np.float32)
          np.savez_compressed(dest/'trajectory.npz',trace=trace,columns=COLS)
          del blocks,trace
          a=np.load(dest/'trajectory.npz');tr=a['trace'];ci={k:i for i,k in enumerate(COLS)}
          for i,row in enumerate(rows):
            if row['prep_class']!='ready':continue
            assert row['outcome']!='pending'
            x=tr[:row['terminal_samples'],i];assert np.isfinite(x).all()
            good=(x[:,ci['radial_m']]<=.00015)&(x[:,ci['depth_m']]>=.0001)&(x[:,ci['reason']]==0)
            bad=np.flatnonzero(~good);tail=len(good)-(int(bad[-1])+1 if len(bad) else 0)
            expected=tail>=101 and int(x[-1,ci['reason']])==0
            assert row['success']==expected,(seed,speed,row['id'],tail)
            assert int(x[-1,ci['reason']])==row['search_reason']
            assert np.allclose(np.diff(x[:,0]),.002,atol=4e-6)
            pos=x[:,[ci['tip_x_m'],ci['tip_y_m']]];actual=np.linalg.norm(np.diff(pos,axis=0),axis=-1)/.002*1000
            # 20ms displacement gives a less noisy actual-motion diagnostic.
            smooth=np.linalg.norm(pos[10:]-pos[:-10],axis=-1)/.02*1000
            req=np.linalg.norm(x[:,[ci['requested_vx'],ci['requested_vy']]],axis=-1)*1000
            assert req.max()<=speed+1e-4
            row.update(audited_dwell_s=max(0,tail-1)*.002,actual_xy_mean_mm_s=float(actual.mean()),actual_xy_p95_20ms_mm_s=float(np.percentile(smooth,95)),
              requested_xy_mean_mm_s=float(req.mean()),requested_xy_max_mm_s=float(req.max()),path_length_mm=float(actual.sum()*.002),
              peak_axial_N=float(x[:,ci['Fz']].max()),peak_radial_N=float(np.linalg.norm(x[:,[ci['Fx'],ci['Fy']]],axis=-1).max()),
              peak_moment_Nm=float(np.linalg.norm(x[:,[ci['Mx'],ci['My'],ci['Mz']]],axis=-1).max()),
              min_radial_mm=float(x[:,ci['radial_m']].min()*1000),ready_fraction=float(x[:,ci['ready']].mean()),
              accel_limited_fraction=float(x[:,ci['accel_limited']].mean()),energy_limited_fraction=float(x[:,ci['energy_limited']].mean()))
          result=dict(seed=seed,speed_mm_s=speed,wall_s=time.monotonic()-wall,reset_exact=reset_ok,audit_passed=True,**summary_rows(rows),rows=rows)
          save_json(dest/'results.json',result);runs.append(result)
          print('RESULT',seed,speed,result['successes'],result['faults'],result['timeouts'],round(result['wall_s'],1),flush=True)
          del tr,a
      verify_frozen(original)
      assert sha(__file__)==plan['source_sha256']
      assert all(sha(v['path'])==v['sha256'] for v in plan['checkpoints'].values())
      save_json(out/'summary.json',dict(complete=True,wall_s=time.monotonic()-start,runs=runs))
      save_json(out/'progress.json',dict(stage='complete',runs=len(runs),wall_s=time.monotonic()-start))
      print('COMPLETE',round(time.monotonic()-start,1),flush=True)
    finally:env.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--pilot',action='store_true');a=p.parse_args()
    run(Path(a.output).resolve(),a.pilot)
