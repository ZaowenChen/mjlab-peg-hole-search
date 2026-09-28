import json, math
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab_contact_prep.search_env import SearchEnv
from mjlab_contact_prep.distribution_env import DistributionEnv
from mjlab_contact_prep.environment import make_env
from mjlab_contact_prep.config import ContactConfig
from mjlab_contact_prep.probe import ProbeConfig
from mjlab_contact_prep.parking import LateralParking
from .shallow_hold import ShallowHold, PROTOCOL_ID, observed_reason

PHYSICS=['qpos','qvel','act','ctrl','qacc_warmstart','mocap_pos','mocap_quat','qfrc_applied','xfrc_applied']
TERM=['command','qvelocity','xy_action','executed_twist']
BUFFERS=['episode_length_buf','last_action','origin','history','returns','previous_distance','inside_samples','ticks','case_ids','nonfinite_seen']
AUDIT_COLUMNS=['time','radial_m','depth_m','reason','Fx','Fy','Fz','Mx','My','Mz','ready','probe_phase','recoveries']
SPIKE013_AUDIT_COLUMNS=AUDIT_COLUMNS+['policy_vx','policy_vy','combined_vx','combined_vy','actual_vx','actual_vy']

class _ReadOnlyControlTap:
    def __init__(self,count,device):
        self.combined=torch.zeros(count,2,device=device)
    def before(self,term,pos,rot):
        pass
    def after(self,term,record):
        self.combined[:]=record['applied_xy_command'][:,:2]

def advance_dwell(samples,inside):
    return torch.where(inside,samples+1,0)

def capture_ready(samples):return samples>=101

def save_json(p,x):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def tensor_fields(obj):return {k:v for k,v in vars(obj).items() if isinstance(v,torch.Tensor)}
def cpu(fields):return {k:v.detach().cpu().clone() for k,v in fields.items()}

def prepared_parking_state(raw):
    """Read a prepared bank; only legacy zero-lateral-request banks may start in TRACK."""
    version=raw.get('parking_schema_version')
    if 'parking' in raw:
        if version != LateralParking.STATE_SCHEMA_VERSION:
            raise ValueError(f'unsupported prepared parking schema: {version}')
        state=raw['parking']
        checker=LateralParking(ContactConfig(),len(raw['rows']),'cpu')
        checker.load_state_dict(state)
        return state,'native_v1'
    if version is not None:
        raise ValueError('prepared bank has parking schema version but no parking state')
    # Legacy formal banks were prepared before any policy lateral request. Their
    # contact servo may have small residual joint velocity; qvel==0 is not required.
    rows=raw['rows']
    if not rows or any(not isinstance(row,dict) or 'eligible' not in row or 'stats' not in row for row in rows):
        raise ValueError('legacy bank lacks preparation records; parking cannot be initialized')
    for group,name in [('term','xy_action'),('controller','last_xy')]:
        value=raw[group][name]
        if len(value) != len(rows):
            raise ValueError(f'legacy bank row count differs from {group}.{name}')
        if not torch.isfinite(value).all() or bool((value.abs()>1e-9).any()):
            raise ValueError(f'legacy prepared bank has active lateral request in {group}.{name}; parking cannot be initialized')
    neutral=LateralParking(ContactConfig(),len(raw['rows']),'cpu').state_dict()
    return neutral,'legacy_zero_lateral_request'

def prepare_bank(cases,out,batch=256,*,contact=None,preparation_s=6.,fingerprint=None):
    from mjlab_contact_prep.data.state_bank import load_state_bank,save_state_bank,STATE_BANK_FORMAT_VERSION
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    contact=contact or ContactConfig()
    def checked(path):
        raw=load_state_bank(path)
        if fingerprint is not None and raw.get('fingerprint')!=fingerprint:
            raise ValueError(f'prepared bank fingerprint mismatch: {path}')
        return raw
    if (out/'bank.pt').exists():
        checked(out/'bank.pt');return out/'bank.pt'
    chunks=[];metadata=[]
    for start in range(0,len(cases),batch):
        subset=cases[start:start+batch];folder=out/f'batch_{start:04d}';folder.mkdir(exist_ok=True)
        part=folder/'state.pt'
        if part.exists():chunk=checked(part)
        else:
            e=DistributionEnv(subset,folder,contact=contact,preparation_s=preparation_s)
            try:
                dist,depth=e.metrics()
                rows=e.preparation
                pos=e.env.sim.data.site_xpos[:,e.term.geometry_id]
                mouth=e.env.sim.data.site_xpos[:,e.term.hole_id]
                rot=e.env.sim.data.site_xmat[:,e.term.hole_id].view(-1,3,3)
                delta=(rot.transpose(1,2)@(pos-mouth)[:,:,None]).squeeze(-1)
                for i,r in enumerate(rows):
                    x=float(delta[i,0]*1000);y=float(delta[i,1]*1000)
                    r.update(actual_handoff_dx_mm=x,actual_handoff_dy_mm=y,
                      actual_handoff_angle_deg=float(math.degrees(math.atan2(y,x))%360),
                      actual_handoff_radius_mm=float(dist[i]*1000),actual_handoff_depth_mm=float(depth[i]*1000))
                chunk=dict(physics=cpu(e.bank),controller=cpu(e.ctrl_bank),term=cpu(e.term_bank),probe=cpu(e.probe_bank),
                           parking=cpu(e.parking_bank),parking_schema_version=LateralParking.STATE_SCHEMA_VERSION,
                           fingerprint=fingerprint,format_version=STATE_BANK_FORMAT_VERSION,
                           format_metadata={'protocol_id':PROTOCOL_ID if fingerprint else None},rows=rows)
                save_state_bank(part,chunk)
            finally:e.close()
        chunk['parking'],_ = prepared_parking_state(chunk)
        chunks.append(chunk);metadata+=chunk['rows']
        save_json(out/'progress.json',dict(prepared=len(metadata),total=len(cases)))
    bank={group:{name:torch.cat([c[group][name] for c in chunks],0) for name in chunks[0][group]} for group in ['physics','controller','term','probe','parking']}
    bank['parking_schema_version']=LateralParking.STATE_SCHEMA_VERSION
    bank['fingerprint']=fingerprint
    bank['format_version']=STATE_BANK_FORMAT_VERSION
    bank['format_metadata']={'protocol_id':PROTOCOL_ID if fingerprint else None}
    bank['rows']=metadata
    save_state_bank(out/'bank.pt',bank)
    save_json(out/'preparation.json',metadata)
    summary=dict(total=len(metadata),eligible=sum(x['eligible'] for x in metadata),failed=sum(not x['eligible'] for x in metadata),
      preparation_captured=sum(x.get('preparation_captured',False) for x in metadata),
      training_eligible=sum(x['eligible'] and not x.get('preparation_captured',False) for x in metadata),
      strict_2s=sum(x['stats']['sustained_2s'] for x in metadata),by_sector=[],by_bucket=[])
    for bucket in range(max((x.get('bucket',0) for x in metadata),default=-1)+1):
        group=[x for x in metadata if x.get('bucket',-1)==bucket]
        summary['by_bucket'].append(dict(bucket=bucket,total=len(group),eligible=sum(x['eligible'] for x in group),
            training_eligible=sum(x['eligible'] and not x.get('preparation_captured',False) for x in group)))
        for sector in range(12):
            r=[x for x in metadata if x.get('bucket',-1)==bucket and int(x['angle_deg']//30)==sector]
            summary['by_sector'].append(dict(bucket=bucket,sector_deg=[sector*30,(sector+1)*30],total=len(r),
                eligible=sum(x['eligible'] for x in r),rejected=sum(not x['eligible'] for x in r),
                training_eligible=sum(x['eligible'] and not x.get('preparation_captured',False) for x in r)))
    save_json(out/'summary.json',summary)
    return out/'bank.pt'

class BankEnv(SearchEnv):
    def __init__(self,bank_path,num_envs,amp,seed,*,autoreset=True,audit=False,horizon=200,
                 contact=None,protocol_id=None,bucket_weights=None,expected_fingerprint=None):
        contact=contact or ContactConfig()
        self.protocol_id=protocol_id;self.shallow_hold=None
        self.device='cuda:0';self.num_envs=num_envs;self.num_actions=2;self.repeat=5;self.dt=contact.dt*20*self.repeat;self.radius=.001
        self.max_episode_length=horizon;self.cfg={};self.autoreset=autoreset;self.audit=audit;self.audit_records=[]
        from mjlab_contact_prep.data.state_bank import load_state_bank
        raw=load_state_bank(bank_path);self.rows=raw['rows']
        if expected_fingerprint is not None and raw.get('fingerprint')!=expected_fingerprint:
            raise ValueError('state bank fingerprint mismatch')
        raw['parking'],self.parking_bank_source=prepared_parking_state(raw)
        self.pool={g:{k:v.to(self.device) for k,v in raw[g].items()} for g in ['physics','controller','term','probe','parking']}
        self.valid=torch.tensor([i for i,x in enumerate(self.rows) if x['eligible']],device=self.device,dtype=torch.long)
        bucket_count=len(bucket_weights) if bucket_weights is not None else 3
        self.bucket_weights=torch.tensor(bucket_weights or [1/bucket_count]*bucket_count,device=self.device)
        self.bucket_ids=[torch.tensor([i for i,x in enumerate(self.rows) if x['eligible'] and
            (protocol_id != PROTOCOL_ID or not x.get('preparation_captured',False)) and x.get('bucket',0)==b],
            device=self.device,dtype=torch.long) for b in range(bucket_count)]
        if autoreset and any(not len(x) for x in self.bucket_ids):raise RuntimeError('A training radius bucket has no usable contacts')
        dummy=[self.rows[i%len(self.rows)] for i in range(num_envs)]
        self.env=ManagerBasedRlEnv(cfg=make_env(dummy,contact,probe=ProbeConfig(amplitude_deg=0)),device=self.device)
        self.env.reset();self.term=self.env.action_manager.get_term('contact');self.term.probe.cfg=replace(self.term.probe.cfg,amplitude_deg=amp)
        if protocol_id == PROTOCOL_ID:
            if abs(self.env.physics_dt-contact.dt)>1e-9 or abs(self.env.step_dt-20*contact.dt)>1e-9 or self.repeat!=5 or amp!=0:
                raise ValueError('SPIKE-013 constructed runtime does not match the effective configuration')
            self.shallow_hold=ShallowHold(num_envs,self.device,contact.dt)
            self.control_tap=_ReadOnlyControlTap(num_envs,self.device)
            self.term.diagnostic_hook=self.control_tap
            self.previous_tip_xy=torch.zeros(num_envs,2,device=self.device)
        self.generator=torch.Generator(device=self.device).manual_seed(seed)
        self.episode_length_buf=torch.zeros(num_envs,dtype=torch.long,device=self.device)
        self.last_action=torch.zeros(num_envs,2,device=self.device);self.origin=torch.zeros(num_envs,3,device=self.device)
        self.history=torch.zeros(num_envs,32,20,device=self.device);self.returns=torch.zeros(num_envs,device=self.device)
        self.previous_distance=torch.zeros(num_envs,device=self.device)
        self.inside_samples=torch.zeros(num_envs,dtype=torch.long,device=self.device)
        self.ticks=torch.zeros_like(self.inside_samples);self.case_ids=torch.zeros_like(self.inside_samples)
        self.nonfinite_seen=torch.zeros(num_envs,dtype=torch.bool,device=self.device)
        self.completed=[]
        original=self.term.apply_actions
        def sampled_actions():
            original() # Controller and physics requests are unchanged.
            distance,depth=self.metrics();reason=self.term.controller.reason
            finite=torch.isfinite(distance)&torch.isfinite(depth)&torch.isfinite(self.term.wrench).all(-1)
            self.nonfinite_seen |= ~finite | (reason==4)
            inside=(distance<=.00015)&(depth>=.0001)&(reason==0)&finite
            self.inside_samples[:]=advance_dwell(self.inside_samples,inside)
            if self.audit:
                self.audit_records.append(torch.cat(((self.ticks.float()*.002)[:,None],distance[:,None],depth[:,None],reason[:,None],self.term.wrench,
                    self.term.controller.ready[:,None],self.term.probe.phase[:,None],self.term.controller.recoveries[:,None]),-1).detach().clone())
            self.ticks+=1
        if protocol_id == PROTOCOL_ID:
            original_step=self.env.sim.step
            def sampled_physics_step():
                original_step()
                self.env.sim.forward() # Refresh post-integration geometry and force sensors together.
                self.term.read_wrench()
                distance,depth=self.metrics()
                reason=observed_reason(self.term.controller.reason,self.term.wrench,
                                       self.term.sensor_wrench,contact)
                self.shallow_hold.sample(distance,depth,self.term.wrench,reason,active=~self.finished)
                current_tip_xy=self.env.sim.data.site_xpos[:,self.term.geometry_id,:2]
                actual_xy=(current_tip_xy-self.previous_tip_xy)/contact.dt
                self.previous_tip_xy[:]=current_tip_xy
                if self.audit:
                    policy_xy=self.term.xy_action/self.term.xy_action.norm(dim=-1,keepdim=True).clamp_min(1)*contact.xy_speed
                    self.audit_records.append(torch.cat(((self.shallow_hold.ticks.float()*contact.dt)[:,None],
                        distance[:,None],depth[:,None],reason[:,None],self.term.wrench,
                        self.term.controller.ready[:,None],self.term.probe.phase[:,None],
                        self.term.controller.recoveries[:,None],
                        policy_xy,self.control_tap.combined,actual_xy),-1).detach().clone())
            self.finished=torch.zeros(num_envs,dtype=torch.bool,device=self.device)
            self.env.sim.step=sampled_physics_step
        else:
            self.term.apply_actions=sampled_actions
        ids=torch.arange(num_envs,device=self.device)
        if autoreset:self.reset(ids)
        else:
            self.restore(ids,ids)
            if self.shallow_hold is not None:
                self.finished[~torch.tensor([self.rows[i]['eligible'] for i in range(num_envs)],device=self.device)]=True

    def restore(self,ids,pick):
        for k,v in self.pool['physics'].items():getattr(self.env.sim.data,k)[ids]=v[pick]
        for group,obj in [('controller',self.term.controller),('term',self.term),('probe',self.term.probe)]:
            for k,v in self.pool[group].items():getattr(obj,k)[ids]=v[pick]
        self.term.parking.load_state_dict(self.pool['parking'],ids,pick)
        self.env.sim.forward();self.term.read_wrench()
        self.origin[ids]=self.env.sim.data.site_xpos[ids,self.term.geometry_id]
        self.episode_length_buf[ids]=0;self.last_action[ids]=0;self.returns[ids]=0
        self.inside_samples[ids]=0;self.ticks[ids]=0;self.nonfinite_seen[ids]=False;self.case_ids[ids]=pick
        if self.shallow_hold is not None:
            self.shallow_hold.reset(ids);self.finished[ids]=False
            self.previous_tip_xy[ids]=self.env.sim.data.site_xpos[ids,self.term.geometry_id,:2]
            self.control_tap.combined[ids]=0
        self.history[ids]=0;self.history[ids,-1]=self._frame()[ids]
        self.previous_distance[ids]=self.metrics()[0][ids]

    def reset(self,ids,randomize=True):
        if not len(ids):return
        buckets=torch.multinomial(self.bucket_weights,len(ids),replacement=True,generator=self.generator)
        pick=torch.empty(len(ids),device=self.device,dtype=torch.long)
        for b,pool in enumerate(self.bucket_ids):
            mask=buckets==b;n=int(mask.sum())
            if n:pick[mask]=pool[torch.randint(len(pool),(n,),generator=self.generator,device=self.device)]
        self.restore(ids,pick)

    def step(self,actions):
        actions=actions.detach().clamp(-1,1);prior=self.last_action.clone()
        for _ in range(self.repeat):self.env.step(actions)
        if self.shallow_hold is not None:
            if self.shallow_hold.nonfinite_seen.any():raise FloatingPointError('Non-finite physics state or controller reason 4; affected task stopped')
        elif self.nonfinite_seen.any():raise FloatingPointError('Non-finite physics state or controller reason 4; affected task stopped')
        self.episode_length_buf+=1;self.last_action[:]=actions
        self.history[:]=torch.roll(self.history,-1,1);self.history[:,-1]=self._frame()
        distance,depth=self.metrics();fault=self.term.controller.reason!=0
        # First valid sample spans zero time: 101 samples are 100 intervals = 0.2 s.
        timeout=self.episode_length_buf>=self.max_episode_length
        if self.shallow_hold is not None:
            done,success,fault,pure_timeout=self.shallow_hold.outcome(timeout)
            done &= ~self.finished;success &= ~self.finished;fault &= ~self.finished;pure_timeout &= ~self.finished
        else:
            success=capture_ready(self.inside_samples)&~fault
            done=success|fault|timeout;pure_timeout=timeout&~success&~fault
        reward=2*(self.previous_distance-distance)/.001-.005*distance/.001-.001*(actions-prior).square().sum(-1)
        reward+=5*success.float()-5*fault.float()-.01*(self.term.wrench[:,2]<5)
        if self.shallow_hold is not None:reward=torch.where(self.finished,0.,reward)
        self.previous_distance[:]=distance;self.returns+=reward
        info={'time_outs':pure_timeout,'log':{'Search/radial_mm':distance.mean()*1000,'Search/force_N':self.term.wrench[:,2].mean()}}
        if self.shallow_hold is not None and not self.autoreset:self.finished |= done
        if self.autoreset:
            ids=torch.nonzero(done).flatten()
            for i in ids.tolist():
                c=int(self.case_ids[i]);self.completed.append(dict(case_index=c,world_id=i,success=bool(success[i]),reason=int(self.shallow_hold.first_fault_reason[i]) if self.shallow_hold and fault[i] else int(self.term.controller.reason[i]),steps=int(self.episode_length_buf[i]),reward=float(self.returns[i]),final_error_mm=float(distance[i]*1000),depth_mm=float(depth[i]*1000),
                  protocol_id=self.protocol_id,first_capture_tick=int(self.shallow_hold.first_capture_tick[i]) if self.shallow_hold else -1,
                  capture_count=int(self.shallow_hold.capture_count[i]) if self.shallow_hold else 0,
                  hold_break_count=int(self.shallow_hold.hold_break_count[i]) if self.shallow_hold else 0))
            if len(ids):
                info['log'].update({'Search/success_rate':success[ids].float().mean(),'Search/fault_rate':fault[ids].float().mean()});self.reset(ids)
        self.last_success=success;self.last_fault=fault
        return self.get_observations(),reward,done.long(),info

    def snapshot(self):
        return dict(physics=cpu({k:getattr(self.env.sim.data,k) for k in PHYSICS}),controller=cpu(tensor_fields(self.term.controller)),
          term=cpu({k:getattr(self.term,k) for k in TERM}),probe=cpu(tensor_fields(self.term.probe)),
          parking=cpu(self.term.parking.state_dict()),parking_schema_version=LateralParking.STATE_SCHEMA_VERSION,
          buffers=cpu({k:getattr(self,k) for k in BUFFERS}),
          generator=self.generator.get_state(),completed=self.completed,
          shallow_hold=None if self.shallow_hold is None else self.shallow_hold.state_dict(),
          finished=None if self.shallow_hold is None else cpu({'finished':self.finished})['finished'])

    def restore_snapshot(self,state):
        if 'parking' not in state:
            if state.get('parking_schema_version') is not None:
                raise ValueError('checkpoint has parking schema version but no parking state')
            if self.term.parking.cfg.parking_enabled:
                raise ValueError('legacy moving checkpoint has no parking state; resume with parking enabled is unsafe')
            # Parking was inactive in the legacy run. Initialize only that unused
            # state; the rest of the moving checkpoint retains its exact values.
            parking=None
        else:
            if state.get('parking_schema_version') != LateralParking.STATE_SCHEMA_VERSION:
                raise ValueError('unsupported checkpoint parking schema version')
            parking=state['parking']
            checker=LateralParking(self.term.parking.cfg,self.num_envs,'cpu')
            checker.load_state_dict(parking)
        for k,v in state['physics'].items():getattr(self.env.sim.data,k)[:]=v.to(self.device)
        for group,obj in [('controller',self.term.controller),('term',self.term),('probe',self.term.probe),('buffers',self)]:
            for k,v in state[group].items():getattr(obj,k)[:]=v.to(self.device)
        if parking is None:self.term.parking.reset()
        else:self.term.parking.load_state_dict(parking)
        self.env.sim.forward();self.term.read_wrench();self.generator.set_state(state['generator'].cpu());self.completed=state['completed']
        if self.shallow_hold is not None:
            if state.get('shallow_hold') is None or state.get('finished') is None:
                raise ValueError('SPIKE-013 snapshot lacks shallow-hold state')
            self.shallow_hold.load_state_dict(state['shallow_hold']);self.finished[:]=state['finished'].to(self.device)
            self.previous_tip_xy[:]=self.env.sim.data.site_xpos[:,self.term.geometry_id,:2]
