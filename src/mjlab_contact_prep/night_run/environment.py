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

PHYSICS=['qpos','qvel','act','ctrl','qacc_warmstart','mocap_pos','mocap_quat','qfrc_applied','xfrc_applied']
TERM=['command','qvelocity','xy_action','executed_twist']
BUFFERS=['episode_length_buf','last_action','origin','history','returns','previous_distance','inside_samples','ticks','case_ids','nonfinite_seen']
AUDIT_COLUMNS=['time','radial_m','depth_m','reason','Fx','Fy','Fz','Mx','My','Mz','ready','probe_phase','recoveries']

def advance_dwell(samples,inside):
    return torch.where(inside,samples+1,0)

def capture_ready(samples):return samples>=101

def save_json(p,x):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def tensor_fields(obj):return {k:v for k,v in vars(obj).items() if isinstance(v,torch.Tensor)}
def cpu(fields):return {k:v.detach().cpu().clone() for k,v in fields.items()}

def prepare_bank(cases,out,batch=256):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'bank.pt').exists():return out/'bank.pt'
    chunks=[];metadata=[]
    for start in range(0,len(cases),batch):
        subset=cases[start:start+batch];folder=out/f'batch_{start:04d}';folder.mkdir(exist_ok=True)
        part=folder/'state.pt'
        if part.exists():chunk=torch.load(part,weights_only=False,map_location='cpu')
        else:
            e=DistributionEnv(subset,folder)
            try:
                dist,depth=e.metrics()
                rows=e.preparation
                for i,r in enumerate(rows):r.update(actual_handoff_radius_mm=float(dist[i]*1000),actual_handoff_depth_mm=float(depth[i]*1000))
                chunk=dict(physics=cpu(e.bank),controller=cpu(e.ctrl_bank),term=cpu(e.term_bank),probe=cpu(e.probe_bank),rows=rows)
                tmp=part.with_suffix('.tmp');torch.save(chunk,tmp);tmp.replace(part)
            finally:e.close()
        chunks.append(chunk);metadata+=chunk['rows']
        save_json(out/'progress.json',dict(prepared=len(metadata),total=len(cases)))
    bank={group:{name:torch.cat([c[group][name] for c in chunks],0) for name in chunks[0][group]} for group in ['physics','controller','term','probe']}
    bank['rows']=metadata
    torch.save(bank,out/'bank.tmp');(out/'bank.tmp').replace(out/'bank.pt')
    save_json(out/'preparation.json',metadata)
    summary=dict(total=len(metadata),eligible=sum(x['eligible'] for x in metadata),failed=sum(not x['eligible'] for x in metadata),
      strict_2s=sum(x['stats']['sustained_2s'] for x in metadata),by_sector=[])
    for bucket in range(3):
        for sector in range(12):
            r=[x for x in metadata if x.get('bucket',-1)==bucket and int(x['angle_deg']//30)==sector]
            if r:summary['by_sector'].append(dict(bucket=bucket,sector_deg=[sector*30,(sector+1)*30],total=len(r),eligible=sum(x['eligible'] for x in r)))
    save_json(out/'summary.json',summary)
    return out/'bank.pt'

class BankEnv(SearchEnv):
    def __init__(self,bank_path,num_envs,amp,seed,*,autoreset=True,audit=False,horizon=200):
        self.device='cuda:0';self.num_envs=num_envs;self.num_actions=2;self.dt=.2;self.repeat=5;self.radius=.001
        self.max_episode_length=horizon;self.cfg={};self.autoreset=autoreset;self.audit=audit;self.audit_records=[]
        raw=torch.load(bank_path,weights_only=False,map_location='cpu');self.rows=raw['rows']
        self.pool={g:{k:v.to(self.device) for k,v in raw[g].items()} for g in ['physics','controller','term','probe']}
        self.valid=torch.tensor([i for i,x in enumerate(self.rows) if x['eligible']],device=self.device,dtype=torch.long)
        self.bucket_ids=[torch.tensor([i for i,x in enumerate(self.rows) if x['eligible'] and x.get('bucket',0)==b],device=self.device,dtype=torch.long) for b in range(3)]
        if autoreset and any(not len(x) for x in self.bucket_ids):raise RuntimeError('A training radius bucket has no usable contacts')
        dummy=[self.rows[i%len(self.rows)] for i in range(num_envs)]
        self.env=ManagerBasedRlEnv(cfg=make_env(dummy,ContactConfig(),probe=ProbeConfig(amplitude_deg=0)),device=self.device)
        self.env.reset();self.term=self.env.action_manager.get_term('contact');self.term.probe.cfg=replace(self.term.probe.cfg,amplitude_deg=amp)
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
        self.term.apply_actions=sampled_actions
        ids=torch.arange(num_envs,device=self.device)
        if autoreset:self.reset(ids)
        else:self.restore(ids,ids)

    def restore(self,ids,pick):
        for k,v in self.pool['physics'].items():getattr(self.env.sim.data,k)[ids]=v[pick]
        for group,obj in [('controller',self.term.controller),('term',self.term),('probe',self.term.probe)]:
            for k,v in self.pool[group].items():getattr(obj,k)[ids]=v[pick]
        self.env.sim.forward();self.term.read_wrench()
        self.origin[ids]=self.env.sim.data.site_xpos[ids,self.term.geometry_id]
        self.episode_length_buf[ids]=0;self.last_action[ids]=0;self.returns[ids]=0
        self.inside_samples[ids]=0;self.ticks[ids]=0;self.nonfinite_seen[ids]=False;self.case_ids[ids]=pick
        self.history[ids]=0;self.history[ids,-1]=self._frame()[ids]
        self.previous_distance[ids]=self.metrics()[0][ids]

    def reset(self,ids,randomize=True):
        if not len(ids):return
        buckets=torch.randint(3,(len(ids),),generator=self.generator,device=self.device)
        pick=torch.empty(len(ids),device=self.device,dtype=torch.long)
        for b,pool in enumerate(self.bucket_ids):
            mask=buckets==b;n=int(mask.sum())
            if n:pick[mask]=pool[torch.randint(len(pool),(n,),generator=self.generator,device=self.device)]
        self.restore(ids,pick)

    def step(self,actions):
        actions=actions.detach().clamp(-1,1);prior=self.last_action.clone()
        for _ in range(5):self.env.step(actions)
        if self.nonfinite_seen.any():raise FloatingPointError('Non-finite physics state or controller reason 4; affected task stopped')
        self.episode_length_buf+=1;self.last_action[:]=actions
        self.history[:]=torch.roll(self.history,-1,1);self.history[:,-1]=self._frame()
        distance,depth=self.metrics();fault=self.term.controller.reason!=0
        # First valid sample spans zero time: 101 samples are 100 intervals = 0.2 s.
        success=capture_ready(self.inside_samples)&~fault
        timeout=self.episode_length_buf>=self.max_episode_length;done=success|fault|timeout
        reward=2*(self.previous_distance-distance)/.001-.005*distance/.001-.001*(actions-prior).square().sum(-1)
        reward+=5*success.float()-5*fault.float()-.01*(self.term.wrench[:,2]<5)
        self.previous_distance[:]=distance;self.returns+=reward
        info={'time_outs':timeout&~success&~fault,'log':{'Search/radial_mm':distance.mean()*1000,'Search/force_N':self.term.wrench[:,2].mean()}}
        if self.autoreset:
            ids=torch.nonzero(done).flatten()
            for i in ids.tolist():
                c=int(self.case_ids[i]);self.completed.append(dict(case_index=c,world_id=i,success=bool(success[i]),reason=int(self.term.controller.reason[i]),steps=int(self.episode_length_buf[i]),reward=float(self.returns[i]),final_error_mm=float(distance[i]*1000),depth_mm=float(depth[i]*1000)))
            if len(ids):
                info['log'].update({'Search/success_rate':success[ids].float().mean(),'Search/fault_rate':fault[ids].float().mean()});self.reset(ids)
        self.last_success=success;self.last_fault=fault
        return self.get_observations(),reward,done.long(),info

    def snapshot(self):
        return dict(physics=cpu({k:getattr(self.env.sim.data,k) for k in PHYSICS}),controller=cpu(tensor_fields(self.term.controller)),
          term=cpu({k:getattr(self.term,k) for k in TERM}),probe=cpu(tensor_fields(self.term.probe)),buffers=cpu({k:getattr(self,k) for k in BUFFERS}),
          generator=self.generator.get_state(),completed=self.completed)

    def restore_snapshot(self,state):
        for k,v in state['physics'].items():getattr(self.env.sim.data,k)[:]=v.to(self.device)
        for group,obj in [('controller',self.term.controller),('term',self.term),('probe',self.term.probe),('buffers',self)]:
            for k,v in state[group].items():getattr(obj,k)[:]=v.to(self.device)
        self.env.sim.forward();self.term.read_wrench();self.generator.set_state(state['generator'].cpu());self.completed=state['completed']
