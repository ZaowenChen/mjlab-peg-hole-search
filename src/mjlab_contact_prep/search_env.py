"""Small GPU PPO feasibility task on the frozen contact controller.

Actor and critic see force/torque history and proprioception only. Hole pose is
used exclusively for reward and evaluation. Success means shallow hole capture,
not completed insertion. Cached GPU contact states remove descent from rollouts.
"""
from dataclasses import asdict
import math
import torch
from tensordict import TensorDict
from mjlab.envs import ManagerBasedRlEnv
from .config import ContactConfig
from .probe import ProbeConfig
from .environment import make_env
from .backend import backend_metadata

class SearchEnv:
    def __init__(self,num_envs=64,radius_mm=1.,amplitude_deg=0.,seed=7,angle_shift=0.,angles_deg=None):
        torch.manual_seed(seed)
        self.device='cuda:0';self.num_envs=num_envs;self.num_actions=2
        self.radius=radius_mm*.001;self.repeat=5;self.dt=.2
        self.max_episode_length=60
        self.cfg=dict(radius_mm=radius_mm,amplitude_deg=amplitude_deg,seed=seed,decision_dt=self.dt,
            history_steps=32,episode_s=12,backend=backend_metadata('contact-fix'),collision_model='partitioned',
            actor_inputs='6D wrench, encoder-derived tip displacement/velocity/tilt, previous XY action, known probe phase, filtered force and ready; 32-frame history',
            reward='privileged hole distance progress and capture, with load/loss/action penalties; truth never in actor or critic observations',
            success='radial error <=0.15mm AND geometric tip depth >=0.1mm for 0.2s; no fault; shallow capture only')
        angles=[2*math.pi*i/num_envs+angle_shift for i in range(num_envs)] if angles_deg is None else [math.radians(angles_deg[i%len(angles_deg)]) for i in range(num_envs)]
        self.cfg['angles_deg']=angles_deg
        cases=[dict(id=f'case_{i}',dx_mm=radius_mm*math.cos(t),dy_mm=radius_mm*math.sin(t)) for i,t in enumerate(angles)]
        self.env=ManagerBasedRlEnv(cfg=make_env(cases,ContactConfig(),probe=ProbeConfig(amplitude_deg=0)),device=self.device)
        self.env.reset();self.term=self.env.action_manager.get_term('contact')
        with torch.no_grad():
            for _ in range(150):self.env.step(torch.zeros(num_envs,2,device=self.device))
        t=self.term
        valid=(t.controller.reason==0)&t.controller.ready
        if not valid.all():
            self.env.close();raise RuntimeError(f'Contact bank has {int((~valid).sum())}/{num_envs} unavailable states; no training launched.')
        self.bank_names=['qpos','qvel','act','ctrl','qacc_warmstart','mocap_pos','mocap_quat','qfrc_applied','xfrc_applied']
        self.bank={n:getattr(self.env.sim.data,n).clone() for n in self.bank_names}
        self.ctrl_bank={n:v.clone() for n,v in vars(t.controller).items() if isinstance(v,torch.Tensor)}
        self.term_bank={n:getattr(t,n).clone() for n in ['command','qvelocity','xy_action']}
        self.probe_bank={n:v.clone() for n,v in vars(t.probe).items() if isinstance(v,torch.Tensor)}
        from dataclasses import replace
        t.probe.cfg=replace(t.probe.cfg,amplitude_deg=amplitude_deg)
        self.episode_length_buf=torch.zeros(num_envs,dtype=torch.long,device=self.device)
        self.last_action=torch.zeros(num_envs,2,device=self.device)
        self.origin=torch.zeros(num_envs,3,device=self.device)
        self.history=torch.zeros(num_envs,32,20,device=self.device)
        self.capture_dwell=torch.zeros(num_envs,device=self.device)
        self.returns=torch.zeros(num_envs,device=self.device)
        self.previous_distance=torch.zeros(num_envs,device=self.device)
        self.completed=[];self.reset(torch.arange(num_envs,device=self.device),randomize=False)

    def metrics(self):
        d=self.env.sim.data;t=self.term
        pos=d.site_xpos[:,t.geometry_id];mouth=d.site_xpos[:,t.hole_id]
        delta=(d.site_xmat[:,t.hole_id].view(-1,3,3).transpose(1,2)@(pos-mouth)[:,:,None]).squeeze(-1)
        return delta[:,:2].norm(dim=-1),-delta[:,2]

    def _frame(self):
        t=self.term;d=self.env.sim.data;t.read_wrench()
        w=t.wrench.clone();w[:,2]-=20
        w=w/torch.tensor([20,20,20,.5,.5,.5],device=self.device)
        pos=(d.site_xpos[:,t.geometry_id]-self.origin)/.001
        velocity=t.executed_twist[:,:3]/.001
        rot=d.site_xmat[:,t.geometry_id].view(-1,3,3)
        tilt=rot[:,:2,2]/.02
        phase=torch.stack((t.probe.phase.sin(),t.probe.phase.cos()),-1)
        return torch.cat((w,pos,velocity,self.last_action,phase,tilt,(t.controller.filtered_force[:,None]-20)/20,t.controller.ready[:,None]),-1).clamp(-20,20)

    def get_observations(self):
        x=self.history.flatten(1).clone() # RSL retains obs until after env.step; never expose mutable history storage.
        return TensorDict({'actor':x,'critic':x},batch_size=[self.num_envs])

    def reset(self,ids,randomize=True):
        if not len(ids):return
        pick=torch.randint(self.num_envs,(len(ids),),device=self.device) if randomize else ids
        for n,v in self.bank.items():getattr(self.env.sim.data,n)[ids]=v[pick]
        for n,v in self.ctrl_bank.items():getattr(self.term.controller,n)[ids]=v[pick]
        for n,v in self.term_bank.items():getattr(self.term,n)[ids]=v[pick]
        for n,v in self.probe_bank.items():getattr(self.term.probe,n)[ids]=v[pick]
        self.env.sim.forward();self.term.read_wrench()
        self.origin[ids]=self.env.sim.data.site_xpos[ids,self.term.geometry_id]
        self.episode_length_buf[ids]=0;self.last_action[ids]=0;self.capture_dwell[ids]=0;self.returns[ids]=0
        self.history[ids]=0;self.history[ids,-1]=self._frame()[ids]
        distance,_=self.metrics();self.previous_distance[ids]=distance[ids]

    def step(self,actions):
        actions=actions.detach().clamp(-1,1);prior=self.last_action.clone()
        for _ in range(self.repeat):
            self.env.step(actions)
            dist_now,depth_now=self.metrics()
            inside=(dist_now<=.00015)&(depth_now>=.0001)&(self.term.controller.reason==0)
            self.capture_dwell[:]=torch.where(inside,self.capture_dwell+self.env.step_dt,0)
        self.episode_length_buf+=1;self.last_action[:]=actions
        frame=self._frame();self.history[:]=torch.roll(self.history,-1,1);self.history[:,-1]=frame
        distance,depth=self.metrics();fault=self.term.controller.reason!=0
        success=(self.capture_dwell>=.2-1e-6)&~fault
        timeout=self.episode_length_buf>=self.max_episode_length
        done=success|fault|timeout
        reward=2*(self.previous_distance-distance)/self.radius-.005*distance/self.radius-.001*(actions-prior).square().sum(-1)
        reward+=5*success.float()-5*fault.float()-.01*(self.term.wrench[:,2]<5)
        self.previous_distance[:]=distance;self.returns+=reward
        ids=torch.nonzero(done).flatten()
        info={'time_outs':timeout&~success&~fault,'log':{'Search/radial_mm':distance.mean()*1000,'Search/force_N':self.term.wrench[:,2].mean()}}
        if len(ids):
            for i in ids.tolist():self.completed.append(dict(world_id=i,success=bool(success[i]),fault=bool(fault[i]),final_error_mm=float(distance[i]*1000),depth_mm=float(depth[i]*1000),steps=int(self.episode_length_buf[i]),reward=float(self.returns[i])))
            info['log']['Search/success_rate']=success[ids].float().mean()
            info['log']['Search/fault_rate']=fault[ids].float().mean()
            self.reset(ids)
        return self.get_observations(),reward,done.long(),info

    def close(self):self.env.close()
