"""Expanded-position experiment; inherits frozen pilot observations and rewards."""
from dataclasses import replace
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from .search_env import SearchEnv
from .environment import make_env
from .config import ContactConfig
from .probe import ProbeConfig
from .execution import COLUMNS
from .metrics import assess

class DistributionEnv(SearchEnv):
    def __init__(self,cases,output,contact=None,preparation_s=6.):
        contact=contact or ContactConfig()
        self.device='cuda:0';self.num_envs=len(cases);self.num_actions=2
        self.radius=.001 # Keep original reward scale, independent of case radius.
        self.repeat=5;self.dt=.2;self.max_episode_length=200;self.cfg={}
        self.env=ManagerBasedRlEnv(cfg=make_env(cases,contact,probe=ProbeConfig(amplitude_deg=0)),device=self.device)
        self.env.reset();self.term=self.env.action_manager.get_term('contact');t=self.term;t.record=True
        zeros=torch.zeros(self.num_envs,2,device=self.device)
        with torch.no_grad():
            for _ in range(round(preparation_s/self.env.step_dt)):self.env.step(zeros)
        trace=torch.stack(t.records).cpu().numpy();t.records.clear();t.probe_records.clear();t.record=False
        self.eligible=((t.controller.reason==0)&t.controller.ready).clone()
        self.valid_ids=torch.nonzero(self.eligible).flatten()
        self.preparation=[]
        for i,c in enumerate(cases):
            reason=int(t.controller.reason[i])
            x=trace[:,i];cols={name:j for j,name in enumerate(COLUMNS)}
            finite=np.isfinite(x[:,[cols[z] for z in ('hole_dx','hole_dy','hole_dz','Fx','Fy','Fz','Mx','My','Mz')]]).all(1)
            inside=finite&(np.hypot(x[:,cols['hole_dx']],x[:,cols['hole_dy']])<=.00015)&(-x[:,cols['hole_dz']]>=.0001)&(x[:,cols['reason']]==0)
            run=0;captured=False
            for valid in inside:
                run=run+1 if valid else 0
                captured |= run>=1+round(.2/contact.dt)
            self.preparation.append(dict(**c,eligible=bool(self.eligible[i]),reason=reason,overload=reason in [1,2,3],
              preparation_s=preparation_s,preparation_captured=bool(captured),
              stats=assess(trace[:,i],COLUMNS,contact)))
        import json
        (output/'preparation.json').write_text(json.dumps(self.preparation,indent=2)+'\n')
        np.savez_compressed(output/'preparation.npz',trace=trace,columns=COLUMNS)
        self.bank_names=['qpos','qvel','act','ctrl','qacc_warmstart','mocap_pos','mocap_quat','qfrc_applied','xfrc_applied']
        self.bank={n:getattr(self.env.sim.data,n).clone() for n in self.bank_names}
        self.ctrl_bank={n:v.clone() for n,v in vars(t.controller).items() if isinstance(v,torch.Tensor)}
        self.term_bank={n:getattr(t,n).clone() for n in ['command','qvelocity','xy_action','executed_twist']}
        self.probe_bank={n:v.clone() for n,v in vars(t.probe).items() if isinstance(v,torch.Tensor)}
        self.parking_bank=t.parking.state_dict()
        self.episode_length_buf=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
        self.last_action=zeros.clone();self.origin=torch.zeros(self.num_envs,3,device=self.device)
        self.history=torch.zeros(self.num_envs,32,20,device=self.device)
        self.capture_dwell=torch.zeros(self.num_envs,device=self.device)
        self.returns=torch.zeros(self.num_envs,device=self.device)
        self.previous_distance=torch.zeros(self.num_envs,device=self.device)
        self.completed=[]
        self.restore(torch.arange(self.num_envs,device=self.device),torch.arange(self.num_envs,device=self.device))

    def restore(self,ids,pick):
        for n,v in self.bank.items():getattr(self.env.sim.data,n)[ids]=v[pick]
        for n,v in self.ctrl_bank.items():getattr(self.term.controller,n)[ids]=v[pick]
        for n,v in self.term_bank.items():getattr(self.term,n)[ids]=v[pick]
        for n,v in self.probe_bank.items():getattr(self.term.probe,n)[ids]=v[pick]
        self.term.parking.load_state_dict(self.parking_bank,ids,pick)
        self.env.sim.forward();self.term.read_wrench()
        self.origin[ids]=self.env.sim.data.site_xpos[ids,self.term.geometry_id]
        self.episode_length_buf[ids]=0;self.last_action[ids]=0;self.capture_dwell[ids]=0;self.returns[ids]=0
        self.history[ids]=0;self.history[ids,-1]=self._frame()[ids]
        distance,_=self.metrics();self.previous_distance[ids]=distance[ids]

    def reset(self,ids,randomize=True):
        if not len(ids):return
        if not len(self.valid_ids):raise RuntimeError('No usable training contact states')
        pick=self.valid_ids[torch.randint(len(self.valid_ids),(len(ids),),device=self.device)] if randomize else self.valid_ids[ids%len(self.valid_ids)]
        self.restore(ids,pick)

    def set_arm(self,amplitude,seed,training):
        torch.manual_seed(seed)
        self.term.probe.cfg=replace(self.term.probe.cfg,amplitude_deg=amplitude)
        ids=torch.arange(self.num_envs,device=self.device)
        if training:self.reset(ids,randomize=False)
        else:self.restore(ids,ids) # Preserve failed test cases in their original slots.
        self.completed=[]
