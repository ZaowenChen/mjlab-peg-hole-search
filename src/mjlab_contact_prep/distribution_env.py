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
    def __init__(self,cases,output):
        self.device='cuda:0';self.num_envs=len(cases);self.num_actions=2
        self.radius=.001 # Keep original reward scale, independent of case radius.
        self.repeat=5;self.dt=.2;self.max_episode_length=200;self.cfg={}
        self.env=ManagerBasedRlEnv(cfg=make_env(cases,ContactConfig(),probe=ProbeConfig(amplitude_deg=0)),device=self.device)
        self.env.reset();self.term=self.env.action_manager.get_term('contact');t=self.term;t.record=True
        zeros=torch.zeros(self.num_envs,2,device=self.device)
        with torch.no_grad():
            for _ in range(150):self.env.step(zeros)
        trace=torch.stack(t.records).cpu().numpy();t.records.clear();t.probe_records.clear();t.record=False
        self.eligible=((t.controller.reason==0)&t.controller.ready).clone()
        self.valid_ids=torch.nonzero(self.eligible).flatten()
        self.preparation=[]
        for i,c in enumerate(cases):
            reason=int(t.controller.reason[i])
            self.preparation.append(dict(**c,eligible=bool(self.eligible[i]),reason=reason,overload=reason in [1,2,3],stats=assess(trace[:,i],COLUMNS,ContactConfig())))
        import json
        (output/'preparation.json').write_text(json.dumps(self.preparation,indent=2)+'\n')
        np.savez_compressed(output/'preparation.npz',trace=trace,columns=COLUMNS)
        self.bank_names=['qpos','qvel','act','ctrl','qacc_warmstart','mocap_pos','mocap_quat','qfrc_applied','xfrc_applied']
        self.bank={n:getattr(self.env.sim.data,n).clone() for n in self.bank_names}
        self.ctrl_bank={n:v.clone() for n,v in vars(t.controller).items() if isinstance(v,torch.Tensor)}
        self.term_bank={n:getattr(t,n).clone() for n in ['command','qvelocity','xy_action','executed_twist']}
        self.probe_bank={n:v.clone() for n,v in vars(t.probe).items() if isinstance(v,torch.Tensor)}
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
