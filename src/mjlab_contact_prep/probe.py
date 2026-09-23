"""Bounded conical reference; consumes contact measurements, never hole pose."""
from dataclasses import dataclass
import math
import torch
from .controller import rotation_error

@dataclass(frozen=True)
class ProbeConfig:
    amplitude_deg: float = .05
    frequency: float = .25
    ramp_time: float = 1.
    contact_dwell: float = .25
    absolute_limit_deg: float = 1.5

    def __post_init__(self):
        if not all(math.isfinite(v) for v in vars(self).values()):
            raise ValueError('nonfinite probe configuration')
        if not 0 <= self.amplitude_deg <= .5:
            raise ValueError('probe amplitude must be in [0, .5] degrees')
        if min(self.frequency,self.ramp_time,self.contact_dwell,self.absolute_limit_deg)<=0:
            raise ValueError('probe times and limits must be positive')

def rotation_exp(v):
    x,y,z=v.unbind(-1);zero=torch.zeros_like(x)
    k=torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),-1).reshape(-1,3,3)
    a=v.norm(dim=-1)
    eye=torch.eye(3,device=v.device,dtype=v.dtype).expand_as(k)
    return eye+torch.sinc(a/math.pi)[:,None,None]*k+.5*torch.sinc(a/(2*math.pi)).square()[:,None,None]*(k@k)

class ConicalReference:
    def __init__(self,cfg,count,device,dt):
        self.cfg,self.dt=cfg,dt
        self.phase=torch.zeros(count,device=device)
        self.amplitude=self.phase.clone();self.dwell=self.phase.clone()
        self.started=torch.zeros(count,device=device,dtype=torch.bool)
        self.target=torch.eye(3,device=device).repeat(count,1,1)
        self.initialized=self.started.clone()

    def reset(self,ids):
        for name in ('phase','amplitude','dwell','started','initialized'):
            getattr(self,name)[ids]=0
        self.target[ids]=torch.eye(3,device=self.phase.device)

    def step(self,mean,ready,usable,contact_present=None):
        c=self.cfg
        previous=torch.where(self.initialized[:,None,None],self.target,mean).clone()
        self.initialized[:]=True
        self.dwell[:]=torch.where(ready,self.dwell+self.dt,0)
        self.started |= self.dwell>=c.contact_dwell
        active=self.started & usable
        contact_present=usable if contact_present is None else contact_present
        self.phase[:]=torch.remainder(self.phase+active*self.dt*2*math.pi*c.frequency,2*math.pi)
        amount=math.radians(c.amplitude_deg)*self.dt/c.ramp_time
        # At high load hold the reference, rather than reversing the tilt on
        # every noisy force sample. Fade out only when contact is truly lost.
        self.amplitude[:]=torch.where(active,self.amplitude+amount,
            torch.where(contact_present,self.amplitude,self.amplitude-amount)).clamp(0,math.radians(c.amplitude_deg))
        v=self.amplitude[:,None]*torch.stack((self.phase.cos(),self.phase.sin(),torch.zeros_like(self.phase)),-1)
        self.target[:]=mean@rotation_exp(v)
        return self.target,rotation_error(self.target,previous)/self.dt,active

PROBE_COLUMNS=['phase','amplitude','started','usable','target_rx','target_ry','target_rz',
               'actual_rx','actual_ry','actual_rz','tracking_error','angular_limited',
               'mean_rx','mean_ry','mean_rz']
