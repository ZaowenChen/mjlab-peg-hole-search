"""Integration check: resetting one world preserves the other world's memory."""
import json
import argparse
from pathlib import Path
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab_contact_prep.config import ContactConfig
from mjlab_contact_prep.environment import make_env
from mjlab_contact_prep.probe import ProbeConfig

parser=argparse.ArgumentParser()
parser.add_argument('--probe',action='store_true')
parser.add_argument('--output',type=Path,default=Path('evaluation/gpu_reset.json'))
args=parser.parse_args()

cases=[dict(id='a',dx_mm=1.,dy_mm=0.),dict(id='b',dx_mm=2.,dy_mm=0.)]
env=ManagerBasedRlEnv(cfg=make_env(cases,ContactConfig(),probe=ProbeConfig() if args.probe else None),device='cuda:0')
try:
 env.reset();term=env.action_manager.get_term('contact')
 for _ in range(130 if args.probe else 75):env.step(torch.zeros(2,2,device=env.device))
 names=['qpos','qvel','act','ctrl']
 before={n:getattr(env.sim.data,n)[1].clone() for n in names}
 command=term.command[1].clone()
 memory={n:v[1].clone() for n,v in vars(term.controller).items() if isinstance(v,torch.Tensor)}
 probe_memory={} if term.probe is None else {n:v[1].clone() for n,v in vars(term.probe).items() if isinstance(v,torch.Tensor)}
 env.reset(env_ids=torch.tensor([0],device=env.device))
 for n,v in before.items():assert torch.equal(v,getattr(env.sim.data,n)[1]),n
 assert torch.equal(command,term.command[1])
 for n,v in memory.items():assert torch.equal(v,getattr(term.controller,n)[1]),n
 for n,v in probe_memory.items():assert torch.equal(v,getattr(term.probe,n)[1]),n
 if term.probe is not None:assert not term.probe.started[0] and term.probe.amplitude[0]==0
 assert term.controller.elapsed[0]==0 and not term.controller.initialized[0]
 env.step(torch.zeros(2,2,device=env.device))
 assert term.controller.elapsed[0]<.05 and term.controller.elapsed[1]>3
 result=dict(passed=True,preserved_world=1,reset_world=0,controller_fields=len(memory),probe_fields=len(probe_memory),physical_arrays=names)
 args.output.write_text(json.dumps(result,indent=2)+'\n');print(result)
finally:env.close()
