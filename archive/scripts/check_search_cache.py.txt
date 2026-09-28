"""GPU integration check for subset cached-contact reset used by PPO."""
import json
from pathlib import Path
import torch
from mjlab_contact_prep.search_env import SearchEnv

env=SearchEnv(4,1,0,7,angles_deg=[90,270])
try:
 for _ in range(3):env.step(torch.ones(4,2,device=env.device)*.2)
 names=['qpos','qvel','act','ctrl']
 before={n:getattr(env.env.sim.data,n)[1:].clone() for n in names}
 history=env.history[1:].clone();controller={n:v[1:].clone() for n,v in vars(env.term.controller).items() if isinstance(v,torch.Tensor)}
 env.reset(torch.tensor([0],device=env.device),randomize=False)
 for n in names:
  assert torch.equal(getattr(env.env.sim.data,n)[0],env.bank[n][0]),n
  assert torch.equal(getattr(env.env.sim.data,n)[1:],before[n]),n
 for n,v in controller.items():assert torch.equal(getattr(env.term.controller,n)[1:],v),n
 assert torch.equal(env.history[1:],history)
 assert env.term.controller.initialized[0] and env.term.controller.ready[0]
 assert env.episode_length_buf[0]==0 and env.episode_length_buf[1]==3
 assert torch.isfinite(env.get_observations()['actor']).all()
 result={'passed':True,'reset_world':0,'preserved_worlds':[1,2,3],'cached_contact_ready':True,'no_approach_repeated':True}
 Path('evaluation/gpu_search_cache_check.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
finally:env.close()
