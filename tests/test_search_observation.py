"""The policy must not gain hole-pose labels through its observation builder."""
import unittest
from types import SimpleNamespace
import torch
from mjlab_contact_prep.search_env import SearchEnv

class ObservationBoundaryTest(unittest.TestCase):
 def test_observation_survives_next_history_write(self):
  env=SearchEnv.__new__(SearchEnv);env.num_envs=2;env.history=torch.randn(2,32,20)
  obs=env.get_observations();saved=obs['actor'].clone();env.history.zero_()
  self.assertTrue(torch.equal(obs['actor'],saved))

 def test_hole_pose_changes_labels_not_observation(self):
  env=SearchEnv.__new__(SearchEnv);env.device='cpu';env.num_envs=2
  env.origin=torch.zeros(2,3);env.last_action=torch.zeros(2,2)
  pos=torch.zeros(2,2,3);pos[:,0,2]=.001
  rotation=torch.eye(3).repeat(2,2,1,1)
  data=SimpleNamespace(site_xpos=pos,site_xmat=rotation)
  env.env=SimpleNamespace(sim=SimpleNamespace(data=data))
  env.term=SimpleNamespace(wrench=torch.tensor([[0.,0,20,0,0,0]]).repeat(2,1),geometry_id=0,hole_id=1,
   executed_twist=torch.zeros(2,6),probe=SimpleNamespace(phase=torch.zeros(2)),
   controller=SimpleNamespace(filtered_force=torch.ones(2)*20,ready=torch.ones(2,dtype=torch.bool)),read_wrench=lambda:None)
  before=env._frame().clone();distance,_=env.metrics()
  pos[:,1,0]=.005
  after=env._frame();changed,_=env.metrics()
  self.assertEqual(before.shape,(2,20));self.assertTrue(torch.equal(before,after))
  self.assertFalse(torch.equal(distance,changed))

if __name__=='__main__':unittest.main()
