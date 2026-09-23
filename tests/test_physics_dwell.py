import unittest
import torch
from mjlab_contact_prep.night_run.environment import advance_dwell,capture_ready
class PhysicsDwellTest(unittest.TestCase):
 def test_first_sample_adds_no_elapsed_interval(self):
  count=torch.zeros(1,dtype=torch.long)
  for _ in range(100):count=advance_dwell(count,torch.tensor([True]))
  self.assertFalse(bool(capture_ready(count)[0])) # 100 samples span 198ms.
  count=advance_dwell(count,torch.tensor([True]));self.assertTrue(bool(capture_ready(count)[0]))
 def test_one_physics_step_gap_breaks_continuity(self):
  count=torch.tensor([100]);count=advance_dwell(count,torch.tensor([False]))
  count=advance_dwell(count,torch.tensor([True]));self.assertEqual(int(count[0]),1)
  self.assertFalse(bool(capture_ready(count)[0]))
