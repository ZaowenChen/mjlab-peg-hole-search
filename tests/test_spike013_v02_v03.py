import unittest
import tempfile
from pathlib import Path

import torch

from mjlab_contact_prep.night_run.shallow_hold import ShallowHold,observed_reason
from mjlab_contact_prep.config import ContactConfig
from mjlab_contact_prep.night_run.spike013 import cases, load_experiment
from mjlab_contact_prep.night_run.environment import prepare_bank


class ShallowHoldTests(unittest.TestCase):
    def setUp(self):
        self.state = ShallowHold(2, "cpu", .002)
        self.radial = torch.tensor([.0001, .0001])
        self.depth = torch.tensor([.0002, .0002])
        self.wrench = torch.zeros(2, 6)
        self.reason = torch.zeros(2, dtype=torch.long)

    def sample(self, count=1):
        for _ in range(count):
            self.state.sample(self.radial, self.depth, self.wrench, self.reason)

    def test_confirmation_and_stable_boundaries(self):
        self.sample(100)
        self.assertEqual(int(self.state.capture_count[0]), 0)
        self.sample()
        self.assertEqual(int(self.state.capture_count[0]), 1)
        self.assertFalse(bool(self.state.outcome(torch.zeros(2,dtype=torch.bool))[1][0]))
        self.sample(249)
        self.assertFalse(bool(self.state.outcome(torch.zeros(2,dtype=torch.bool))[1][0]))
        self.sample()
        self.assertTrue(bool(self.state.outcome(torch.ones(2,dtype=torch.bool))[1][0]))
        self.assertFalse(bool(self.state.outcome(torch.ones(2,dtype=torch.bool))[3][0]))

    def test_break_requires_full_reconfirmation_and_end_boundary(self):
        self.sample(351)
        self.radial[0] = .001
        self.sample()
        self.assertEqual(int(self.state.hold_break_count[0]), 1)
        self.assertFalse(bool(self.state.outcome(torch.zeros(2,dtype=torch.bool))[1][0]))
        self.radial[0] = .0001
        self.sample(100)
        self.assertEqual(int(self.state.capture_count[0]), 1)
        self.sample()
        self.assertEqual(int(self.state.capture_count[0]), 2)
        self.state.reset(torch.tensor([0]))
        self.assertEqual(int(self.state.ticks[0]), 0)
        self.assertGreater(int(self.state.ticks[1]), 0)

    def test_fault_latch_and_snapshot(self):
        self.sample(351)
        saved = self.state.state_dict()
        self.reason[0] = 2
        self.sample()
        self.reason[0] = 0
        self.sample()
        done, success, fault, timeout = self.state.outcome(torch.ones(2,dtype=torch.bool))
        self.assertTrue(bool(fault[0]))
        self.assertFalse(bool(success[0]))
        self.assertFalse(bool(timeout[0]))
        self.assertEqual(int(self.state.first_fault_reason[0]), 2)
        self.state.load_state_dict(saved)
        self.assertTrue(bool(self.state.outcome(torch.zeros(2,dtype=torch.bool))[1][0]))

    def test_nonfinite_has_highest_priority(self):
        self.sample(351)
        self.wrench[0, 0] = float('nan')
        self.sample()
        done, success, fault, timeout = self.state.outcome(torch.ones(2,dtype=torch.bool))
        self.assertTrue(bool(done[0]))
        self.assertFalse(bool(success[0] or fault[0] or timeout[0]))

    def test_last_physics_load_fault_is_visible_before_next_control_action(self):
        tip=torch.zeros(2,6);sensor=torch.zeros_like(tip)
        tip[0,2]=40.;sensor[1,3]=8.
        reason=observed_reason(torch.zeros(2,dtype=torch.long),tip,sensor,ContactConfig())
        self.assertEqual(reason.tolist(),[1,3])
        self.state.sample(self.radial,self.depth,tip,reason)
        self.assertEqual(self.state.first_fault_reason.tolist(),[1,3])
        self.assertEqual(self.state.outcome(torch.zeros(2,dtype=torch.bool))[2].tolist(),[True,True])


class DistributionTests(unittest.TestCase):
    def test_five_buckets_and_precise_far_grid(self):
        document, contact = load_experiment()
        groups = cases(document['effective'])
        self.assertEqual([len(groups[x]) for x in ('train','validation','test')],[2400,40,280])
        self.assertEqual([sum(c['bucket']==b for c in groups['train']) for b in range(5)],[480]*5)
        self.assertEqual(sum(c['radius_mm']==10 for c in groups['test']),32)
        self.assertEqual(contact.xy_speed,.001)

    def test_invalid_speed_is_rejected(self):
        with self.assertRaises(ValueError):load_experiment(['contact.xy_speed=0.0002'])

    def test_existing_bank_with_other_fingerprint_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            torch.save({'fingerprint':'old','physics':{},'controller':{},'term':{},'probe':{},'rows':[]},path/'bank.pt')
            with self.assertRaisesRegex(ValueError,'fingerprint mismatch'):
                prepare_bank([],path,fingerprint='new')


if __name__ == '__main__':unittest.main()
