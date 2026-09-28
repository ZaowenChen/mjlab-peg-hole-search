import unittest

import torch

from mjlab_contact_prep.config import ContactConfig
from mjlab_contact_prep.night_run.environment import prepared_parking_state
from mjlab_contact_prep.parking import LateralParking


class ParkingStateTest(unittest.TestCase):
    def setUp(self):
        self.parking = LateralParking(ContactConfig(), 2, 'cpu')

    def test_partial_restore_preserves_other_world_and_all_fields(self):
        original = self.parking.state_dict()
        original['state'][0] = self.parking.BRAKE
        original['goal'][0, 0] = .013
        original['last_output'][0, 0] = .0004
        original['resume_time'][0] = .034
        self.parking.load_state_dict(original)
        self.parking.reset(torch.tensor([0]))
        self.assertEqual(int(self.parking.state[0]), self.parking.TRACK)
        self.parking.load_state_dict(original, torch.tensor([0]), torch.tensor([0]))
        for name, value in original.items():
            self.assertTrue(torch.equal(getattr(self.parking, name), value), name)

    def test_missing_or_incompatible_state_is_rejected_before_mutation(self):
        before = self.parking.state_dict()
        missing = dict(before)
        del missing['goal']
        with self.assertRaisesRegex(ValueError, 'fields'):
            self.parking.load_state_dict(missing)
        wrong = dict(before)
        wrong['goal'] = wrong['goal'].double()
        with self.assertRaisesRegex(ValueError, 'goal'):
            self.parking.load_state_dict(wrong)
        for name, value in before.items():
            self.assertTrue(torch.equal(getattr(self.parking, name), value), name)

    def test_legacy_prepared_bank_requires_zero_lateral_request(self):
        raw = {'rows': [{'eligible': True, 'stats': {}}, {'eligible': False, 'stats': {}}], 'term': {'xy_action': torch.zeros(2, 2)},
               'controller': {'last_xy': torch.zeros(2, 3)}}
        neutral, source = prepared_parking_state(raw)
        self.assertEqual(source, 'legacy_zero_lateral_request')
        self.assertEqual(set(neutral), set(self.parking.STATE_FIELDS))
        self.assertTrue(torch.equal(neutral['load_fraction'], torch.ones(2)))
        raw['term']['xy_action'][0, 0] = .2
        with self.assertRaisesRegex(ValueError, 'active lateral request'):
            prepared_parking_state(raw)
        raw['term']['xy_action'][0, 0] = 0
        del raw['rows'][0]['stats']
        with self.assertRaisesRegex(ValueError, 'preparation records'):
            prepared_parking_state(raw)


if __name__ == '__main__':
    unittest.main()
