import unittest
from dataclasses import replace

import torch

from mjlab_contact_prep.config import ContactConfig
from mjlab_contact_prep.parking import LateralParking


class ParkingTests(unittest.TestCase):
    def setUp(self):
        self.cfg = replace(ContactConfig(), xy_speed=.001)
        self.p = LateralParking(self.cfg, 2, 'cpu')
        self.axis = torch.tensor([[0., 0., -1.]]).repeat(2, 1)
        self.pos = torch.zeros(2, 3)
        self.velocity = torch.zeros_like(self.pos)
        self.lead = torch.zeros_like(self.pos)
        self.request = torch.zeros_like(self.pos)
        self.permitted = torch.zeros(2, dtype=torch.bool)
        self.fault = torch.zeros_like(self.permitted)
        self.reference = torch.zeros_like(self.pos)

    def step(self):
        return self.p.step(self.pos, self.velocity, self.lead, self.axis,
                           self.request, torch.where(self.permitted[:, None], self.request, 0),
                           self.permitted, self.fault, self.reference,
                           self.request, torch.zeros(2))

    def start_and_stop(self):
        self.permitted[0] = True
        self.request[0, 0] = .001
        self.step()
        self.request.zero_()
        self.velocity[0, 0] = .001
        self.step()

    def test_initial_zero_and_one_time_goal(self):
        for _ in range(5): self.step()
        self.assertEqual(int(self.p.state[0]), self.p.TRACK)
        self.start_and_stop()
        self.assertEqual(int(self.p.event[0]), 1)
        goal = self.p.goal[0].clone()
        self.pos[0, 0] = .0001
        self.step()
        self.assertEqual(int(self.p.event[0]), 0)
        self.assertTrue(torch.equal(goal, self.p.goal[0]))

    def test_hold_disturbance_and_resume(self):
        self.start_and_stop()
        self.velocity.zero_()
        self.pos[0, 0] = self.p.goal[0, 0]
        for _ in range(round(self.cfg.parking_still_dwell/self.cfg.dt)+1): self.step()
        self.assertEqual(int(self.p.state[0]), self.p.HOLD)
        goal = self.p.goal[0].clone()
        self.pos[0, 0] += .0001
        self.step()
        self.assertEqual(int(self.p.state[0]), self.p.BRAKE)
        self.assertTrue(torch.equal(goal, self.p.goal[0]))
        self.request[0, 0] = .001
        for _ in range(round(self.cfg.parking_resume_dwell/self.cfg.dt)-1): self.step()
        previous = self.p.last_output[0].clone()
        self.step()
        self.assertEqual(int(self.p.event[0]), 3)
        self.assertLessEqual(float((self.p.last_output[0]-previous).norm()),
                             self.cfg.parking_recovery_acceleration*self.cfg.dt+1e-9)
        self.assertLess(float((self.reference[0]-self.pos[0])[:2].norm()),
                        self.cfg.parking_command_lead+.00001)

    def test_fault_and_partial_reset(self):
        self.start_and_stop()
        self.fault[0] = True
        output = self.step()
        self.assertEqual(float(output[0].norm()), 0.)
        self.p.reset(torch.tensor([0]))
        self.assertEqual(int(self.p.state[0]), self.p.TRACK)
        self.assertFalse(bool(self.p.tracked[0]))
        self.assertEqual(float(self.p.goal[0].norm()), 0.)
        self.assertEqual(float(self.p.load_fraction[0]), 1.)
        self.assertEqual(int(self.p.state[1]), self.p.TRACK)

    def test_ready_loss_while_request_remains(self):
        self.permitted[0] = True
        self.request[0, 0] = .001
        self.step()
        self.permitted[0] = False
        for _ in range(round(self.cfg.parking_gate_loss_dwell/self.cfg.dt)): self.step()
        self.assertEqual(int(self.p.event[0]), 1)
        self.assertEqual(int(self.p.state[0]), self.p.BRAKE)

    def test_ready_chatter_does_not_repeat_resume(self):
        self.start_and_stop()
        self.request[0, 0] = .001
        events = 0
        for i in range(100):
            self.permitted[0] = i % 4 < 2
            self.step()
            events += int(self.p.event[0] == 3)
        self.assertEqual(events, 0)
        self.assertEqual(int(self.p.state[0]), self.p.BRAKE)


if __name__ == '__main__':
    unittest.main()
