import unittest
from dataclasses import replace
from types import SimpleNamespace

import torch

from mjlab_contact_prep.config import ContactConfig
from mjlab_contact_prep.controller import ContactController
from mjlab_contact_prep.execution import ContactAction, COLUMNS
from mjlab_contact_prep.parking import LateralParking
from mjlab_contact_prep.xy_tracking import advance_reference, limit_request, unapplied_advance


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.axis = torch.tensor([[0., 0., 1.]], dtype=torch.float64)
        self.zero = torch.zeros(1, 3, dtype=torch.float64)

    def test_norm_cap_and_tilted_plane(self):
        axis = torch.tensor([[0., .6, .8]], dtype=torch.float64)
        raw = torch.tensor([[.0013, .002, -.001]], dtype=torch.float64)
        result = limit_request(raw, axis, .001)
        self.assertLessEqual(result.norm().item(), .001+1e-12)
        self.assertAlmostEqual((result*axis).sum().item(), 0., places=12)

    def test_blocked_position_bounds_reference_and_keeps_axial_anchor(self):
        ref = torch.tensor([[0., 0., .17]], dtype=torch.float64)
        request = torch.tensor([[.001, .001, .2]], dtype=torch.float64)
        for _ in range(1000):
            ref, _ = advance_reference(ref, self.zero, self.axis, request, .002, .0003)
        self.assertLessEqual(ref[:, :2].norm().item(), .0003+1e-12)
        self.assertEqual(ref[0, 2].item(), .17)

    def test_zero_input_does_not_move_far_hold_target(self):
        ref = torch.tensor([[.003, .001, .17]], dtype=torch.float64)
        new, delta = advance_reference(ref, self.zero, self.axis, self.zero, .002, .0003)
        self.assertTrue(torch.equal(new, ref))
        self.assertTrue(torch.equal(delta, self.zero))

    def test_outside_leash_allows_reversal_but_not_more_forward_advance(self):
        ref = torch.tensor([[.001, 0., 0.]], dtype=torch.float64)
        request = torch.tensor([[.001, 0., 0.]], dtype=torch.float64)
        new, _ = advance_reference(ref, self.zero, self.axis, request, .002, .0003)
        self.assertTrue(torch.equal(new, ref))
        new, _ = advance_reference(ref, self.zero, self.axis, -request, .002, .0003)
        self.assertAlmostEqual(new[0, 0].item(), .000998, places=12)

    def test_application_feedback_only_cancels_this_step(self):
        delta = torch.tensor([[2e-6, 0., 0.]], dtype=torch.float64)
        wanted = torch.tensor([[.0013, 0., 0.]], dtype=torch.float64)
        applied = torch.tensor([[.001, 0., .1]], dtype=torch.float64)
        rollback = unapplied_advance(delta, wanted, applied, self.axis, .002)
        self.assertAlmostEqual(rollback[0, 0].item(), .6e-6, places=12)
        rollback = unapplied_advance(delta, wanted, -wanted, self.axis, .002)
        self.assertTrue(torch.equal(rollback, delta))
        self.assertTrue(torch.equal(unapplied_advance(delta, wanted, 2*wanted, self.axis, .002), self.zero))
        self.assertTrue(torch.equal(unapplied_advance(self.zero, wanted, self.zero, self.axis, .002), self.zero))


def action_harness(cfg, count=2):
    """Run the real action method with an identity-J kinematic boundary.

    No plant response is simulated; these tests verify routing, limits, and
    reference feedback through the actual production method.
    """
    t = SimpleNamespace()
    t.cfg = SimpleNamespace(contact=cfg, damping=.01)
    t.controller = ContactController(cfg, count, 'cpu')
    t.controller.diagnostic_free_hold = True
    t.controller.diagnostic_xy_permission = torch.ones(count, dtype=torch.bool)
    t.parking = LateralParking(cfg, count, 'cpu')
    t.xy_action = torch.zeros(count, 2)
    t.command = torch.zeros(count, 6, dtype=torch.float64)
    t.qvelocity = torch.zeros(count, 6)
    t.wrench = torch.zeros(count, 6)
    t.sensor_wrench = t.wrench.clone()
    t.executed_twist = t.wrench.clone()
    t.energy_limited = torch.zeros(count, dtype=torch.bool)
    t.accel_limited = t.energy_limited.clone()
    t.diagnostic_lateral = t.diagnostic_parking_speed = t.diagnostic_parking_mask = None
    t.diagnostic_parking_active = torch.zeros(count, dtype=torch.bool)
    t.diagnostic_parking_goal = torch.zeros(count, 3)
    t.probe = None
    t.record = False
    t._point_torch = torch.zeros(count, 3)
    t._joint_ids = t._joint_dof_ids = list(range(6))
    t.act_addresses = torch.arange(6)
    t.stiffness = torch.full((6,), 1000., dtype=torch.float64)
    t._env = SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(
        qfrc_bias=torch.zeros(count, 6), qfrc_gravcomp=torch.zeros(count, 6), act=torch.zeros(count, 6))))
    t._entity = SimpleNamespace(data=SimpleNamespace(
        joint_pos=torch.zeros(count, 6), joint_vel=torch.zeros(count, 6)),
        set_joint_position_target=lambda *args, **kwargs: None)
    t._joint_upper = torch.full((count, 6), 10.)
    t._joint_lower = -t._joint_upper
    t._jacp_torch = torch.eye(6)[:3].repeat(count, 1, 1)
    t._jacr_torch = torch.eye(6)[3:].repeat(count, 1, 1)
    t._compute_jacobian = lambda: None
    t.read_wrench = lambda: (torch.zeros(count, 3), torch.eye(3).repeat(count, 1, 1))
    t.diagnostic_hook = SimpleNamespace(before=lambda *args: None,
        after=lambda term, data: setattr(term, 'last_data', {k: v.clone() for k, v in data.items()}))
    # Initialize through the real path before seeding a reference.
    ContactAction.apply_actions(t)
    return t


class ExecutionTests(unittest.TestCase):
    def test_candidate_is_not_enabled_by_default(self):
        self.assertFalse(ContactConfig().xy_tracking_limits_enabled)

    def test_combined_request_cap_and_legacy_opt_out(self):
        for enabled in (False, True):
            t = action_harness(replace(ContactConfig(), xy_speed=.001, xy_tracking_limits_enabled=enabled))
            t.controller.xy_reference[:, 0] = .00003
            t.xy_action[:, 0] = 1.
            ContactAction.apply_actions(t)
            data = t.last_data
            speed = data['final_twist'][:, :3].norm(dim=-1)
            if enabled:
                self.assertTrue((speed <= .001+1e-9).all())
                self.assertTrue((data['reference_rollback'][:, 0] > 0).all())
            else:
                self.assertTrue((speed > .0013).all())

    def test_joint_position_saturation_does_not_accumulate_reference(self):
        t = action_harness(replace(ContactConfig(), xy_speed=.001, xy_tracking_limits_enabled=True))
        t._joint_upper.zero_(); t._joint_lower.zero_()
        t.xy_action[:, 0] = 1.
        for _ in range(100):
            ContactAction.apply_actions(t)
        self.assertTrue(torch.equal(t.controller.xy_reference, torch.zeros(2, 3)))
        self.assertTrue(torch.equal(t.command, torch.zeros_like(t.command)))

    def test_leash_also_removes_forward_feedforward(self):
        t = action_harness(replace(ContactConfig(), xy_speed=.001, xy_tracking_limits_enabled=True))
        t.controller.xy_reference[:, 0] = .0003
        # Match the scaled effective target so feedback is zero at the leash.
        t.command[:, 0] = t.controller.xy_reference[:, 0].double()*t.cfg.contact.pose_error_scale
        t.xy_action[:, 0] = 1.
        ContactAction.apply_actions(t)
        self.assertLess(t.last_data['final_twist'][:, :3].abs().max().item(), 1e-8)

    def test_acceleration_feedback_and_zero_request_keep_target(self):
        t = action_harness(replace(ContactConfig(), xy_speed=.001, joint_acceleration=.01, xy_tracking_limits_enabled=True))
        t.xy_action[:, 0] = 1.
        ContactAction.apply_actions(t)
        self.assertTrue(t.accel_limited.all())
        self.assertTrue((t.last_data['reference_rollback'][:, 0] > 0).all())
        t.xy_action.zero_()
        fixed = t.controller.xy_reference.clone()
        ContactAction.apply_actions(t)
        self.assertTrue(torch.equal(t.controller.xy_reference, fixed))

    def test_parking_brake_envelope_is_not_clipped_to_tracking_speed(self):
        t = action_harness(replace(ContactConfig(), xy_speed=.001, parking_enabled=True, xy_tracking_limits_enabled=True))
        t.parking.state[:] = t.parking.BRAKE
        t.parking.goal[:, 0] = -.001
        t.parking.last_output[:, 0] = -.0015
        ContactAction.apply_actions(t)
        self.assertTrue((t.last_data['final_twist'][:, 0] < -.001).all())
        self.assertTrue(torch.equal(t.last_data['reference_rollback'], torch.zeros(2, 3)))

    def test_merged_track_stops_once_and_brake_does_not_advance(self):
        cfg = replace(ContactConfig(), xy_speed=.001, parking_enabled=True,
                      xy_tracking_limits_enabled=True)
        t = action_harness(cfg)
        t.xy_action[:, 0] = 1.
        ContactAction.apply_actions(t)
        data = t.last_data
        self.assertTrue(data['tracking_limits_active'].all())
        self.assertTrue((data['reference_increment'][:, 0] > 0).all())
        self.assertTrue((data['final_twist'][:, :3].norm(dim=-1) <= .001+1e-9).all())
        # An immobile plant with saturated commands cannot bank advance.
        t._joint_upper.zero_(); t._joint_lower.zero_()
        t.controller.xy_reference.zero_(); t.command.zero_(); t.qvelocity.zero_()
        t.parking.last_output.zero_()
        for _ in range(10):
            ContactAction.apply_actions(t)
        self.assertTrue(torch.equal(t.controller.xy_reference, torch.zeros(2, 3)))
        t.xy_action.zero_()
        ContactAction.apply_actions(t)
        self.assertTrue((t.parking.event == 1).all())
        self.assertTrue((t.parking.state == t.parking.BRAKE).all())
        self.assertFalse(t.last_data['tracking_limits_active'].any())
        self.assertTrue(torch.equal(t.last_data['reference_increment'], torch.zeros(2, 3)))
        goal = t.parking.goal.clone()
        ContactAction.apply_actions(t)
        self.assertTrue(torch.equal(t.parking.goal, goal))
        self.assertTrue(torch.equal(t.last_data['reference_rollback'], torch.zeros(2, 3)))

    def test_merged_mixed_phase_and_resume_uses_new_reference(self):
        cfg = replace(ContactConfig(), xy_speed=.001, parking_enabled=True,
                      xy_tracking_limits_enabled=True)
        t = action_harness(cfg)
        t.parking.state[0] = t.parking.BRAKE
        t.parking.goal[0, 0] = -.001
        t.parking.last_output[0, 0] = -.0015
        t.xy_action[1, 0] = 1.
        ContactAction.apply_actions(t)
        self.assertEqual(t.last_data['tracking_limits_active'].tolist(), [False, True])
        self.assertEqual(float(t.last_data['reference_increment'][0].norm()), 0.)
        self.assertGreater(float(t.last_data['reference_increment'][1, 0]), 0.)
        self.assertLess(float(t.last_data['final_twist'][0, 0]), -.001)
        # Resume after the dwell: stale parked goal is far from the measured
        # position, so the first TRACK output must use the reanchored reference.
        t.parking.goal[0, 0] = .01
        t.parking.last_output[0] = 0
        t.parking.resume_time[0] = cfg.parking_resume_dwell-cfg.dt
        t.xy_action[0, 0] = 1.
        ContactAction.apply_actions(t)
        self.assertEqual(int(t.parking.event[0]), 3)
        self.assertTrue(bool(t.last_data['tracking_limits_active'][0]))
        self.assertLess(float(t.controller.xy_reference[0, :2].norm()), .00031)
        self.assertLessEqual(float(t.last_data['final_twist'][0, :3].norm()),
                             cfg.parking_recovery_acceleration*cfg.dt+1e-8)

    def test_merged_fault_isolated_from_running_environment(self):
        cfg = replace(ContactConfig(), xy_speed=.001, parking_enabled=True,
                      xy_tracking_limits_enabled=True)
        t = action_harness(cfg)
        t.controller.state[0] = ContactController.FAULT
        t.parking.state[0] = t.parking.BRAKE
        t.parking.tracked[0] = True
        t.xy_action[:, 0] = 1.
        ContactAction.apply_actions(t)
        self.assertEqual(t.last_data['parking_event'].tolist(), [5, 0])
        self.assertTrue(torch.equal(t.last_data['final_twist'][0], torch.zeros(6)))
        self.assertEqual(t.last_data['tracking_limits_active'].tolist(), [False, True])
        self.assertEqual(float(t.last_data['reference_increment'][0].norm()), 0.)
        self.assertGreater(float(t.last_data['reference_increment'][1, 0]), 0.)

    def test_fault_and_unpermitted_env_do_not_advance(self):
        t = action_harness(replace(ContactConfig(), xy_speed=.001, xy_tracking_limits_enabled=True))
        t.controller.state[0] = ContactController.FAULT
        t.controller.diagnostic_xy_permission[1] = False
        t.xy_action[:, 0] = 1.
        ContactAction.apply_actions(t)
        self.assertTrue(torch.equal(t.controller.xy_reference, torch.zeros(2, 3)))
        self.assertTrue(torch.equal(t.last_data['final_twist'], torch.zeros(2, 6)))

    def test_record_columns_match_the_real_action_output(self):
        t = action_harness(replace(ContactConfig(), xy_speed=.001, xy_tracking_limits_enabled=True))
        t.record = True
        t.records = []
        t.geometry_id, t.hole_id = 0, 1
        t.raw_wrench = torch.zeros(2, 6)
        t._env.sim.data.site_xpos = torch.zeros(2, 2, 3)
        t._env.sim.data.site_xmat = torch.eye(3).repeat(2, 2, 1, 1).reshape(2, 2, 9)
        t.xy_action[:, 0] = 1.
        ContactAction.apply_actions(t)
        self.assertEqual(t.records[0].shape, (2, len(COLUMNS)))
        self.assertTrue((t.records[0][:, COLUMNS.index('tracking_limits_active')] == 1).all())


if __name__ == '__main__':
    unittest.main()
