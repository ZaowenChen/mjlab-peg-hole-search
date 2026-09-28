"""Per-environment lateral stop, fixed hold and continuous resume.

All vectors are world-frame vectors projected onto the current control plane.
The owner is ContactAction; ContactController only supplies the gated request.
"""
import torch

from .controller import bounded


class LateralParking:
    TRACK, BRAKE, HOLD = range(3)
    STATE_SCHEMA_VERSION = 1
    STATE_FIELDS = (
        'state', 'tracked', 'request_active', 'was_moving', 'was_permitted',
        'ramping', 'event', 'goal', 'still_origin', 'last_output',
        'actual_velocity', 'raw_correction', 'limited_correction',
        'still_time', 'gate_loss_time', 'resume_time', 'load_fraction',
        'speed_limited', 'acceleration_limited', 'lead_limited',
        'downstream_limited',
    )

    def __init__(self, cfg, count, device):
        self.cfg = cfg
        self.state = torch.zeros(count, device=device, dtype=torch.long)
        self.tracked = torch.zeros(count, device=device, dtype=torch.bool)
        self.request_active = torch.zeros_like(self.tracked)
        self.was_moving = torch.zeros_like(self.tracked)
        self.was_permitted = torch.zeros_like(self.tracked)
        self.ramping = torch.zeros_like(self.tracked)
        self.event = torch.zeros(count, device=device, dtype=torch.long)  # 1 stop, 2 hold, 3 resume, 4 disturbed, 5 fault
        self.goal = torch.zeros(count, 3, device=device)
        self.still_origin = torch.zeros_like(self.goal)
        self.last_output = torch.zeros_like(self.goal)
        self.actual_velocity = torch.zeros_like(self.goal)
        self.raw_correction = torch.zeros_like(self.goal)
        self.limited_correction = torch.zeros_like(self.goal)
        self.still_time = torch.zeros(count, device=device)
        self.gate_loss_time = torch.zeros(count, device=device)
        self.resume_time = torch.zeros(count, device=device)
        self.load_fraction = torch.ones(count, device=device)
        self.speed_limited = torch.zeros_like(self.tracked)
        self.acceleration_limited = torch.zeros_like(self.tracked)
        self.lead_limited = torch.zeros_like(self.tracked)
        self.downstream_limited = torch.zeros_like(self.tracked)
        assert {k for k, v in vars(self).items() if isinstance(v, torch.Tensor)} == set(self.STATE_FIELDS)

    def reset(self, ids=None):
        ids = slice(None) if ids is None else ids
        for name in self.STATE_FIELDS:
            getattr(self, name)[ids] = 0
        self.load_fraction[ids] = 1

    def state_dict(self):
        """All per-environment state that can affect the next parking output."""
        return {name: getattr(self, name).detach().clone() for name in self.STATE_FIELDS}

    def load_state_dict(self, state, ids=None, source_ids=None):
        """Restore all fields, optionally from selected bank rows into selected worlds."""
        if set(state) != set(self.STATE_FIELDS):
            raise ValueError('parking state fields do not match schema version 1')
        if (ids is None) != (source_ids is None):
            raise ValueError('ids and source_ids must be supplied together')
        for name in self.STATE_FIELDS:
            target, source = getattr(self, name), state[name]
            if not isinstance(source, torch.Tensor) or source.dtype != target.dtype or source.shape[1:] != target.shape[1:]:
                raise ValueError(f'incompatible parking state field: {name}')
            if ids is None and source.shape != target.shape:
                raise ValueError(f'incompatible parking state size: {name}')
        for name in self.STATE_FIELDS:
            target, source = getattr(self, name), state[name].to(self.state.device)
            if ids is None:
                target[:] = source
            else:
                target[ids] = source[source_ids]

    @staticmethod
    def project(value, axis):
        return value - (value * axis).sum(-1, keepdim=True) * axis

    def begin(self, pos, measured_velocity, lead, axis, upstream_speed,
              permitted, fault, reference):
        c, dt = self.cfg, self.cfg.dt
        project = lambda v: self.project(v, axis)
        velocity, command_lead = project(measured_velocity), project(lead)
        self.actual_velocity[:] = velocity
        self.event.zero_()
        self.raw_correction.zero_()
        self.limited_correction.zero_()
        self.speed_limited.zero_()
        self.acceleration_limited.zero_()
        self.lead_limited.zero_()
        self.downstream_limited.zero_()

        # Hysteresis is applied to the upstream request, never to feedback twist.
        magnitude = upstream_speed.norm(dim=-1)
        active = torch.where(self.request_active,
                             magnitude > c.parking_zero_tolerance,
                             magnitude >= c.parking_resume_tolerance)
        moving = active & permitted & ~fault
        started = (self.state == self.TRACK) & moving
        self.tracked |= started
        self.gate_loss_time[:] = torch.where(self.tracked & active & ~permitted,
                                              self.gate_loss_time+dt, 0)
        self.resume_time[:] = torch.where((self.state != self.TRACK) & moving,
                                          self.resume_time+dt, 0)
        request_stop = self.was_moving & ~active
        gate_stop = self.was_permitted & (self.gate_loss_time >= c.parking_gate_loss_dwell)
        stop = (self.state == self.TRACK) & self.tracked & (request_stop | gate_stop) & ~fault
        resume = (self.state != self.TRACK) & moving & (self.resume_time >= c.parking_resume_dwell)
        self.event[:] = torch.where(stop, 1, self.event)
        self.event[:] = torch.where(resume, 3, self.event)
        lookahead = bounded(velocity * c.parking_lookahead, c.parking_lookahead_distance)
        self.goal[:] = torch.where(stop[:, None], pos + lookahead, self.goal)
        self.still_origin[:] = torch.where(stop[:, None], pos, self.still_origin)
        self.still_time[:] = torch.where(stop, 0, self.still_time)
        self.state[:] = torch.where(stop, self.BRAKE, self.state)
        self.state[:] = torch.where(resume, self.TRACK, self.state)
        self.resume_time[:] = torch.where(stop | resume, 0, self.resume_time)
        self.ramping[:] = torch.where(resume, True, self.ramping)

        parked = (self.state != self.TRACK) & ~fault
        # Only transverse reference is synchronized; the axial anchor is untouched.
        reference[:] = torch.where(parked[:, None], reference + project(self.goal-reference), reference)
        # On resume, start from the current effective command point, with bounded lead.
        reference[:] = torch.where(resume[:, None], reference + project(pos + bounded(command_lead, c.parking_command_lead)-reference), reference)
        self.was_moving[:] = torch.where(stop, False, self.was_moving | moving)
        self.was_permitted[:] = torch.where(stop, False, self.was_permitted | (permitted & ~fault))
        self.request_active[:] = active
        return (self.state == self.TRACK) & ~fault

    def output(self, pos, lead, axis, normal_lateral, load, fault):
        c, dt = self.cfg, self.cfg.dt
        project = lambda v: self.project(v, axis)
        velocity, command_lead = self.actual_velocity, project(lead)
        parked = (self.state != self.TRACK) & ~fault

        error = project(self.goal-pos-command_lead)
        raw = c.parking_position_gain*error-c.parking_velocity_gain*velocity
        self.raw_correction[:] = torch.where(parked[:, None], raw, 0)
        desired = bounded(raw, c.parking_recovery_speed)
        self.speed_limited[:] = parked & ((desired-raw).norm(dim=-1) > 1e-10)
        # Do not increase an already excessive command lead. Below the cap,
        # limit the outward velocity to the available lead distance per step.
        lead_norm = command_lead.norm(dim=-1).clamp_min(1e-12)
        direction = command_lead/lead_norm[:, None]
        outward = (desired*direction).sum(-1)
        allowance = ((c.parking_command_lead-lead_norm)/dt).clamp(min=0)
        excess = (outward-allowance).clamp(min=0)
        self.lead_limited[:] = parked & (excess > 1e-10)
        desired = desired-excess[:, None]*direction
        # Near the working load ceiling, slow only the newly requested lateral
        # change. Reverse braking remains possible through the minimum fraction.
        fraction = ((c.parking_load_end-load.abs())/(c.parking_load_end-c.parking_load_start)).clamp(0, 1)
        fraction = c.parking_load_min_fraction+(1-c.parking_load_min_fraction)*fraction
        self.load_fraction[:] = torch.where(parked, fraction, 1)
        desired = desired*fraction[:, None]
        delta = bounded(desired-self.last_output, c.parking_recovery_acceleration*dt)
        braking = self.last_output+delta
        self.acceleration_limited[:] = parked & ((desired-self.last_output).norm(dim=-1) > c.parking_recovery_acceleration*dt+1e-10)
        self.limited_correction[:] = torch.where(parked[:, None], braking, 0)

        track_delta = bounded(normal_lateral-self.last_output, c.parking_recovery_acceleration*dt)
        tracking = torch.where(self.ramping[:, None], self.last_output+track_delta, normal_lateral)
        self.ramping[:] = self.ramping & ((normal_lateral-tracking).norm(dim=-1) > 1e-10)
        output = torch.where(parked[:, None], braking, tracking)
        output = torch.where(fault[:, None], 0, output)
        self.last_output[:] = output

        drift = project(pos-self.still_origin).norm(dim=-1)
        stable = (velocity.norm(dim=-1) <= c.parking_still_speed) & (drift <= c.parking_still_position) & \
                 (project(pos-self.goal).norm(dim=-1) <= c.parking_still_position)
        self.still_time[:] = torch.where(parked & stable, self.still_time+dt, 0)
        # A drifting brake observation starts a new dwell interval without
        # changing the fixed goal.
        self.still_origin[:] = torch.where(parked[:, None] & ~stable[:, None], pos, self.still_origin)
        held = (self.state == self.BRAKE) & (self.still_time >= c.parking_still_dwell)
        disturbed = (self.state == self.HOLD) & ~stable
        self.state[:] = torch.where(held, self.HOLD, self.state)
        self.state[:] = torch.where(disturbed, self.BRAKE, self.state)
        self.event[:] = torch.where(held, 2, self.event)
        self.event[:] = torch.where(disturbed, 4, self.event)
        self.event[:] = torch.where(fault & self.tracked, 5, self.event)
        return output

    def step(self, pos, measured_velocity, lead, axis, upstream_speed, gated_speed,
             permitted, fault, reference, normal_lateral, load):
        """Compatibility entry point for callers without a separate TRACK owner."""
        track = self.begin(pos, measured_velocity, lead, axis, upstream_speed,
                           permitted, fault, reference)
        reference[:] += torch.where(track[:, None], gated_speed*self.cfg.dt, 0)
        return self.output(pos, lead, axis, normal_lateral, load, fault)

    def commit(self, applied, saturated):
        # Use the achieved command increment as rate-limit memory whenever a
        # downstream limit clipped the requested increment.
        self.downstream_limited[:] = saturated
        self.last_output[:] = torch.where(saturated[:, None], applied, self.last_output)
