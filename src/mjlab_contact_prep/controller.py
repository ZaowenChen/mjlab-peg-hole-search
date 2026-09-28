"""Three-state, batched force controller. No hole pose enters this module."""
import math
import torch
from .config import ContactConfig


def bounded(x, limit):
    return x * (limit / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1)


def rotation_error(target, actual):
    # World-frame small rotation error; preparation keeps relative angles small.
    return .5 * torch.linalg.cross(actual, target, dim=1).sum(dim=2)


class ContactController:
    APPROACH, ACQUIRE, REGULATE, FAULT = range(4)
    REASONS = ('none', 'axial_load', 'radial_load', 'moment_load', 'nonfinite',
               'travel', 'approach_timeout', 'recovery_budget', 'acquisition_timeout', 'probe_attitude')

    def __init__(self, cfg: ContactConfig, count, device, dtype=torch.float32):
        self.cfg, self.count, self.device = cfg, count, device
        self.state = torch.zeros(count, device=device, dtype=torch.long)
        self.reason = torch.zeros_like(self.state)
        self.recoveries = torch.zeros_like(self.state)
        for n in ('initialized', 'ever_contacted', 'touch_event', 'reanchor_event', 'ready'):
            setattr(self, n, torch.zeros(count, device=device, dtype=torch.bool))
        for n in ('elapsed', 'filtered_force', 'target', 'integral', 'velocity',
                  'acquire_time', 'support_time', 'loss_time', 'recovery_time',
                  'recovery_start', 'travel', 'fault_time', 'angular_drift'):
            setattr(self, n, torch.zeros(count, device=device, dtype=dtype))
        self.anchor_pos = torch.zeros(count, 3, device=device, dtype=dtype)
        self.anchor_rot = torch.eye(3, device=device, dtype=dtype).repeat(count, 1, 1)
        self.last_rot = self.anchor_rot.clone()
        self.xy_reference = self.anchor_pos.clone()
        self.last_xy = self.anchor_pos.clone()
        self.last_twist = torch.zeros(count, 6, device=device, dtype=dtype)
        # Optional experiment-only permission; the production ready gate is unchanged.
        self.diagnostic_xy_permission = None
        self.diagnostic_free_hold = False
        self.history = torch.zeros(count, round(cfg.ready_window/cfg.dt), device=device, dtype=dtype)
        self.history_count = torch.zeros_like(self.state)
        self.cursor = torch.zeros_like(self.state)

    def reset(self, ids=None):
        ids = slice(None) if ids is None else ids
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                value[ids] = 0
        self.anchor_rot[ids] = torch.eye(3, device=self.device, dtype=self.anchor_rot.dtype)
        self.last_rot[ids] = torch.eye(3, device=self.device, dtype=self.anchor_rot.dtype)

    def step(self, wrench, sensor_wrench, pos, rot, xy_action, control_rotation=None, defer_xy_reference=False):
        c, dt = self.cfg, self.cfg.dt
        fresh = ~self.initialized
        self.anchor_pos[:] = torch.where(fresh[:, None], pos, self.anchor_pos)
        self.xy_reference[:] = torch.where(fresh[:, None], pos, self.xy_reference)
        self.anchor_rot[:] = torch.where(fresh[:, None, None], rot, self.anchor_rot)
        self.last_rot[:] = torch.where(fresh[:, None, None], rot, self.last_rot)
        if self.diagnostic_free_hold:
            # Free-space rig only: keep initial axial pose and attitude while
            # exercising the ordinary XY reference update and execution chain.
            frame = self.anchor_rot if control_rotation is None else control_rotation
            local = torch.cat((bounded(xy_action, 1.) * c.xy_speed,
                               torch.zeros_like(xy_action[:, :1])), -1)
            requested = (frame @ local[:, :, None]).squeeze(-1)
            permission = self.diagnostic_xy_permission
            if permission is None:
                permission = torch.zeros_like(self.ready)
            requested = torch.where(permission[:, None], requested, 0)
            if not defer_xy_reference:
                self.xy_reference += requested * dt
            self.last_xy[:] = requested
            self.velocity.zero_()
            self.ready.zero_()
            self.last_twist.zero_()
            self.initialized[:] = True
            self.elapsed += dt
            return self.last_twist
        raw = wrench[:, 2]
        self.filtered_force[:] = torch.where(fresh, raw, (1-c.filter_alpha)*self.filtered_force+c.filter_alpha*raw)
        self.initialized[:] = True
        frame = self.anchor_rot if control_rotation is None else control_rotation
        axis = rot[:, :, 2] if control_rotation is None else frame[:, :, 2]
        self.travel[:] = ((pos-self.anchor_pos)*frame[:, :, 2]).sum(-1)
        finite = torch.isfinite(wrench).all(-1) & torch.isfinite(sensor_wrench).all(-1) & torch.isfinite(pos).all(-1) & torch.isfinite(rot).all(-1).all(-1)
        reason = torch.zeros_like(self.reason)
        reason = torch.where(self.travel > c.max_travel, 5, reason)
        reason = torch.where((self.state == self.APPROACH) & (self.elapsed > c.approach_timeout), 6, reason)
        reason = torch.where(torch.maximum(wrench[:, 3:5].norm(dim=-1), sensor_wrench[:, 3:5].norm(dim=-1)) > c.moment_stop, 3, reason)
        reason = torch.where(wrench[:, :2].norm(dim=-1) > c.radial_stop, 2, reason)
        reason = torch.where(raw.abs() >= c.axial_stop, 1, reason)
        reason = torch.where(~finite, 4, reason)
        new_fault = (reason != 0) & (self.state != self.FAULT)
        self.reason[:] = torch.where(new_fault, reason, self.reason)
        self.state[:] = torch.where(new_fault, self.FAULT, self.state)
        active = self.state != self.FAULT

        touch = active & (self.state == self.APPROACH) & (self.filtered_force >= c.touch_force)
        self.touch_event[:] = touch
        self.reanchor_event[:] = touch | new_fault
        self.ever_contacted |= touch
        self.state[:] = torch.where(touch, self.ACQUIRE, self.state)
        self.target[:] = torch.where(touch, c.buffer_force, self.target)
        self.acquire_time[:] = torch.where(touch, 0, self.acquire_time)
        acquiring = self.state == self.ACQUIRE
        self.acquire_time += acquiring * dt
        self.support_time[:] = torch.where(acquiring & (self.filtered_force >= c.touch_force), self.support_time + dt, 0)
        enter = acquiring & (self.acquire_time >= c.buffer_time) & (self.support_time >= .02)
        self.state[:] = torch.where(enter, self.REGULATE, self.state)
        self.target[:] = torch.where(enter, self.filtered_force.clamp(c.buffer_force, c.target_force), self.target)
        self.integral[:] = torch.where(enter, (self.velocity-c.kp*(self.target-self.filtered_force)).clamp(-c.integral_limit,c.integral_limit), self.integral)
        regulating = self.state == self.REGULATE
        self.target[:] = torch.where(regulating, (self.target+c.target_rate*dt).clamp(max=c.target_force), self.target)
        self.loss_time[:] = torch.where(regulating & (self.filtered_force < c.loss_force), self.loss_time+dt, 0)
        recover = regulating & (self.loss_time >= c.loss_dwell)
        self.recoveries += recover.long()
        self.recovery_start[:] = torch.where(recover, self.travel, self.recovery_start)
        self.recovery_time[:] = torch.where(recover, 0, self.recovery_time)
        self.acquire_time[:] = torch.where(recover, 0, self.acquire_time)
        self.target[:] = torch.where(recover, c.buffer_force, self.target)
        self.integral[:] = torch.where(recover, 0, self.integral)
        self.state[:] = torch.where(recover, self.ACQUIRE, self.state)
        recovering = (self.state == self.ACQUIRE) & (self.recoveries > 0)
        self.recovery_time += recovering*dt
        exhausted = recovering & ((self.recoveries > c.max_recoveries) | (self.recovery_time > c.reacquire_timeout) | (self.travel-self.recovery_start > c.reacquire_travel))
        acquire_failed = (self.state == self.ACQUIRE) & (self.acquire_time > c.acquisition_timeout)
        for mask, code in ((exhausted,7),(acquire_failed,8)):
            new = mask & (self.state != self.FAULT)
            self.reason[:] = torch.where(new, code, self.reason)
            self.state[:] = torch.where(new, self.FAULT, self.state)
            self.reanchor_event |= new

        regulating = self.state == self.REGULATE
        error = self.target-self.filtered_force
        desired = c.kp*error+self.integral
        clipped = desired.clamp(-c.retreat_speed,c.advance_speed)
        integrate = regulating & (self.filtered_force >= c.support_force) & ((desired == clipped) | (error*desired < 0))
        self.integral[:] = torch.where(regulating, (self.integral+integrate*c.ki*error*dt).clamp(-c.integral_limit,c.integral_limit), self.integral)
        smooth = self.velocity+(1-math.exp(-dt/c.velocity_tau))*(clipped-self.velocity)
        smooth = torch.where(recovering & (self.filtered_force < c.touch_force), torch.clamp(smooth,max=c.reacquire_speed), smooth)
        approach = torch.where(self.travel < c.initial_gap-c.slow_zone, c.fast_speed,c.slow_speed)
        request = torch.where(self.state == self.APPROACH, approach, smooth)
        next_v = request.clamp(self.velocity-c.acceleration*dt,self.velocity+c.acceleration*dt)
        # Raw overload and loss have priority over smoothing; no delayed advance into overload.
        next_v = torch.where(raw >= c.operating_upper, next_v.clamp(max=0), next_v)
        next_v = torch.where((self.state != self.APPROACH) & (raw < 1.), next_v.clamp(min=0), next_v)
        fault = self.state == self.FAULT
        self.fault_time += fault*dt
        next_v = torch.where(fault, torch.where(self.fault_time <= .2,-c.retreat_speed,0), next_v)
        next_v = torch.where(finite, next_v, 0)
        self.velocity[:] = next_v

        self.history.scatter_(1,self.cursor[:,None],self.filtered_force[:,None])
        self.cursor[:] = (self.cursor+1)%self.history.shape[1]
        self.history_count[:] = torch.where(regulating,self.history_count+1,0)
        self.ready[:] = (regulating & (self.target >= c.target_force-1e-4)
                         & (self.history_count >= self.history.shape[1])
                         & ((self.history.mean(-1)-c.target_force).abs() <= c.band_half_width)
                         & ((self.history >= c.support_force).float().mean(-1) >= .9)
                         & (raw >= c.support_force) & (raw < c.operating_upper))
        requested_xy = bounded(xy_action,1.) * c.xy_speed
        requested_xy = torch.cat((requested_xy,torch.zeros_like(raw[:,None])),dim=-1)
        requested_world = (frame @ requested_xy[:,:,None]).squeeze(-1)
        permission = self.ready if self.diagnostic_xy_permission is None else self.diagnostic_xy_permission
        requested_world = torch.where(permission[:,None],requested_world,0)
        if not defer_xy_reference:
            self.xy_reference += requested_world*dt
        delta = self.xy_reference-pos
        lateral = delta-(delta*axis).sum(-1)[:,None]*axis
        lateral = bounded(c.lateral_hold_gain*lateral,c.lateral_hold_speed)
        angular = c.orientation_gain*rotation_error(self.anchor_rot,rot)-c.orientation_damping*rotation_error(rot,self.last_rot)/dt
        angular = bounded(angular,c.angular_speed)
        self.angular_drift[:] = rotation_error(self.anchor_rot,rot).norm(dim=-1)
        self.last_rot[:] = rot
        translation = next_v[:,None]*axis+lateral+requested_world
        translation = torch.where(fault[:,None],next_v[:,None]*axis,translation)
        angular = torch.where(fault[:,None],0,angular)
        self.last_xy[:] = requested_world
        self.last_twist[:] = torch.cat((translation,angular),-1)
        self.last_twist[:] = torch.where(finite[:,None],self.last_twist,0)
        self.elapsed += dt
        return self.last_twist
