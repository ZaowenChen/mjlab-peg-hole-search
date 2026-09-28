"""Read-only, physics-sampled shallow-entry completion protocol."""
from __future__ import annotations

import math
import torch

PROTOCOL_ID = "autonomous_shallow_hold_v1"


def observed_reason(controller_reason, tip_wrench, sensor_wrench, contact):
    """Read current post-integration load using the controller's hard limits."""
    finite = torch.isfinite(tip_wrench).all(-1) & torch.isfinite(sensor_wrench).all(-1)
    reason = torch.zeros_like(controller_reason)
    moment = torch.maximum(tip_wrench[:, 3:5].norm(dim=-1),
                           sensor_wrench[:, 3:5].norm(dim=-1))
    reason = torch.where(moment > contact.moment_stop, 3, reason)
    reason = torch.where(tip_wrench[:, :2].norm(dim=-1) > contact.radial_stop, 2, reason)
    reason = torch.where(tip_wrench[:, 2].abs() >= contact.axial_stop, 1, reason)
    reason = torch.where(~finite, 4, reason)
    return torch.where(controller_reason != 0, controller_reason, reason)


def interval_count(seconds: float, dt: float) -> int:
    ratio = seconds / dt
    rounded = round(ratio)
    return int(rounded if math.isclose(ratio, rounded, rel_tol=0, abs_tol=1e-9) else math.ceil(ratio))


class ShallowHold:
    def __init__(self, count: int, device: str, dt: float, capture_s: float = .2, hold_s: float = .5):
        self.dt = dt
        self.capture_samples = 1 + interval_count(capture_s, dt)
        self.stable_samples = self.capture_samples + interval_count(hold_s, dt)
        self.valid_run_samples = torch.zeros(count, dtype=torch.long, device=device)
        self.max_run_samples = torch.zeros_like(self.valid_run_samples)
        self.ever_captured = torch.zeros(count, dtype=torch.bool, device=device)
        self.first_capture_tick = torch.full((count,), -1, dtype=torch.long, device=device)
        self.capture_count = torch.zeros_like(self.valid_run_samples)
        self.hold_break_count = torch.zeros_like(self.valid_run_samples)
        self.fault_seen = torch.zeros(count, dtype=torch.bool, device=device)
        self.first_fault_reason = torch.zeros_like(self.valid_run_samples)
        self.nonfinite_seen = torch.zeros(count, dtype=torch.bool, device=device)
        self.ticks = torch.zeros_like(self.valid_run_samples)
        self.first_stable_tick = torch.full((count,), -1, dtype=torch.long, device=device)

    def reset(self, ids):
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                value[ids] = -1 if name in ("first_capture_tick", "first_stable_tick") else 0

    def state_dict(self):
        return {k: v.detach().cpu().clone() for k, v in vars(self).items() if isinstance(v, torch.Tensor)}

    def load_state_dict(self, state):
        for name, target in vars(self).items():
            if isinstance(target, torch.Tensor):
                if name not in state or state[name].shape != target.shape:
                    raise ValueError(f"missing or incompatible shallow-hold state: {name}")
                target[:] = state[name].to(target.device)

    def sample(self, radial, depth, wrench, reason, *, active=None):
        if active is None:
            active = torch.ones_like(self.ticks, dtype=torch.bool)
        finite = torch.isfinite(radial) & torch.isfinite(depth) & torch.isfinite(wrench).all(-1)
        nonfinite = active & (~finite | (reason == 4))
        self.nonfinite_seen |= nonfinite
        first_fault = active & (reason != 0) & ~self.fault_seen
        self.first_fault_reason[first_fault] = reason[first_fault]
        self.fault_seen |= active & (reason != 0)
        valid = active & finite & (radial <= .00015) & (depth >= .0001) & (reason == 0)
        prior = self.valid_run_samples.clone()
        broke = active & ~valid & (prior >= self.capture_samples)
        self.hold_break_count[broke] += 1
        self.valid_run_samples[active] = torch.where(valid[active], prior[active] + 1, 0)
        self.max_run_samples = torch.maximum(self.max_run_samples, self.valid_run_samples)
        captured = active & (self.valid_run_samples == self.capture_samples)
        first = captured & ~self.ever_captured
        self.first_capture_tick[first] = self.ticks[first] + 1
        self.capture_count[captured] += 1
        self.ever_captured |= captured
        stable = active & (self.valid_run_samples == self.stable_samples) & (self.first_stable_tick < 0)
        self.first_stable_tick[stable] = self.ticks[stable] + 1
        self.ticks[active] += 1
        return finite

    def outcome(self, timeout):
        success = (self.valid_run_samples >= self.stable_samples) & ~self.fault_seen & ~self.nonfinite_seen
        fault = self.fault_seen & ~self.nonfinite_seen
        pure_timeout = timeout & ~success & ~fault & ~self.nonfinite_seen
        done = success | fault | timeout | self.nonfinite_seen
        return done, success, fault, pure_timeout


def audit_trace(trace, columns, dt, stable_samples):
    """Rebuild the final run using saved measurements, without online counters."""
    if len(trace) == 0:
        return dict(valid_run_samples=0, fault=False, nonfinite=False)
    idx = {name: columns.index(name) for name in ("radial_m", "depth_m", "reason", "Fx", "Fy", "Fz", "Mx", "My", "Mz")}
    r = trace[:, idx["radial_m"]]
    d = trace[:, idx["depth_m"]]
    reason = trace[:, idx["reason"]]
    wrench = trace[:, [idx[k] for k in ("Fx", "Fy", "Fz", "Mx", "My", "Mz")]]
    import numpy as np
    finite = np.isfinite(r) & np.isfinite(d) & np.isfinite(wrench).all(axis=1)
    valid = finite & (r <= .00015) & (d >= .0001) & (reason == 0)
    bad = np.flatnonzero(~valid)
    tail = len(valid) - (int(bad[-1]) + 1 if len(bad) else 0)
    if "time" in columns and len(trace) > 1:
        times = trace[:, columns.index("time")]
        if not np.allclose(np.diff(times), dt, rtol=0, atol=4e-6):
            raise AssertionError("physics trace has a duplicate or missing sample")
    return dict(valid_run_samples=tail, success=tail >= stable_samples,
                fault=bool(np.any(reason != 0)), nonfinite=bool(np.any(~finite) or np.any(reason == 4)),
                final_dwell_s=max(0, tail - 1) * dt)
