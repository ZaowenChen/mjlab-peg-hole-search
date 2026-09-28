"""One effective configuration for the standalone contact experiment."""
from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class ContactConfig:
    dt: float = .002
    target_force: float = 20.
    initial_gap: float = .002
    fast_speed: float = .003
    slow_speed: float = .0003
    slow_zone: float = .0012
    acceleration: float = .02
    touch_force: float = 3.
    support_force: float = 5.
    buffer_force: float = 6.
    buffer_time: float = .06
    target_rate: float = 40.
    kp: float = .00015
    ki: float = .0001
    integral_limit: float = .0003
    velocity_tau: float = .03
    advance_speed: float = .001
    retreat_speed: float = .001
    filter_alpha: float = .3
    loss_force: float = 2.
    loss_dwell: float = .15
    reacquire_speed: float = .0003
    reacquire_timeout: float = 1.
    reacquire_travel: float = .0003
    max_recoveries: int = 2
    max_travel: float = .005
    approach_timeout: float = 6.
    acquisition_timeout: float = 2.
    operating_upper: float = 30.
    axial_stop: float = 40.
    radial_stop: float = 50.
    moment_stop: float = 7.5
    orientation_gain: float = 4.
    orientation_damping: float = .2
    angular_speed: float = .02
    pose_error_scale: float = 5.
    lateral_hold_gain: float = 2.
    lateral_hold_speed: float = .0003
    xy_speed: float = .0002
    xy_tracking_limits_enabled: bool = False  # candidate: plane support regression; opt in
    xy_reference_lead: float = .0003  # m, measured control-point reference leash
    parking_enabled: bool = False  # opt in only after load/support acceptance
    parking_zero_tolerance: float = 1e-7  # m/s, upstream request
    parking_resume_tolerance: float = 2e-7  # m/s
    parking_gate_loss_dwell: float = .01  # s
    parking_resume_dwell: float = .04  # s
    parking_position_gain: float = 5.  # 1/s
    parking_velocity_gain: float = .8  # dimensionless
    parking_recovery_speed: float = .0015  # m/s
    parking_recovery_acceleration: float = .015  # m/s^2
    parking_command_lead: float = .0003  # m
    parking_lookahead: float = .04  # s
    parking_lookahead_distance: float = .00005  # m
    parking_still_speed: float = .00004  # m/s
    parking_still_position: float = .00002  # m
    parking_still_dwell: float = .2  # s
    parking_load_start: float = 22.  # N
    parking_load_end: float = 28.  # N
    parking_load_min_fraction: float = .25
    joint_acceleration: float = 3.
    joint_speed: float = .5
    command_energy: float = 1.
    ready_window: float = .1
    band_half_width: float = 4.
    mean_tolerance: float = 2.
    rmse_limit: float = 4.
    min_in_band: float = .85
    max_contact_loss: float = .05
    stable_window: float = .25
    sustained_window: float = 2.
    max_attitude_drift_deg: float = .5

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool):
                continue
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if self.parking_zero_tolerance >= self.parking_resume_tolerance:
            raise ValueError('parking request hysteresis is invalid')
        if not 0 < self.parking_load_min_fraction <= 1 or not self.parking_load_start < self.parking_load_end <= self.operating_upper:
            raise ValueError('parking load envelope is invalid')
        if not self.touch_force < self.buffer_force < self.target_force < self.operating_upper < self.axial_stop:
            raise ValueError('inconsistent force targets/envelopes')
        if self.slow_zone >= self.initial_gap or self.slow_speed > self.fast_speed:
            raise ValueError('invalid approach schedule')
        if max(self.filter_alpha, self.min_in_band, self.max_contact_loss) > 1:
            raise ValueError('invalid fraction')
        if int(self.max_recoveries) != self.max_recoveries:
            raise ValueError('max_recoveries must be an integer')
        for t in (self.ready_window, self.stable_window, self.sustained_window):
            if not math.isclose(t / self.dt, round(t / self.dt)):
                raise ValueError('window must contain integral physics samples')
