"""Offline force stability; fixed criteria, both raw and filtered channels."""
import numpy as np
from .controller import ContactController


def rolling_mean(x,n):
    s=np.concatenate(([0.],np.cumsum(x,dtype=np.float64)))
    return (s[n:]-s[:-n])/n


def stability_mask(trace,columns,cfg,window):
    d={name:trace[:,i] for i,name in enumerate(columns)}
    n=round(window/cfg.dt)
    if len(trace)<n:return np.empty(0,dtype=bool)
    eligible=(d['state']==ContactController.REGULATE)&(d['reason']==0)&(d['target']>=cfg.target_force-1e-4)&(d['attitude_drift']<=np.deg2rad(cfg.max_attitude_drift_deg))
    good=rolling_mean(eligible,n)>=1-1e-10
    for key in ['Fz','filtered_force']:
        f=d[key];err=f-cfg.target_force
        good &= np.abs(rolling_mean(f,n)-cfg.target_force)<=cfg.mean_tolerance
        good &= rolling_mean(err*err,n)<=cfg.rmse_limit**2
        good &= rolling_mean(np.abs(err)<=cfg.band_half_width,n)>=cfg.min_in_band-1e-10
        good &= rolling_mean(f<cfg.support_force,n)<=cfg.max_contact_loss+1e-10
    return good


def assess(trace,columns,cfg):
    if not np.isfinite(trace).all():raise ValueError('non-finite trace')
    d={name:trace[:,i] for i,name in enumerate(columns)}
    if not np.allclose(np.diff(d['time']),cfg.dt,rtol=.002,atol=1e-5):raise ValueError('non-contiguous trace')
    first=lambda mask:float(d['time'][np.flatnonzero(mask)[0]]) if np.any(mask) else None
    # Physical contact can trigger an immediate fault before entering ACQUIRE.
    touched=d['Fz']>=cfg.touch_force
    fault=d['reason']!=0
    short=stability_mask(trace,columns,cfg,cfg.stable_window)
    long=stability_mask(trace,columns,cfg,cfg.sustained_window)
    ns=round(cfg.stable_window/cfg.dt);nl=round(cfg.sustained_window/cfg.dt)
    # Every 250 ms sub-window inside the same 2 s interval must also pass.
    if len(long):long &= rolling_mean(short,nl-ns+1)>=1-1e-10
    first_short=int(np.flatnonzero(short)[0])+ns-1 if np.any(short) else None
    first_long=int(np.flatnonzero(long)[0])+nl-1 if np.any(long) else None
    post_short=short[first_long-ns+1:] if first_long is not None else np.empty(0,dtype=bool)
    before_fault=np.flatnonzero(fault)[0] if np.any(fault) else len(trace)
    last=trace[max(0,before_fault-nl):before_fault]
    initial_xy=d['hole_dx'][0:1]**2+d['hole_dy'][0:1]**2
    out={'first_touch_s':first(touched),'first_ready_s':first(d['ready']>0),
         'first_250ms_stable_s':None if first_short is None else float(d['time'][first_short]),
         'stable_contact_s':None if first_long is None else float(d['time'][first_long-nl+ns]),
         'sustained_confirmation_s':None if first_long is None else float(d['time'][first_long]),
         'sustained_2s':first_long is not None,
         'post_confirmation_stable_fraction':float(post_short.mean()) if len(post_short) else None,
         'stable_through_end':bool(first_long is not None and np.all(post_short)),'first_fault_s':first(fault),
         'fault':ContactController.REASONS[int(d['reason'][np.flatnonzero(fault)[0]])] if np.any(fault) else None,
         'peak_axial_N':float(np.abs(d['Fz']).max()),'peak_radial_N':float(np.hypot(d['Fx'],d['Fy']).max()),
         'max_attitude_drift_deg':float(np.rad2deg(d['attitude_drift']).max()),
         'initial_relative_angle_deg':float(np.rad2deg(d['relative_angle'][0])),
         'initial_radial_error_mm':float(np.sqrt(initial_xy[0])*1000),
         'final_radial_error_mm':float(np.hypot(d['hole_dx'][-1],d['hole_dy'][-1])*1000),
         'recoveries':int(d['recoveries'].max()),'duration_s':float(d['time'][-1]+cfg.dt),
         'peak_measured_angular_speed_rad_s':float(np.sqrt(d['actual_wx']**2+d['actual_wy']**2+d['actual_wz']**2).max())}
    out['final_pre_fault_window_samples']=len(last)
    if len(last):
        f=last[:,columns.index('Fz')]
        out['final_pre_fault_force_mean_N']=float(f.mean())
        out['final_pre_fault_force_rmse_N']=float(np.sqrt(((f-cfg.target_force)**2).mean()))
        out['final_pre_fault_loss_fraction']=float((f<cfg.support_force).mean())
    return out
