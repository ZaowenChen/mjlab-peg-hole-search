#!/usr/bin/env python3
"""Paired conical-probe experiment, with deployable signals separate from labels."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab_contact_prep.config import ContactConfig
from mjlab_contact_prep.environment import make_env
from mjlab_contact_prep.backend import backend_metadata
from mjlab_contact_prep.execution import COLUMNS
from mjlab_contact_prep.probe import ProbeConfig, PROBE_COLUMNS

def save(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def summarize(trace,probe):
    c={k:trace[:,i] for i,k in enumerate(COLUMNS)}
    p={k:probe[:,i] for i,k in enumerate(PROBE_COLUMNS)}
    started=np.flatnonzero(p['started'])
    window=c['time']>=c['time'][-1]-6
    if len(started):window &= c['time']>=c['time'][started[0]]+.5
    def number(x):return float(x)
    row={'started_s':number(c['time'][started[0]]) if len(started) else None,
         'fault_code':int(c['reason'].max()),'peak_force_N':number(c['Fz'].max()),
         'recovery_count':int(c['recoveries'].max()),'window_s':number(window.sum()*.002)}
    faults=np.flatnonzero(c['reason'])
    row['fault_s']=number(c['time'][faults[0]]) if len(faults) else None
    row['phase_turns']=number((np.unwrap(p['phase'])[-1]-p['phase'][0])/(2*np.pi))
    if window.any():
        phase=p['phase'][window]
        design=np.stack((np.ones_like(phase),np.sin(phase),np.cos(phase)),1)
        actual=np.stack((p['actual_rx'][window],p['actual_ry'][window]),1)
        coef=np.linalg.lstsq(design,actual,rcond=None)[0]
        row.update(force_mean_N=number(c['Fz'][window].mean()),force_rmse_N=number(np.sqrt(np.mean((c['Fz'][window]-20)**2))),
            contact_loss_fraction=number(np.mean(c['Fz'][window]<5)),usable_fraction=number(p['usable'][window].mean()),
            tracking_rmse_deg=number(np.rad2deg(np.sqrt(np.mean(p['tracking_error'][window]**2)))),
            actual_harmonic_amplitude_deg=np.rad2deg(np.linalg.norm(coef[1:],axis=0)).tolist() if not len(faults) and np.linalg.matrix_rank(design)==3 else [],
            angular_limited_fraction=number(p['angular_limited'][window].mean()),
            xy_drift_mm=number(1000*np.linalg.norm(np.array([c['tip_x'][window][-1]-c['tip_x'][window][0],c['tip_y'][window][-1]-c['tip_y'][window][0]]))),
            moment_mean_Nm=[number(c[k][window].mean()) for k in ('Mx','My')])
    return row

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--output',type=Path,required=True)
    a.add_argument('--amplitudes',type=float,nargs='+',default=[0,.05])
    a.add_argument('--offsets-mm',type=float,nargs='+',default=[5])
    a.add_argument('--directions',type=int,default=4)
    a.add_argument('--seconds',type=float,default=12)
    a.add_argument('--plane',action='store_true')
    a.add_argument('--collision-model',choices=['legacy','partitioned'],default='partitioned')
    a.add_argument('--tilt-deg',type=float,default=0)
    a.add_argument('--frequency',type=float,default=.25)
    a.add_argument('--ramp-time',type=float,default=1.)
    a.add_argument('--physics-backend',choices=['stock','contact-fix'],default='contact-fix')
    args=a.parse_args()
    if not 0<args.seconds<=40 or not 1<=args.directions<=16:a.error('invalid duration or directions')
    if not all(math.isfinite(x) and 0<x<=20 for x in args.offsets_mm):a.error('invalid offsets')
    args.output.mkdir(parents=True,exist_ok=False)
    cfg=ContactConfig()
    cases=[dict(id=f'{r:g}mm_{angle:g}deg',dx_mm=r*math.cos(math.radians(angle)),dy_mm=r*math.sin(math.radians(angle)),ry_deg=args.tilt_deg)
           for r in args.offsets_mm for angle in np.arange(args.directions)*360/args.directions]
    root=Path(__file__).resolve().parents[1]
    hashes={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for folder in ('src','scripts','assets') for f in (root/folder).rglob('*') if f.is_file() and '__pycache__' not in f.parts}
    save(args.output/'manifest.json',{'physics_backend':backend_metadata(args.physics_backend),'cases':cases,'plane':args.plane,'collision_model':args.collision_model,'contact':asdict(cfg),'seconds':args.seconds,'source_sha256':hashes,
         'frame':'fixed nominal world-horizontal support, rotation diag(1,-1,-1); compressive Fz positive; moments at geometric peg tip',
         'start':'controller.ready continuously for 0.25s; this is NOT strict sustained raw-force stability',
         'labels':'hole_dx/dy/dz and relative_angle are offline labels only, never controller inputs'})
    rows=[]
    for amplitude in args.amplitudes:
        probe=ProbeConfig(amplitude_deg=amplitude,frequency=args.frequency,ramp_time=args.ramp_time)
        out=args.output/f'amp_{amplitude:g}';out.mkdir()
        save(out/'config.json',{'probe':asdict(probe)})
        save(out/'status.json',{'status':'running'})
        env=None
        try:
            env=ManagerBasedRlEnv(cfg=make_env(cases,cfg,plane=args.plane,seconds=args.seconds,probe=probe,collision_model=args.collision_model,physics_backend=args.physics_backend),device='cuda:0')
            env.reset();term=env.action_manager.get_term('contact');term.record=True
            start=time.monotonic()
            for k in range(math.ceil(args.seconds/env.step_dt)):
                env.step(torch.zeros(len(cases),2,device=env.device))
                if (k+1)%50==0:print('PROGRESS',amplitude,round((k+1)*env.step_dt,2),flush=True)
            torch.cuda.synchronize()
            tr=torch.stack(term.records).cpu().numpy();pr=torch.stack(term.probe_records).cpu().numpy()
            assert tr.shape[-1]==len(COLUMNS) and pr.shape[-1]==len(PROBE_COLUMNS)
            np.savez_compressed(out/'trace.npz',samples=tr,columns=np.array(COLUMNS),probe=pr,probe_columns=np.array(PROBE_COLUMNS))
            result=[dict(case=case,amplitude_deg=amplitude,**summarize(tr[:,i],pr[:,i])) for i,case in enumerate(cases)]
            rows.extend(result)
            save(out/'results.json',{'rows':result,'loop_wall_s':time.monotonic()-start})
            save(out/'status.json',{'status':'complete','physics_executed':True})
            print('RESULT',json.dumps(result),flush=True)
        except BaseException as error:
            save(out/'status.json',{'status':'failed','error':repr(error)});raise
        finally:
            if env is not None:env.close()
    save(args.output/'results.json',{'rows':rows})

if __name__=='__main__':main()
