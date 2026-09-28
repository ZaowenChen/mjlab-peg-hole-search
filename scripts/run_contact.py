#!/usr/bin/env python3
"""Measure time to sustained 20 N contact at exact lateral offsets on GPU."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab_contact_prep.config import ContactConfig
from mjlab_contact_prep.configuration import parse_typed_override
from mjlab_contact_prep.environment import make_env
from mjlab_contact_prep.backend import backend_metadata
from mjlab_contact_prep.execution import COLUMNS
from mjlab_contact_prep.metrics import assess


def save(path,x):path.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--offsets-mm',type=float,nargs='+',default=[1,2,5])
    p.add_argument('--directions',nargs='+',choices=['xp','xm','yp','ym'],default=['xp','xm','yp','ym'])
    p.add_argument('--tilt-deg',type=float,default=0.)
    p.add_argument('--plane',action='store_true')
    p.add_argument('--collision-model',choices=['legacy','partitioned'],default='partitioned')
    p.add_argument('--seconds',type=float,default=12.)
    p.add_argument('--xy-pulse',action='store_true',help='bounded 0.5 s +X pulse starting at t=6 s')
    p.add_argument('--set',action='append',default=[],help='ContactConfig key=value candidate override')
    p.add_argument('--physics-backend',choices=['stock','contact-fix'],default='contact-fix')
    args=p.parse_args()
    if not math.isfinite(args.seconds) or not 0<args.seconds<=40:p.error('seconds must be in (0,40]')
    if not all(math.isfinite(x) and 0<=x<=20 for x in args.offsets_mm):p.error('invalid offsets')
    if not math.isfinite(args.tilt_deg) or abs(args.tilt_deg)>2:p.error('invalid tilt')
    cfg=ContactConfig()
    updates={}
    for item in args.set:
        if '=' not in item:p.error('setting must be KEY=VALUE')
        key,value=item.split('=',1)
        if key not in asdict(cfg):p.error(f'unknown setting {key}')
        try:updates[key]=parse_typed_override(value,getattr(cfg,key))
        except ValueError as exc:p.error(str(exc))
    cfg=replace(cfg,**updates)
    directions={'xp':(1,0),'xm':(-1,0),'yp':(0,1),'ym':(0,-1)}
    cases=[dict(id=f'{offset:g}mm_{direction}',dx_mm=offset*directions[direction][0],dy_mm=offset*directions[direction][1],ry_deg=args.tilt_deg) for offset in args.offsets_mm for direction in args.directions]
    if args.plane:cases=[dict(id='plane',dx_mm=0.,dy_mm=0.,ry_deg=args.tilt_deg)]
    args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    versions={'python':sys.version,'torch':torch.__version__}
    import mujoco, mjlab
    versions['mujoco']=mujoco.__version__
    hashes={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for folder in ['src','assets','scripts'] for f in (root/folder).rglob('*') if f.is_file() and '__pycache__' not in f.parts}
    save(args.output/'config.json',{'physics_backend':backend_metadata(args.physics_backend),'collision_model':args.collision_model,'contact':asdict(cfg),'cases':cases,'plane':args.plane,'seconds':args.seconds,'xy_pulse':args.xy_pulse,'backend':'MJLab / MuJoCo-Warp CUDA','versions':versions,'source_sha256':hashes,'criteria':'both raw compensated and filtered; each 250 ms subwindow must pass throughout a 2 s interval; force mean +/-2 N, RMSE<=4 N, >=85% in 16..24 N, <=5% below 5 N; attitude drift<=0.5 deg; REGULATE/no fault/target20'})
    save(args.output/'status.json',{'status':'running'})
    env=None
    try:
        env=ManagerBasedRlEnv(cfg=make_env(cases,cfg,plane=args.plane,seconds=args.seconds,collision_model=args.collision_model,physics_backend=args.physics_backend),device='cuda:0')
        env.reset()
        term=env.action_manager.get_term('contact');term.record=True
        save(args.output/'runtime.json',{'module':str(Path(sys.modules[type(term).__module__].__file__).resolve()),'device':torch.cuda.get_device_name(),'nq':env.sim.mj_model.nq,'nv':env.sim.mj_model.nv,'ngeoms':env.sim.mj_model.ngeom,'dt':env.sim.mj_model.opt.timestep,'num_envs':env.num_envs,'ccd_iterations':env.sim.mj_model.opt.ccd_iterations,'enableflags':env.sim.mj_model.opt.enableflags,'env_spacing':0.,'peg_detection_margin_m':0.,'peg_detection_gap_m':0.})
        torch.cuda.synchronize();start=time.monotonic()
        for k in range(math.ceil(args.seconds/env.step_dt)):
            action=torch.zeros(len(cases),2,device=env.device)
            if args.xy_pulse and 6<=k*env.step_dt<6.5:action[:,0]=1
            env.step(action)
            if (k+1)%50==0:print('PROGRESS',round((k+1)*env.step_dt,2),flush=True)
        torch.cuda.synchronize()
        elapsed=time.monotonic()-start
        traces=torch.stack(term.records).cpu().numpy()
        assert traces.shape[-1]==len(COLUMNS),(traces.shape,len(COLUMNS))
        np.savez_compressed(args.output/'trace.npz',samples=traces,columns=np.asarray(COLUMNS))
        rows=[dict(case=case,**assess(traces[:,i],COLUMNS,cfg)) for i,case in enumerate(cases)]
        save(args.output/'results.json',{'rows':rows,'loop_wall_s':elapsed,'stable_count':sum(x['sustained_2s'] for x in rows),'stable_through_end_count':sum(x['stable_through_end'] for x in rows),'num_cases':len(rows)})
        save(args.output/'status.json',{'status':'complete','physics_executed':True,'performance_all_passed':all(x['stable_through_end'] for x in rows)})
        for r in rows:print('RESULT',json.dumps(r,ensure_ascii=False),flush=True)
    except BaseException as e:
        save(args.output/'status.json',{'status':'failed','error':repr(e)})
        raise
    finally:
        if env is not None:env.close()


if __name__=='__main__':main()
