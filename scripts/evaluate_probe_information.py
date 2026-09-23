#!/usr/bin/env python3
"""Trajectory-held-out direction diagnostic. Never trains on adjacent test frames."""
import argparse
import json
from pathlib import Path
import numpy as np

def extract(root,horizon):
    manifest=json.loads((root/'manifest.json').read_text());rows=[]
    for run in sorted(root.glob('amp_*')):
        if not (run/'trace.npz').exists():continue
        cfg=json.loads((run/'config.json').read_text())['probe']
        data=np.load(run/'trace.npz');names=list(data['columns']);pn=list(data['probe_columns'])
        samples=data['samples'];probes=data['probe'];data.close()
        for i,case in enumerate(manifest['cases']):
            x=samples[:,i];p=probes[:,i];t=x[:,names.index('time')]
            start=np.flatnonzero(p[:,pn.index('started')])
            row=dict(case=case,amplitude=cfg['amplitude_deg'],plane=manifest['plane'],run=str(run),valid=False)
            if len(start):
                begin=t[start[0]]+cfg['ramp_time']
                mask=(t>=begin)&(t<begin+horizon)
                if mask.sum()>=int(.99*horizon/.002) and not np.any(x[t<begin+horizon,names.index('reason')]):
                    force=x[mask][:,[names.index(k) for k in ('Fx','Fy','Fz','Mx','My','Mz')]]
                    phase=p[mask,pn.index('phase')]
                    s=np.sin(phase);s-=s.mean();c=np.cos(phase);c-=c.mean()
                    # Correlations, not ill-conditioned partial-cycle sinusoid fits.
                    centered=force-force.mean(0)
                    z=x[mask,names.index('tip_z')];z=z-z[0]
                    ang=p[mask][:,[pn.index('actual_rx'),pn.index('actual_ry')]]
                    f=np.r_[force.mean(0),force.std(0),(centered*s[:,None]).mean(0),(centered*c[:,None]).mean(0),
                            ang.mean(0),ang.std(0),np.mean(z*s),np.mean(z*c)]
                    # Physical true direction at the beginning of this window,
                    # offline only. Horizontal hole axes match world XY in yaw.
                    k=np.flatnonzero(mask)[0]
                    label=-x[k,[names.index('hole_dx'),names.index('hole_dy')]]
                    row.update(valid=True,features=f,label=label/np.linalg.norm(label))
            rows.append(row)
    return rows

def fit(train):
    x=np.stack([r['features'] for r in train]);y=np.stack([r['label'] for r in train])
    mu=x.mean(0);scale=x.std(0);scale=np.maximum(scale,1e-6)
    z=np.c_[np.ones(len(x)),(x-mu)/scale]
    reg=np.eye(z.shape[1]);reg[0,0]=0
    weights=np.linalg.solve(z.T@z+reg,z.T@y)
    return mu,scale,weights

def evaluate(rows,model):
    mu,scale,w=model;out=[]
    for r in rows:
        pred=np.r_[1,(r['features']-mu)/scale]@w
        cosine=float(np.clip(pred@r['label']/max(np.linalg.norm(pred),1e-12),-1,1))
        out.append(dict(case=r['case'],angle_error_deg=float(np.degrees(np.arccos(cosine))),correct_halfplane=cosine>0,prediction=pred.tolist()))
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train',type=Path,nargs='+',required=True)
    p.add_argument('--test',type=Path,required=True)
    p.add_argument('--plane',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(exist_ok=False,parents=True)
    results=[]
    for horizon in [.5,1,2,4]:
        train=sum([extract(r,horizon) for r in args.train],[]);test=extract(args.test,horizon)
        amplitudes=sorted(set(r['amplitude'] for r in train))
        key=lambda r:(str(Path(r['run']).parent),r['case']['id'])
        common_train=set.intersection(*[{key(r) for r in train if r['valid'] and r['amplitude']==amp} for amp in amplitudes])
        common_test=set.intersection(*[{key(r) for r in test if r['valid'] and r['amplitude']==amp} for amp in amplitudes])
        for amplitude in amplitudes:
            cardinal=lambda r:abs(r['case']['dx_mm'])<1e-5 or abs(r['case']['dy_mm'])<1e-5
            tr=[r for r in train if r['valid'] and key(r) in common_train and r['amplitude']==amplitude and cardinal(r)]
            te=[r for r in test if r['valid'] and key(r) in common_test and r['amplitude']==amplitude and not cardinal(r)]
            total=sum(r['amplitude']==amplitude and not cardinal(r) for r in test)
            if len(tr)<4 or not te:continue
            model=fit(tr);pred=evaluate(te,model)
            result=dict(horizon_s=horizon,amplitude_deg=amplitude,train_count=len(tr),test_eligible=len(te),test_total=total,
                median_angle_error_deg=float(np.median([r['angle_error_deg'] for r in pred])),
                correct_halfplane_fraction=float(np.mean([r['correct_halfplane'] for r in pred])),predictions=pred)
            if args.plane:
                plane=[r for r in extract(args.plane,horizon) if r['valid'] and r['amplitude']==amplitude]
                result['plane_diagnostic_only']=evaluate(plane,model)
            results.append(result)
            np.savez(args.output/f'model_amp{amplitude:g}_history{horizon:g}.npz',mean=model[0],scale=model[1],weights=model[2])
    (args.output/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k not in ('predictions','plane_diagnostic_only')} for r in results],indent=2))

if __name__=='__main__':main()
