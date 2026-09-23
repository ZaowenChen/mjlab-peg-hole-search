#!/usr/bin/env python3
"""Plot measured probe tracking and force histories without mixing label columns."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('runs',type=Path,nargs='+')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for root in args.runs:
        manifest=json.loads((root/'manifest.json').read_text())
        for run in sorted(root.glob('amp_*')):
            if not (run/'trace.npz').exists():continue
            data=np.load(run/'trace.npz');x=data['samples'];p=data['probe']
            cols=list(data['columns']);pc=list(data['probe_columns'])
            records=json.loads((run/'results.json').read_text())['rows']
            fig,ax=plt.subplots(3,len(records),figsize=(4*len(records),8),squeeze=False)
            for i,r in enumerate(records):
                if r['fault_code']:
                    # Post-fault constant phase cannot identify a harmonic.
                    r['actual_harmonic_amplitude_deg']=[]
                rows.append(dict(run=str(run),plane=manifest['plane'],**r))
                t=x[:,i,cols.index('time')]
                ax[0,i].plot(t,x[:,i,cols.index('Fz')],lw=.5,label='measured')
                ax[0,i].plot(t,x[:,i,cols.index('filtered_force')],lw=.8,label='filtered')
                ax[0,i].axhline(20,c='k',ls='--',lw=.7)
                ax[0,i].set_title(r['case']['id'])
                for k,color in [('rx','C0'),('ry','C1')]:
                    ax[1,i].plot(t,np.rad2deg(p[:,i,pc.index('target_'+k)]),c=color,ls='--',label='target '+k)
                    ax[1,i].plot(t,np.rad2deg(p[:,i,pc.index('actual_'+k)]),c=color,lw=.7,label='actual '+k)
                for k in ['Mx','My']:ax[2,i].plot(t,x[:,i,cols.index(k)],lw=.5,label=k)
                ax[2,i].set_xlabel('simulation time (s)')
                for j in range(3):ax[j,i].grid(alpha=.2)
            for j,label in enumerate(['normal force (N)','probe angle (deg)','tip moment (Nm)']):
                ax[j,0].set_ylabel(label);ax[j,0].legend(fontsize=7)
            fig.suptitle(f'{root.name} / {run.name}');fig.tight_layout()
            fig.savefig(args.output/f'{root.name}_{run.name}.png',dpi=150);plt.close(fig)
    (args.output/'summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    lines=['# 锥形探测首批数据汇总','', '以下为实际GPU仿真。四方向小样本不构成留出方向泛化证明。统计窗口为末尾6秒（且启动后至少0.5秒）；故障后的力统计包含退离，不能当成运行中的恒力性能。故障工况不报告谐波幅值。','',
           '|几何|摆幅°|工况|启动s|故障|力RMSE N|实际X/Y谐波摆幅°|跟踪RMSE°|低于5N比例|',
           '|---|---:|---|---:|---:|---:|---|---:|---:|']
    for r in rows:
        amp='/'.join(f'{x:.3f}' for x in r.get('actual_harmonic_amplitude_deg',[]))
        start='未启动' if r['started_s'] is None else f"{r['started_s']:.3f}"
        lines.append(f"|{'平面' if r['plane'] else '有孔'}|{r['amplitude_deg']}|{r['case']['id']}|{start}|{r['fault_code']}|{r.get('force_rmse_N',0):.2f}|{amp}|{r.get('tracking_rmse_deg',0):.3f}|{r.get('contact_loss_fraction',0):.1%}|")
    (args.output/'SUMMARY.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
