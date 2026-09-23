"""Build tables and plots directly from retained, unfiltered experiment traces."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mjlab_contact_prep.config import ContactConfig
from mjlab_contact_prep.metrics import assess

root=Path(__file__).resolve().parents[1];ev=root/'evaluation';fig=root/'figures';fig.mkdir(exist_ok=True)
load=lambda name:json.loads((ev/name/'results.json').read_text())
runs=['final_offsets','final_repeat2','final_repeat3']
allrows=[]
for name in runs:
 for r in load(name)['rows']:allrows.append(dict(run=name,**r))
with (ev/'final_summary.csv').open('w') as f:
 keys=['run','case','first_touch_s','stable_contact_s','sustained_confirmation_s','sustained_2s','stable_through_end','peak_axial_N','final_pre_fault_force_rmse_N','max_attitude_drift_deg','fault']
 w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
 for r in allrows:w.writerow({k:r['case']['id'] if k=='case' else r[k] for k in keys})
summary=[]
for offset in [1,2,5]:
 group=[r for r in allrows if np.hypot(r['case']['dx_mm'],r['case']['dy_mm'])==offset]
 good=[r for r in group if r['stable_through_end']]
 stable=[r for r in group if r['sustained_2s']]
 summary.append(dict(offset_mm=offset,trials=len(group),first_touch_range_s=[min(r['first_touch_s'] for r in group),max(r['first_touch_s'] for r in group)],stable_count=len(stable),through_end_count=len(good),stable_time_range_s=[min(r['stable_contact_s'] for r in good),max(r['stable_contact_s'] for r in good)] if good else None,peak_force_N=max(r['peak_axial_N'] for r in group),force_rmse_range_N=[min(r['final_pre_fault_force_rmse_N'] for r in group),max(r['final_pre_fault_force_rmse_N'] for r in group)]))
(ev/'aggregate.json').write_text(json.dumps(summary,indent=2)+'\n')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
z=np.load(ev/'final_offsets'/'trace.npz');a=z['samples'];cols=list(z['columns']);idx={k:i for i,k in enumerate(cols)}
f,axs=plt.subplots(3,4,figsize=(15,9),sharex=True,sharey=True,constrained_layout=True)
for j,(ax,r) in enumerate(zip(axs.flat,load('final_offsets')['rows'])):
 t=a[:,j,idx['time']];ax.axhspan(16,24,color='#e6f4ec');ax.plot(t,a[:,j,idx['Fz']],color='#8997a5',lw=.5,label='Raw compensated')
 ax.plot(t,a[:,j,idx['filtered_force']],color='#176b70',lw=.9,label='Filtered');ax.plot(t,a[:,j,idx['target']],color='#cc8730',lw=1,ls='--',label='Target')
 if r['stable_contact_s'] is not None:ax.axvline(r['stable_contact_s'],color='#15803d' if r['stable_through_end'] else '#c87513',lw=1)
 status='held to end' if r['stable_through_end'] else 'temporary only' if r['sustained_2s'] else 'no 2 s interval'
 ax.set_title(r['case']['id']+' | '+status,fontsize=10);ax.set_ylim(-1,41);ax.set_xlim(0,12);ax.grid(alpha=.15)
 if j%4==0:ax.set_ylabel('Axial force (N)')
 if j>=8:ax.set_xlabel('Simulation time (s)')
axs[0,0].legend(fontsize=7,loc='upper left');f.suptitle('20 N contact preparation | exact lateral offsets | primary run',fontsize=15)
f.savefig(fig/'force_offsets.png',dpi=150);plt.close(f)
z=np.load(ev/'final_plane'/'trace.npz');b=z['samples'][:,0];cc=list(z['columns']);get=lambda k:b[:,cc.index(k)];t=get('time')
f,axs=plt.subplots(3,1,figsize=(10,7),sharex=True,constrained_layout=True)
axs[0].axhspan(16,24,color='#e6f4ec');axs[0].plot(t,get('Fz'),label='Raw compensated',lw=.8);axs[0].plot(t,get('target'),'--',label='Target');axs[0].set_ylabel('Axial force (N)');axs[0].legend()
axs[1].plot(t,-get('cmd_vz')*1000,label='Command');axs[1].plot(t,-get('actual_vz')*1000,label='Measured');axs[1].set_ylabel('Downward speed (mm/s)');axs[1].legend()
axs[2].plot(t,np.rad2deg(get('attitude_drift')));axs[2].set_ylabel('Attitude drift (deg)');axs[2].set_xlabel('Simulation time (s)')
for ax in axs:ax.grid(alpha=.2)
f.suptitle('Plane control check | same final controller');f.savefig(fig/'plane_contact.png',dpi=150);plt.close(f)

# Compare actual motion in the accepted local-X direction, not just state flags.
z=np.load(ev/'final_handoff'/'trace.npz');b=z['samples'];cc=list(z['columns']);ci={k:i for i,k in enumerate(cc)};handoff=[]
cfg=ContactConfig(**json.loads((ev/'final_handoff'/'config.json').read_text())['contact'])
for j,r in enumerate(load('final_handoff')['rows']):
 tr=b[:,j];t=tr[:,ci['time']];i0=int(np.argmin(abs(t-6)));i1=int(np.argmin(abs(t-6.5)))
 req=tr[:,[ci['requested_xy_x'],ci['requested_xy_y'],ci['requested_xy_z']]]
 requested=req[i0:i1].sum(0)*cfg.dt;axis=requested/max(np.linalg.norm(requested),1e-15)
 p=tr[:,[ci['tip_x'],ci['tip_y'],ci['tip_z']]]
 post=assess(tr[i1:],cc,cfg)
 handoff.append(dict(case=r['case']['id'],requested_admitted_mm=float(np.linalg.norm(requested)*1000),measured_during_pulse_mm=float((p[i1]-p[i0])@axis*1000),measured_at_7p5_s_mm=float((p[np.argmin(abs(t-7.5))]-p[i0])@axis*1000),post_pulse_stable_s=post['stable_contact_s'],post_pulse_held_to_end=post['stable_through_end'],fault=r['fault']))
(ev/'handoff_summary.json').write_text(json.dumps(handoff,indent=2)+'\n')
print(json.dumps({'aggregate':summary,'handoff':handoff},indent=2))
