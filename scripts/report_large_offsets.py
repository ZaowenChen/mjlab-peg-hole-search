"""Summarize the unchanged 20 N controller at 10/20 mm lateral offsets."""
import json,csv,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];ev=root/'evaluation'
runs=[f'large_offsets_r{i}' for i in [1,2,3]];rows=[]
base=json.loads((ev/'final_offsets/config.json').read_text())
for name in runs:
 cfg=json.loads((ev/name/'config.json').read_text())
 assert cfg['contact']==base['contact']
 for p,h in base['source_sha256'].items():
  if p.startswith(('src/','assets/')):assert cfg['source_sha256'][p]==h==hashlib.sha256((root/p).read_bytes()).hexdigest()
 for r in json.loads((ev/name/'results.json').read_text())['rows']:rows.append(dict(run=name,**r))
summary=[]
for offset in [10,20]:
 rr=[r for r in rows if r['case']['id'].startswith(f'{offset}mm')];good=[r for r in rr if r['stable_through_end']];stable=[r for r in rr if r['sustained_2s']]
 def span(xs):return [min(xs),max(xs)] if xs else None
 summary.append(dict(offset_mm=offset,trials=len(rr),touch_count=sum(r['first_touch_s'] is not None for r in rr),touch_range_s=span([r['first_touch_s'] for r in rr if r['first_touch_s'] is not None]),sustained_count=len(stable),held_to_end_count=len(good),held_stable_time_range_s=span([r['stable_contact_s'] for r in good]),sustained_time_range_s=span([r['stable_contact_s'] for r in stable]),peak_force_N=max(r['peak_axial_N'] for r in rr),fault_count=sum(r['fault'] is not None for r in rr),max_attitude_drift_deg=max(r['max_attitude_drift_deg'] for r in rr)))
(ev/'large_offsets_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
keys=['run','case','first_touch_s','stable_contact_s','sustained_confirmation_s','sustained_2s','stable_through_end','peak_axial_N','final_pre_fault_force_rmse_N','max_attitude_drift_deg','fault']
with (ev/'large_offsets_summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
 for r in rows:w.writerow({k:r['case']['id'] if k=='case' else r[k] for k in keys})
z=np.load(ev/runs[0]/'trace.npz');a=z['samples'];c=list(z['columns']);idx={k:i for i,k in enumerate(c)}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
f,axs=plt.subplots(2,4,figsize=(15,6.5),sharex=True,sharey=True,constrained_layout=True)
for j,ax in enumerate(axs.flat):
 r=rows[j];t=a[:,j,idx['time']];ax.axhspan(16,24,color='#e6f4ec');ax.axhline(40,ls=':',color='#bd3b3b',lw=.8)
 ax.plot(t,a[:,j,idx['Fz']],color='#8997a5',lw=.5,label='Raw compensated');ax.plot(t,a[:,j,idx['filtered_force']],color='#176b70',lw=.8,label='Filtered');ax.plot(t,a[:,j,idx['target']],color='#cc8730',ls='--',lw=1,label='Target')
 if r['stable_contact_s'] is not None:ax.axvline(r['stable_contact_s'],color='#15803d' if r['stable_through_end'] else '#c87513',lw=1)
 status='held to end' if r['stable_through_end'] else 'temporary only' if r['sustained_2s'] else 'no 2 s interval'
 ax.set_title(r['case']['id']+' | '+status,fontsize=10);ax.set_ylim(-1,43);ax.set_xlim(0,12);ax.grid(alpha=.15)
 if j%4==0:ax.set_ylabel('Axial force (N)')
 if j>=4:ax.set_xlabel('Simulation time (s)')
axs[0,0].legend(fontsize=7);f.suptitle('20 N contact | 10 and 20 mm offsets | first repeat',fontsize=15)
f.savefig(root/'figures/large_offsets_force.png',dpi=150);plt.close(f)
fmt=lambda x:'未通过' if x is None else f'{x:.2f}'
lines=['# 10 mm、20 mm 横向偏差接触实验','', '日期：2026-09-22。延续 [1/2/5 mm 实验报告](REPORT.md)，本次未修改控制器、模型或稳定判据。','', '## 结果','', '每种偏差包含孔位 X+/X−/Y+/Y− 四个方向，每个方向重复三次，共 24 个工况。每个回合观察 12 s。以下时间是从复位后首次控制开始计算的仿真时间。','', '| 偏差 | 首次接触 s | 稳定建立 s，仅最终保持合格样本 | 连续稳定 2 s | 保持到 12 s | 轴向峰值 N | 硬故障 |','|---|---|---|---|---|---|---|']
for g in summary:
 t=g['touch_range_s'];s=g['held_stable_time_range_s'];st='无' if s is None else f'{s[0]:.2f}–{s[1]:.2f}'
 lines.append(f"| {g['offset_mm']} mm | {t[0]:.2f}–{t[1]:.2f} | {st} | {g['sustained_count']}/12 | {g['held_to_end_count']}/12 | {g['peak_force_N']:.2f} | {g['fault_count']}/12 |")
lines += ['', '**可以形成接触，但两种偏差都尚未达到全部方向、全部重复持续稳定的目标。**合格样本的时间范围不包含失败样本，不能视为整个偏差组的可靠准备时间。固定方向的重复次数较少，表中比例不是随机总体成功率。','',
'## 判据与条件','',
'- 原 20 N 控制配置、原详细孔与轴资产、2 ms 物理步长均保留；代码和资产哈希逐项核对，与此前正式实验一致。每轮 8 个 GPU 环境，世界原点重合且各环境独立。',
'- 初始间隙参数 2 mm；不施加额外孔倾角，实际模型初始相对倾角约 0.539°；无 XY/旋转搜索指令。偏差是径向精确偏移，不是随机误差上限。',
'- 首次接触：原始补偿轴向力达到 3 N。稳定：原始力与滤波力同时满足每 250 ms 窗口均值 20±2 N、RMSE≤4 N、至少 85% 样本在 16–24 N，低于 5 N 的样本≤5%，且无故障、目标为 20 N、姿态漂移≤0.5°。',
'- 连续 2 s 的所有短窗口均合格后，回溯其首个 250 ms 窗口结束时刻作为“稳定建立”；确认时间晚 1.75 s。另检查确认后是否保持到 12 s。',
'- 很晚才稳定的回合，确认后的观察很短。例如约 10.25 s 才建立、约 12 s 才确认的结果，仅证明末段 2 s 合格，不能称为长期稳定。','',
'## 各方向三轮结果','', '| 方向 | 三轮稳定建立时间 s | 连续 2 s 通过 | 保持至结束 |','|---|---|---|---|']
for offset in [10,20]:
 for direction in ['xp','xm','yp','ym']:
  rr=[r for r in rows if r['case']['id']==f'{offset}mm_{direction}'];lines.append(f"| {offset}mm_{direction} | {' / '.join(fmt(r['stable_contact_s']) for r in rr)} | {sum(r['sustained_2s'] for r in rr)}/3 | {sum(r['stable_through_end'] for r in rr)}/3 |")
lines += ['', '下图为首轮；三轮完整数值见 `evaluation/large_offsets_summary.csv`。','', '![大偏差原始力曲线](figures/large_offsets_force.png)','',
'## 如何理解大偏差','',
'详细孔模型的碰撞几何外包围范围约为孔中心 X/Y 各 ±45 mm。轴半径约 17.45 mm，中心偏移 20 mm 时，轴端面投影仍在这个外包围范围内；实际模拟也验证所有测试方向都能触碰实体。包围范围本身不证明内部每一点都有支撑，实际接触仍由原详细模型计算。','',
'这里主要是在孔口周边实体表面建立压靠。偏差变大，不必然延长下降接触时间；接触位置、倒角支撑与接触点生成会影响稳定性。**接触成功不等于可以从 10/20 mm 偏差寻孔，更不代表完成插入或 PPO 已能学会。**','',
'部分回合的原始力仍明显跳变。此前已在 2 mm 工况复现 CPU/GPU 接触点不一致；本次没有对 10/20 mm 再做同状态 CPU 对照，因此不能把本轮所有抖动直接归因为同一引擎问题。','',
'本轮结论支持继续把注意力放在接触稳定性；暂不扩大到更大的 PPO 搜索范围，也没有为了通过实验修改力带或硬故障阈值。','',
'## 证据与复现','',
'- 原始结果与轨迹：`evaluation/large_offsets_r1`、`large_offsets_r2`、`large_offsets_r3`，各含配置、源码/资产哈希、运行信息、结果、状态和每 2 ms 的轨迹。',
'- 汇总：`evaluation/large_offsets_summary.json`、`large_offsets_summary.csv`。',
'- 几何检查：`evaluation/large_offsets_geometry.json`。',
'- 报告生成：`scripts/report_large_offsets.py`。','',
'```bash','cd /path/to/mjlab-peg-hole-search','bash scripts/run_local.sh python scripts/run_contact.py --offsets-mm 10 20 --seconds 12 --output evaluation/large_offsets_new','```','', '输出目录必须是新目录。']
(root/'REPORT_10_20mm.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,indent=2))
