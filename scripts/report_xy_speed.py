"""Summarize audited paired speed evaluations without changing their scores."""
import argparse,json,csv
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(out):
 s=json.loads((out/'summary.json').read_text());assert s['complete'];runs=s['runs'];rows=[x for r in runs for x in r['rows']]
 speeds=[.2,.5,1.];radii=[.5,1,2,5]
 def metric(rr):
  ready=[x for x in rr if x['prep_class']=='ready'];ppo=[x for x in ready if x['success']]
  return dict(n=len(rr),success=sum(x['success'] for x in rr),ppo=sum(x['outcome']=='success' for x in rr),eligible=len(ready),
   prep_capture=sum(x['prep_class']=='captured' for x in rr),prep_fail=sum(x['prep_class']=='failed' for x in rr),
   faults=sum(x['outcome']=='fault' for x in rr),timeouts=sum(x['outcome']=='timeout' for x in rr),
   mean_success_ppo_s=float(np.mean([x['search_s'] for x in ppo])) if ppo else None,
   mean_capped_s=float(np.mean([x['capped_search_s'] for x in rr])),
   mean_actual_speed=float(np.mean([x['actual_xy_mean_mm_s'] for x in ready])) if ready else None)
 lines=['# XY速度对照实验结果','',f"全部 {len(rows)} 个模型—位置—速度组合已完成，重训次数为0。评估脚本完整运行耗时 {s['wall_s']/60:.2f} 分钟（不含试跑及报告整理）。",'',
 '模型：冻结无摆动种子7/17/27；偏差0.5/1/2/5 mm；每半径16个全圆新位置；请求速度0.2/0.5/1.0 mm/s。物理步长2ms、PPO周期0.2s、40s寻孔时限及保护不变。',
 '同一独立接触状态配对测试所有模型与速度。逐次核对完整缓存恢复，保存2ms轨迹并独立复核连续0.2s浅入孔判据。物理源码、原实验源码及权重哈希运行前后不变。','',
 '## 总体结果','', '| 请求速度 mm/s | 整体成功/全部 | PPO成功/需寻孔 | 准备已捕获 | 准备失败 | 故障 | 超时 | PPO成功平均s | 失败计40s均值 | 实际XY均速mm/s |',
 '|---|---|---|---|---|---|---|---|---|---|']
 agg={}
 for speed in speeds:
  a=metric([r for r in rows if r['speed_mm_s']==speed]);agg[str(speed)]=a
  lines.append(f"| {speed} | {a['success']}/{a['n']} | {a['ppo']}/{a['eligible']} | {a['prep_capture']} | {a['prep_fail']} | {a['faults']} | {a['timeouts']} | {a['mean_success_ppo_s']:.2f} | {a['mean_capped_s']:.2f} | {a['mean_actual_speed']:.3f} |")
 lines+=['','整体成功包含准备阶段已捕获；它们在每个速度与种子中相同，不能算作PPO成功。寻孔耗时不含固定6s准备，已在准备捕获者寻孔耗时记0。失败按40s计用于比较，并非故障实际发生时间。实际XY均速为各需寻孔回合终止前轨迹路径长/时间的算术平均，包含暂停、修正，不能等同于请求上限。','',
 '## 分偏差结果','', '| 半径mm | 速度mm/s | 整体成功/全部 | PPO成功/需寻孔 | 故障 | 超时 | 成功PPO平均s | 失败计40s均值 |','|---|---|---|---|---|---|---|---|']
 for radius in radii:
  for speed in speeds:
   a=metric([r for r in rows if r['radius_mm']==radius and r['speed_mm_s']==speed])
   avg='—' if a['mean_success_ppo_s'] is None else f"{a['mean_success_ppo_s']:.2f}"
   lines.append(f"| {radius} | {speed} | {a['success']}/{a['n']} | {a['ppo']}/{a['eligible']} | {a['faults']} | {a['timeouts']} | {avg} | {a['mean_capped_s']:.2f} |")
 prep=json.loads((out/'preparation_audit.json').read_text())
 lines+=['','## 准备后实际接手误差','', '| 名义半径mm | 需寻孔位置数 | 准备已捕获位置数 | 需寻孔者实际接手半径范围mm |','|---|---|---|---|']
 for radius in radii:
  rr=[r for r in prep if r['radius_mm']==radius];ready=[r for r in rr if r['prep_class']=='ready']
  span=f"{min(r['actual_handoff_radius_mm'] for r in ready):.3f}–{max(r['actual_handoff_radius_mm'] for r in ready):.3f}" if ready else '—'
  lines.append(f"| {radius} | {len(ready)} | {sum(r['prep_class']=='captured' for r in rr)} | {span} |")
 lines+=['','名义偏差是准备前设置值，不能当成PPO接手时精确误差。全部速度组使用相同的接手状态。','', '## 各个种子','', '| 种子 | 速度mm/s | 成功/全部 | 故障 | 超时 | 失败计40s均值 |','|---|---|---|---|---|---|']
 for r in runs:lines.append(f"| {r['seed']} | {r['speed_mm_s']} | {r['successes']}/{r['total']} | {r['faults']} | {r['timeouts']} | {r['mean_capped_search_s']:.2f} |")
 lines+=['','## 配对变化与失败','']
 baseline={(r['seed'],r['id']):r for r in rows if r['speed_mm_s']==.2}
 for speed in [.5,1.]:
  rr=[r for r in rows if r['speed_mm_s']==speed]
  lost=sum(baseline[(r['seed'],r['id'])]['success'] and not r['success'] for r in rr)
  gained=sum(not baseline[(r['seed'],r['id'])]['success'] and r['success'] for r in rr)
  reasons=Counter(str(r['search_reason']) for r in rr if r['outcome']=='fault')
  lines.append(f'- {speed} mm/s相对0.2：原成功变失败 {lost} 次，原失败变成功 {gained} 次；故障编码 {dict(reasons)}。')
 lines+=['','编码1=轴向超载、2=径向超载、3=力矩超载、7=接触恢复失败（可由时间/行程/次数预算触发）。所有失败保存在逐案结果中。','',
 '## 结论适用范围','',
 '- 此实验判断旧策略直接改变请求速度后的表现；不能据高速失败断言重新训练后的高速策略不可行。',
 '- 三个模型共享64个位置，各速度有192个评估组合，不是192个独立孔位。',
 '- 终点是浅入孔捕获，不是完整插入；新结果不覆盖10–20mm。',
 '- 原始日志中的EPA等数值警告必须随报告保留，缺少对应案例信息时不能据警告行数推断失败因果。',
 '- 保存文件：plan.json、preparation_audit.json、summary.json、每种子/速度的results.json与trajectory.npz。','']
 (out/'REPORT.md').write_text('\n'.join(lines))
 (out/'aggregate.json').write_text(json.dumps(agg,indent=2)+'\n')
 keys=['id','seed','radius_mm','angle_deg','speed_mm_s','prep_class','success','outcome','search_s','search_reason','capped_search_s','actual_xy_mean_mm_s','peak_axial_N','peak_radial_N','peak_moment_Nm','final_error_mm','final_depth_mm']
 with (out/'results.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
 fig,axs=plt.subplots(1,2,figsize=(10,4))
 x=np.arange(4);width=.25
 for i,speed in enumerate(speeds):
  aa=[metric([r for r in rows if r['radius_mm']==radius and r['speed_mm_s']==speed]) for radius in radii]
  axs[0].bar(x+(i-1)*width,[100*a['success']/a['n'] for a in aa],width,label=f'{speed} mm/s')
  axs[1].plot(radii,[a['mean_capped_s'] for a in aa],marker='o',label=f'{speed} mm/s')
 axs[0].set_xticks(x,[str(r) for r in radii]);axs[0].set_ylabel('Pipeline success (%)');axs[0].set_ylim(0,105)
 axs[1].set_ylabel('Mean search time, failures = 40 s')
 for ax in axs:ax.set_xlabel('Nominal offset (mm)');ax.legend();ax.grid(axis='y',alpha=.2)
 fig.tight_layout();fig.savefig(out/'speed_comparison.png',dpi=160);plt.close(fig)
 print(json.dumps(agg,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('output');a=p.parse_args();main(Path(a.output))
