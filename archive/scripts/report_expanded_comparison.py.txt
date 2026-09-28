"""Summarize the predeclared experiment; never select checkpoints or alter training."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];out=root/'evaluation/gpu_expanded_ab_v3'
runs=json.loads((out/'results.json').read_text());assert len(runs)==6
manifest=json.loads((out/'manifest.json').read_text())
prep=json.loads((out/'test_bank/preparation.json').read_text())
trainprep=json.loads((out/'training_bank/preparation.json').read_text())
lines=['# 扩展孔位与大偏差：PPO 无摆动／有摆动对照','', '## 实验范围','', '- 物理模型、20 N 力控、XY 最大请求速度 0.2 mm/s、观测和奖励公式保持不变。缓存复位同时恢复实际速度观测，避免上一回合的速度残留。','- 训练：96 个位置；半径 0.6、0.8、1.0、1.2、2、5 mm；角度为 55° 至 125°、235° 至 305°，间隔 10°。仍是两个角度区间，不是全圆覆盖。','- 测试：72 个独立位置；半径 0.7、0.9、1.1、1.3、2、5 mm；角度为 60° 至 120°、240° 至 300°，间隔 12°。无任何训练孔位组合重复。','- 两组各从头训练 3 个种子（7、17、27），相同种子的初始 actor 参数完全一致；每次 48 次更新、每次 32 步、96 个环境，共 147,456 个决策样本，是旧实验单次 16,384 个样本的 9 倍。','- 两组唯一区别为固定锥形摆动幅度 0°／0.05°（频率 0.25 Hz）。不包含无力觉 C 组，也没有扫描其他摆动参数。','- 寻孔时限统一为 40 秒：5 mm 在当前速度下直达也需要约 25 秒，旧的 12 秒预算不足。成功阈值沿用径向 ≤0.15 mm、深度 ≥0.1 mm、持续 0.2 秒，无故障；只是浅入孔，不是完整插入。','- 训练用共同新建接触状态库；所有训练完成后，另建测试状态库。测试状态只在各策略之间作公平配对，绝未用于训练。6 秒准备失败仍计入总分母。','- 固定使用训练预算结束时的权重，没有根据测试挑选最好轮次。接触准备、在线 ready 与严格 2 秒稳定判据分别报告。','', '## 接触准备','',f'- 训练在线接手通过 {sum(x["eligible"] for x in trainprep)}/{len(trainprep)}，严格 2 秒稳定通过 {sum(x["stats"]["sustained_2s"] for x in trainprep)}/{len(trainprep)}。',f'- 测试在线接手通过 {sum(x["eligible"] for x in prep)}/{len(prep)}，严格 2 秒稳定通过 {sum(x["stats"]["sustained_2s"] for x in prep)}/{len(prep)}。','', '## 各训练种子的独立测试','', '| 种子 | 摆动幅度 | 成功／全部 | 准备失败 | 寻孔超载 | 成功耗时中位数 s | 失败记 40 s 的平均完成时间 |','|---:|---:|---:|---:|---:|---:|---:|']
for r in runs:
 lines.append(f"| {r['seed']} | {r['amplitude_deg']}° | {r['successes']}/{r['total']} | {r['prep_failed']} | {r['search_overloads']} | {r['median_success_s']} | {r['mean_capped_completion_s']:.2f} s |")
lines+=['', '时间从策略接手计起，不含 6 秒接触准备。成功样本中位数存在幸存样本差异；平均完成时间把失败统一记作 40 秒，用于同预算比较，不等于实际故障发生时刻。','', '## 分半径表现（三种子合计）','', '| 半径 mm | 无摆动成功 | 有摆动成功 | 无摆动 90°／270°区间 | 有摆动 90°／270°区间 |','|---:|---:|---:|---|---|']
rad=[.7,.9,1.1,1.3,2.,5.]
strata=[]
for radius in rad:
 vals=[]
 for amp in [0.,.05]:
  rows=[x for r in runs if r['amplitude_deg']==amp for x in r['rows'] if x['radius_mm']==radius]
  a=[x for x in rows if x['angle_deg']<180];b=[x for x in rows if x['angle_deg']>180]
  vals.append((sum(x['success'] for x in rows),len(rows),f"{sum(x['success'] for x in a)}/{len(a)} ／ {sum(x['success'] for x in b)}/{len(b)}"))
 lines.append(f'| {radius:g} | {vals[0][0]}/{vals[0][1]} | {vals[1][0]}/{vals[1][1]} | {vals[0][2]} | {vals[1][2]} |')
 strata.append(dict(radius_mm=radius,arms=vals))
deltas=[]
for seed in manifest['seeds']:
 a=next(r for r in runs if r['seed']==seed and r['amplitude_deg']==0.)
 b=next(r for r in runs if r['seed']==seed and r['amplitude_deg']==.05)
 deltas.append(dict(seed=seed,success_difference_pp=100*(b['overall_success_rate']-a['overall_success_rate']),capped_time_difference_s=b['mean_capped_completion_s']-a['mean_capped_completion_s']))
lines+=['', '## 配对比较与结论边界','']
for x in deltas:lines.append(f"- 种子 {x['seed']}：摆动相对无摆动的成功率差 {x['success_difference_pp']:+.1f} 个百分点，平均完成时间差 {x['capped_time_difference_s']:+.2f} 秒（负值为更快）。")
lines+=['', '只有 3 个训练种子，同一批孔位重复评估不能当作大量独立训练实验；这些是描述性比较。尚不能推出所有摆动幅度都有效或无效，也不能证明直接力觉的因果贡献。扩展后训练分布、预算和寻孔时限均有变化，因此与旧实验的总体成功率不能直接归因于某一项改动。','', '## 数据与复现','', '- `evaluation/gpu_expanded_ab_v3/manifest.json`：训练前固定的样本、预算、评价方式与源文件哈希。','- `training_bank`、`test_bank`：各自独立的接触准备记录。','- `seed_*`：权重、训练日志、逐回合记录、逐孔位测试结果及完整 2 ms 轨迹。','- `integrity.json`：冻结物理与实验源码校验。','- `gpu_expanded_ab_v1` 已被用户新增大偏差要求替代；`v2` 中断并作废，修复缓存速度恢复后统一从头运行 `v3`。这两者不用于比较。','']
(root/'REPORT_EXPANDED_AB.md').write_text('\n'.join(lines))
(out/'comparison.json').write_text(json.dumps(dict(paired_seed_differences=deltas,strata=strata),indent=2)+'\n')
fig,ax=plt.subplots(figsize=(8,4.5))
for amp,color,label,shift in [(0.,'tab:blue','No rocking',-.08),(.05,'tab:orange','Rocking 0.05 deg',.08)]:
 values=np.array([[np.mean([x['success'] for x in r['rows'] if x['radius_mm']==v])*100 for v in rad] for r in runs if r['amplitude_deg']==amp])
 xx=np.arange(len(rad))+shift
 ax.plot(xx,values.mean(0),color=color,label=label,marker='o')
 for row in values:ax.scatter(xx,row,color=color,alpha=.4,s=20)
ax.set_xticks(range(len(rad)),[str(v) for v in rad]);ax.set(xlabel='Initial radius (mm)',ylabel='Held-out capture success (%)',ylim=(-3,103),title='Three seeds, equal budget; dots show individual seeds')
ax.grid(axis='y',alpha=.2);ax.legend();fig.tight_layout();fig.savefig(out/'success_by_radius.png',dpi=170)
print(json.dumps(deltas,indent=2))
