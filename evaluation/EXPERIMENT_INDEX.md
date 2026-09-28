# 实验索引

## 当前正式基线

`night_fullcircle_5mm_v1`（2026-09-23）：全方向 0.5–5 mm、20 N、XY 请求速度 0.2 mm/s；无摆动和固定 0.05°/0.25 Hz 摆动各三个种子。

- 原始目录：`evaluation/night_fullcircle_5mm_v1`
- 分析：`evaluation/night_fullcircle_5mm_v1/ANALYSIS_AND_NEXT_STEPS.md`
- 原始计划与实际配置：`plan.json`、各 `amp_*/seed_*/config.json`
- 最终权重：各 `amp_*/seed_*/policy.pt`，共六个
- 汇总：无摆动 667/672；摆动 635/672
- 成功口径：连续 0.2 s 稳定浅入孔；不是完整插入

## 当前开发实验：SPIKE-013

`spike013_1mms_10mm/20260924_overnight_256env_v1`：1 mm/s 策略请求、0.5–10 mm 五桶全方向，40 s 预算。三种子训练和 12 次固定 40 案验证完成；2026-09-28 预先固定 seed 27 / update 256 的独立 280 案测试完成，247/280 成功、33 故障（恢复耗尽 29、轴向超载 4），无准备失败和超时。固定 10 mm 为 23/32。

成功协议为连续 351 个 2 ms 有效采样点（0.2 s 捕获 + 0.5 s 保持），不能与旧正式 0.2 s 协议直接合并。三个训练模型共用初始化；历史 `combined_vx/vy` 无效。公开报告见 [独立最终测试](../docs/reports/SPIKE013_FINAL_TEST_20260928.md)。原始报告、逐案结果和轨迹保留在本机实验目录。该目录的早期验证 `REPORT.md` / `summary.json` 是最终测试前快照，最新状态以 `final_test_status.json` 与 `FINAL_TEST_REPORT.md` 为准。

## 重要历史结果

`spike012_execution_v1`（2026-09-24）：同一 1 mm/s PPO 路线的底层执行诊断。已完成 E0、48 回合 E1/E2 最小矩阵、W2／关通路扩展及一个 A/B 停车候选；候选因余移未达 0.05 mm 且平面载荷退化被拒绝。孔口与原／新 48 回归尚未运行。详见 `evaluation/spike012_execution_v1/REPORT.md`。

`spike012_execution_v1/20260924_i`（2026-09-24）：修正提前停车触发和低力统计后，完成 F0 快照／A/A 审计、24 次同停车前状态 A/B、6 次同上游输入关通路 G。B 保留余移收益，但平面 E2 Dmax 0.385 mm、Fz 峰值 35.08 N，仍拒绝准入；G 余移大于 A。见 `evaluation/spike012_execution_v1/20260924_i/FINAL_REPORT.md`。f/g 为采集失败批次，h 的 G 对照设计无效，最终判断只用 i 批。

`xy_handoff_spike010_v5`（2026-09-23）：高速到真实孔距 1 mm 后减速／制动的开发诊断对照，24 位置 × 3 冻结模型 × 3 组，共 216 回合。捕获后额外保持 0.5 s 的新口径；A/B/C 分别保持通过 71/72、41/72、26/72。详见 `evaluation/xy_handoff_spike010_v5/REPORT.md`。旧 `xy_handoff_spike010_v4` 正式 B/C 因速度限幅错误作废，不用于结论。

| 实验 | 说明 | 入口 |
|---|---|---|
| `gpu_contact_candidate_v1` | 冻结 GPU 接触物理候选 | [历史报告](../docs/history/REPORT_GPU_HANDOFF.md) |
| `gpu_expanded_ab_v3` | 近孔、2 mm、5 mm 扩展分布 A/B | [历史报告](../docs/history/REPORT_EXPANDED_AB.md) |
| `gpu_unseen_holes_v1` | 独立新孔位测试 | [历史报告](../docs/history/REPORT_UNSEEN_HOLES.md) |
| `probe_*` / `repaired_*` | 锥形探测与接触定位 | [历史报告](../docs/history/REPORT_CONICAL_PROBE.md)、[历史报告](../docs/history/REPORT_CONTACT_LOCALIZATION.md) |

其他 `offsets_*`、`partitioned_*` 与 `final_*` 目录是诊断或中间证据，不能替代当前正式基线。
