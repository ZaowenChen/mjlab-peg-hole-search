# 实验索引

## 当前正式基线

`night_fullcircle_5mm_v1`（2026-09-23）：全方向 0.5–5 mm、20 N、XY 请求速度 0.2 mm/s；无摆动和固定 0.05°/0.25 Hz 摆动各三个种子。

- 原始目录：`evaluation/night_fullcircle_5mm_v1`
- 分析：`evaluation/night_fullcircle_5mm_v1/ANALYSIS_AND_NEXT_STEPS.md`
- 原始计划与实际配置：`plan.json`、各 `amp_*/seed_*/config.json`
- 最终权重：各 `amp_*/seed_*/policy.pt`，共六个
- 汇总：无摆动 667/672；摆动 635/672
- 成功口径：连续 0.2 s 稳定浅入孔；不是完整插入

## 重要历史结果

| 实验 | 说明 | 入口 |
|---|---|---|
| `gpu_contact_candidate_v1` | 冻结 GPU 接触物理候选 | `REPORT_GPU_HANDOFF.md` |
| `gpu_expanded_ab_v3` | 近孔、2 mm、5 mm 扩展分布 A/B | `REPORT_EXPANDED_AB.md` |
| `gpu_unseen_holes_v1` | 独立新孔位测试 | `REPORT_UNSEEN_HOLES.md` |
| `probe_*` / `repaired_*` | 锥形探测与接触定位 | `REPORT_CONICAL_PROBE.md`、`REPORT_CONTACT_LOCALIZATION.md` |

其他 `offsets_*`、`partitioned_*` 与 `final_*` 目录是诊断或中间证据，不能替代当前正式基线。
