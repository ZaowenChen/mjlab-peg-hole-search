# 配置合同

当前入口的配置来源如下：

| 入口 | 实际来源 |
|---|---|
| `run_contact.py` | `ContactConfig()` → 显式 `--set KEY=VALUE`；完整字段写入运行目录的 `config.json` |
| `run_probe.py` | `ContactConfig()` 与显式探测参数；写入 `manifest.json` 和每组 `config.json` |
| `spike013_v02_v03.py` | `configs/spike013_default.json` → `configs/spike013_1mms_10mm_v1.json` → `KEY=VALUE`；补全 ContactConfig 字段并保存 `effective_config.json` |
| PPO | `configs/ppo.json` → seed、环境数、更新数等运行参数；写入每种子的 `config.json` |
| 历史重放 | 原 `plan.json`、原种子配置和原源码快照；新增 `replay_provenance.json` |

通用 `resolve_config()` 支持“默认文件 → 实验文件 → 显式覆盖”，但只有实际调用它的入口适用该合同。`configs/default.json` 是通用接触基线文档，不会自动覆盖每个入口。

覆盖值按 JSON 解析。接触命令使用小写 `true`/`false`，例如 `--set parking_enabled=false`；字符串 `"false"` 和数值 `0` 不当作布尔值接受。版本当前为 1，未知文件版本或版本覆盖必须拒绝。

PPO 配置已从原 pilot 的参数提取并纳入 Git，新训练不依赖 `evaluation/gpu_ppo_pilot_v2`。历史冻结源码仍按它当时的配置来源重放。

四份旧设计/范围配置归档到 `archive/configs/`。其中旧 10/20 mm 配置采用两个扇区、0.2 mm/s 及不同预算，不适用于当前 SPIKE-013 的五桶全方向 1 mm/s 协议。

已完成的旧正式实验以原始 `plan.json` 和各种子的 `config.json` 为最终事实；`configs/experiments/night_fullcircle_5mm_v1.json` 只提供索引。
