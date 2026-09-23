# MJLab Peg-Hole Search 统一工作区

[![checks](https://github.com/ZaowenChen/mjlab-peg-hole-search/actions/workflows/checks.yml/badge.svg)](https://github.com/ZaowenChen/mjlab-peg-hole-search/actions/workflows/checks.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

这是 MJLab Peg-Hole Search 的统一代码仓库。旧工作区和原始目录仅作为历史参考，不再作为新修改入口；详见 [历史目录索引](docs/HISTORICAL_WORKSPACES.md)。

## 当前基线

正式基线为 `evaluation/night_fullcircle_5mm_v1`：20 N、全方向 0.5–5 mm、XY 请求速度 0.2 mm/s，无摆动及固定 0.05°/0.25 Hz 摆动各三个种子。独立 224 孔位评估合计：

- 无摆动：667/672；
- 有摆动：635/672；
- 成功口径：连续 0.2 s 稳定浅入孔，不是完整插入。

结论、准备准入异常和下一步见 [当前状态](CURRENT_STATE.md) 与 [实验索引](evaluation/EXPERIMENT_INDEX.md)。当前建议以无摆动配置作为近孔基线；5 mm 尚未完全可靠，完整插入和 2 秒物理准备尚未验证。

## 开始使用

在本目录执行。运行输出必须使用新目录，不能覆盖已有实验。

```bash
# 只读核对源码哈希、六个权重、状态库、日志、轨迹和正式统计
bash scripts/run_local.sh python scripts/validate_workspace.py

# 全部单元与兼容测试
bash scripts/run_local.sh python -m unittest discover -s tests -v

# 20 N 接触实验
bash scripts/run_local.sh python scripts/run_contact.py \
  --seconds 12 --output evaluation/my_contact_run

# 锥形探测配对实验
bash scripts/run_local.sh python scripts/run_probe.py \
  --seconds 16 --output evaluation/my_probe_run
```

准备阶段和训练队列的历史入口保留在 `scripts/night_experiment.py`；正式实验的实际参数以原始 `plan.json` 和每种子的 `config.json` 为准。重放现有最终权重时必须写入新标签：

```bash
bash scripts/run_local.sh python scripts/evaluate_checkpoint.py \
  --seed 7 --amplitude 0 --tag recheck_seed7
```

该命令使用冻结测试状态库做独立评估，并保存轨迹审计和完整有效配置。它需要 GPU，但不会训练或覆盖原 `evaluation/` 结果。

## 代码结构

| 路径 | 职责 |
|---|---|
| `src/mjlab_contact_prep/simulation/` | 模型、碰撞、传感器、坐标系、后端和复位 |
| `src/mjlab_contact_prep/control/` | 下降、姿态、力控、保护、接触恢复与执行 |
| `src/mjlab_contact_prep/envs/` | PPO 观测、动作、奖励与终止 |
| `src/mjlab_contact_prep/data/` | 完整状态库、格式版本与旧缓存转换 |
| `src/mjlab_contact_prep/training/` | PPO、检查点与训练队列 |
| `src/mjlab_contact_prep/evaluation/` | 独立评估、轨迹审计与统计 |
| `scripts/` | 薄运行入口 |
| `evaluation/` | 模型、状态库、日志和轨迹等运行产物 |

正式实验冻结时使用的顶层模块与 `night_run/` 文件保持不变，新职责子包提供后续开发入口，旧导入仍兼容。详细边界见 [架构说明](docs/ARCHITECTURE.md)。

本轮测试、权重重评和训练链冒烟见 [DES-005 兼容验证](docs/REFACTOR_VALIDATION.md)。无摆动代表模型逐案例一致；摆动代表模型在冻结代码下重复评估仍有成功/故障分类波动，原正式统计未被改写。

## 配置与环境

新实验使用“`configs/default.json` → 实验配置 → 命令行覆盖”的顺序，并把最终有效配置保存到运行目录。合同见 [配置说明](docs/CONFIGURATION.md)。不要修改全局默认值来复现历史实验。

本机启动器默认使用 `${MPHS_RUNTIME:-$HOME/.local/share/mphs/runtime-20260917}`，并把当前工作区 `src` 放入导入路径。已记录版本见 `reference/runtime_versions.json`；跨机器需要准备兼容的 CUDA、MJLab 1.3.0、MuJoCo 3.7.0、MuJoCo-Warp 3.7.0.1 和 RSL-RL 5.0.1 环境。

独立 Git 基线和来源记录见 `reference/workspace_origin.json`。原始模型、缓存、日志、轨迹和源码快照继续保留在本机，不进入普通 Git。

## 许可

项目原创代码按 [MIT License](LICENSE) 发布。`src/mjlab_contact_prep/_vendor/` 保留其 Apache-2.0 许可；机器人、末端和孔模型资产不自动适用 MIT，使用前请阅读 [资产与第三方说明](ASSETS_AND_THIRD_PARTY.md)。
