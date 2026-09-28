# MJLab Peg-Hole Search

[![checks](https://github.com/ZaowenChen/mjlab-peg-hole-search/actions/workflows/checks.yml/badge.svg)](https://github.com/ZaowenChen/mjlab-peg-hole-search/actions/workflows/checks.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

本仓库是当前开发入口，Python 包名为 `mjlab_contact_prep`。运行入口见 [入口清单](docs/ENTRYPOINTS.md)，实验进展见 [当前状态](CURRENT_STATE.md) 和 [实验索引](evaluation/EXPERIMENT_INDEX.md)。

## 开始使用

在仓库根目录执行，实验输出使用新目录。

```bash
# 单元与兼容测试，无需启动 GPU 仿真
bash scripts/run_local.sh python -m unittest discover -s tests -v

# 20 N 接触实验
bash scripts/run_local.sh python scripts/run_contact.py \
  --seconds 12 --output evaluation/my_contact_run

# 锥形探测配对实验
bash scripts/run_local.sh python scripts/run_probe.py \
  --seconds 16 --output evaluation/my_probe_run

# 为新的 SPIKE-013 寻孔实验生成配置、位置清单与源码快照
bash scripts/run_local.sh python scripts/spike013_v02_v03.py \
  plan evaluation/my_search_run
```

寻孔的准备、训练和评估命令见 [入口清单](docs/ENTRYPOINTS.md)。接触、探测、状态准备、训练和评估需要兼容的 CUDA/GPU 环境。`plan` 和源码快照检查不启动物理仿真。

## 当前证据与成功协议

- 原正式基线 `night_fullcircle_5mm_v1`：20 N、全方向 0.5–5 mm、XY 请求 0.2 mm/s；无摆动 667/672，有摆动 635/672。成功为连续 0.2 s 稳定浅入孔。
- SPIKE-013：XY 策略请求 1 mm/s、0.5–10 mm；预先选定 seed 27 / update 256 的独立测试为 247/280，10 mm 固定方向为 23/32。成功需 0.2 s 捕获再保持 0.5 s。

两种协议均未验证完整插入。SPIKE-013 三种子共用初始化，历史合成速度字段无效；1 mm/s 是策略请求上限，实测速度可以超过它。停车和 XY 跟踪候选默认关闭。来源和限制见 [当前状态](CURRENT_STATE.md)。

## 历史复现

历史源码与当前开发代码分开核验。已有本机实验产物时运行：

```bash
# 校验历史源码快照、权重、状态库、日志和统计
bash scripts/run_local.sh python scripts/validate_workspace.py

# 仅核验重放所需源码，无 GPU、无写入
bash scripts/run_local.sh python scripts/evaluate_checkpoint.py \
  --seed 7 --amplitude 0 --tag source_check --check-only

# 使用原始冻结源码重评，写入新标签
bash scripts/run_local.sh python scripts/evaluate_checkpoint.py \
  --seed 7 --amplitude 0 --tag recheck_seed7
```

重放从实验的 `source/` 或本机 `reference/*.tar.gz` 读取匹配原哈希的文件，在新结果目录建立独立源码树。原实验清单和结果不改写。`validate_workspace.py --compare-working-tree` 额外检查当前开发源码是否等于旧正式版本；开发后出现差异是预期情况。见 [兼容说明](docs/COMPATIBILITY.md)。

## 文件布局

| 路径 | 用途 |
|---|---|
| `src/mjlab_contact_prep/` | 当前控制、仿真、环境、状态库、训练和评估代码 |
| `scripts/` | 当前入口及仍用于开发的诊断脚本；见目录 README |
| `configs/` | 当前配置；`ppo.json` 保存受版本管理的 PPO 参数 |
| `docs/` | 架构、配置、入口和兼容说明 |
| `docs/history/` | 注明实验阶段的历史计划与报告 |
| `archive/` | 退出当前入口的源码文本和旧配置，保存迁移前哈希 |
| `evaluation/` | 本机权重、状态库、日志和轨迹；普通 Git 只包含索引 |
| `reference/` | 来源清单；大源码压缩快照只保留在本机 |

归档脚本保存为 `.py.txt`，用于查阅和来源核验。历史脚本中的路径和参数属于当时的版本。新实验从上面的当前入口开始。

## 配置与环境

接触/探测使用 `ContactConfig`，SPIKE-013 使用其专属默认配置 → 实验配置 → 显式覆盖，并保存完整有效配置。PPO 参数来自 `configs/ppo.json`，无需读取旧试验目录。具体合同见 [配置说明](docs/CONFIGURATION.md)。

本机启动器使用 `${MPHS_RUNTIME:-$HOME/.local/share/mphs/runtime-20260917}`，把当前 `src` 放入导入路径。运行时版本见 `reference/runtime_versions.json` 和 `pyproject.toml`。跨机器需自行准备兼容运行时；普通 Git 不包含已训练权重、历史状态库、轨迹和本机源码压缩包。

## 许可

原创代码按 [MIT License](LICENSE) 发布；`_vendor/` 保留 Apache-2.0。模型资产的许可见 [资产与第三方说明](ASSETS_AND_THIRD_PARTY.md)。
