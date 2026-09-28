# 当前入口清单

日期：2026-09-28。在仓库根目录运行，使用新的实验目录或结果标签。

## 运行与检查

| 入口 | 用途 | 前提 |
|---|---|---|
| `scripts/run_local.sh` | 选择本机运行时并加载当前源码 | 已安装兼容运行时 |
| `scripts/run_contact.py` | 20 N 接触实验 | CUDA/GPU |
| `scripts/run_probe.py` | 锥形探测配对实验 | CUDA/GPU |
| `scripts/spike013_v02_v03.py` | 新 SPIKE-013 的计划、准备、训练、评估和报告 | 计划不启动仿真；其余物理阶段需要 GPU |
| `scripts/evaluate_checkpoint.py` | 已有权重的原版本重放 | 本机原计划、冻结源码、权重和测试状态库 |
| `scripts/validate_workspace.py` | 原正式实验来源与产物核验 | 本机原始实验数据与源码快照 |
| `python -m unittest discover -s tests -v` | 全部 CPU 单元与兼容测试 | 项目 Python 依赖；不启动物理实验 |

```bash
bash scripts/run_local.sh python scripts/run_contact.py \
  --output evaluation/new_contact --seconds 12 \
  --set parking_enabled=false --set xy_tracking_limits_enabled=false

bash scripts/run_local.sh python scripts/spike013_v02_v03.py plan evaluation/new_search
bash scripts/run_local.sh python scripts/spike013_v02_v03.py prepare evaluation/new_search train
bash scripts/run_local.sh python scripts/spike013_v02_v03.py prepare evaluation/new_search validation
bash scripts/run_local.sh python scripts/spike013_v02_v03.py train evaluation/new_search 7 --envs 64 --iterations 1
bash scripts/run_local.sh python scripts/spike013_v02_v03.py evaluate evaluation/new_search 7 validation
```

上面的 1 次训练更新是链路试跑示例；正式预算应在新实验设计中明确。独立最终集使用 `prepare ... test`、`evaluate ... SEED test`；在查看最终测试前固定模型选择。重复评估使用新目录或标签，当前底层 `evaluate_job` 本身仍允许同名目录，调用方必须保护原结果。

## 历史重放

`evaluate_checkpoint.py --check-only` 按原计划哈希核对源码，不需要 GPU，不创建输出。实际重放把匹配源码写入新结果标签的 `frozen_source/`，由独立进程导入该版本；不会把当前控制实现冒充成历史冻结实现。可用 `--experiment-root` 选择 SPIKE-013 等已有实验，并用 `--checkpoint` 固定检查点。

原正式实验快照存于本机 `reference/*.tar.gz`；SPIKE-013 使用各实验的 `source/`。这些大文件及权重、状态库不在普通 Git 中。克隆仓库可以生成新实验，但要重放本机历史结果需先取得对应产物。

`smoke_training_pipeline.py` 保留为 DES-005 专用冒烟入口，按原正式哈希要求运行；当前开发源码下它会拒绝该旧计划。它不是新实验训练入口。

## 开发诊断

SPIKE-012 执行、停车、XY 跟踪诊断，以及 SPIKE-009/010 速度与交接对照，保留在 `scripts/`。这些工具按明确的历史输入运行；有真值驱动的离线干预不能当成部署控制器。相关共享函数仍被测试和诊断脚本使用。

`spike013_overnight.py` 绑定原小规模实验库，`spike013_viewer_replay.py` 绑定既有验证实验。当前配置变更后旧队列的冻结核验会拒绝重用；新实验使用上面的计划入口。

## 退役文件

29 个旧一次性脚本存为 `archive/scripts/*.py.txt`，四份旧配置存于 `archive/configs/`。原字节与路径映射见 [归档清单](../archive/manifest.json)。删除了 `run_checks.sh` 和 `run_repeats.sh`，它们固定写入历史目录。

早期计划和报告集中在 [docs/history](history/README.md)。新使用者先阅读项目 README 与 CURRENT_STATE。
