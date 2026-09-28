# 当前脚本

常用命令、前提和输出规则见 [入口清单](../docs/ENTRYPOINTS.md)。

常用入口为 `run_local.sh`、`run_contact.py`、`run_probe.py`、`spike013_v02_v03.py`、`evaluate_checkpoint.py` 和 `validate_workspace.py`。`smoke_training_pipeline.py` 是 DES-005 历史训练链冒烟工具，需要对应冻结源码和本机实验库。

`spike012_*`、`analyze_spike012_*`、`audit_spike012_e0.py`、`validate_spike012_*`、`validate_xy_tracking.py`、`plot_spike012.py`、`summarize_spike012.py` 用于执行层诊断和候选回归。部分函数被测试或其他诊断脚本复用。

`evaluate_xy_speed.py`、`evaluate_xy_handoff.py`、对应报告脚本及 `compare_low_speed_budgets.py` 用于有明确来源的开发对照。`spike013_overnight.py` 和 `spike013_viewer_replay.py` 绑定特定本机来源实验，分别用于训练队列和可视重放；新实验先使用 `spike013_v02_v03.py`。

旧的一次性训练、物理诊断和报告脚本已退出本目录，存于 [archive/scripts](../archive/scripts/)，保存为源码文本。不要以脚本名称相近为依据替换实验协议。
