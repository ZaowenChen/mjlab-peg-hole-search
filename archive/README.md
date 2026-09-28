# 历史源码与配置档案

此目录保存退出当前入口的文件。当前命令见 [入口清单](../docs/ENTRYPOINTS.md)。

`scripts/*.py.txt` 保留迁移前的原始源码字节，避免被误当成当前 Python 入口；`configs/` 保存原设计草案、旧有效配置及旧 10/20 mm 扇区训练方案。原路径、新路径和 SHA-256 见 [manifest.json](manifest.json)。归档内容不接入当前配置解析或训练命令。

已删除 `scripts/run_checks.sh` 与 `scripts/run_repeats.sh`：它们固定写入已有历史目录，是一次性命令；项目测试使用 `python -m unittest discover -s tests -v`。

需要复现历史实验时，使用实验冻结源码和原始计划。`scripts/evaluate_checkpoint.py` 可核验并重放已有权重；历史代码也保留在相应 Git 提交和本机源码快照中。归档源码中的旧路径是当时的记录，不代表当前文件布局。
