# 历史兼容与验证

旧顶层导入继续可用。六个正式最终权重、训练/测试状态库、TensorBoard 日志和原轨迹保留原路径；旧状态库可在内存中升级，原文件不重写。

当前开发源码已扩展停车、XY 跟踪和 SPIKE-013，文件整理也移除了旧入口。历史 `plan.json` 的哈希仍作为原版本合同，不替换成开发版哈希。

```bash
# 校验原哈希对应的源码快照和原实验产物
bash scripts/run_local.sh python scripts/validate_workspace.py

# 严格检查当前源码是否还等于旧正式源码；当前开发版预期失败
bash scripts/run_local.sh python scripts/validate_workspace.py --compare-working-tree

# 单元、配置、公共导入、状态恢复和来源边界测试
bash scripts/run_local.sh python -m unittest discover -s tests -v

# 原版本重放来源检查；只读、无 GPU
bash scripts/run_local.sh python scripts/evaluate_checkpoint.py \
  --seed 7 --amplitude 0 --tag source_check --check-only
```

源码解析依次检查实验 `source/`、归档路径、当前文件与本机源码压缩包，只接受原 SHA-256 匹配的字节。重放在新标签内构建独立源码目录，并由独立 Python 进程导入；历史数据留在原路径。

普通 Git 不包含大源码压缩包和实验数据，因此本机历史产物检查不是干净克隆的 CI 检查。CI 运行语法、JSON，以及不依赖 GPU/历史产物的配置、源码来源和目录边界测试。

2026-09-23 的 [DES-005 验证](REFACTOR_VALIDATION.md)是当时版本的历史记录，不能作为后来源码的物理回归证明。摆动模式存在既有 GPU 重复性限制；本次整理不重训或改写统计。
