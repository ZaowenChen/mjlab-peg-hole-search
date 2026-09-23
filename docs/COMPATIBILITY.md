# 历史兼容与验证

## 保留内容

- 顶层旧导入（如 `mjlab_contact_prep.environment`、`search_env`、`night_run`）继续可用。
- 正式实验引用的源文件未改动，`plan.json` 中冻结哈希仍能通过。
- 六个最终 `policy.pt`、训练/测试状态库、TensorBoard 日志和六份轨迹保留原路径。
- 状态库版本 0 可由 `data.load_state_bank()` 显式升级为内存中的版本 1，原文件不重写。

## 检查命令

```bash
bash scripts/run_local.sh python scripts/validate_workspace.py
bash scripts/run_local.sh python -m unittest discover -s tests -v
```

第一条只读取源码哈希和正式产物，不启动训练或仿真。第二条包含既有控制/观测测试和新增的配置、缓存、公共导入兼容测试。

涉及 GPU 动力学的短评估与训练冒烟仍应在明确需要时单独运行；结构重构本身不把历史训练重跑一遍。
