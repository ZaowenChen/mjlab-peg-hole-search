# 配置合同

新实验按以下顺序解析配置：

1. `configs/default.json`；
2. `configs/experiments/<experiment>.json`；
3. 明确的命令行 `KEY=VALUE` 覆盖。

`mjlab_contact_prep.configuration.resolve_config()` 返回完整有效配置以及三个来源，运行入口应使用 `write_effective_config()` 将结果原样保存到输出目录。覆盖值按 JSON 解析，例如 `contact.xy_speed=0.0005`、`experiment.seeds=[7]`；不能靠修改默认值复现历史实验。

`format_version` 当前为 1。不认识的版本必须拒绝，不能静默猜测。

已完成的 `night_fullcircle_5mm_v1` 以其原始 `plan.json` 和每个种子的 `config.json` 为最终事实；`configs/experiments/night_fullcircle_5mm_v1.json` 只提供可发现索引，不覆盖原始记录。
