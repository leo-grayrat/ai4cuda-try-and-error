# OpenCode + DeepSeek V4.1 Flash 采集检查点（2026-09-27）

## 当前结论

本机 OpenCode 1.18.12 使用 `deepseek/deepseek-flash`（界面名 DeepSeek V4.1 Flash）已完成最小模型调用、文件读写调用，以及 `vector_affine` 三种提示条件各一次运行。JSON 事件能记录模型步骤、每步输入与输出 token、缓存 token、工具调用与反馈；独立评测另行检查候选代码。这证明了采集链路可用，**不证明 skill 有效或无效**。

`row_reduce` 的无额外 skill 运行完成；完整 skill 运行达到 300 秒上限，精简 skill 未运行。试跑在此暂停，以保存仓库检查点。没有封存任务集、重复运行或固定 token 上限；不能据此比较三种提示的效果，也不能启动责任迁移实验。

## 本机试跑

所有运行采用独立任务副本、相同本机 CUDA Python 环境、`opencode run --pure --auto --format json` 和模型标识 `deepseek/deepseek-flash`。下表的输入、输出均包括对应缓存或推理用量；单次性能计时只作为候选状态记录。

| 任务与提示 | 状态 | 步骤 | 工具调用 | 输入 token | 输出 token | 总 token | 独立正确性 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `vector_affine`，无额外 skill | 完成 | 5 | 7 | 84,615 | 1,333 | 85,948 | 3/3 形状通过 |
| `vector_affine`，完整 skill | 完成 | 5 | 5 | 85,990 | 1,561 | 87,551 | 3/3 形状通过 |
| `vector_affine`，精简 skill | 完成 | 6 | 5 | 101,504 | 1,145 | 102,649 | 3/3 形状通过 |
| `row_reduce`，无额外 skill | 完成 | 30 | 32 | 934,469 | 24,701 | 959,170 | 3/3 形状通过 |
| `row_reduce`，完整 skill | 300 秒超时 | 20 | 29 | 717,968 | 25,969 | 743,937 | 未评测 |

OpenCode 的 `input` 字段不含缓存读取，`output` 字段不含推理 token；表内按 `run.json` 的统一口径相加，保留这些分项在原始记录。`row_reduce` 无额外 skill 候选虽然正确，但独立计时中位数为参考实现的约 3 倍；agent 的最终文字却声称性能相当或更快。这说明需要独立性能评测，仍不能由此推断 skill 的作用。`vector_affine` 三次运行的单次计时起伏较大，不做提速比较。

原始 `events.jsonl`、`run.json`、提示、源码快照和评测结果留在本机被 Git 忽略的 `runs/opencode-*/`，不会随代码提交。后续若继续试验，应先核查 `row_reduce` 超时与当前任务难度、300 秒上限的关系，再确定正式任务和重复次数。此次检查点不包含正式研究结论。
