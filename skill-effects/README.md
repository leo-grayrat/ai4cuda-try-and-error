# CUDA Agent skill 效果测试

以原始 CUDA Agent skill 为基准，比较格式化、去重优化、原文规则子集及权限和检查器处理后的表现、成本与遗漏。研究动机是追加禁令造成的臃肿；当前主任务是评测原文及其处理版。

- [研究起点与范围](RESEARCH_FRAME.zh.md)
- [研究设计](DESIGN.zh.md)
- [执行计划](PLAN.zh.md)
- [原文派生材料清单](materials/original-conditions/manifest.json)
- [原始要求与来源核对](materials/SOURCE_AUDIT.zh.md)
- [保留题及独立验证](TASKS.zh.md)
- [文献核对](LITERATURE.zh.md)
- [执行记录](RUN_LOG.zh.md)

分支为 `research/skill-effects`。原始 skill 为 3,392 文字 token；仅格式化版为 3,400，核对后的优化版为 2,768。两者保留全部九个原始示例。[正式清单](protocol/v1/manifest.json)已按 11 条件、四题、两次重复封存，共 88 次；[分析约定](FORMAL_PROTOCOL.zh.md)先于候选运行固定。正式序列已启动，原始运行保存在被忽略的 `runs/formal-series-01` 和 `runs/f01-*`。

此前构造的追加历史、长篇重复压力和相关压缩材料保留在旧材料与 evidence 中，作为探索档案，不进入正式主实验。
