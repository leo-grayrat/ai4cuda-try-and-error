# FlashRT 说明文档效果研究

当前入口：[FlashRT 材料与实测](flashrt/README.md)、[研究起点](RESEARCH_FRAME.zh.md)、[执行计划](PLAN.zh.md)。研究回到原始 FlashRT AGENTS.md 及其整理方式，先核实源码、已有检查与可运行任务，再冻结比较方案。

## 已停止的 CUDAAgent 方案记录

**本方案已停止，研究对象待纠正。** 回读用户原始思考后确认，长篇 AGENTS.md 与图论架构的讨论指向 FlashRT；此前错误选用 CUDAAgent 作为正式主体。后台实验与后处理已停止，以下保留旧方案记录，不能自动续跑。详见[研究对象纠正](SCOPE_CORRECTION.zh.md)。

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

独立分析工具：`collect_formal.py` 汇总全部预定槽位并保留未完成、失败和说明缺失；`prepare_blind_review.py` 生成隐藏条件的可见证据包，对应表与审阅目录分开。`trace_formal.py` 在正式序列全部结束后，对独立正确的最后候选及 eager、compiled 参考核查实际 kernel 启动。它仅在预热后的单次调用范围内采集，记录 kernel 名称、启动数量和正耗时；性能比较仍使用正式运行的完整调用 CUDA event 计时。

运行后处理时先刷新 collection，再生成审阅包。GPU 核查命令如下；正式序列未结束会拒绝执行，避免干扰计时。

```powershell
python skill-effects/collect_formal.py --output skill-effects/runs/formal-series-01/collection.json
python skill-effects/prepare_blind_review.py --collection skill-effects/runs/formal-series-01/collection.json --output skill-effects/runs/formal-series-01/blind-review-final
python skill-effects/trace_formal.py --collection skill-effects/runs/formal-series-01/collection.json --output skill-effects/runs/formal-series-01/native-activity
python skill-effects/analyze_formal.py --collection skill-effects/runs/formal-series-01/collection.json --output skill-effects/runs/formal-series-01/analysis.json
```

`analyze_formal.py` 仅在全部槽位结束后输出条件汇总和逐题、逐次配对表。成本包含失败运行；没有数值交付时，每次交付成本留空。有一侧未通过独立正确性或没有有效计时时，配对表保留双方结果，但不编造性能差值。文风与适用要求的审阅结果另行合并。

`review_packet.py` 可对单份隐藏条件的材料进行新上下文 API 审阅，逐字核对引用、报告是否存在和条目完整性，保存单独用量。其输出仍需证据复核。[首份审阅校准](evidence/review-calibration-01/decisions.zh.md)保留了引用来源错误、清理判断过宽及测试命名归因错误；未把这些初步意见当作最终合规率。JSON 输出参数依据 [DeepSeek 官方文档](https://api-docs.deepseek.com/guides/json_mode/)。

当前审阅材料为 schema 2，附完整工作区文件清单、候选源码哈希和可见回复轮次，明确区分模型操作与研究方最终测量。早期 schema 1 材料及校准意见留档；新审阅器要求 schema 2，旧证据包的内容不会被覆盖。

`postprocess_when_ready.py` 可在同一正式序列结束后顺序运行汇总、审阅材料生成、实际 kernel 核查及配对分析，保存各阶段日志与子进程信息。它不重跑候选，遇到正式控制器基础设施错误或后处理失败会留档停止；完成后仍需审阅意见的证据复核、图表和研究报告。
