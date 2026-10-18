# skill 效果测试

研究从 AI 遇到问题便往说明文档里补“不要”“必须”，最终积成几千行这一现象出发，考察规则累积对任务表现、要求遗漏和成本的影响。压缩以及把部分职责交给权限、检查器，是随后针对同一批要求的对照。

- [研究起点与 Nature 流程](RESEARCH_FRAME.zh.md)
- [研究设计](DESIGN.zh.md)
- [调查与环境实测](INVESTIGATION.zh.md)
- [执行计划](PLAN.zh.md)
- [真实要求映射与内容核对](materials/SOURCE_AUDIT.zh.md)
- [第一次失败驱动的规则追加](materials/GROWTH_CASE_01.zh.md)
- [第四次追加中的解释传播](materials/GROWTH_CASE_04.zh.md)
- [相关文献全文核对](LITERATURE.zh.md)
- [执行记录与实验选择](RUN_LOG.zh.md)
- [保留题、原始困难维度与独立验证](TASKS.zh.md)
- [CUDA Agent 原始 skill](sources/CUDA-Agent.SKILL.md)与[来源](sources/provenance.json)
- [FlashRT 原始文档](sources/FlashRT.AGENTS.md)与[来源](sources/FlashRT.provenance.json)

本轮位于 `research/skill-effects` 分支。当前已有真实来源的 45 项要求映射，归档的前四次追加形成了 509 行文档；四道保留题的参考基线、独立验证和权限/提交门槛也已有实测。八轮失败驱动的累积过程正在运行，正式效果对照尚未开始。执行记录保存在本目录被忽略的 `runs/`，核对后的汇总与证据摘录保存在 `evidence/`。

环境探针：`python skill-effects/environment_probe.py --name environment-03`。每次使用新的名称保存完整过程；Windows 下使用已安装 MSVC 与 CUDA 12.9，适配内容逐项记录。原始资料不被探针修改。
