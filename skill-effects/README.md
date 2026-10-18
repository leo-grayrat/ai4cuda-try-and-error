# skill 效果测试

研究从 AI 遇到问题便往说明文档里补“不要”“必须”，最终积成几千行这一现象出发，考察规则累积对任务表现、要求遗漏和成本的影响。压缩以及把部分职责交给权限、检查器，是随后针对同一批要求的对照。

- [研究起点与 Nature 流程](RESEARCH_FRAME.zh.md)
- [研究设计](DESIGN.zh.md)
- [调查与环境实测](INVESTIGATION.zh.md)
- [执行计划](PLAN.zh.md)
- [CUDA Agent 原始 skill](sources/CUDA-Agent.SKILL.md)与[来源](sources/provenance.json)
- [FlashRT 原始文档](sources/FlashRT.AGENTS.md)与[来源](sources/FlashRT.provenance.json)

本轮位于 `research/skill-effects` 分支。当前完成来源冻结、设计首稿和 CUDA C++ 开发示例的编译及正确性验证；正式效果实验尚未运行。执行记录保存在本目录被忽略的 `runs/`；核对后的汇总与证据摘录另行入库。

环境探针：`python skill-effects/environment_probe.py --name environment-03`。每次使用新的名称保存完整过程；Windows 下使用已安装 MSVC 与 CUDA 12.9，适配内容逐项记录。原始资料不被探针修改。
