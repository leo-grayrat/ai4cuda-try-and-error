# FlashRT 说明文档效果研究

研究从 FlashRT 的长篇 AGENTS.md 出发：增加要求能否改善交付，收益是否抵得上阅读和生成成本，要求增多后哪些内容容易遗漏，以及权限和程序检查能否代替反复提醒。原有图表示与调度方法属于被研究项目的基础设施；比较文档处理时保留这套基础。

## 已核实的材料

来源固定为 [FlashRT dd9dae0](https://github.com/Infini-AI-Lab/FlashRT/tree/dd9dae0072a72964a2bdc3ca3f06f68afa122a7f)。完整源码存入被忽略的 `.cache/flashrt/dd9dae0072a72964a2bdc3ca3f06f68afa122a7f/repository`。514 个文件均与 Git tree 的 blob 校验值一致，见 [核验记录](evidence/repository-verification.json)及[下载来源](../sources/FlashRT.repository-provenance.json)。

| 原文件位置 | 行数 | token 数 |
| --- | ---: | ---: |
| AGENTS.md | 1,749 | 21,800 |
| flashrt/apps/AGENTS.md | 622 | 7,071 |
| flashrt/ir/AGENTS.md | 213 | 2,481 |
| assets/AGENTS.md | 18 | 191 |

合计 31,543 token，按同一分词器计算。各文件的适用目录不同，这个合计不能充当每次试验的实际输入量。

[逐字材料清单](materials/verbatim-v1/manifest.json)保留四份原文及其目录优先级，按原有标题划出 90 个片段。每份文档的片段连接后与原文件逐字节一致。这一步用于追溯来源，尚未形成压缩版或按需加载方案。

已另行制作 [Phase 4 整理草稿](materials/phase4-draft-v1/replacement.md)与[来源对照](materials/phase4-draft-v1/AUDIT.zh.md)：经第二轮核对，该段由 8,128 token 改为 2,704，整份根说明由 21,800 改为 16,376。区间外字节与另外三份说明保留原样，见[构建核验](materials/phase4-draft-v1/build.json)。该稿把重复限制集中到队列、工作范围、退出证据、诊断和迭代流程中，恢复了完成检查引用的六个标题；实际效果仍待比较。

[原文核对单](REQUIREMENTS.zh.md)覆盖四份说明，共 69 个研究核对单元；[索引](materials/requirements-v1.json)固定每项来源范围和校验值。这些单元区分字段、执行与语义要求，并非原文原子规则计数。[来源张力记录](SOURCE_TENSIONS.zh.md)保留路径、基线、失败判决等原有歧义。[比较草案](COMPARISON_DRAFT.zh.md)据此安排四问的证据与条件。

[仅排版对照](materials/format-v1/manifest.json)只合并普通正文中的软换行，四份文档由 31,543 变为 30,973 token。全部非空白词序、代码块原字节和 CommonMark 渲染结构核对一致；未用改写要求来制造排版效果。

## 原项目已有检查

[CPU 实测](evidence/ir-state-probe-01.json)直接使用原仓库执行器：未声明的状态读取、未声明的状态写入均被拒绝；声明写入后可以执行；声明读取后取出列表再原地修改，也能执行。最后一种边界在源码注释中已有说明。原依赖分析对共享状态采取保守处理，本次实测没有证明该边界造成错误调度。

这组记录只回答已有检查管到哪里。完整应用的正确性、流式交付和性能，仍需单独验证。

## 接下来的实验入口

先确定能实际运行的顺序参考后端，保持适配器、输入输出和流式状态约定一致，再比较原文、整理版及相应机制。检查原有程序已经约束什么，才能判断额外提醒或新增检查带来的收益。每项整理须映射回原文，保留要求强度、适用条件和示例。

原仓库五个现成应用依赖大型视频或多模态模型，但框架允许新增简单、顺序、单卡应用。按用户纠正，现采用[本机任务](LOCAL_TASKS.zh.md)，不再等待远端资源。已实测原因果卷积组成的小网络、原 IR 执行器的逐块数值一致性与重置，以及 Windows 跨进程共享内存传输。正式参考应用和独立计时继续据此搭建。

详见[源码审视](SOURCE_AUDIT.zh.md)和[当前执行计划](../PLAN.zh.md)。旧 CUDAAgent 条件及其试验保持停止，不纳入本研究的正式比较。

复现本阶段材料：

```powershell
python skill-effects/flashrt/prepare_documents.py --repository .cache/flashrt/dd9dae0072a72964a2bdc3ca3f06f68afa122a7f/repository --tokenizer .cache/skill-effects-tokenizer/tokenizer.json --output .cache/flashrt-verbatim-check
python skill-effects/flashrt/probe_ir.py --repository .cache/flashrt/dd9dae0072a72964a2bdc3ca3f06f68afa122a7f/repository --output .cache/flashrt-ir-probe-check.json
python skill-effects/flashrt/build_phase4_draft.py --tokenizer .cache/skill-effects-tokenizer/tokenizer.json
python skill-effects/flashrt/index_requirements.py
python skill-effects/flashrt/prepare_format_control.py --tokenizer .cache/skill-effects-tokenizer/tokenizer.json --output .cache/flashrt-format-check
python skill-effects/flashrt/probe_small_workloads.py --repository .cache/flashrt/dd9dae0072a72964a2bdc3ca3f06f68afa122a7f/repository --output .cache/flashrt-small-probe-check.json
```

第一条命令要求输出目录尚不存在，以免覆盖既有材料。分词器校验值记录在材料清单中。
