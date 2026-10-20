# FlashRT 说明文档效果研究

研究起点是用户对“遇到问题就追加规则”的质疑：几千行告诫会不会越来越难用，定量要求是否会压倒需要理解和解释的要求，论文中的图方法与长篇说明各有多少贡献。两份原始思考明确以FlashRT为对象，见[原文摘录与校验值](../sources/research-origin-verified-20261020.json)。本目录比较这些说明对交付、成本与遗漏的影响，以及权限和程序检查能否替代反复提醒。原有图表示与调度方法作为共同基础保留。

## 已核实的材料

来源固定为 [FlashRT dd9dae0](https://github.com/Infini-AI-Lab/FlashRT/tree/dd9dae0072a72964a2bdc3ca3f06f68afa122a7f)。完整源码存入被忽略的 `.cache/flashrt/dd9dae0072a72964a2bdc3ca3f06f68afa122a7f/repository`。514 个文件均与 Git tree 的 blob 校验值一致，见 [核验记录](evidence/repository-verification.json)及[下载来源](../sources/FlashRT.repository-provenance.json)。

| 原文件位置 | 行数 | token 数 |
| --- | ---: | ---: |
| AGENTS.md | 1,749 | 21,800 |
| flashrt/apps/AGENTS.md | 622 | 7,071 |
| flashrt/ir/AGENTS.md | 213 | 2,481 |
| assets/AGENTS.md | 18 | 191 |

合计 31,543 token，按同一分词器计算。各文件的适用目录不同，这个合计不能充当每次试验的实际输入量。

## 当前实验与证据入口

[显式预算批次](REPEAT_BATCH.zh.md)比较八种文档/机制条件、两个任务、各三次重复，目标为48个完整观察。外部中断保留原记录和消耗，以追加槽位补足观察机会；[执行清单](materials/repeat-batch-recovery-v1.json)记录原槽位与补充关系。实时运行目录为被忽略的`skill-effects/runs/flashrt-repeat-v1/`，已提交原始记录见[归档索引](evidence/repeat-archive-index.json)。

按用户确认，现有批次收完后直接切到[自然工作对照](NATURAL_WORK.zh.md)：同一真实任务配不同说明，模型正常实施和结束，用量由研究方在外记录。两阶段独立比较，不把前后变化全归因于预算提醒。分析口径见[分析计划](ANALYSIS_PLAN.zh.md)。

[进行中快照及比较图](evidence/repeat-partial-20261020/figure/README.zh.md)保留37个完整观察的逐轮成本与最终性能；其余9次外部中断和11个未运行机会列入源数据，八组未齐。36个有效后端中34个通过旧负载最高档200请求/秒，图中明确这是测量上限。

已经核对的具体过程如下；它们是逐例证据，不代表各条件的总体频率。

| 观察 | 证据 |
| --- | --- |
| 程序失败后，报告丢失已知的启动诊断事实 | [t004](evidence/repeat-review-t004.zh.md) |
| 后端正确、较快，图验证却被安排到最后并未完成 | [t005](evidence/repeat-review-t005.zh.md) |
| 只给任务也有长思考，并为了间隔检查调整发布 | [t006](evidence/repeat-review-t006.zh.md) |
| 低负载门槛与倒计时促成一次使吞吐回退的改动 | [t009](evidence/repeat-review-t009.zh.md) |
| 长思考中提前发现真实历史拼接错误 | [t012](evidence/repeat-review-t012.zh.md) |
| 两张图的失败被混淆，思考反复围绕错误位置推演 | [t017](evidence/repeat-review-t017.zh.md) |
| 十种策略实际测量，但队列时序与分析解释仍有遗漏 | [t021](evidence/repeat-review-t021.zh.md) |
| 检查器促使补齐独立证据；新增文档脚本也触发重复测量 | [两次补齐](evidence/repeat-checker-review-01.zh.md)、[t032重测](evidence/repeat-checker-review-02.zh.md) |
| 为满足未测入口清理规则，又因目录版本绑定而重新验证和测量 | [t042收尾](evidence/repeat-checker-review-03.zh.md) |

只读汇总命令不调用模型或重跑候选：

```powershell
python skill-effects/flashrt/summarize_repeats.py --manifest skill-effects/flashrt/materials/repeat-batch-recovery-v1.json --output .cache/flashrt-repeat-progress.json
python skill-effects/flashrt/analyze_workflow.py --manifest skill-effects/flashrt/materials/repeat-batch-recovery-v1.json --output .cache/flashrt-workflow-progress.json
python skill-effects/flashrt/analyze_enforcement.py --manifest skill-effects/flashrt/materials/repeat-batch-recovery-v1.json --output .cache/flashrt-enforcement-progress.json
```

## 文档处理材料

[逐字材料清单](materials/verbatim-v1/manifest.json)保留四份原文及其目录优先级，按原有标题划出 90 个片段。每份文档的片段连接后与原文件逐字节一致。这一步用于追溯来源，尚未形成压缩版或按需加载方案。

已另行制作 [Phase 4 整理草稿](materials/phase4-draft-v1/replacement.md)与[来源对照](materials/phase4-draft-v1/AUDIT.zh.md)：经第二轮核对，该段由 8,128 token 改为 2,704，整份根说明由 21,800 改为 16,376。区间外字节与另外三份说明保留原样，见[构建核验](materials/phase4-draft-v1/build.json)。该稿把重复限制集中到队列、工作范围、退出证据、诊断和迭代流程中，恢复了完成检查引用的六个标题；实际效果仍待比较。

[原文核对单](REQUIREMENTS.zh.md)覆盖四份说明，共 69 个研究核对单元；[索引](materials/requirements-v1.json)固定每项来源范围和校验值。这些单元区分字段、执行与语义要求，并非原文原子规则计数。[来源张力记录](SOURCE_TENSIONS.zh.md)保留路径、基线、失败判决等原有歧义。[比较草案](COMPARISON_DRAFT.zh.md)据此安排四问的证据与条件。

[仅排版对照](materials/format-v1/manifest.json)只合并普通正文中的软换行，四份文档由 31,543 变为 30,973 token。全部非空白词序、代码块原字节和 CommonMark 渲染结构核对一致；未用改写要求来制造排版效果。

## 原项目已有检查

[CPU 实测](evidence/ir-state-probe-01.json)直接使用原仓库执行器：未声明的状态读取、未声明的状态写入均被拒绝；声明写入后可以执行；声明读取后取出列表再原地修改，也能执行。最后一种边界在源码注释中已有说明。原依赖分析对共享状态采取保守处理，本次实测没有证明该边界造成错误调度。

这组记录只回答已有检查管到哪里。完整应用的正确性、流式交付和性能，仍需单独验证。

## 接下来的实验入口

先确定能实际运行的顺序参考后端，保持适配器、输入输出和流式状态约定一致，再比较原文、整理版及相应机制。检查原有程序已经约束什么，才能判断额外提醒或新增检查带来的收益。每项整理须映射回原文，保留要求强度、适用条件和示例。

原仓库五个现成应用依赖大型视频或多模态模型，但框架允许新增简单、顺序、单卡应用。按用户纠正，现采用[本机任务](LOCAL_TASKS.zh.md)，不再等待远端资源。[完整参考应用](workloads/causal_features/CONTRACT.zh.md)已通过适配器完成六请求数值检查，包括连续会话、末尾短块和重置；原 IR 执行器也与这些实际输出比较通过。工具保留原始时序、输入输出、清理记录及代码快照，接着加入负载测量与候选比较。

详见[源码审视](SOURCE_AUDIT.zh.md)和[当前执行计划](../PLAN.zh.md)。旧 CUDAAgent 条件及其试验保持停止，不纳入本研究的正式比较。

## 已执行的模型试跑

第二批开发记录见 [48,000 预算轮](evidence/pilot-calibration-02/collection.json)。该轮同时启用写入权限限制和提交检查，独立最终验证通过，但未尝试提交报告；因此没有检验提交检查的实际作用。它与首批的预算和配置不同，不作条件效果比较。

另有五次小预算运行专门调试报告阶段，见 [报告流程校准](evidence/report-calibration-01/collection.json)。完整报告在检查器拒绝成果时也保留，缺少正文与成果被拒收分别记录。两类完整轨迹均归档并核对逐文件校验值。当前还有任务准备和校准结果，尚无正式对照结论。

[首轮校准](evidence/pilot-calibration-01/FINDINGS.zh.md)保存三次真实 DeepSeek API 运行。可比的两轮使用同一任务、工具及24,000生成token预算：原文轮未写出候选，Phase 4整理稿轮写出一个后端并通过独立六请求验证；两轮均未提交报告。完整请求、响应、思考用量、工具事件、候选代码和独立测量归档于该目录，逐文件校验通过。

[负载校准](evidence/reference-load-01/result.json)按10、25、50、100、200请求/秒施加有限窗口负载，分别记录积压和输出间隔，验证过载可被检出。当前是开发校准，不将一秒窗口表述为无限稳定性结论。

工具边界[第一轮](evidence/tool-boundary-probe-01.json)实测可写区、只读拒绝、路径逃逸和未验证交付拒绝；[第二轮](evidence/tool-boundary-probe-02.json)补验证据目录及后端入口绑定。工具文件权限不等同于原生Python的操作系统隔离，来源变更和独立数值检查另行记录。

复现本阶段材料：

```powershell
python skill-effects/flashrt/prepare_documents.py --repository .cache/flashrt/dd9dae0072a72964a2bdc3ca3f06f68afa122a7f/repository --tokenizer .cache/skill-effects-tokenizer/tokenizer.json --output .cache/flashrt-verbatim-check
python skill-effects/flashrt/probe_ir.py --repository .cache/flashrt/dd9dae0072a72964a2bdc3ca3f06f68afa122a7f/repository --output .cache/flashrt-ir-probe-check.json
python skill-effects/flashrt/build_phase4_draft.py --tokenizer .cache/skill-effects-tokenizer/tokenizer.json
python skill-effects/flashrt/index_requirements.py
python skill-effects/flashrt/prepare_format_control.py --tokenizer .cache/skill-effects-tokenizer/tokenizer.json --output .cache/flashrt-format-check
python skill-effects/flashrt/probe_small_workloads.py --repository .cache/flashrt/dd9dae0072a72964a2bdc3ca3f06f68afa122a7f/repository --output .cache/flashrt-small-probe-check.json
python skill-effects/flashrt/run_local_reference.py --repository .cache/flashrt/dd9dae0072a72964a2bdc3ca3f06f68afa122a7f/repository --output .cache/flashrt-reference-check
python skill-effects/flashrt/verify_local_ir.py --repository .cache/flashrt/dd9dae0072a72964a2bdc3ca3f06f68afa122a7f/repository --run .cache/flashrt-reference-check
```

第一条命令要求输出目录尚不存在，以免覆盖既有材料。分词器校验值记录在材料清单中。
