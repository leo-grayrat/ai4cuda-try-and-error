# AI4CUDA 研究试验仓库

本仓库按四个部分记录信用分配、skill 效率、Qwen 程序变换自动化探索，以及低概率 token 归因的工作。程序变换探索承接了第一部分提出的问题。

## 方向一：从 token 级信用分配到程序级决策信用

这一方向真正的起点不是“设计一个更好的程序变换 action space”，而是 CUDA Agent 训练里的 **credit assignment**：Actor 逐 token 生成，Critic/GAE 也沿 token 时间步估值和回传 advantage；但 kernel 的性能变化往往来自“融合两段计算、消除中间写回、改变每线程工作量、调整精度/布局/归约方式”这样的程序级决策，而不是某一个孤立 token。

最初的问题因此是：

> **价值到底应该分给 token，还是分给程序级决策？**

现有 [`action-credit/`](action-credit/README.md) 记录了第一轮尝试：先从真实 parent→child 版本中定位真正影响性能的代码修改，再考虑把性能反馈集中到这些位置。它最终没有回答训练问题：源码位置不等于程序决策；真实 CUDA Agent 的逐 token Critic value / advantage 等训练观测量又没有公开；即使画出更“漂亮”的信用分布，也不能证明 Actor 因此学得更好。

[`research-notes/program-transformation-action-space.md`](research-notes/program-transformation-action-space.md) 是在这之后产生的一个**派生方向**：既然一个程序决策可能跨多个代码位置，能否把 `Fuse(...)`、`Tile(...)`、`IncreaseWorkPerThread(...)` 之类的程序变换作为更自然的中间单位？这里后来发生了一次重要偏移：

- 原问题是 **“训练信用应该落在哪个单位上？”**；
- 派生问题逐渐变成 **“Agent 应该在哪种动作空间里搜索？”**。

二者相关，但不是同一个问题。完全可以让 LLM 继续自由生成 token，只在训练时识别程序级决策并让 Critic/advantage 围绕这些决策工作；并不要求先把生成接口改成一个显式 transformation DSL。

Qwen 的自动化探索位于分支 [`qwen/transform-action-space`](https://github.com/leo-grayrat/ai4cuda-try-and-error/tree/qwen/transform-action-space/qwen-transform-action-space)。它主要从 AdaExplore 现成 parent→child 轨迹中检查“少数重复的程序变换是否看起来存在”。这最多是对**程序级信用单位是否可能存在**的一次很前置、很弱的摸底；它没有研究 Critic 如何给这些决策估值，也没有比较新的 credit assignment 是否改善训练。

关于这次偏移的完整反思原样保存在 [`retrospectives/2026-10-05-credit-assignment-origin-and-drift.md`](retrospectives/2026-10-05-credit-assignment-origin-and-drift.md)。

## 方向二：CUDA agent 的 skill 效率与执行责任

[`cuda-skill-efficiency/`](cuda-skill-efficiency/README.md) 关注现实 AI4CUDA agent 中很长的 skill / 操作说明：它们占多少上下文，实际给 CUDA 开发任务带来多少收益，又让后续 agent 多花多少 token；如果真实 skill 确实存在利用不足、规则违反或明显成本，再继续问哪些规则适合交给权限、流程或检查器执行。

这条线已经搭起 OpenCode 逐轮轨迹采集、任务隔离、真实 kernel 启动检查、配对 GPU 计时等实验设施，但第一次正式对照为了适配本机环境，把研究对象从真实 CUDA Agent `SKILL.md` 换成了自写 Triton 提示，因此 18 次运行不能回答原问题。

两条方向此前做过什么、为什么都没有完成原始验证，见 [阶段性复盘](RESEARCH_RETROSPECTIVE.md)。

**简而言之AI牛大了，我只是做简单的探索性试跑实验都死掉了，而且都是中途扎进技术细节一去不复返，任务偏移到姥姥家了，最终成果只能说是没用，无论是证实还是证伪，最后还给我拉一堆AI防御性文案**

旧研究已有的本机虚拟环境 `.venv*`、下载缓存 `.cache/` 和原始记录 `runs/` 留在仓库根目录，均不入库。目录迁移不改变这些历史文件；旧研究的新运行可在其子目录下生成自己的 `runs/`。

## 方向三：Qwen 的程序变换自动化探索

[`qwen-transform-action-space/`](qwen-transform-action-space/README.md) 记录了 Qwen3.8-Flash 围绕程序变换动作空间进行的一轮自动化研究。它承接方向一中的派生问题，尝试从真实优化轨迹中观察反复出现的变换类别，并完成文献调研、数据标注、统计分析、论文撰写、评审和修订。

数据来自 AdaExplore 的公开 parent→child 轨迹：从 2,510 条候选边中选取 55 条，由 AI 按预先冻结的变换词表标注。词表覆盖了 90.9% 的样本，分析还检查了同一变换类别与代码修改位置、diff 结构之间的对应关系。修订中发现并纠正了 Miller–Madow 偏差项的误用，语义标签的置换检验结果随之反转；后续盲重标得到 74.5% 的一致率，头条检验的 p 值也从原标注下的 0.0005 变为重标下的 0.106。标注和统计口径怎样影响结论，是这轮探索留下的一项具体结果。

这一部分的入口包括[英文论文](qwen-transform-action-space/paper/main.md)、[PDF](qwen-transform-action-space/paper/latex/main.pdf)和[中文对照稿](qwen-transform-action-space/paper/zh/full.md)。[实验报告](qwen-transform-action-space/STAGE1_5_REPORT.md)、[修订记录](qwen-transform-action-space/stage4_revision/REVISION_NOTES.md)及[最终核对记录](qwen-transform-action-space/STAGE4_5_FINAL_INTEGRITY.md)保留了从初稿到修订稿的证据与变化。

## 方向四：CUDA 低概率 token 的分布与归因

[`lowprob-attribution/`](lowprob-attribution/README.md) 研究 CUDA Agent 对训练失稳提出的解释：CUDA 在预训练中稀少，是否会使生成代码中的相关 token 落到极低概率区域？这一分支通过多轮内核编写与评测，采集实际生成的逐 token 概率，比较领域词汇、代码、注释、解释文字及不同反馈上下文中的分布。

主要实验包含四道逐元素或归约任务的 128 次运行，以及五道 KernelBench Level 2 算子链任务的 40 次运行，分别得到 98,533 和 542,611 个 token。严格词表下，CUDA/Triton 词汇的低概率率为 0.26%，其余 token 为 0.84%；算子链实验中，代码块内为 0.60%，去除注释后为 0.33%，代码块外的解释文字为 13.56%。逐项阅读低概率位置后，尾部主要落在注释、变量命名、标点和异常续写中；在这批生成输出里，文本区域比领域词汇更能区分概率代价的分布。

采集工作同时保留了两条路线。DeepSeek API 实验将代码放在回复正文，由驱动程序写文件和执行评测；本地 vLLM 记录则直接保存工具调用参数中的代码与概率。对已有本地记录的重新核对恢复了 40 个代码参数、3,162 个带概率的生成 token，参数内容均与对应工具事件的输入一致。这部分补充将早期本地采集的成果接回了论文。

当前稿件已完成正文文风重写、长公式独立展示、四张结果图的制作，以及参考文献回源核对。阅读入口为 [v2 论文 PDF](lowprob-attribution/v2/stage5_finalize/main.pdf)、[Word](lowprob-attribution/v2/stage5_finalize/paper.docx)和[正文源稿](lowprob-attribution/v2/stage2_write/paper.md)；[本地记录核对](lowprob-attribution/v2/local_tool_evidence/REPORT.zh.md)、[配图与绘图数据](lowprob-attribution/v2/figures/README.md)及[引用修正说明](lowprob-attribution/v2/stage4_revision/REFERENCE_AUDIT.zh.md)提供对应的复查入口。
