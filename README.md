# AI4CUDA 研究试验仓库

本仓库记录两条主要研究线，以及第一条线后来派生出的 `program transformation` 前置探索。

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
