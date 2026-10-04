# Stage 1 — RQ Brief

体裁：Position / Analysis Paper。模式：quick（用户指示：文献检索非关键路径，全程自动化）。

---

## 1. 命题（一句话）

对 LLM 驱动的 CUDA kernel 优化，把搜索与学习的**动作单位**从 token 序列上移到
**参数化程序变换** `a = (τ, ℓ, θ)`，是否真的显著改变搜索效率与学习效率——
以及，真实优化轨迹是否呈现出「code diff 熵高、transformation 熵低」的压缩现象。

## 2. 命题的精确层次（三层必须分开）

研究笔记把三层揉在一起，本论文必须拆开，否则判读树失效：

| 层次 | 主张 | 本文状态 |
|------|------|----------|
| L1 表示层 | 程序变换是比 token 更稳定的**描述单位** | 有初步证据（Stage 1.5 熵分析） |
| L2 搜索层 | 先决定「做什么」再实现，能改变**搜索效率** | 只有框架设计，无实验 |
| L3 学习层 | 把变换而非 token 作为 **RL action**，能改变学习效率 | 只有框架设计，无实验 |

L1 是 L2/L3 的必要非充分条件。本文只声称对 L1 给出证据，对 L2/L3 给出可证伪协议。

## 3. 差异点（与最近邻工作的正面区分）

必须直面的先行工作不是编译器，而是**「以 edit 为动作单位的代码 RL/LLM 优化」**：

| 最近邻 | 它的动作单位 | 本文的差别 |
|--------|--------------|------------|
| Learning Performance-Improving Code Edits (arXiv 2302.07867, ICLR 2024) | 整段函数级代码 edit（文本），面向 CPU 竞赛程序 | 动作单位是**类型化语义变换**而非文本 diff；一个 action 强制跨多位置协调（kernel body + host launch），且词汇跨任务复用 |
| Neural Networks for Modeling Source Code Edits (arXiv 1904.02818, ICLR 2019) | 细粒度光标级 edit 序列 | 同上；他们的粒度退化到 token 邻域，我们的粒度刻意停在「约束实现但不等于 diff」的中间层 |
| To Diff or Not to Diff? (arXiv 2604.27296, ACL 2026 Findings) | **输出格式**（full-code vs block-diff） | 它证明「坐标系=格式」有成本影响；本文主张「坐标系=语义 action」有**归纳偏置**影响，二者不同层次 |
| CudaPerf (arXiv 2607.20908) | token 策略 + **CUDA 结构性奖励项**（coalescing / occupancy / AI / 同步） | 它把 CUDA 性质放进 **reward**，本文质疑 **action space** 本身；两者正交且可叠加 |
| CUDA Agent (arXiv 2602.24286) | token（PPO 在 131k 上下文上） | 本文的批判对象：训练层面设计与 CUDA 特性无关，信用分配黑洞未处理 |
| TVM / Ansor (arXiv 1802.04799 / 2006.06762) | schedule space / 搜索空间上的程序变换 | 血缘关系明确。**若结论只是「compiler pass 比 token 更语义化」则无论文价值**；本文命题是 LLM-in-the-loop 界面上移到变换层是否改变效率 |
| Option-Critic (arXiv 1609.05140) / When Do Options Help? (arXiv 2609.05508) | options（时间扩展动作） | 理论框架同源；2609.05508 给出「高层 action 何时反而有害」的反例条件，正是本文判读树最后一支的先验依据 |

**一句话差异**：编辑动作（edit-based）文献把「动作」定义为**文本层面的操作**，
本文把「动作」定义为**语义层面、带类型参数、跨位置协调、跨任务词汇复用**的变换对象；
判定标准是 `a → {e₁,e₂,…}` 的多对多关系是否真实存在（Stage 1.5 直接检验这一点）。

## 4. 可证伪性

四条件对照（Direct / Free-plan / Structured-action / Oracle-action）+ 判读树，
其中 **Oracle-action ≈ Direct 直接杀死方向**（笔记原话）。
前置经验问题独立可证伪：若 30–60 条真实成功边聚不成稳定 vocabulary，
或必须不断发明更细的 action 才能覆盖，则 L1 不成立，方向应死。

本文自身的主要证伪风险（写进 Limitations，不掩盖）：
1. 好优化可能根本不可离散分解（整体重构，tile/layout/mapping 强耦合）；
2. LLM 内部可能已隐式做该 abstraction，显式化只是把 reasoning 打印出来（Free-plan 对照存在的原因）；
3. 选 action 容易而 realization 才是瓶颈（Oracle-action 会直接暴露）。

## 5. 停止条件问句（仓库方法论红线）

> 如果这个实验完美完成，它会改变我们对原研究问题的哪一个判断？

- Stage 1.5 熵分析：**能**改变对 L1 的判断（vocabulary 存在性）。✅ 合格。
- Stage 1 文献检索：**不能**改变任何研究判断 → 纯基础设施，故按用户指示压缩为 quick。
  （据此删除了自建 OpenAlex 检索脚本 `scripts/openalex_search.py`。）

## 6. 数据边界

只用 AdaExplore 公开父子轨迹（arXiv 2604.16625；本地 `.cache/adaexplore-l{2,3}.tar.gz`）。
CUDA Agent 除论文正文与 6k 数据集外**未开源** Agent 轨迹 / 逐 token Critic value /
原始概率 / advantage，故任何需要其训练观测量的路线均不可执行（已在 RESEARCH_RETROSPECTIVE 记录）。
