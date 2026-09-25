# Oracle semantic credit：当前状态与更正

本页取代此前的“Baseline (CUDA Agent) vs. Oracle Semantic Credit 分配对比报告”。此前报告中的 84%–100% 样板信用浪费不能作为实验结论，相关数字已经撤回。

## 为什么撤回

先前实现把 CUDA Agent 的 baseline 简化成“终局奖励按 (gamma * lambda)^k 向前指数衰减”，并把这些权重重新归一化到终局 reward。这个做法等价于忽略 Critic 的真实价值估计，而且额外加入了“优势之和必须等于终局奖励”的约束。

真正的 GAE 需要：

delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

A_t = delta_t + gamma * lambda * A_{t+1}

因此，没有实际 rollout 上的 Critic 值（或论文系统直接导出的逐 token advantage），就不能声称自己复现了 CUDA Agent 的信用分配。Advantage 也不是需要守恒的一袋 reward。

先前的三个“oracle cases”也不满足真实 token 级实验要求：两个来自仓库早期人为构造的 smoke test，并非 Agent 轨迹；所谓 2_20 案例重写成了玩具程序，却挂上了复杂真实父子版本的整体 1.301x 加速；raw_response 和 token 序列则是后补的文本与正则分词，不是模型当时真实生成的回复和 tokenizer 输出。这些文件已从当前分支删除。

## 当前分支真正完成了什么

当前只保留两个很窄的实验部件：

1. kernel_lab/code_credit.py 冻结为朴素启发式基线，不再针对新漏报继续补正则。
2. kernel_lab/oracle_credit.py 只负责表达人工给出的优化决策 span、把它对齐到真实模型 tokenizer 的 offset，以及在已经给定真实 Critic 值时计算标准 GAE。

compute_oracle_token_signal 只是“如果人工 oracle 已知，怎样构造一个集中在该 span 上的实验信号”的定义。它本身不证明这种信号更合理，也不证明它能改善学习。

## 现在还没有什么结论

当前没有数据支持以下说法：

- CUDA Agent 把 84%–100% 的信用错误分给了胶水代码；
- Oracle semantic credit 优于 CUDA Agent；
- 公开 AdaExplore 档案足以做 token 级信用比较；
- 人工 oracle 集中权重能改善后续 kernel 学习。

要做真正的比较，至少需要一条真实生成轨迹同时保留：原始模型回复、生成时 tokenizer 的 token/offset、终局性能 reward，以及 baseline Critic 的逐 token value（或直接的 advantage）。随后才能在人不知道 baseline advantage 的前提下人工标注程序级优化决策，并比较二者。

在拿到这种数据以前，这个分支停在“把研究问题形式化清楚”的位置，不继续扩 synthetic case，不继续造训练数据，也不启动本机小模型训练。
