# 引文与参考文献核对

核对日期：2026-10-18。

本次同时对照了 WorkBuddy 的原始聊天与检索返回、此前保存的相关工作材料、当前正文及文献表，并重新打开原论文或作者页面核实身份和内容。

## 主要发现和修改

| 旧稿问题 | 查回的来源关系 | 本次处理 |
|---|---|---|
| CUDA Agent 在文献表出现两个近似标题 | 一条是 Dai 等人的论文，一条是 Jianyu Huang 的阅读笔记。博客开头明确写着所读的论文。 | 正文中涉及原论文的机制、训练和成绩统一引用原论文；移除这条重复转述的博客。 |
| “第三方对消融的解读”引用 `logprobtracking` | 历史搜索返回中的对应文字来自 alphaXiv 对 CUDA Agent 的概述；《Log Probability Tracking of LLM APIs》讨论的是用概率监测 API 变化。 | 查回 CUDA Agent 的消融内容，直接引用原论文；将概率追踪论文放回 API 测量的相关工作。 |
| `Precision Mismatch in Reasoning Models` 被列成独立论文 | 可追溯的记录是博客标题 `Jet-RL and the Precision Mismatch in Reasoning Models`，其实际论文链接通向 Jet-RL。 | 删掉这个没有独立身份依据的条目，引用 Jet-RL 原论文。 |
| `First-Order Approximation for Stable LLM-RL Training` 被当作论文题名 | 它是另一篇阅读笔记的标题，链接指向 Zheng 等人的《Stabilizing Reinforcement Learning with LLMs: Formulation and Practices》。 | 改为原论文的正式标题、全体作者、2025 年份和 arXiv 编号。 |
| 多条作者使用 `... Authors` 占位文字 | 历史中补作者的脚本注明是为了提供 plainnat 所需的 author/key，并直接写入这些占位值。 | 根据 arXiv 元数据及原文恢复作者，删除占位文字。 |
| KernelBench 作者和编号错误 | 原论文为 Anne Ouyang 等人，arXiv:2502.10517；旧表写成 `Omsk, Keren` 和另一个编号。 | 修正作者、编号与链接。 |
| 括号内的章节、表格标注不可靠 | CUDA Agent 旧引文写成 `Sec. 5.1–5.2` 和 `Tab. 2`，重新核对的训练机制位于 3.3 节，主成绩为表 1。 | 正文统一纯数字引用；定位核对信息保存在本说明中。 |
| 引言中 CUDA Agent 的部分成绩不符 | 旧稿把不同指标和分级数字混在一起。原文主表给出整体通过率 98.8%、比 `torch.compile` 更快的比例 96.8%。 | 引言改用这两个可直接核对的整体指标。 |

## 历史中已经找到、后来没有接回正文的材料

历史检索返回曾同时包含《Log Probability Tracking of LLM APIs》《How Open Must Language Models be to Enable Reliable Scientific Inference?》和《Surrogate Fidelity: When Can Open LLMs Explain Closed Ones?》。随后保存的相关工作文档也记录了它们，以及 Carlini 等人的模型提取研究。Kevin 则出现在另一轮内核强化学习检索中。

这些记录能够确认它曾经检索到或保存过这些材料。当前修订分别将它们用于 API 概率测量、研究对象与可观测接口、跨模型解释的对应关系、概率输出所含信息，以及多轮内核强化学习背景。每处新增引文对应正文中的具体论述。

另外补入了 PPO 和广义优势估计的原始文献，并沿 CUDA Agent 引用的失稳研究作者页面，找到其关联的 Trust Region Masking 正式预印本。新补的基础文献与从历史中恢复的材料在此分开记录。

## 现稿文献及核实出处

| 文献 | 原始出处 |
|---|---|
| Kevin | [arXiv:2507.11948](https://arxiv.org/abs/2507.11948) |
| CUDA Agent | [arXiv:2602.24286](https://arxiv.org/abs/2602.24286) |
| KernelBench | [arXiv:2502.10517](https://arxiv.org/abs/2502.10517) |
| Proximal Policy Optimization Algorithms | [arXiv:1707.06347](https://arxiv.org/abs/1707.06347) |
| High-Dimensional Continuous Control Using Generalized Advantage Estimation | [arXiv:1506.02438](https://arxiv.org/abs/1506.02438) |
| Dr. Kernel | [arXiv:2602.05885](https://arxiv.org/abs/2602.05885) |
| Jet-RL | [arXiv:2601.14243](https://arxiv.org/abs/2601.14243) |
| Stabilizing Reinforcement Learning with LLMs | [arXiv:2512.01374](https://arxiv.org/abs/2512.01374) |
| Your Efficient RL Framework Secretly Brings You Off-Policy RL Training | [原作者技术报告](https://fengyao.notion.site/off-policy-rl) |
| Trust Region Masking for Long-Horizon LLM Reinforcement Learning | [arXiv:2512.23075](https://arxiv.org/abs/2512.23075) |
| Log Probability Tracking of LLM APIs | [arXiv:2512.03816](https://arxiv.org/abs/2512.03816)、[ICLR 2026 记录](https://openreview.net/forum?id=hFxivbAgVP) |
| Stealing Part of a Production Language Model | [arXiv:2403.06634](https://arxiv.org/abs/2403.06634) |
| How Open Must Language Models be to Enable Reliable Scientific Inference? | [arXiv:2603.26539](https://arxiv.org/abs/2603.26539) |
| Surrogate Fidelity: When Can Open LLMs Explain Closed Ones? | [arXiv:2606.32008](https://arxiv.org/abs/2606.32008) |

正文引文位置从 9 处增加至 29 处，覆盖 14 条文献；每条都在正文实际使用。PDF 与 Word 按首次引用顺序编号，文献表补齐作者、年份和可追溯链接。旧文献表和修改前稿件保存在同目录，便于比对。

本次检查也发现，之前修订虽然重新组织了文字，仍然沿用了错误的文献身份和引用对应关系。本轮已将核对推进到原始出处及具体论述。
