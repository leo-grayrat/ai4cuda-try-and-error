# CUDA 语法稀有性不是训练不稳定的根因：来自 98,533 个 token 的条件概率证据

**作者：** leo-grayrat
**机构：** 独立研究
**日期：** 2026-10-17

---

## 摘要

**背景**：CUDA Agent 将 PPO 训练不稳定归因于"CUDA 在预训练语料中占比低于 0.01%，导致基础模型对 CUDA 词元的条件概率极低，进而使重要性比率爆炸"，并把该论断作为论文标题中的"根因"。该因果链的起点——CUDA 专属词汇是否构成低概率尾部——此前未被独立检验。

**目的**：检验该因果链起点是否成立，若否，识别低概率真正的来源。

**方法**：不使用 agent 框架（其工具调用参数不受 `logprobs` 覆盖），改为多轮纯对话采集：模型在回复正文中直接输出 kernel 代码，由驱动程序写入文件、调用独立评测器、并构造反馈文本进入下一轮。模型为 `deepseek-chat`（非推理），4 道 Triton 任务 × 2 种反馈条件（完整/截断）× 16 次重复，共 127 次有效运行、98,533 个携带 logprob 的正文 token。

**发现**：（1）CUDA/Triton 词汇的低概率率**显著低于**其他 token——严格定义下 0.26% vs 0.84%（Fisher p = 0.0007），宽松定义下 0.49% vs 0.84%（p = 0.0070），两种边界方向一致。（2）上下文效应远强于 token 类型：`after_failure`（收到评测失败反馈后）低概率率 **4.30%**，是 `initial`（0.94%）的 4.6 倍。（3）全库最低的 token 全部是自然语言词，无一为代码 token。（4）压缩反馈**未能**降低低概率率（完整 0.71% vs 截断 0.95%）。

**意义**：CUDA Agent 的因果链起点不成立；其声称的"根因"更可能来自多轮 agent 轨迹中失败反馈带来的分布外上下文，而非 CUDA 语法的稀有性。

**关键词**：CUDA Agent, 训练不稳定, 逐 token logprob, 分布外上下文, Triton

---

## 1. 引言

### 1.1 背景

CUDA kernel 的自动优化是 LLM agent 的一个重要应用场景：agent 反复生成 kernel 代码、编译、测量、根据反馈修改，最终目标是产出比 `torch.compile` 更快的实现。这一流程需要在多轮工具调用与环境反馈中维持稳定的策略改进。

CUDA Agent 报告称，这类训练在约第 17 步出现崩溃，并给出了一个因果链：CUDA 在预训练语料中占比 < 0.01% → 基础模型与 CUDA 分布严重错配 → 采样出大量低概率 CUDA 词元 → PPO 重要性比率爆炸 → 训练崩溃。该论断被置于论文标题"训练不稳定的根因"之下。

### 1.2 问题陈述

该因果链的起点依赖一个可检验的假设：**CUDA 专属词汇在生成时构成低概率尾部**。这与三项直觉相抵触：

第一，CUDA 代码中的绝大多数 token 并不专属 CUDA（`for`、`if`、`float`、`int`、运算符），真正专属的只有 `threadIdx.x`、`__shared__`、`__syncthreads()` 等少数标识符，而它们"恰恰非常固定、非常模式化"。

第二，生成时的关键量是**当前上下文下的条件概率**，与预训练语料中的绝对占比没有直接关系。

第三，论文自身的数据构成反证：其基础模型并非"不会写 CUDA"（原始模型总体正确率 74%，Level 1 达 90%）。

### 1.3 研究缺口

CUDA Agent 未公开逐 token 的 Critic value、advantage 或生成概率，因此无法从内部观测验证该因果链。而在闭源 API 侧，工具调用参数不受 `logprobs` 覆盖（这是多家厂商的一致行为，被普遍解读为防止蒸馏），使得基于 agent 框架的逐 token 分析在参数部分不可行。

### 1.4 研究问题

本研究检验以下问题：

- **RQ1**：CUDA/Triton 专属词汇的低概率率是否显著高于其他 token？
- **RQ2**：低概率率是否随上下文类型变化？若变化，哪一类上下文最高？
- **RQ3**：压缩环境反馈能否降低低概率率？

### 1.5 研究意义

若 RQ1 的答案为否，则 CUDA Agent 的因果链起点不成立，其"根因"叙述需要改写，而针对"多喂 CUDA 语料"一类干预的合理性也随之减弱。

---

## 2. 相关工作

**CUDA Agent**：提出上述因果链，但未公开可验证的逐 token 训练观测量。

**Off-policy 与分布外上下文**：Liu 等人的工作指出 RL 框架可能因长轨迹中累积的分布外输入而使训练偏离。这为本研究 RQ2 的替代假设提供了理论依据。

**闭源 API 的概率可得性边界**：多项证据表明，工具调用参数不被 `logprobs` 覆盖是厂商的策略选择而非技术限制。三篇关于"闭源模型能否支持科学推断"的论文均将"隐藏完整概率分布"列为障碍之一。这解释了为何本研究采用无工具的多轮对话采集。

**DeepSeek Harness / OpenCode 等 agent 框架**：这些框架提供工具调用能力，但其参数不受 `logprobs` 覆盖，因此不适合本研究的问题。

---

## 3. 方法

### 3.1 采集架构

不使用 agent 框架。驱动程序 `loop_driver.py` 直接向 chat-completions 端点发起多轮对话：

1. 第 1 轮：发送任务描述（含 `reference.py`）与要求（必须使用 Triton kernel；在正文中以围栏代码块输出完整 `candidate.py`；不使用任何工具）。
2. 模型在**正文**中输出代码。工具未被提供，因此代码必然落在 `message.content` 中——而该字段受 `logprobs` 覆盖。
3. 驱动解析代码块、写入 `candidate.py`、调用独立评测器。
4. 驱动将评测结果构造为反馈文本，作为新的 user message 进入下一轮。

**前置验证**：在正式采集前以单轮请求确认可行性，测得 **448/448** 个代码块内 token 均携带 logprob。该验证通过后路线方成立。

**模型**：`deepseek-chat`（非推理）。选用理由为实测结果：`deepseek-flash`（推理模型）在 `max_tokens=4000` 时预算被思考耗尽、`content` 返回空；`deepseek-chat` 的 `reasoning_tokens` 恒为 0，内容直接落在 `content`。

### 3.2 任务

4 道 Triton 任务：`row_reduce`（行归约）、`vector_affine`（逐元素仿射）、`layernorm_rows`（行归一化）、`softmax_rows`（行 softmax）。前两道为常规集，后两道为留出集。

### 3.3 反馈条件（干预）

- **完整（full）**：评测器的完整输出（编译错误全文、正确性对比、计时）。
- **截断（truncated）**：单行摘要（`compilation failed` / `output mismatch` / `correct`）。

两条件在采集开始前定死并实现，采集期间未修改。

### 3.4 分类与统计

**token 类型**：按关键词表在**全部 content token** 上匹配，不依赖任何分段信息。为检验稳健性，同时在两种定义边界下各计算一次：

- **宽松表**：包含 Triton API 名（`tl`、`load`、`store`、`zeros`、`arange`、`atomic_add`、`num_warps` 等 28 项）。
- **严格表**：仅包含无歧义拼写（`tl`、`num_warps`、`constexpr`、`program_id` 及 CUDA 原生标识符）。

**上下文类型**：由轮次与上一轮评测结果决定——`initial`（第 1 轮）、`after_success`（上轮评测通过）、`after_failure`（上轮评测失败）。

**观测量**：低概率定义为 `logprob ≤ -2`（自然对数）。组间比较使用 Fisher 精确检验（双侧）。

### 3.5 规模

4 任务 × 2 条件 × 16 重复 = 128 次运行，其中 127 次产出有效数据。

---

## 4. 结果

共 98,533 个 content token。

### 4.1 RQ1：CUDA 词汇的概率显著更高

| 分组 | n | 平均 logprob | 低概率率(≤-2) |
|---|---|---|---|
| 宽松表命中 | 4,524 | -0.038 | **0.49%** |
| 宽松表未命中 | 94,009 | -0.062 | 0.84% |
| 严格表命中 | 2,344 | -0.013 | **0.26%** |
| 严格表未命中 | 96,189 | -0.062 | 0.84% |

Fisher 精确检验：宽松表 **p = 0.0070**，严格表 **p = 0.0007**。

**CUDA/Triton 词汇的低概率率显著低于其他 token，且两种定义边界下方向一致。**

作为对照，结构 token（`for`、`if`、`return` 等无替代写法的词）的低概率率为 1.54%，高于其他 token 的 0.81%——即"低概率"与"是否为 CUDA 词汇"无关，甚至方向相反。

### 4.2 RQ2：上下文效应远强于 token 类型

| 上下文 | n | 平均 logprob | 低概率率 | p10 |
|---|---|---|---|---|
| **after_failure** | 3,143 | -0.276 | **4.30%** | -0.898 |
| after_success | 46,719 | -0.033 | 0.48% | -0.000 |
| initial | 48,671 | -0.074 | 0.94% | -0.049 |

**`after_failure` 的低概率率是 `initial` 的 4.6 倍、`after_success` 的 9 倍。**

按任务：`softmax_rows` 1.20% > `row_reduce` 0.99% > `layernorm_rows` 0.53% > `vector_affine` 0.47%。

### 4.3 RQ3：压缩反馈未能降低低概率率

| 条件 | n | 低概率率 |
|---|---|---|
| 完整反馈 | 50,238 | 0.71% |
| 截断反馈 | 48,295 | 0.95% |

截断反馈不仅未降低低概率率，反而略高（0.95% vs 0.71%）。**该干预无效。**

### 4.4 低概率 token 的构成

全库最低的 token：

```
' constants'   -17.746   row_reduce    full       after_success
' backed'      -11.271   row_reduce    full       after_success
' wrong'       -10.056   softmax_rows  full       after_success
' columns'      -9.527   layernorm_rows truncated initial
```

**无一为代码 token；全部为自然语言词。**

---

## 5. 讨论

### 5.1 因果链起点不成立

RQ1 的答案是否定的，且方向与假设相反。CUDA/Triton 专属词汇不仅不构成低概率尾部，其低概率率还显著低于其他 token。这直接否证了"CUDA 语法稀有 → 低概率"这一环节。

可能的机制是：这些 token 高度模式化、上下文约束极强（`tl.load` 的后续位置几乎确定），因而模型在它们上的条件概率接近 1。这与引言中"非常固定、非常模式化"的观察一致。

### 5.2 低概率来自失败反馈后的上下文

RQ2 显示上下文效应强于 token 类型一个数量级。`after_failure` 的低概率率是 `initial` 的 4.6 倍，这与"长 agent 轨迹中累积的分布外输入导致偏离"的解释一致：评测失败的反馈文本（编译错误、不匹配的输出）本身是训练分布中稀有的输入形态。

值得注意的是，`after_success` 反而低于 `initial`——成功反馈收窄了后续的候选空间。

### 5.3 干预无效的含义

RQ3 表明，压缩反馈的**信息量**并不减少低概率。结合 5.2，低概率的来源是"失败这一事件本身改变了上下文分布"，而非"反馈文本太长"。因此针对反馈长度的压缩并非有效干预。

### 5.4 方法启示

本研究早期的一轮（本地 vLLM + 4B 模型）曾得出"CUDA token 低概率率 10%、p = 0.0156"的结论，后经核查为**分类偏差**：分类器在散文段刻意排除了若干 Triton 关键词（以避免与英文散文混淆），使该类别仅捕获 20/451 个 token，而其中恰含两个极端采样值。同一个词 `load` 因此被切进两类：归入散文的 6 次平均 -0.044、零低概率；归入 Triton 的 4 次平均 -1.034、25% 低概率。

**教训**：任何按类别做的统计，必须先做"放宽定义边界后结论是否仍成立"的稳健性检查。本研究的正式结果因此报告了两种定义边界。

---

## 6. 局限

1. **模型**：使用 `deepseek-chat`，而非 CUDA Agent 的 Seed1.6。结论不能直接外推到后者——尽管 `deepseek-chat` 已是公开 API 中能力最接近的一档。
2. **任务**：4 道，均为元素级或归约级，未包含算子融合、tiling、launch 配置等更接近真实优化的任务类型。
3. **样本分布**：`after_failure` 仅占 4.3% 的 token，其 4.30% 的低概率率基于 3,143 个样本。
4. **未测**：低概率是否真的导致 PPO 重要性比率爆炸（需实际训练）；train-inference mismatch 的贡献比重（需双引擎对照）；Seed1.6 特定权重上的表现（需其权重）。
5. **单一采集路径**：所有数据来自同一驱动与同一 API，未做跨引擎复现。

---

## 7. 结论

本研究以 127 次运行、98,533 个携带条件概率的 token 检验了 CUDA Agent 因果链的起点，发现：

1. CUDA/Triton 专属词汇的低概率率**显著低于**其他 token（0.26% vs 0.84%，p = 0.0007），因果链起点不成立。
2. 低概率真正集中在**失败反馈之后**的上下文（4.30%），约为初始上下文的 4.6 倍。
3. 压缩环境反馈**不能**降低低概率率。

因此，把 CUDA 训练的困难归因于"CUDA 语法稀有"缺乏经验支持；更可能的来源是多轮 agent 轨迹中的失败反馈改变了输入分布。后续工作应以此为出发点设计干预，而非增加 CUDA 语料。

---

## 参考文献

1. CUDA Agent: *The Root Cause of Training Instability*（原文标题所引论断）
2. Liu et al. *Your Efficient RL Framework Secretly Brings You Off-Policy RL Training*
3. Carlini et al. *Stealing Part of a Production Language Model*
4. *How Open Must Language Models be to Enable Reliable Scientific Inference?* (2026)
5. *Log Probability Tracking of LLM APIs* (INRIA, OpenReview)

---

## 附录：复现

```bash
# 最小验证（路线是否成立）
python lowprob-attribution/scripts/check_plaintext_logprobs.py --model deepseek-chat --max-tokens 16000

# 正式采集
python lowprob-attribution/scripts/loop_driver.py \
    --tasks row_reduce vector_affine \
    --conditions full truncated --repeats 16 --max-turns 8 \
    --model deepseek-chat --max-tokens 8192 \
    --runs-root lowprob-attribution/runs-formal
python lowprob-attribution/scripts/loop_driver.py \
    --heldout --tasks layernorm_rows softmax_rows \
    --conditions full truncated --repeats 16 --max-turns 8 \
    --model deepseek-chat --max-tokens 8192 \
    --runs-root lowprob-attribution/runs-formal

# 分析
python lowprob-attribution/scripts/analyze_loop.py lowprob-attribution/runs-formal
```
