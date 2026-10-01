# Stage 1 — 逐篇核查表（Source Verification）

检索时间：2026-10-02。方法：arXiv Atom API（`export.arxiv.org/api/query`）直查 ID / 标题；
`nature-academic-search` skill 的 OpenAlex 降级脚本在本环境连续两次读超时，按 skill
错误处理规则改走 arXiv API + WebSearch（用户亦指示文献检索非关键路径）。

状态定义：**VERIFIED** = API 返回了标题+作者+日期；**UNVERIFIED** = 检索不到，禁止引用。

---

## A. Spec §5.2 特别核查清单

| # | 论文 | Spec 给的 ID | 实际 ID | 状态 | 实际标题 / 备注 |
|---|------|--------------|---------|------|-----------------|
| 1 | CUDA Agent | 2602.xxxxx（未定） | **2602.24286v1** | VERIFIED | *CUDA Agent: Large-Scale Agentic RL for High-Performance CUDA Kernel Generation*; Dai, Wu, Yu, et al.; 提交 2026-02-27。摘要确认 Level-1/2/3 相对 torch.compile 提速率 100%/100%/92%，Level-3 领先 Claude Opus 4.5 与 Gemini 3 Pro 约 40%。**与笔记论述一致** |
| 2 | FlashRT | 2607.18171 | **2607.18171v2** | VERIFIED | *FlashRT: Agent Harness for Guiding Agents to Deploy Real-Time Multimodal Applications*。标题确含 "Multimodal"，与笔记「标题写多模态而后文消失」的可疑点相符（该点需读正文才能断言，本文不作强断言） |
| 3 | HFSV | 2606.08960 | **2606.08960v1** | VERIFIED | *Hardening Agent Benchmarks with Adversarial Hacker-Fixer Loops*; 2026-06-08 |
| 4 | Correctness Illusion | 2606.20128 | **2606.20128v2** | VERIFIED | *The Correctness Illusion in LLM-Generated GPU Kernels*; 2026-06-18 |
| 5 | Contract-Grade Verifier | 2608.12700 | **2608.12700v2** | VERIFIED | *A Contract-Grade Verifier for LLM-Generated GPU Kernels, and a Native Blackwell Backward for the Gated-Linear-Recurrence Family*; 2026-08-13 |
| 6 | KernelBench-Verified | 2607.16241 | **2607.16241v1** | VERIFIED | *KernelBench-Verified: Do LLM-Generated Kernels Actually Beat PyTorch?*; 提交日 2026-06-26（ID 前缀 2607 与提交日不一致，引用时以 arXiv 记录为准） |
| 7 | Atrex-Bench | 2607.14541 | **2607.14541v1** | VERIFIED | *Are LLM-Generated GPU Kernels Production-Ready? A Trace-Driven Benchmark and Optimization Agent*; 2026-07-16 |
| 8 | ProofWright (Stanford) | 无 ID | — | **UNVERIFIED** | arXiv 标题/全文检索未命中。⚠️ 论文中**不引用** |
| 9 | "When Speed Kills Stability" | 无 ID | — | **UNVERIFIED** | arXiv `all:"When Speed Kills Stability"` 未命中任何论文。⚠️ 该标题来自 spec §0.3 的对话叙述，本会话无法证实存在，论文中**不引用**；train–inference mismatch 论点改用下面 #12/#13 的真实文献 |

> **诚信提示（Stage 2.5 会重复检查）**：spec 背景叙事里「CUDA Agent 发表仅半年」与
> 2602.24286（2026-02-27）一致；但同期中文媒体报道集中在 2026-08-14~18，
> 那是后续版本/传播而非首发，引用一律以 v1 提交日为准。

## B. 数据源与基准

| 论文 | ID | 状态 | 备注 |
|------|----|------|------|
| AdaExplore | **2604.16625v2** | VERIFIED | *AdaExplore: Failure-Driven Adaptation and Diversity-Preserving Search for Efficient Kernel Generation*; 2026-04-17。**本文数据来源** |
| KernelBench | **2502.10517v1** | VERIFIED | 2025-02-14 |
| CUDA-L1 | **2507.14111v12** | VERIFIED | *Improving CUDA Optimization via Contrastive RL* |
| Dr. Kernel | **2602.05885v2** | VERIFIED | *RL Done Right for Triton Kernel Generations* |
| NKI-Agent | **2607.04395v1** | VERIFIED | 把 CUDA-Agent 框架迁移到 AWS Trainium/NKI；报告 GRPO+二进制编译奖励未能超过 SFT。**支持「范式可跨硬件迁移」的批判性论据** |

## C. Related Work 四类图谱的核心锚点

### (1) Compiler IR / schedule space
- TVM **1802.04799v3** ✅ ｜ Ansor **2006.06762v5** ✅ ｜ Halide（PLDI 2013，非 arXiv，引用需走 DOI）

### (2) RL credit assignment / edit-based action space ← **最需直面**
- Learning Performance-Improving Code Edits **2302.07867v5** ✅（ICLR 2024 Spotlight; Shypula, Madaan, Zeng, Alon, Gardner, Hashemi, Neubig, Ranganathan, Bastani, Yazdanbakhsh）
- Neural Networks for Modeling Source Code Edits **1904.02818v1** ✅（ICLR 2019; Zhao, Bieber, Swersky, Tarlow）
- SWE-RL **2502.18449v2** ✅（软件演化数据上的 RL）
- To Diff or Not to Diff? **2604.27296v1** ✅（ACL 2026 Findings；输出格式=block-diff vs full-code）

### (3) LLM agent action space / skill library
- Voyager **2305.16291v2** ✅（技能库=可复用程序化 action）
- Option-Critic Architecture **1609.05140v2** ✅（时间扩展动作的理论基座）
- When Do Options Help? **2609.05508v1** ✅（高层 action 何时无效——本文反面先验）

### (4) CUDA-specific methods / verifier hardening
- CUDA Agent **2602.24286** ✅ ｜ CudaPerf **2607.20908** ✅（结构性奖励）｜ HFSV **2606.08960** ✅ ｜
  Correctness Illusion **2606.20128** ✅ ｜ Contract-Grade Verifier **2608.12700** ✅ ｜
  Measuring the Checker (mutation analysis for GPU-kernel oracles) **2609.22220** ✅ ｜
  Real-workload coverage **2609.21058** ✅

### (5) train–inference mismatch（替代 spec 中未证实的那两篇）
- Rethinking Training-Inference Mismatch in LLM RL **2609.32444v1** ✅
- Score Centering Stabilizes Off-policy RL **2609.20807v1** ✅
- $T^5$: Twin-Critic Training for Token-Level Thoughts **2609.32791v1** ✅

## D. 无先验工作做过同一命题——论证要点

1. edit-based 代码 RL（2302.07867 / 1904.02818 / 2604.27296）动作单位停留在**文本操作层**，
   未提出带类型参数的语义变换对象，也未检验「一个 action ↔ 多个分散 diff hunk」的多对多结构；
2. CUDA-specific RL 工作（2602.24286 / 2607.20908 / 2507.14111 / 2602.05885）改变的是
   **reward / 训练流程 / 对比目标**，动作空间始终是 token；CudaPerf 把 CUDA 性质放进 reward，
   恰是「问题不在 action space」的隐含假设；
3. 编译器谱系（TVM/Ansor/Halide）确实以变换为操作单位，但其变换由**人写好的 pass** 枚举，
   不涉及 LLM 策略，也没有「界面上移是否改变搜索/学习效率」的经验问题；
4. options / skill-library 谱系给出理论可能性和反例条件，但没有落到 GPU kernel 优化轨迹的
   实测 vocabulary 上。

**因此本文的净增量**：用真实公开优化轨迹（AdaExplore）测量 transformation 熵 vs code-diff 熵，
并给出四条件证伪协议。这是一个**测量 + 协议**贡献，不是新算法贡献——论文明确按此定位，不夸大。
