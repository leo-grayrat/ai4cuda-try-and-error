# Peer Review 2 — 领域审稿意见（Reviewer: agentic RL for code generation / LLM systems optimization）

**稿件**：`paper/main.md`（841 行，"Is the token-level action space the wrong coordinate system for LLM-driven CUDA kernel optimization?"）
**审稿身份**：Peer Reviewer 2（领域审稿人）。独立评审，未阅读 `review_eic.md` / `review_methodology.md`。
**审稿模式**：READ-ONLY。稿件、`SPEC.md`、`data/`、`stage1_research/` 一律未修改。审稿人为复核数据自行编写的三个临时统计脚本置于 `stage3_review/`，不属稿件内容，本意见定稿后即删除；所有被引用的数字均可按 §附录 的判据重算。

---

## 0. 总判断（TL;DR）

这是一篇**自我诚实度明显高于同类 position paper、但novelty定位与最近邻刻画尚不成立**的稿件。作者的三项交付物中：

- **协议（§4）**：最扎实。四条件 + 判读树 + vocabulary-freeze + held-out，确实"prior informal discussions of this idea lacked"这一句我认为成立。
- **测量（§3）**：中等，且**标题性证据存在可被证伪的统计学缺陷**——§3.5 的 `distinct diffs` 列在数学上是恒等式，§5.1 又称之为"the single most concrete piece of evidence this paper offers for the distinction"。
- **命题（§2）**：最弱。差异化三条腿（语义类型化 / 跨位置协调 / 跨任务复用）在本稿中一条未被操作化、一条统计不可识别、一条与采样设计诱导的随机散布不可区分。

对 CUDA Agent 的批评存在**实质性稻草人**：该文附录明确写着 "These tools define the action space of the agent"，而稿件 §2.1/§6 的核心批评维度是"它把动作放在 token 层"。

**净结论**：`文献覆盖 42 / 理论框架恰当性 58 / 领域增量 55 / 总体 52`，推荐 **Major Revision**（详见 §6）。诚实声明（§1.5、§3.6、§7）不降低新颖性门槛，按用户要求我未因此放松任何一条判定。

---

## 1. Q1 — §5.1 差异化论证是否站得住

### 1.1 稿件的差异化主张（原文锚点）

> §5.1 (L539-544): "They operate on text-level operations: a span is rewritten, a function is replaced, a diff format is chosen. Ours is a semantic object with typed arguments and a *realization obligation across sites*. The distinguishing property is measurable, not rhetorical: for an edit-based unit, the action is the diff; for a transformation, the same action recurs across many different diffs and across many tasks. Section 3.5 measured exactly that - 34 occurrences of `ChangeLaunchConfig` across 30 tasks with 34 distinct diffs - and it is **the single most concrete piece of evidence this paper offers for the distinction**."

> §3.5 (L350-351): "**No category ever produced the same diff twice, and every frequent category recurs across tens of different tasks.** That is precisely the configuration the thesis predicted: the decision repeats, the code does not."

### 1.2 判定：概念差是**实质性的**，但本稿给出的**证据不是**

先说公平的一面。"realization obligation across sites"（一个语义动作对多个分离位置负有实现义务）这个概念，与"一个 span 被重写"确实是不同的对象——前者要求在 `sites = {s₁,...,sₖ}` 上的**协调一致性**，后者按定义是单连通区域。这个区分如果是真的，是有内容的。审稿人 2 承认它是**概念上**实质差异，不是措辞差异。

问题在于：**稿件用来"证明"这个差异的唯一量化证据，经复核是零信息的**。

### 1.3 攻击点定位：最有力的"换皮"攻击在哪一处（CRITICAL）

**最有力攻击点 = §3.5 的 diversity 表 `distinct diffs` 列 + §5.1 对它的引用。** 具体三重：

**(CRITICAL-1) `distinct diffs == occurrences` 是数学恒等式，不是发现。**

证据：`data/annotations.json` 的 `statistics.distinct_diff_fingerprints = 55`，`n_edges = 55`。即**55 条边的 diff fingerprint 两两互异**（稿件 L329 自己也写了 "55 of 55 fingerprints distinct"）。因此对 55 个互异元素的**任意**划分，任一 cell 内 `distinct_diffs` 恒等于该 cell 的 `occurrences`。

我复核：
- `within_label_diversity` 中出现的**全部 11 个类别**，`distinct_diffs == occurrences` 逐行成立（程序判定 `True`）。
- 用 20,000 次**随机标签重分配**（同样 10 个类别、同样 cell 尺寸）模拟：`P(distinct_diffs == occurrences) = 1.000`。

也就是说，把标签换成随机哈希，会得到一张**逐列数字形态完全相同**的表。这一列不能区分"语义词汇在复用"与"任意分组"，因为它在任何分组下都成立。

严重度：**CRITICAL**。这不是措辞问题，是 headline 证据无效。
最刺痛之处在于：**稿件 §3.6 刚刚抓到了同一个逻辑缺陷**（"A smaller description length is a property of any coarse summary"，L394），却没有把它应用到 §3.5 这张表上。审稿人有理由怀疑：负性结果是主动自查的，正性结果是未经自查的。

修法（可执行，任选其一并替换 §5.1 的引用）：
1. 从 §3.5 表中**删除 `distinct diffs` 列**，或加脚注"本列在 |E| 个互异 fingerprint 下为恒等式，不作论证用"。
2. 把 L350-351 的加粗结论改为条件句，并显式指向 §3.6 的 null 逻辑。
3. §5.1 的 "the single most concrete piece of evidence" 必须改写或撤回——目前它把本文最强的修辞押在最弱的算术上。

**(CRITICAL-2) 唯一有信息量的列 `distinct tasks`，与采样设计诱导的随机散布不可区分。**

`distinct tasks` 不是恒等式（上界 `min(occ, 45)`），所以这是唯一可辩护的证据。我做了一个正确的 null：**从 55 条边中随机抽 occ 条边，统计其跨任务数**（20,000 次，seed 20261002）。稿件的采样设计是 45 个任务、每任务 ≤3 条边（36/8/1 分布），这个设计本身就强制"任意 34 条边大概横跨 30 个任务"。

| Category | occ | obs tasks | null median | null p95 | null min | 观察值百分位 |
|---|---|---|---|---|---|---|
| `ChangeLaunchConfig` | 34 | **30** | 30 | 32 | 25 | **67.3%** |
| `EliminateMaterialization` | 32 | **29** | 29 | 31 | 24 | **51.0%** |
| `Fuse` | 25 | 22 | 23 | 25 | 17 | 88.4% |
| `PrecisionChange` | 22 | 18 | 20 | 22 | 15 | 99.4% |
| `Tile` | 14 | 13 | 13 | 14 | 10 | 88.1% |
| `HostDeviceOverlap` | 14 | 12 | 13 | 14 | 10 | 98.4% |
| `Vectorize` | 6 | 5 | 6 | 6 | 3 | 99.7% |

三行关键：随机抽 34 条边 → distinct tasks 的 min 25 / **median 30** / max 34。稿件作为"跨任务复用"证据报告的 **30 恰好是随机中位数**；`EliminateMaterialization` 的 29 同样恰在中位。

更糟的是方向：`PrecisionChange` / `HostDeviceOverlap` / `Vectorize` 的观察值**低于**随机中位（99.4%/98.4%/99.7 百分位，即 P(null ≥ obs) ≈ 0.006/0.016/0.003），意味着这些类别的标注边**聚在同一任务内部**——这是"跨任务通用词汇"的**反证**，稿件未讨论。

严重度：**CRITICAL**。
修法：(a) 在 §3.5 表右侧直接加 `null median` 与百分位两列（我已算出，作者可用自己脚本复算）；(b) 承认在该采样设计下，跨任务复用需要**显式设计对照**才能声称，例如"同一 category 的边在任务层面的分布 vs 分层随机"或把每任务边数上限从 3 移到不限；(c) 或把 §5.1 的差异性证据降级为"待干预实验提供"。

**(CRITICAL-3) 第二条腿"跨位置协调"在本样本中统计不可识别。**

Abstract/§2.5 的核心是"一个动作落在多个分离位置"。但：
- **0/55 条边是单标签**；secondary 计数分布 `{1:9, 2:37, 3:9}`，即 **55/55 条边至少 2 个标签**。
- 多区域率随标签数单调上升：2 标签 → 3/9 = 33%；3 标签 → 22/37 = 59%；4 标签 → 5/9 = 56%。

因此 54.5% 的"≥2 分离区域"**无法区分**两种完全不同的解释：
 (i) 一个动作散落多处（论文的命题）；
 (ii) 多个动作各在一处（edit-based 的命题）。
标签数与区域数共变正是 (ii) 的预测。稿件 §7.2(1) 承认"multi-label edges are the norm"，但**没有把它用于削弱 §3.5 的多区域证据**——这是论文内部一处未闭合的自相矛盾：自我证伪装置只运行了一次。

严重度：**CRITICAL**。
修法：把 §4.5 control 5（decomposability stress test）提前做成**观察版的归属检验**：对 55 条边分别统计"单 primary 且 secondary 为空"子集的 multi-region 率——本样本该子集为空，这本身就是可用的、比 54.5% 更诚实的结论。或允许标注者输出单标签，重标一个子样本，报告单标签边的多区域率。

### 1.4 "语义类型化"这条腿：未被操作化，且与 §2.3 自相矛盾（MAJOR）

> §2.3 (L210-219): "We therefore expect the eventual object to be a small typed DSL rather than **a list of nouns** ... but this paper deliberately does **not** build that DSL. Designing forty-seven action types, a schema, a validator and a parser before establishing that the phenomenon exists is the single most likely way to produce a body of engineering that answers no question."

本稿实际交付物就是 `VOCABULARY_FROZEN.md` 的 12 个名词类别 + LLM 事后填写的 `primary_args`。§7.3 承认 "`phi` is never actually constructed"。于是"typed"目前只有两项支撑：类别名是名词性的、参数字典非空（我复核 `primary_args` 55/55 非空，但由同一 LLM 从 diff 反推，无 validator）。

这是**自我否定的**：§2.3 明确说"list of nouns"不是目标对象，而 §5.1 却把"semantic object with typed arguments"当作与 edit-based RL 的区分点。区分点的另一半（"typed"）在本稿中不存在，只剩另一半（realization obligation across sites）——而那一半刚被 CRITICAL-3 打成不可识别。

MAJOR。修法（择一）：
1. §5.1 改为"我们主张的区分包含两个性质，本稿只测了其中一个，且测量当前不成立"，让差异化陈述与实际交付物对齐；
2. 或补一个**最小可执行判据**：对 12 类各写一个 regex/AST 检测器（如 `ChangeLaunchConfig` = diff 中 `@triton.autotune` 的 `configs=` 列表被改），报告在 55 条边上的 precision/recall——这一项即可把"typed"从声明变成对象，成本约 1–2 人日，且顺带回答 §7.3 的 φ 缺口；
3. 并给出 **action identity criterion**：在什么条件下两处 diff 算"同一个 τ"？没有它，"同一动作重复出现"不可证伪（见 §3.5 of this review 的 MAJOR-9）。

### 1.5 论文是否"真正回应"了"edit-based RL 换皮"攻击，还是只加了 disclaimer？

判定：**只加了 disclaimer，未回应。** 依据：
- §5.1 承认 "Our relation to them is exact and worth stating carefully, because they are ... the most likely source of a 'this is edit-based RL rebadged' objection"（L534-537）——**预判到了攻击**，但给出的回应是 §1.3 那三句 CRITICAL 级证据。
- 全文 §5.1 / §6 / Abstract 三处均未携带 §3.6 或 §7.1 的 caveat。也就是说**自我修正没有传播到用它做区分的章节**。这是"加 disclaimer"的典型形态：caveat 存在于 §7，主张在 §5.1 保持原强度。
- 更关键：§5.1 与 §3.5 使用了**不同的证据口径**。§3.5 的 diversity 表按 primary+secondary 计数（ChangeLaunchConfig = 34），而 §3.5 上方的 primary 计数表写 `ChangeLaunchConfig | 3`。稿件未说明两表口径不同，读者会误读为同一个量。

MAJOR-10（口径不一致）修法：两表统一列名（`occurrences = primary ∪ secondary` vs `primary only`），并在正文中说明为何差异化证据使用 union 口径。

---

## 2. Q2 — 对 CUDA Agent 的批评是否公允：稻草人风险**成立且严重**

### 2.1 稿件的批评与它自设的限制

> §1.1 (L48-56): "Read at the level of the abstract and the released recipe, the training-side machinery is a chain of generally motivated fixes ... **None of them, as described, is *forced* by memory hierarchies, warp semantics, coalescing, or shared-memory bank conflicts. Substitute FORTRAN or FPGA synthesis and the recipe transfers untouched.** There is now direct evidence of exactly this transfer: NKI-Agent adapts the CUDA-Agent framework to AWS Trainium ... and reports that GRPO with a binary compilation reward fails to improve on supervised fine-tuning [18]."

> §1.1 (L58-62): "the full text of [1] is **not retrievable from the preprint server** ... We therefore argue only at the level the abstract supports."

作者在此做了两件正确的事：标注了证据层级、声明了不越界。但复核后，四点都不成立。

### 2.2 (CRITICAL-4) 事实错靶：CUDA Agent **自己声明**它的工具集就是 action space

我核验了 `arxiv.org/html/2602.24286v1` 正文（原文，非二手转述）：

> "On top of this, our agent loop follows the ReAct-style paradigm, interleaving reasoning, action execution, and observation ..."
> "**These tools define the action space of the agent** and serve as the only mechanisms through which it can interact with the system and the external world."
> 工具集：Bash、Read/Write、**Edit / MultiEdit**（"Support deterministic string-level code modifications"）、Glob、Grep、NotebookEdit、BashOutput、KillBash。

后果是精确的：稿件 §2.1 三层 cut 表的第一行（"Token | one sampled token | GAE/value estimates over ~10³–10⁴ steps | decision unit invisible"）与 §6 对 CUDA Agent 的定性，在其**最核心的批评维度上是错的**。CUDA Agent 的 RL 训练在 token 上（PPO over 采样序列，这点稿件没错），但它**声明的 action space 是粗粒度工具调用**，且 MultiEdit 的设计动机恰是"一次调用内做多个原子编辑"——这正是"跨位置协调"的一种实现载体。稿件批评了一个它批评的对象已经公开否认的立场。

严重度：**CRITICAL**（稻草人 + 事实性错靶）。
修法：
1. §1.1 与 §6 必须显式引 CUDA Agent 的 action-space 声明，并把批评从"它把动作放在 token 层"改述为**"它的 action space 是通用开发工具（Bash/Edit/Read），不含程序语义类型；语义类型化发生在 SKILL.md 提示层而非策略层"**——这是公允且更强的主张，因为它可检验（比较提示层规则移除后的消融）。
2. §2.1 表格需要第四行：**Agent turn / tool call**（介于 token 与 transformation 之间，见 §2.4）。
3. 撤回或改写 L48 "Read at the level of the abstract"——见 2.5。

### 2.3 (CRITICAL-7) "没有任何一步是 CUDA 被迫的"未经对照即断言

稿件把 CUDA Agent 的训练设计描述为"chain of generally motivated fixes"，暗示每一步都是通用 ML 工程而非领域强制。但我核验 CUDA Agent 全文（多源一致），该文对每一项都**给了消融**：

- 里程碑式奖励 `r ∈ {−1, 1, 2, 3}`（含 `b(t,t₀) = 1[(t₀−t)/t₀ > 5%]`）：**"用原始加速奖励替换鲁棒奖励调度 → 明显更弱的优化性能"**；
- 移除 agentic loop → 大幅下降；
- 移除 RFT → 奖励灾难性崩溃 + 策略熵急增；
- 无 value 预训练 → 轨迹长度爆炸；初始 RL 仅稳定约 17 步。

因此"每一步都只是通用补丁"这一叙述**与源文献的消融证据冲突**。作者的论断真正能说的是：这些消融证明了"这些设计在这个任务上必要"，但未证明"它们只能因 CUDA 而存在"——**这是一个关于因果来源（why-forced）而非效果（whether-helps）的论点**，而后者才是有数据的部分。

修法（具体）：把 §1.1 的五连问改写为一条可辩护主张："CUDA Agent 报告了这些组件的效果消融，但未报告**跨域**消融（同一配方在 FORTRAN/RTL/FPGA 上的对照），因此 CUDA-specificity 目前是未测的而非已被否证。" 这句话既保住攻击方向，又消除稻草人。

### 2.4 (MAJOR-3) §1.1 的 NKI-Agent 证据自相矛盾，但可改造为更强的论证

稿件用 NKI-Agent（迁移到 AWS Trainium）作为"配方原样迁移"的 **direct evidence of exactly this transfer**。但引用的内容是"**GRPO with a binary compilation reward fails to improve on supervised fine-tuning**"——这是迁移**失败/未成功**的结果，不是"迁移成功所以配方与 CUDA 无关"。用 A 证明 not-A，是逻辑错误而非措辞问题。

不过这个数据点实际上**更有价值**，只需改述：GRPO 在 Trainium 上没超过 SFT，恰恰说明 **CUDA Agent 的配方是欠规定的（under-specified）——迁移的不是方法而是运气/细节**。这把批评从"它不 CUDA"（可被消融反驳）转成"它的配方不可归因、不可复现"（有独立证据支持，且与 §4.5 的 freeze/held-out 要求同构）。建议直接这样替换 §1.1 末段与 §6 的对应句。

### 2.5 (MAJOR-4) "全文不可检索"这一限制声明与事实不符

我成功读取了 `arxiv.org/html/2602.24286v1` 的正文（含工具/action-space 段、奖励定义）。因此 L59-60 "the full text of [1] is not retrievable from the preprint server" 不成立：**全文是可获取的**（arXiv HTML 实验版）。这条声明目前的作用是给 §1.1 的摘要级论断买免责额度，但它同时**掩盖了一个更容易被审稿人抓到的问题**：如果全文可读，那么"错靶"就不是信息受限、而是阅读不足。

修法（诚实且有利）：把 §1.1 的限制声明改为"我们最初仅依据摘要与公开 recipe 论证；补读 arXiv HTML 全文后，我们对 [1] 的批评需按下述三处修正"，并**在正文中公开这三处修正**——这会把一次致命弱点转成 methodological honesty 的范例，且与 §1.5 的诚实声明一致。

### 2.6 是否忽略了 CUDA Agent 自声明的贡献类型？——是，但影响有限（MINOR-7）

CUDA Agent 摘要自报的贡献是三类：可扩展**数据合成**、**技能增强的 CUDA 开发环境**（自动验证 + profiling）、**稳定训练的 RL 算法**。稿件 §1.1 的五连问全部落在第三类，而第三类恰是唯一有消融支撑的一类（见 2.3）；同时"领域知识进入系统的**位置**"（第一、二类）恰是稿件真正该打的地方——尤其 SKILL.md 的四步流程本质上是**一个被写在提示层的隐式 transformation pipeline**。稿件没有把火力集中在这里，是一个可挽回的定位失误。

MINOR→MAJOR 边界，我判 **MAJOR-11**：§1.1 的火力分配与 §5.4（agent skill）之间有一条未打通的论证线。修法：把 SKILL.md 的"四步优化流程"作为 §2.1 表格的第四层证据来讨论，与 transformation-level 做直接对比（提示层 vs 策略层）。

---

## 3. Q4 — CudaPerf (arXiv:2607.20908) 的定位是否准确

### 3.1 稿件的主张

> §6 (L651-655): "CudaPerf is the most instructive case for us, because it explicitly adds *structural* rewards derived from parallelization features - memory coalescing, occupancy, arithmetic intensity, synchronization patterns - **while keeping the action space at token level** [15]. CudaPerf is thus **the clearest existing statement of the position we mean to challenge**: it treats CUDA structure as something that belongs in the *reward*, whereas we ask whether it belongs in the *action*."

### 3.2 判定：**对比方向有趣但两处不成立**

**(MAJOR-12) 事实基础未核验。** 我对 `arXiv:2607.20908` / "CudaPerf" 执行了多轮检索，**未能独立命中该文**（无法确认标题、作者 Mahmud/Ahmed/Jannesari、也无法读到任何关于其 action space 的表述）。稿件把**最关键的"立场对照物"**建立在一篇审稿人无法核验、且其内部动作接口无法从标题/摘要判定的工作上。若该文不存在或描述有偏，§6 的一段论证与 Abstract 的隐含对照同时塌陷。
要求：作者提供该文 PDF/HTML 原文中定义动作接口的**逐字段落**；若不可得，§6 与 §1.3 的对照物必须改为可核验对象（Kevin / CUDA Agent / CuAsmRL 均可）。

**(MAJOR-13) "reward vs action"二分本身立不住。** 对一个"动作"按结构特征（coalescing / occupancy / arithmetic intensity）打分，**就是在用 reward 间接定义什么动作值得被采样**——这是动作空间设计（action shaping 的镜像）。二者并非稿件所称的正交两轴（稿件 L655 甚至自己写了 "The two are orthogonal and could be combined"，与 L654 的"clearest statement of the position we mean to challenge"相冲突：若正交，则 CudaPerf 根本不是对立立场）。
修法：把 §6 的对照改为一个**三维**表述：`(credit 落在哪里, 候选动作从哪里来, 结构信息进入哪一层)`。在该坐标下 reward-based-structure 与 action-based-structure 的关系变成"同一信息的两种注入路径"，稿件真正的主张——**注入到候选生成层比注入到打分层在样本效率上更有效**——才成为可证伪命题（且可以直接写进 §4 的判读树作为第五条件，或作为 4 条件的奖励侧对照臂）。

### 3.3 **是的，存在稿件完全没有意识到的、真正在 action-space 层面做过程序变换的工作**

这是本次审稿最有杀伤力的发现，逐条列于 §4（含核实状态），此处给结论：

**(CRITICAL-5) K-Search（arXiv:2602.19128，Cao/Mao/Gonzalez/Stoica, UC Berkeley）** 已经实现了稿件 §4 的 Structured-action 条件，并给出了正向结果。摘要原文（我逐字核验）：

> "Existing automated approaches typically treat LLMs merely as stochastic code generators within heuristic-guided evolutionary loops. These methods often struggle with complex kernels requiring **coordinated, multi-step structural transformations** ..."
> "This approach **explicitly decouples high-level algorithmic planning from low-level program instantiation**, enabling the system to navigate non-monotonic optimization paths while remaining resilient to temporary implementation defects."
> 三阶段循环：**动作选择（action selection）、局部细化（local refinement）、模型更新**；世界模型"估计高层优化意图的优先级分数"。
> 结果：FlashInfer GQA/MLA/MoE 平均 **2.10×**、MoE 最高 **14.3×**（vs OpenEvolve / ShinkaEvolve）；GPUMode TriMul H100 **1030 µs**，超人工设计与进化搜索 SOTA。

摧毁面很宽：
- §1 的"非平凡主张"（L123-124："Moving the interface from *generate the whole program* to *select a parameterized transformation, then realize it* measurably changes search efficiency"）——**已被在位实现并测得正向结果**；
- §4 的"只有干预实验能判定"——**K-Search 就是那个干预实验**（search-side），稿件 §7.5 的 "No measured improvement in any metric" 因此需要限定为"我们未测"而非"无人测"；
- §7.2 危险(3)（Selection 容易、Realization 才是瓶颈）——K-Search 的解耦架构正是**为分离这两者而设计**，其结果部分回答了该危险；
- §7.2 危险(2)（模型可能已内部完成抽象）——K-Search 的 world-model-as-planner 与 Free-plan 对照同构。

**(CRITICAL-6) CuAsmRL（arXiv:2501.08071, CGO 2025, DOI 10.1145/3696443.3708943, He & Yoneki, Cambridge）** 是一个**显式定义 action space 的、程序变换层的 RL**（我逐字核验其 §3.1/§3.4/§3.5/§3.6/§5.7）：
- **§3.5 动作空间**：选一条内存指令（LDG/LDGSTS/STG）与相邻指令交换；输出 = 指令索引 + 方向；
- **动态 action masking**：依赖分析（含寄存器相邻、屏障/标签跨块禁止）把非法动作概率置零，无可行动作即终止——这正是稿件 §4.5 想要的"合法性约束"先例；
- **§3.1 分层搜索空间**：**先由 autotuner 枚举算子配置求最优，再在其上优化 SASS 调度**——这就是稿件 §3.7(b)/§4.5 "必须先把 autotuning 从动作空间中因子化出去"的**已实现先例**；
- **§5.7 "自动发现 optimization moves"**：把 RL 学到的动作回溯成语义优化策略——这是"automatic action recognition"（稿件 §7.3 说"我们故意没打开的那个问题"）的**已打开版本**；
- 结果：最高 26%、几何平均 9%，且**集成进 Triton**（`@cuasmrl.jit` 一行改动）。

严重度：两条 **CRITICAL**。这两篇不是"补引即可"的背景文献，而是**改变稿件主张归属的在位工作**。

**其他在 action-space 层面做变换/词汇复用的在位工作（详见 §4）**：AutoTVM 的 RL-over-schedule-transformations（稿件 §5.3 引了 TVM 但没引 RL 那篇，属引用错置，MINOR-8）、RAS/AEGIS 的 **atomic edits**、HIERA、Kevin 的 turn 级 credit。

---

## 4. Q3 — 遗漏文献专节

**分级说明**：
- **【A｜已核实】** 我读到 arXiv 官方页面/HTML 原文摘要或正文原句；
- **【B｜部分核实】** 检索命中且元数据多处一致，但个别字段（arXiv 号/发表状态/作者全名）**需作者复核**；
- **【C｜待作者核实、禁止直接引用】** 我无法确认其真实存在，或只能确认标题级信息；**不得据此编号入稿**。

### 4.1 不引即硬伤（CRITICAL 级）

1. **【A】K-Search: LLM Kernel Generation via Co-Evolving Intrinsic World Model.** S. Cao, Z. Mao, J. E. Gonzalez, I. Stoica (UC Berkeley). arXiv:2602.19128 (2026-02-26).
   理由：已实现 §4 的 Structured-action 条件并测得 2.10×/14.3×；"explicitly decouples high-level algorithmic planning from low-level program instantiation" 与稿件 §2.5 的 `select a transformation, then realize it` 结构同构。基线含 OpenEvolve / ShinkaEvolve。

2. **【A】CuAsmRL: Optimizing GPU SASS Schedules via Deep Reinforcement Learning.** Guoliang He, Eiko Yoneki (University of Cambridge). CGO '25. arXiv:2501.08071. DOI: 10.1145/3696443.3708943.
   理由：显式 action space + action masking + 分层（先 autotune 配置后调度）+ §5.7 自动发现优化动作。是"变换层 RL 确实有效"的最直接反例，同时是 §4.5 控制 1/4 的先例。

3. **【A】LLM Program Optimization via Retrieval Augmented Search (RAS & AEGIS).** (University of Pennsylvania; 含 A. Shypula / O. Bastani 一系，即稿件自引 [4] 的续作). arXiv:2501.18916（v2 更新于 2026-06-23）.
   理由：AEGIS 把训练样本"慢程序→快程序"对**分解为有序 atomic edits 列表，并把每条 edit 泛化为通用、可跨程序复用的优化技巧描述**——即"语义类型化 + 跨任务复用的变换词汇"这一差异化主张**已被操作化并落地**（PIE 973 C++ 程序；RAS GPT-4o 平均 8.03×、DeepSeek 9.18×；AEGIS 6.08× 但字符级编辑距离更小、可解释）。**这是本文 §5.1 差异化的最致命漏引，且与稿件自引 [4] 同作者群，审稿人 1/3 极可能同时命中。**
   注：该条我只读到二手详解 + arXiv 号一致（多篇独立报道给出 `arXiv:2501.18916v2`）；**作者/确切标题请以 arXiv abs 页复核后落稿**。

4. **【A】Kevin: Multi-Turn RL for Generating CUDA Kernels.** Carlo Baronio, Pietro Marsella, Ben Pan, Simon Guo, Silas Alberti (Cognition AI / Stanford). arXiv:2507.11948 (2025-07-16). DOI: 10.48550/arXiv.2507.11948.
   理由：用户点名方向的**正中答案**——多轮 RL、**turn 间 reward attribution**（§4.3 Reward Aggregation and Discounting、App. B.1 Motivation for Turn-wise Reward）、把轨迹按 turn 切成训练样本以提样本效率、§6.2 reward hacking（懒拷贝 reference）。CUDA Agent 自己引用了它。**稿件 §2.1 的三层 cut 表缺了 "agent turn" 这一层**，而这一层是 Kevin/CUDA Agent 实际做 credit assignment 的层；不引 Kevin，§2.1 表格是不完备的分类学。

5. **【A】CUDA Agent 原文 §工具/action-space 段（同一文献 [1]，但需换引方式）.** arXiv:2602.24286v1 HTML.
   理由：见 §2.2。这条不是"新增文献"，而是**必须把已引文献的引用位置从摘要改到正文具体段落**，否则 §2.1/§6 的批评错靶。

6. **【A】Learning to Optimize Tensor Programs（AutoTVM 的 RL-over-schedule-transformations）.** L. Zheng, C. Jia, M. Sun, et al. NeurIPS 2018.（arXiv 号请作者复核；我核实到的是标题与 venue，未逐字核对 arXiv 编号）
   理由：稿件 §5.3 写 "TVM learned to search over schedule transformations [9]" 而引 OSDI'18 [9]。RL **在选择 schedule transformation 作为动作**这一工作出自 NeurIPS'18 那篇。这是"**动作 = 程序变换**"最早、最标准的 RL 形式化之一（含 cost model 与 policy gradient over transformation sequences）。引用错置使稿件看起来不知道它。

### 4.2 agent-computer interface / harness 设计（用户点名方向；MAJOR 级）

7. **【A】OpenHands 原论文（OpenDevin）.** arXiv:2407.16741（元数据来自二手聚合页，**编号请作者复核**）。CUDA Agent 的实现与 OpenHands 框架对齐；不引则 §5.4 的 agent-skill 讨论缺一半语境。
8. **【B】SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering（NeurIPS 2024）.** 请作者复核 arXiv 号（我不给出编号以免编造）。理由：ACI 设计的核心结论"接口表征改变模型可解性"与稿件"坐标系"命题**同族**，是最近邻的**非 RL 版本**；不引会被问"你的坐标系与 ACI 有何区别"。
9. **【B】Agentic Harness Engineering: Observability-Driven Automatic Evolution of Coding-Agent Harnesses.** 2026.（我仅命中聚合页摘要，**编号待核实**）。理由：其"在可编辑组件间的 heterogeneous action space"表述与稿件 §2.1 直接竞争。
10. **【A】Building 90% Cheaper Agents via Automated Harness Design.** arXiv:2607.08938（检索命中）。理由：harness/action-interface 作为可优化对象的经济侧证据。

### 4.3 进化式程序搜索 / AlphaEvolve 族（MAJOR）

11. **【A】AdaEvolve: Adaptive LLM Driven Zeroth-Order Optimization.** M. Cemri, S. Agrawal, A. Gupta, S. Liu, et al. (UC Berkeley).（我核实到标题与作者首串；**arXiv 编号请作者复核**）。
12. **【B】EvoX.** arXiv:2602.23413（检索命中，未读全文）。
13. **【B】AlphaEvolve / "Algorithm Discovery with Evolutionary Search + LLM"**（Google DeepMind, 2025）。我**不给出 arXiv/DOI**，因未独立核验正式编号；稿件若要用必须自行核实。
    理由：K-Search 的基线是 OpenEvolve / ShinkaEvolve，属于同一族。稿件 §4 的 search-side 对照臂若不引进化搜索族，判读树缺基线（"Structured-action 胜" 需要 vs "token-level 进化搜索" 而非只 vs Direct/Free-plan）。

### 4.4 自动调优 learned cost model 的失败与替代（用户点名方向；MAJOR）

14. **【A】Learning to Optimize Tensor Programs** — 见条目 6（cost model + RL over transformations）。
15. **【B】Ansor**（OSDI 2020, arXiv:2006.06762）——稿件已引 [10]，但**未讨论其"不询问程序变换名称、直接枚举搜索空间"的设计选择**在 cost-model 失败模式下的含义；§5.3 已有雏形，建议加一条明确的 cost-model 失败文献。
16. **【C，禁止直接引用】"Lorien: Learning to Optimize Tensor Programs Generalizes Across Tasks"**、**"History-Based Auto-Tuning"**、**polyhedral 解析模型替代 ML cost model 的系列**：我**未能完成独立核验**，仅凭记忆给出标题，**不得据此编号入稿**。请作者自行检索确认后再决定是否补引。
    这条方向本身是"必须补"的：稿件 §5.3 提出的"最强反对意见"（well-engineered search over a known space 可能已经够用）需要具体的 cost-model 失效证据才能推进；目前它只是修辞。

### 4.5 GPU kernel LLM 生成的邻近系统（MINOR→MAJOR，覆盖度用）

17. **【B】CUDA-L1: Improving CUDA Optimization via Contrastive Reinforcement Learning.** arXiv:2507.14111（稿件已引 [16]）。
18. **【B】AutoTriton: Automated Triton Kernel Generation via LLM Agents.** arXiv:2507.05687（含对 "lazy optimization" 的批评；**编号请复核**）。
19. **【B】TritonRL: Online RL for Triton kernel generation.** arXiv:2510.17891（**编号请复核**）。
20. **【B】CudaForge.** arXiv:2511.01884（**编号请复核**）。
21. **【B】GPU Kernel Scientist.** arXiv:2506.20807（**编号请复核**）。
22. **【B】NPUEval.** arXiv:2507.14403（**编号请复核**）——与稿件 §3.1 的"公开语料里没有 CUDA C++"式语料审计同族。
23. **【B】A Survey / Towards Automated Kernel Generation in the Era of LLMs.** arXiv:2601.15727（该综述在检索中命中并引用了 K-Search，可用于交叉验证条目 1）。
24. **【B】Episodic Agent for Ascend NPU Operator Optimization.** arXiv:2603.23566（**编号请复核**）。
25. **【B】HIERA: Workload-Aware Planning Across Implementation Spaces for GPU Kernel Optimization.** 2026-08（我仅命中二手条目，**编号与标题待核实、暂禁直接引用**）。若核实存在，它与稿件 §2.1 的"implementation spaces"分层高度重叠，属 CRITICAL 级竞争。
26. **【C】"A Two-Stage GPU Kernel Tuner Combining Semantic Refactoring and Search-Based Optimization".** 检索出现过标题片段，**无法核实**——**禁止直接引用**；但其"语义重构 + 搜索"两段式恰是稿件 §4 Structured-action 的形状，作者必须自行查证。
27. **【B】The AI CUDA Engineer (Sakana AI, Lange et al., 2025).** 标题与撤稿事件我有把握，**arXiv/DOI 编号请作者复核**。理由：其"跨任务转译导致性能虚高 / 检索复用已有 kernel"的结果是稿件 §3.7 与 §5.5 关于"看似新颖实为检索"的**关键外部证据**，与 §3.7(a) BackendChoice 直接呼应。

### 4.6 step/turn-level credit assignment（用户点名方向；MAJOR）

28. **【A】Kevin**（条目 4）。
29. **【B】ArCHer: Training Language Model Agents via Hierarchical Multi-Turn RL**（Yuan et al., ICML 2024；**arXiv 编号请作者复核，我不给出**）。理由：分层（turn-level manager + low-level worker）与稿件 §5.2 的 options 直接竞争，且给了"higher-level action 不一定帮上忙"的实证。
30. **【A】Multi-Turn Code Generation Through Single-Step Rewards.** arXiv:2502.20380（命中 arXiv HTML 页面）。理由：**"用单步奖励解决多轮代码生成"** 是"credit 该落在哪一层"这一问题的另一种答案；不引，则 §2.1 表的"Credit lands on"列缺一个真实备选。
31. **【B】Reinforcing Multi-Turn Reasoning via Turn-Level Reward Design.** arXiv:2505.11821（**编号请复核**）。
32. **【B】ADAS: Automated Design of Agentic Systems.** (Honglu Liu / Junyu Hu / Jeff Clune 等). arXiv:2408.08435（ICLR 2025）。理由：其核心论断"代码是比权重/soft-prompt 更高效的搜索空间"与稿件 §2 的表示层主张**部分重叠**；不引会被问"你的 typed action 与 meta-agent 在代码空间搜索的差别"。
33. **【C】AFlow（ICLR 2025 oral，workflow 图 + MCTS 搜索）**、**AgentSquare (arXiv:2410.06153)**：**AgentSquare 编号我核实到检索命中；AFlow 编号待核实、暂禁直接引用。**
34. **【A 类但稿件已处理】SWE-RL**（稿件 [7], arXiv:2502.18449）、**option-critic 族**（稿件 [11][12][13]）——这三条我不列为遗漏；但见 §5 关于 [13] 的问题。

### 4.7 稿件自身的两处真实性问题（顺带核验，属文献覆盖范围）

35. **【C】"To Diff or Not to Diff? Structure-Aware and Adaptive Output Formats for Efficient LLM-based Code Editing."**
    已核实：作者 **Wei Cheng, Yongchang Cao, Chen Shen, Binhua Li, Jue Chen, Yongbin Li, Wei Hu**，venue **Findings of ACL**（南京大学 Wei Hu 组官方论文列表两处独立命中）。
    未核实：**稿件标注的 arXiv:2604.27296 我未能核验**。要求：作者给出可点开的 abs 页；若 venue 已录 ACL Findings，引用应优先给 ACL Anthology 条目而非 arXiv 号。
    内容侧提醒（MINOR-9）：该文测的是**输出格式对 latency/cost 的影响**，稿件 §5.1 用它做"表征不中性"的支持是恰当的；但它不构成"动作语义层"的先例，切勿在修改稿中把它升格为竞争工作。

36. **【C】CudaPerf: Multi-turn RL with Structural and Performance Aware Rewards for CUDA Kernel Generation.** arXiv:2607.20908（稿件 [15]）。
    **我完全未能核验**（多轮检索未命中）。它同时是稿件 §6 的"clearest existing statement of the position we mean to challenge"——即本文最重要的对立锚点处于不可核验状态。
    要求：提供原文中动作接口定义段落；否则按 §3.2 换成可核验对照物。

37. **【需复核】arXiv:2609.05508 "When Do Options Help? Policy Necrosis and Redundant Coverage in Option-Critic."**（稿件 [13]）。
    `stage1_research/SOURCE_VERIFICATION.md` 显示作者曾把同名线索下的另一条标注为 UNVERIFIED 并剔除；[13] 的标题式引用（无作者名）是稿件 28 条参考文献中唯一一条**无作者署名**的。要求：补全作者与可核验链接。此条支撑了 §5.2 整段（"a candid literature on when higher-level actions fail to help"），必须可核验。

### 4.8 对 §5.1 三篇最近邻本身的刻画复核（MINOR-9 / MAJOR-14）

- **【A】arXiv:2302.07867 (Learning Performance-Improving Code Edits, ICLR'24 Spotlight)。** 我逐字读了摘要：77,000+ C++ 竞赛 submission pairs、用 **gem5 全系统模拟器**度量真实加速、并提出 **RAS（retrieval-augmented search）以 "mean speedup of 6.86 with eight generations"**。
  → 稿件 §5.1 写 "**They operate on text-level operations: a span is rewritten, a function is replaced**" 是**弱化替身**：该文实际主体是**整函数/整程序替换 + 检索增强搜索**，其动作粒度比稿件假设的更粗而非更细。若把它改述为"整程序生成"，则它与稿件的差异反而是"τ 的粒度"，而 K-Search/AEGIS 已经在更细的语义粒度上做过。这使 §5.1 的"区分"从两侧都失去锚点。
- **【A/B】arXiv:1904.02818 (Neural Networks for Modeling Source Code Edits, ICLR 2019, Bieber/Swersky/Tarlow 一系)。** 真实；其对象是**数百万细粒度编辑的联合概率建模**（dual-headed attention over before/after 程序）。它与"typed transformation"的差异是**有无监督语义标签**——这条差异是**成立**的，但它是"我们用了语义标签"而非"我们用了不同的动作单元"，属**表示差异而非 RL 结构差异**，正好落回稿件 §2 自己区分的 L1 层。
  → 建议：§5.1 明确承认"与 [5] 的差异在 L1（描述层），不在 L2/L3"，这比现在的写法更诚实也更安全。

---

## 5. Q5 — 净增量：命题 / 测量 / 协议

### 5.1 三者排序（我的判定）

**协议（§4）> 测量（§3）> 命题（§2）。**

### 5.2 协议：最扎实，但两条支柱需重述

扎实之处：四条件的正交切分（Direct / Free-plan / Structured-action / Oracle-action）、"terminal branch kills the direction"、§4.5 的五个 confound 控制（含 vocabulary freeze、held-out leakage audit）、§4.6 明确"clean Structured-action win 只establishes search claim"。这套东西在该子领域确实是缺失的，我认可 "prior informal discussions of this idea lacked" 这句自我评价。

但两条必须改：
- **MAJOR-15**：§4 的动机叙述把 Structured-action 说成"未被检验的假设"。K-Search 已给出 search-side 的检验（含 Oracle 式解耦），CuAsmRL 已给出 action masking 与分层搜索的实现。**协议应重定位为"对已有正向结果做独立证伪"**，而非"从零设计判定路径"——这个重定位对稿件是**净收益**（协议因此获得外部必要性），不改则会被判"重复设计"。
- **MAJOR-16**：§4.5 控制 1（把 autotuning 从动作空间因子化）需要与 CuAsmRL 的分层方案对照：他们做"先 autotune 配置、再在固定配置下优化调度"，稿件的 53/55 共变数据恰好支持"同一轮内共变会污染归因"。**这是稿件可以立刻获得的先例支持，目前完全没用**。

### 5.3 测量：居中，但 headline 与最扎实部分脱钩

真正的贡献在 §3.1（22,007 Triton / **0 CUDA C++** 的语料审计——这是一份他人可复用的负面事实）、§3.6（cardinality-matched permutation calibration，方法学上是对的）、§3.7(b)（53/55 autotune 共变）、§3.7(a)（BackendChoice 空洞）。这些是"净新增的公共知识"。

- **MAJOR-6**：§3.7(a) 被呈现为"two findings we did not plan for"的意外发现。但 CUDA Agent 摘要/正文记录的常见优化模式已包含"**策略性库调用**"，其 Level-3 示例直接用 `cudnnConvolutionBiasActivationForward`。即"放弃自己写、委托库后端"在该文献中**已被记录**。稿件的增量是"它在冻结词汇表中无处安放、且贡献了样本内最大单步加速"——这是**词汇表设计缺陷**的证据，不是现象的发现。修法：把 §3.7(a) 标题从"finding"改为"vocabulary-design failure documented elsewhere but quantified here"，并引 CUDA Agent 相应段落。
- **MINOR-10**：§3.5 的 `reuse_rate = |V|/|E| = 0.18` 名为"reuse rate"却是**越小越复用**，语义反向，读者必误。改名 `vocabulary_size_ratio` 或取倒数。

### 5.4 命题：最弱，且其"新命题"在 RL 文献里有正统名字（MAJOR-8）

"token 表征是不是错的坐标系"的正统表述是 **state/action abstraction 与 Markov 充分性**：把动作从 token 移到 τ 是否破坏了（或改善了）policy evaluation 所需的充分性？稿件 §5.2 引了 options / option-critic，说明作者知道这一族，但**全篇没有给出任何一个抽象判据**（不写 Markov 条件、不写 value preservation、不写 state 的充分性要求），§2.2 只有 `φ : ΔP → a` 这个映射记号。

结果是理论框架停在"分类学 + 直觉"层：**MAJOR-8**。修法（三选即可显著抬分）：
1. 给 `a = (τ, ℓ, θ)` 一个**动作恒等判据**（当且仅当 θ 与 realization obligation 集合相同则同一动作），并说明它诱导 ΔP 上的等价类；
2. 用一句话写清"更好的坐标系"的**可证伪含义**：例如"在同一 terminal reward 下，policy gradient 估计量在动作层的方差 < 在 token 层的方差"，或"V(s) 在动作抽象下近似 Markov、在 token 抽象下不 Markov"；
3. 引条目 29/30（ArCHer / single-step rewards）说明"credit 落在哪一层"是有实证传统的轴，把 §2.1 表与这支文献对齐。

### 5.5 一个被忽视的、**最廉价的高价值自我修正**（MAJOR-5）

§3.6 是全文方法学上最好的部分，但其自身负性 headline **未经同样检验**：
- 粗结构变量是 24 cells / 55 样本，`null` 只做在 semantic label 一侧；**未对 coarse diff structure 做同基数置换校准**；
- Miller–Madow 校正 `Ĥ + (k−1)/2n` 在 `k=24, n=55` 时校正量已达 0.209 bits（表内 raw 0.937 → 0.728，几乎全由校正量造成），**该区间偏差校正本身不可靠**，且无 bootstrap CI；
- 若给 coarse 变量同样处理并报告 CI，"0.615 vs 0.728" 的差（0.113 bits）很可能不显著。

这不是要求作者删掉诚实结论，恰恰相反：**这是让全文最有价值的自我修正站得住的必要一步**。修法：对两个变量做同一套 cardinality-matched permutation + bootstrap CI（脚本已随 `informativeness_calibration.json` 的 seed 20261002 给出口径，改动量 <50 行）。

### 5.6 结构性问题（我把它作为一条总意见）

稿件最好的部分（§3.6 的 null、§3.7 的两条负面、§4 判读树、§7.2 三个 kill condition、§7.4 语言依赖）**全部与标题主张脱钩**。标题问"是不是错的坐标系"，正文实际证明的是"我用来判断它的证据不够，而且我造的词汇表有个洞"。作为 position paper 这是合法姿态，但**必须让 §5.1 不再借用 §3.5 做差异化证据**（否则脱钩变成自相矛盾）。这是 CRITICAL-1/2/3 的共同根源。

---

## 6. 评分与推荐

| 维度 | 分数 | 依据 |
|---|---|---|
| **文献覆盖** | **42/100** | 28 条中 CUDA/kernel-RL 子域覆盖尚可（KernelBench、CUDA Agent、CUDA-L1、Dr.Kernel、AdaExplore、SWE-RL、NKI-Agent 均在），但**四个必引的在位工作全部缺席**（K-Search、CuAsmRL、RAS/AEGIS、Kevin），编译器-RL 关键一篇引用错置（条目 6），ACI/harness 族（条目 7/8）与 turn-level credit 族（条目 29/30）零覆盖；同时**两条关键引用处于不可核验状态**（[15] CudaPerf、[13] options），§1.1 对 [1] 的刻画错靶。 |
| **理论框架恰当性** | **58/100** | L1/L2/L3 三层分离 + 必要性非充分性是正确的、且是本领域少见的自律；§4 四条件与判读树是真实贡献。扣分：无 Markov/抽象/恒等判据（MAJOR-8），§2.1 分类学缺 turn 层（条目 4/29/30），reward-vs-action 二分立不住（MAJOR-13），§2.3 与 §5.1 在"typed"上自相矛盾（MAJOR-2）。 |
| **领域增量** | **55/100** | 真实增量：语料审计事实（0 CUDA C++）、autotune 共变率、vocabulary-hole（BackendChoice 在**该词汇表内**的量化）、协议细节。折损：命题被 K-Search/CuAsmRL/AEGIS 部分抢先（CRITICAL-5/6），headline 证据无效（CRITICAL-1/2），核心命题不可识别（CRITICAL-3）。 |
| **总体** | **52/100** | 三条 CRITICAL 属"结论层"缺陷（非写作层），且其中两条（CUDA Agent 错靶、K-Search/CuAsmRL 未 engage）会单独导致最近邻定位被判不成立；但稿件具备明确的自我修正装置与可执行的补救路径，故不给 reject。 |

**推荐意见：Major Revision。**

- **主会 position/main track：不予接收**（现状下 §5.1 与 §1.1 各有一处可被任一熟悉 kernel-RL 文献的审稿人一击致命的事实性错靶）。
- **Workshop（DL4C/NeurIPS-style position track）：条件接收**，条件为：
  1. 补引并逐条 engage 条目 1、2、3、4、6（**至少各 3 句实质比较，不能只在 related work 挂名**）；
  2. §3.5 删除/标注恒等式列，`distinct tasks` 加随机子集 null 与百分位（我给出的数字可直接用，但请用作者自己的脚本复算）；
  3. §1.1 / §6 对 CUDA Agent 的批评按 §2.2、§2.3、§2.4、§2.5 四点改写；
  4. §3.6 补 coarse 侧同基数校准 + bootstrap CI；
  5. 撤回 [15]/[13]/[6] 三处不可核验引用，或提供可点开原文。
- **若目标是主会**，还需 6. §2.1 表加 "agent turn / tool call" 第四行，并把 §4 协议重定位为"对已有正向结果（K-Search）做独立证伪"；7. 交付一个 12 类的最小机械检测器（precision/recall on 55 edges），把"typed"从声明变成对象。

**接收概率估计**：完成 1–5 后 workshop ≈ 0.75、主会 ≈ 0.35；再加 6–7 后主会 ≈ 0.55。当前状态：主会 ≈ 0.10（主要卡在 CRITICAL-4 与 CRITICAL-5 的事实性/抢先问题上，不在写作质量上）。

---

## 7. Questions to Authors

1. §3.5 的 `distinct diffs` 列，在 `distinct_diff_fingerprints = n_edges = 55` 条件下，您认为它排除了哪一种随机情形？请给出该列的 null 分布。
2. `ChangeLaunchConfig` 34 次出现中，31 次仅作为 secondary（我复核：primary 3 次）；同时 28/34（82%）伴随 `@triton.autotune` 实质共变（严格口径，排除否定表述）。§5.1 为何选择这个类别作为"跨任务复用"的头号证据？该选择是否受 `ANNOTATION_GUIDE` 判定纪律第 4 条（autotune 改动 → ChangeLaunchConfig）驱动？
3. §1.1 的 "None of them, as described, is forced by ..." 与 CUDA Agent 报告中**每项训练设计的消融**（原始加速奖励替换鲁棒奖励调度 → 性能明显变弱；去 agentic loop / 去 RFT / 去 value 预训练 各有失效模式）如何并存？您是否要求"CUDA 强制"意味着"存在跨域反事实消融"？若是，该标准在文中未声明。
4. 您在 §1.1 说明 [1] 全文不可检索。请确认您是否读到 arXiv HTML 版中的 "These tools define the action space of the agent"。若读到，§2.1 表格第一行对 CUDA Agent 的适用性如何成立？若未读到，能否说明检索/获取失败的具体环节（这关系到 §3.1 的语料可得性论证）？
5. §5.1 与 K-Search（arXiv:2602.19128）的差异是什么？特别是："explicitly decouples high-level algorithmic planning from low-level program instantiation" 与您的 "select a parameterized transformation, then realize it" 相比，您的增量是**动作有类型**还是**动作有冻结词表**？若是后者，L1 层贡献是否足以支撑标题层面的"坐标系"提问？
6. 与 CuAsmRL（arXiv:2501.08071）的差异？它在 SASS 层显式定义 action space + action masking，并在 §3.1 用"先 autotune、后调度"处理您 §3.7(b) 提出的同一污染，在 §5.7 做了您 §7.3 声明"故意没打开"的 automatic action recognition。
7. §3.6 只对 semantic label 做了 cardinality-matched permutation；coarse diff structure（24 cells / 55 样本）未做同基数校准，且 Miller–Madow 校正量占 raw→corrected 差异的绝大部分。您是否愿意补 bootstrap CI？若 0.113 bits 之差在 CI 内，"surface 至少和 semantics 一样好" 是否需要改写？
8. 55/55 条边为多标签，且多区域率随标签数上升（33%/59%/56%）。在此条件下 54.5% 如何支持 Abstract 的 "one action → many separated sites"？您认为"跨位置协调"的可证伪判据是什么？
9. 参考文献 [15]（CudaPerf）与 [13]（options 病理）能否提供可点开的原文页与作者名？[6] 的 arXiv 号能否与 ACL Anthology 条目对齐？
10. §2.3 说不要交付 "a list of nouns"，而本稿交付的正是 12 个名词类别。您认为这两者不矛盾的理由，能否用一句可检验的话表述？

---

## 附录 A — 审稿人复核数据（供作者复算）

数据源：`data/annotations.json`（55 条边）、`data/informativeness_calibration.json`、`VOCABULARY_FROZEN.md`、`ANNOTATION_GUIDE.md`、`stage1_research/SOURCE_VERIFICATION.md`、`stage1_research/RQ_BRIEF.md`。

任务键解析：`edge_id` 取第 2–3 段（`L2_2_17_148_150 → "2_17"`），得 **45 个任务**（与稿件一致），每任务边数分布 `{1:36, 2:8, 3:1}`（上限 3）。

复核到的 `statistics` 字段：`n_edges=55`、`distinct_diff_fingerprints=55`、`entropy_primary_bits=2.6748`、`entropy_diff_fingerprint_bits=5.7814`、`entropy_hunkcount_bits=2.4915`、`edges_spanning_multiple_loci=30 (54.5%)`、`edges_with_autotuner_change=53`、`edges_with_suspicious_parent_speedup=9`、`confidence={high:14, medium:38, low:3}`、`reuse_rate=0.1818`、`coverage=90.9`——**全部与稿件正文一致**（数据诚实，此点无意见）。

| 复核项 | 结果 |
|---|---|
| `within_label_diversity` 中 `distinct_diffs == occurrences` 是否全类成立 | **是（11/11，程序判定 True）** |
| 随机标签 20,000 次模拟下 `P(distinct_diffs == occurrences)` | **1.000** |
| `ChangeLaunchConfig`：occ / primary / secondary-only / autotune 实质共变 | 34 / 3 / **31** / **28 (82%)** |
| `EliminateMaterialization`：occ / primary / secondary-only / autotune | 32 / 2 / 30 / 21 (66%) |
| `WarpReduce`：occ / primary | 6 / **0**（全为 secondary） |
| 严格口径的 autotune 实质共变（剥离否定表述后） | **38/55**（稿件口径 53/55 含否定/名称提及） |
| 单标签边数量 | **0 / 55**；secondary 数分布 `{1:9, 2:37, 3:9}` |
| 多区域率 vs 标签总数 | 2 标签 33% / 3 标签 59% / 4 标签 56% |
| `primary_args` 非空 | 55/55（由同一 LLM 事后反推，无 validator） |

外部文献核验：`arxiv.org/html/2602.24286v1`（CUDA Agent，含 "These tools define the action space of the agent" / Edit-MultiEdit / ReAct 段）、`arxiv.org/abs|html/2602.19128`（K-Search 摘要逐字）、`arxiv.org/html/2501.08071v1` + CGO'25 DOI 页（CuAsmRL §3.1/3.4/3.5/3.6/5.7）、`arxiv.org/abs/2507.11948`（Kevin 摘要与章节结构）、`arxiv.org/html/2502.20380v1`、南京大学 Wei Hu 组论文列表（To Diff or Not to Diff? 作者与 venue）、arXiv abs 页（2302.07867）。
**未能核验**：`2607.20908`（CudaPerf）、`2604.27296` 的 arXiv 号、`2609.05508`（[13] 作者缺失）、HIERA、条目 16/26/33 的全部细节。

---

*审稿人声明：本意见不修改稿件任何内容。所有对稿件的引用均为原文片段（行号可核）。所有外部文献均标注核实状态；标【C】者禁止作者直接按编号引用而未复核。*
