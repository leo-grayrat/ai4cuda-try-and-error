# Review — Journal-Fit Reviewer (领域编辑视角)

**稿件**：*Is Token-Level the Wrong Coordinate System? Program Transformation as the Action Space for LLM-Driven Kernel Optimization*
**体裁**：Position / Analysis paper
**目标场合**：arXiv preprint + MLSys / ICML / NeurIPS workshop position track
**评审角色**：Journal-Fit Reviewer（独立、只读、不看其他审稿人意见、不改稿）
**评审范围**：① 期刊/会议契合度 ② 原创性与重要性的双重判断 ③ "so what" 测试 ④ 篇幅与体裁纪律 ⑤ 标题/主张强度 vs 证据强度
（方法学细节 / 文献覆盖 / 跨学科适配分别交给 R1 / R2 / R3，本文只在触及"契合度与主张强度"时引用。）

---

## 0. 证据核对（evidence audit，先声明我看了什么）

审稿前我逐项核对了论文 Section 3 的核心数字是否**真实存在于仓库**，结论：**存在且内部自洽**，不是凭空数字。

| 论文主张 | 来源 | 仓库核对 |
|---|---|---|
| `H(primary label)=2.67` / `H(diff fingerprint)=5.78` | Abstract / 3.5 | `annotations.json:16,20` → 2.6748 / 5.7814 ✓ |
| 10/12 类别覆盖 90.9% | Abstract / 3.5 | `annotations.json:14-15` reuse 0.1818 / coverage 90.9 ✓ |
| 54.5% 边跨 ≥2 分离区域 | Abstract / 3.5 | `annotations.json:54-55` → 30 / 54.5 ✓ |
| 语义标签 0.615 bit vs 粗结构 0.728 bit | Abstract / 3.6 | `informativeness_calibration.json:10,18` ✓ |
| 随机划分 98.8 百分位 / 20,000 shuffle / seed 20261002 | 3.6 | `informativeness_calibration.json:11,26-27` ✓ |
| 53/55 边共改 `@triton.autotune` | 3.7(b) | `annotations.json:61` → 53 ✓ |
| 全部 5 条 `Other` = 后端切换；含 327x / 69x | 3.7(a) | `annotations.json` 边 `L2_2_55_36_37`(69.4x) / `L2_2_56_4_9`(327.2x) primary=`Other`、描述均为 "Backend swap … cuBLAS" ✓ |

复现脚本 `scripts/select_edges.py`、`analyze_entropy.py`、`calibrate_informativeness.py` 与冻结词表 `VOCABULARY_FROZEN.md` 均在仓库。**因此本评审不质疑数据真实性；质疑的是数据的"可解释范围"与稿件的体裁/分量匹配。**

---

## 1. 总体判断与推荐

这是一篇**异常诚实、可复现、但体裁身份漂移**的稿件。它的实证节（Section 3）自己证明了"支撑论题的证据不足以支撑论题"（3.6/3.8），于是全稿唯一的正向可交付物变成了一份**尚未执行的实验协议**（Section 4）＋**两个非预期的负向/警示性发现**（3.7）。对 workshop position track 而言，这个组合是**可发表的**；但它现在以"coordinate-system 论文"的姿态出现，而 coordinate-system 命题恰恰是作者自己承认无法用现有证据判定。名实之间存在系统性错位。

- **推荐（workshop position track）**：**Major Revision**（可救，方向明确）。
- **推荐（主会主轨）**：**Reject / 转 workshop**（无干预实验＋循环自标注，rigor 天花板锁死）。
- **总分（0–100，分维度）**：见第 6 节。overall = **57**。

一句话：诚实是本稿的必要条件而非贡献；把"我很诚实"当成卖点，掩盖不了"我提出的命题我没能证成，也还没去证伪"这一事实。真正值得保护、且能让读者改变行为的，是 3.7 的两条负向发现和 4.3 的自杀式 reading tree——**建议把论文的重心从"coordinate-system 命题"迁移到"用一次自我证伪换来的三条方法论禁令"上**，名实即可对齐。

---

## 2. 逐维度评估

### 2.1 契合度（Fit）

**主题契合、体裁不纯。** "token 是不是选错坐标系"对 MLSys/ICML 的 kernel-RL 与 code-RL 读者是**恰当且当下热门**的问题（CUDA Agent、KernelBench、AdaExplore、CudaPerf 都是近作），arXiv 预印本毫无门槛。问题在体裁：

- 它**不是**纯粹的 position paper：Section 3 有真实的测量研究（语料审计、熵、置换校准），篇幅与细节量像实验报告。
- 它也**不是**实验报告：3.6 把测量结论自我作废，Section 4 的协议一个都没跑，7.5 明说"No search experiments. No training. No measured improvement in any metric."
- 它是 **position + 未跑完的实验 + 预注册协议** 的混合体，三种身份各占约三分之一，且互相拆台（3.5 建立 → 3.6 拆台 → 4 悬置）。

对 position track，委员会要的是一个**可被领域采纳的立场**。本稿的可采纳立场其实存在，但**不在标题处**——它在 3.7 与 4.5。见 FIT-1。

### 2.2 原创性 × 重要性（双重判断）

**命题状态：既未被社区默认接受，也未被默认否定——是"真开放"，这点对 position paper 有利。** 论文用 CudaPerf（把 CUDA 结构放进 *reward*）vs 本文（放进 *action*）的对照（Section 6）清楚划出了未决地带，这是**恰当的定位**。

但"新"要分两层：

- **framing 层不新**：作者自认。1.4 "compiler people have known for decades … we would have rediscovered IR"；5.3 "the idea is not new, and any novelty claim must rest on the LLM interface." "transformation > token" 在编译器（Halide/TVM/Ansor）和分层 RL（options）里都是旧知识，`a=(τ,ℓ,θ)` 自己就是一个带参数的 option（5.2 承认）。
- **观察层有新**：三处是本稿真正的原创贡献，且有数据支撑——(i) 基数匹配校准显示**语义标签在预测收益上不优于粗表面结构**（3.6，0.615 vs 0.728）；(ii) **内省构造的动作空间会漏掉收益最高的动作**（3.7(a) 全部 5 条 Other = 后端选择）；(iii) **真实语料里结构变换与超参搜索经验上不可分**（3.7(b)，53/55）。这三条是可复现、可引用、可证伪的。

**重要性上限被"so what"卡住**（见 2.3）：因为命题被作者悬置，读者即便全盘接受，也不会因此改用 transformation 接口——只会更谨慎。谨慎是有价值的，但对 high-impact 场合分量偏轻。

### 2.3 "so what" 测试（若社区完全接受本文结论，行为如何改变？）

我要求"具体、可执行、可观察的行为改变"，逐条给结论：

**会改变的行为（真实存在，但窄）：**
1. 任何做 code-RL 动作级 credit assignment 的人，**必须先证明语义标签击败粗结构，再谈抽象价值**（3.6 的教训）。→ 改变评审/自检标准。
2. 任何人枚举代码动作空间，**必须把 BackendChoice（"要不要自己实现"）列为一等公民**（3.7(a)）。→ 改变动作空间设计清单。
3. 任何人基于 Triton/kernel 轨迹做归因，**必须先把 autotune 从动作维度里剥离**（3.7(b) + 4.5 控制 1）。→ 改变实验对照设计。

**不会改变的行为（关键）：**
- coordinate-system 命题本身 → 无行为改变，因为论文说它"decidable only by intervention"（Abstract / 4.6 / 7.5），而干预还没做。
- 现有系统（CUDA Agent / CudaPerf）不会因本文改动接口——本文未证成任何接口更优。

**"so what" 裁决**：如果只接受本文，学界会变得更**谨慎**（三条方法论禁令），但不会变得更**靠近正确答案**（命题仍未决）。所以本文的净贡献是 **"禁令 + 协议"型**，而非 **"结论 + 方法"型**。直说：**它的 so-what 是负向/防守性的，不是建构性的**——这决定了它的分量天花板 ≈ workshop，而非主会。由此得出第 5 节的**重心迁移建议**：把最能改变行为的 3.7 提到主张层，把尚待验证的 coordinate-system 降为"动机/待判问题"。

### 2.4 篇幅与体裁纪律

8,800 词对"8–12 页 workshop position paper"**超标**。按双栏 ~600–700 词/页估，正文（不含图表/参考文献）已 ≈ 13–15 页；含 8 个一级节 + 大量小节 + 28 条参考文献，视觉体量像 full paper。**Section 5 与 Section 6 存在实质性重叠**：Section 6 几乎退化为 Section 5 的指针索引——"Compiler search spaces. Halide [8], TVM [9], Ansor [10]. **Section 5.3**."、"Hierarchical RL … **Section 5.2**."、"Benchmark integrity … **Section 5.5**."、"Agent harnesses and skills … **Section 5.4**."。而 Section 5 自身把 related-work 定位（5.1–5.3）与 broader implications（5.4–5.6）混编。具体砍/并建议见 LEN-1、LEN-3。

### 2.5 标题与主张强度 vs 证据强度

**标题强度是合适的**——"Is Token-Level the Wrong Coordinate System?" 用问句、7.5 声明"The title is deliberately a question"，与"证据不足以断定"匹配。**Abstract 与 3.5 的强度越界**：把在 55/55 唯一指纹下**恒真/分箱产物**的熵比较，当作论据前置（详见 CLAIM-1，本稿最实的一条技术性问题）。作者自己在 3.6 承认"Low entropy is therefore not yet an argument"，但 Abstract 仍以 2.67-vs-5.78 开篇立论。**主张（Abstract）与自我反驳（3.6）之间的错位，是本稿最需要修的一致性问题。**

---

## 3. 发现清单（按严重度，每条含证据锚点 + 可执行修改建议）

### CRITICAL
本稿没有"数据造假/不可复现"级别的致命伤（证据已核对为真、脚本齐备）。以下 MAJOR 中，**CLAIM-1** 与 **SIG-1** 若不在修订中解决，应升级为 CRITICAL，因为它们直接决定"主张是否被自己的证据支撑"。

---

#### [MAJOR] FIT-1 — 体裁身份漂移：三重身份互相拆台
- **锚点**：1.5 "It contains **no training experiments, no new algorithm, and no search experiments**."；3.5 建立压缩证据 → 3.6 自我作废 → 4.6 "that is not measurable observationally"；7.5 "Section 3 is an existence and non-sufficiency result … Section 4 is a protocol."
- **问题**：一篇 position paper 同时是（a）测量报告、（b）自我否定的分析、（c）预注册协议。三者的"贡献归属"不同：position 的功劳应记在"立场"上，而本文立场被自己悬置。委员会读到 3.6 会问"那这篇的 thesis 到底是什么"。
- **可执行建议**：在 Abstract 第 3–4 句**明写主交付物 = 三条可采纳的方法论禁令（3.6/3.7）＋ 一份能杀死本方向的协议（4.3）**，把 coordinate-system 明确标注为"本文主张其可判定、而非已判定"的开放命题。标题保留问句即可，但摘要的落点要从"我们证明了压缩存在"改为"我们证明了什么不足以立论、以及怎样才能立"。

#### [MAJOR] CLAIM-1 — 头条熵证据在唯一指纹前提下恒真，被误当论据
- **锚点**：Abstract "transformation-label entropy is 2.67 bits against a 5.78-bit diff-fingerprint entropy"；3.5 表 `H(diff fingerprint)=5.78 bits — 55 of 55 fingerprints distinct`；3.5 正文 "**No category ever produced the same diff twice**"；5.1 称此为 "the single most concrete piece of evidence this paper offers"。
- **问题（技术内核）**：既然 55 条指纹两两不同，`H(fingerprint)=log₂55=5.781` 恰为**恒等于最大熵的 identity 分箱**；任何把 55 条映射到 <55 类的标签，其熵**必然 ≤** 它——`2.67 < 5.78` 与 "no category repeats a diff" 都是**分箱/命名的直接推论，不提供任何超出"指纹互异"的信息**。作者其实**在 3.6 已承认**"A smaller description length is a property of any coarse summary"，却在 Abstract/3.5/5.1 反把它当前沿论据。这是名实错位的根点，也是标题-证据匹配的核心破口。
- **可执行建议**：
  1. Abstract/3.5/8 **把 2.67-vs-5.78 与 0.615-vs-0.728 并列呈现**（用自我设限的校准数平衡熵差数），不要让"压缩存在"单独出现在结论位。
  2. **删除**把 "distinct diffs / 不重复同一 diff" 当证据的措辞——它在数学上不可证伪。
  3. 若想保留"决策重复、代码不重复"的现象主张，改用**真正非平凡**的量：仓库里已有的 `distinct_tasks`（Fuse 22 / ChangeLaunchConfig 30 跨任务复现＝真信号）与 `distinct_region_sets`（Fuse 25→23、ElimMat 32→29，**小于** occurrences＝非恒真），或补一个**类内 AST/结构距离**指标。5.1 的"最具体证据"应重指到"跨任务复现"，不是"跨 diff 互异"。

#### [MAJOR] SIG-1 — 净正向贡献 ≈ 一份未跑的协议；"so what" 为防守性
- **锚点**：4.6 "Even a clean Structured-action win would establish the *search* claim only … We do not have the infrastructure for that here."；7.5 全节；3.8 "Non-sufficiency"。
- **问题**：论文把"决定性证据"整体外包给 Section 4（未执行）。对 rigor 敏感的委员会会认为核心贡献尚未兑现；对 position 宽容的委员会则看协议质量。天花板由"诚实但悬置"锁死。
- **可执行建议**：把 3.7 两条发现**升格为一等主张**并各自给一句祈使式 takeaway（"assume a BackendChoice action until shown otherwise" / "factor autotuning out before per-action credit"）。这两条**现在就**改变他人行为，比未决命题更有 so-what。协议保留为 Section 4 但**不作为唯一卖点**。

#### [MAJOR] LEN-1 — Section 5 与 Section 6 大面积重叠；6 退化为指针索引
- **锚点**：Section 6 "**Compiler search spaces.** Halide [8], TVM [9], Ansor [10]. **Section 5.3**." 等 5 处 "Section 5.x"；Section 5 混编 5.1–5.3（related work）与 5.4–5.6（implications）。
- **问题**：同一批文献被"先讨论一遍（S5）再索引一遍（S6）"，既占篇幅又制造"有两个 related work 节"的错觉。
- **可执行建议**：**合并为单一 Related Work**，按主题（kernel-RL / edit-RL / 编译器 / 分层RL / 基准完整性 / 技能设计）分块，每块内**就地**给出本文关系句（吸收现 5.1/5.2/5.3 的定位内容）；把 5.4（技能文件合规）与 5.6（训练基底批评）另立为短 § "Design constraints & reframing"，5.5（验证）并入 Section 4 作前置条件。目标：砍掉一整节 + ~600–900 词。

#### [MAJOR] CLAIM-2 / FIT-3 — Abstract 与结论省略样本偏置与循环性限定，读感强于 3.2/7.1 所能支撑
- **锚点**：Abstract "54.5% of edges land on two or more separated code regions" **无**"multi-hunk-enriched biased sample"限定；正相反 3.2 "**This is a biased sample: it can support an existence claim but not a population estimate.**"（采样刻意富集 33/55 多 hunk）。7.1 声明词表与标注同出一个模型家族（循环）。
- **问题**：Abstract 的 54.5% 是**偏样本上的比例**，被呈现为一般事实；54.5%（多区域）恰是最依赖"多 hunk 富集"这一偏置的数字。叠加"AI 标注 AI 词表"，3.5 的可引用性进一步打折。
- **可执行建议**：
  1. Abstract 中 54.5% / 90.9% 各加 4–6 字限定（"on a multi-hunk-enriched sample"），或改述为"跨任务复现 + 多区域落地**可被观察到**"（存在性措辞）。
  2. 明确"本文 Section 3 的标签分布在人工独立标注前**不可被他人引用为总体估计**"——把 7.1 的循环性从限制节**上提**到 3.3/Abstract 的可见位置（现在藏得太深）。

#### [MAJOR] FIT-4 — 对唯一批判标靶 CUDA Agent 的内部不可得，削弱 position 的落点
- **锚点**：1.1 "the full text of [1] is not retrievable … the internals are not released … we flag the unverifiable parts rather than asserting them"；1.1 只能在"摘要层面"论证。
- **问题**：Introduction 的核心批判（"recipe 只是训练基底的产物、不针对 CUDA 语义"）建立在一个**只能读摘要**的系统上；虽然作者诚实标注，但这使 Section 1 的批判更像"对一个重建版对手的批判"。对 fit 的影响：委员会会问"你挑战的具体设计选择，你能确认它们真实存在吗"。
- **可执行建议**：把可确认的锚点前移、承担举证——用**可复现的公开系统**（CudaPerf 的结构化 reward、AdaExplore 的轨迹、NKI-Agent 的 GRPO 负结果 [18]）作 Section 1 的**主**证据，CUDA Agent 退为动机案例，并在 1.1 一句话点明"本文批判的接口选择由 [15]/[18] 的公开设计直接证实，不依赖 [1] 的内部"。

#### [MINOR] LEN-2 — 5.4/5.6 为未量化轶事与不可证伪推测，稀释纪律信号
- **锚点**：5.4 "we have not measured either observations; they are design constraints we adopted"（尤其 "compliance appears to degrade with instruction-file length … we observed this first-hand"）；5.6 "the argument above is therefore a *framing* claim … not an empirical one."
- **问题**：一篇要求 Free-plan 对照、基数匹配校准的稿子，正文却含两处明说"未测量/不可测"的论断，风格上自相矛盾，给审稿人"注水"观感。
- **可执行建议**：5.4 两条观察压成 1 段并入 4.5 的"控制 2 动机"；5.6 缩为 Conclusion 前 3–4 句"reframing 备注"，标题去掉 "training-substrate criticism" 的批判性暗示。合计再省 ~400–600 词。

#### [MINOR] LEN-3 — Section 3 细节量超出 position 体裁（3.1/3.4/3.7 的审计类内容）
- **锚点**：3.1 语料审计（23,731 文件 / 22,007 Triton / **0 CUDA C++**）、3.4 红线表、3.7 integrity note（9/55 >10x、fast_p 病态父）。
- **问题**：这些都是**好**的内容，但体量像 full-paper 的数据完整性节，挤占 8–12 页预算。
- **可执行建议**：语料审计与 integrity note 移附录，正文保留一句"0/23,731 为 CUDA C++ → 本文可测的是 schedule 级变换"这一**影响结论**的事实即可；红线表（3.4）保留（它是预注册纪律的亮点，值得放正文）。

#### [MINOR] ORIG-1 — φ 的形式化未交付，易被读作"比直觉更严格"
- **锚点**：2.2 `φ: ΔP → a`、`a → {e₁,e₂,...}`；7.3 "`φ` is never actually constructed in this paper — we assert that the diffs have labels, not that a mechanical map … exists."
- **问题**：形式化给了"存在一个简单 φ"的问题结构，但作者并未构造、也未测自动识别；Section 2 的记号严格性略高于其承载的证据。
- **可执行建议**：2.2 明确标注 φ 在本稿中是**研究问题的陈述对象**（open：automatic action recognition），不是本文贡献；把 7.3 末句的这个诚实限定**上提**到 2.2 就地声明，避免读者误读。

---

## 4. 真实优点（有证据、非凭空平衡）

1. **可复现的自我证伪，罕见且扎实。** 3.6 的基数匹配置换校准（20,000 shuffle、Miller-Madow 校正、同-k 零分布、98.8 百分位）**主动去证伪自己的 motivating 证据**并如实报告"语义不优于粗结构"。多数 position paper 不会做对自己不利的对照，本稿做了，且数据可核（`informativeness_calibration.json`）。这是全文最有价值的部分，作者自评（1.5）于此为真。
2. **自杀式 reading tree 是真·可证伪设计。** 4.3 五行中两行直接判本方向死刑（"Oracle-action ~ Direct → direction is dead"；"Free-plan ~ Structured → **no paper here**"）。这满足 position 体裁对"承诺负结果也算数"的最高要求。
3. **两条非预期发现可复现、可引用、能立刻改变他人行为。** 3.7(a) 全部 5 条 Other = 后端选择（含 327x/69x，`annotations.json` 边 `L2_2_55_*`/`L2_2_56_*` 已核）；3.7(b) 53/55 共改 autotune（已核）。这两条是**本文最硬的 so-what**。
4. **预注册纪律被真正执行。** 3.4 两条红线（>40% 单一类 / Other>25%）在见分布**前**设定，均"未触发"仍如实报告，并声明"不得事后调阈值"——这在同类稿里少见。
5. **三claim 分离（descriptive / search / learning）是清晰的概念贡献。** 2.x + 1.4 把"变换更语义"这一混淆拆成必要（descriptive，本文支持）、非充分（search/learning，本文未测），为领域提供了可复用的论证卫生学。

---

## 5. 给作者的重心迁移建议（一句话）

把论文从 *"coordinate-system 命题＋一个支撑它前提又自我拆台的测量＋一个待跑协议"*，重写为 *"三条由可复现证据支撑的方法论禁令（语义须胜粗结构 / BackendChoice 须入空间 / autotune 须剥离）＋ 一份能判定 coordinate-system 命题的协议"*——**前者是现在Abstract的卖点，后者才是现在本文实际贡献**。标题问句可保留。

---

## 6. 评分（0–100，分维度）

> 说明：本稿承认无干预实验，rigor 天花板被锁死，故 overall 不因"诚实"而抬分；诚实是必要条件不是贡献。下列分数针对**workshop position track**；主会主轨 overall 应再降约 10–12 分（建议 Reject/转轨）。

| 维度 | 分数 | 依据（锚点） |
|---|---|---|
| **Fit（场合契合）** | **60** | 主题高度契合、体裁三重漂移（FIT-1）、篇幅超标（LEN-1/3）、标靶部分不可核（FIT-4） |
| **Originality（原创性）** | **58** | framing 不新（作者自认 1.4/5.3）；但 3.6 校准 + 3.7 两发现是真原创观察 |
| **Significance（重要性 / so-what）** | **45** | 净贡献≈未跑协议；so-what 防守性、受众窄（SIG-1）；三条禁令有价值但不足以撑起命题 |
| **Clarity（清晰度）** | **62** | 单句写作极佳；但体量过载、S5/S6 重叠、Abstract 与 3.6 错位（CLAIM-1/2）拉低 |
| **Rigor（方法严谨，非用户列维度但影响 overall）** | **50** | 校准/预注册纪律加分；循环自标注（7.1）、偏样本（3.2）、无干预（4.6/7.5）封顶 |
| **Overall** | **57** | Major Revision（workshop）；可救，方向明确，须做重心迁移与一致性修正 |

**推荐意见：Major Revision**（workshop position track）。若委员会按主会标准，则为 **Reject（建议转 workshop，并先补 3.7/4.3 为主张）**。

---

## 7. 修改优先级清单（供作者按序处理）

1. CLAIM-1：删/重述恒真的熵与"distinct diff"证据；Abstract 并列 0.615-vs-0.728；改用 `distinct_tasks` / `distinct_region_sets` / 类内语义距离。
2. FIT-1 + SIG-1：Abstract/Introduction 落点从"压缩存在"改为"三条禁令 + 可判定协议"，coordinate-system 明标为开放命题。
3. CLAIM-2/FIT-3：把 7.1 循环性与 3.2 偏样本限定上提到 Abstract/3.3 可见处；54.5%、90.9% 加限定语。
4. FIT-4：Section 1 举证前移到可核公开系统（[15]/[18]），CUDA Agent 降为案例。
5. LEN-1/2/3：合并 S5/S6，压缩语料审计与 integrity 至附录，5.4/5.6 降级为备注——目标净砍 1,500–2,000 词，落入 8–12 页。
6. ORIG-1：2.2 就地声明 φ 为待解研究问题、非本文贡献。

*（本评审全程只读，未改动 `paper/main.md`；所有引用片段均可在稿件对应小节与仓库 `data/` 中定位。）*
