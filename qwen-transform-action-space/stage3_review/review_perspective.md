# Peer Reviewer 3 — 跨学科视角独立评审
### 视角：资深 GPU 编译器 / 自动调优系统工程师（TVM / Halide / Ansor 生产落地背景，兼关注 LLM agent 系统）

> **独立性与 READ-ONLY 声明。** 本评审严格独立，**未阅读** `review_domain.md` / `review_eic.md` / `review_methodology.md` 中任何其他审稿人意见；未修改稿件任何内容。
> **评审对象。** `paper/main.md`（重点 §2、§4、§5.3、§5.4、§7），并为一手核验技术论断，直接核对了随稿数据与协议文件：`data/annotations.json`（含 `statistics` 与逐边 `reasoning/notes`）、`data/informativeness_calibration.json`、`VOCABULARY_FROZEN.md`、`ANNOTATION_GUIDE.md`、`SPEC.md`。凡引用系统层事实，我尽量给到"论文原文片段 + 数据字段/行"双锚点，以便作者与编辑核验。
> **口径声明。** 本视角的若干判据（"必须与 enumerate+cost model 基线比较""反馈通道应以硬件计数器为强基线"）是**编译器/自动调优社区的方法论规范**，对一个纯 ML/SE 的 position-track 读者未必同等硬性。凡属此类"仅对本社区成立"的批评，我在条目内明确标注 *community-relative*，以免把我的领域偏好当作客观标准。

---

## 0. 一句话裁决（so what 先说）

这是一篇**协议诚实度显著高于领域平均、但坐标系论题对其最致命的反诘（§5.3）只提出未检验、且 §4 主交付按其自身说明当前不可执行**的 position/analysis 论文。目标读者行为改变分两类：**Triton kernel 基准/调优从业者**本周即可采纳 §3.7 的两条"基准卫生"（值得单独保留）；**研究社区**则应把 §4 读作"设计草图"而非"ready-to-run 协议"，把 §5 的跨域外推读作"待检验假设"而非"启示"。若这些重述到位，本文可作为一篇体面的、带负面自证的坐标系立场论文发表；若不就 §5.3 的反诘补对照臂、不解决 Oracle-action 的不可自动填输入，则对系统向读者而言核心问题仍"未判定"。

**推荐：Major Revision（跨学科/系统向）**；若目标为允许"仅协议设计"的纯 ML position track，可 **Borderline Accept**，但压缩 §5 外推与 §4 可执行性表述为**接收前置条件**。

---

## 1. 分维度评分（0–100，本视角）

| 维度 | 分数 | 依据（详见 §2–§6） |
|---|---|---|
| 跨学科连接 | **58** | 意图清晰且诚实（options/Halide/TVM/Ansor/edit-RL 定位准确）；但缺失两大系统接口（profiler 观测通道、enumerate+cost model 基线），且外推与自身 §7.4 语言依赖结论相互拆台 |
| 实践相关性 | **48** | §3.7 两条对 Triton 从业者即用、真实；主论题与 §4 对生产 kernel 团队本周零落地，且论文自认未跑 |
| 可行性（按 §4 现文所设执行） | **35** | Oracle-action / Structured-action 依赖论文自认未造的 `φ`；无搜索基线对照臂；无计算预算/样本量/统计功效 |
| 总体（本视角） | **55** | 诚实与自我拆台数据是加分；可判定性主张与其实际可执行性之间的落差是主要减分 |

> 说明：以上为**单一跨学科审稿人**的主观刻度，不代表会议底线，也不与其他审稿人对齐。

---

## 2. 发现清单（按任务的五个评审角度组织）

严重度图例：**CRITICAL**=不解决则核心主张对系统读者不成立；**MAJOR**=显著削弱说服力/可执行性，须修；**MINOR**=打磨项。每条含：证据锚点 → 问题 → 可执行修法 → *community-relative?*。

---

### 角度一：从编译器/自动调优实践者看，"坐标系"论述是否可信

#### **P3-C1 —（CRITICAL）§5.3 自封的"最强反诘"从未在 §4 被操作化；四条件框架里根本没有 enumerate+cost-model 这条对照臂。** *community-relative: 是（对系统社区是硬伤，对纯 LLM position 读者可能视为超范围，但论文自己称其为"最强反诘"，就必须自证回应）*

- **证据锚点。**
  - §5.3 原文（main.md L570–575）："Autotuning systems deliberately *do not* ask the programmer to name the transformation: they **enumerate a space and search it with a learned cost model**. If a well-engineered search over a known space already finds the schedule, an LLM choosing transformations must be adding something the cost model lacked ... **That is a real hypothesis with a real chance of being false, and it is the strongest objection we know of to this entire direction.**"
  - §4.2 条件表（main.md L455–460）：四条件为 **Direct / Free-plan / Structured-action / Oracle-action**，**全部是 LLM-in-the-loop 的接口变体**。无任何"外部枚举 + 学习型 cost model"的臂。
  - 核验：对 `main.md` 与 `SPEC.md` 全文 grep `cost model` / `random search` / `Ansor` / `TVM` / `baseline` / `brute` —— `cost model` 仅出现在 §5.3（L572–573）；SPEC 中**零命中**。即该反诘在设计文档里也没有落成实验臂。
- **问题。** 论文把"是否外部化结构对 LLM 有增益"当作可证伪命题（§4.3 读树），但它证伪的是**"LLM 的各变体之间"**的差异，而不是**"LLM 选变换 vs 工程化搜索+cost model"**这一它自己点名的真正竞争者。一个 Ansor/TVM 背景工程师的直觉答案正是 §5.3 那句：*"我们从没让模型自己命名变换；我们枚举 + 搜索 + cost model"*。§4 无法回答这句话，因为它没把这句话放进对照组。结果是 §5.3 提出了反诘、随即在 §4 放下——这正是任务要我判断的"提了就放下"。**判断：确实放下了。**
- **可执行修法。**
  1. 在 §4.2 增设第 5 条 **Brute-search 臂**：对同一组冻结旋钮（`BLOCK_* / num_warps / num_stages`）跑 random / grid search，配一个学习型 cost model（XGBoost 或小型 MLP，吃静态特征 + 可选 NCU 计数器），在 **matched rollout / matched wall-clock 预算**下与 Structured-action 比较"first-to-≥1.5× faster 命中率"。
  2. 若作者论证"该臂超出本文范围"，则须把 §4 的"可判定性"主张（Abstract L24；§4.3 L478）降级为**"本协议能在 LLM 接口内部各变体间做判别，但不涵盖 vs 工程化搜索的对比"**，并在 §5.3 末尾明写此缺口。二者取其一，不能两头都占。

#### **P3-M2 —（MAJOR）在 Triton 中"由模型命名变换"与"枚举 + 搜索"并非对立，而是同一对象的两种写法；因此 §4.2 把 Direct 当作"无结构/无搜索"基线是错的。** *community-relative: 部分是（这是对所选 testbed 语言语义的技术指认，任何在该 testbed 上主张对比公平的读者都须接受）*

- **证据锚点。**
  - `VOCABULARY_FROZEN.md` 行内定义：`Tile`→Triton realization 为 `BLOCK_* constexpr + program_id 网格映射`（L26）；`ChangeLaunchConfig`→`num_warps / num_stages / BLOCK / grid 尺寸`（L29）；`Vectorize`→`让 BLOCK 维与连续内存对齐 + tl.multiple_of/max_contiguous`（L28）；`IncreaseWorkPerThread`→`内层 tl.static_range 或更大的 BLOCK 覆盖更多列`（L27）。
  - `ANNOTATION_GUIDE.md` 判定纪律 4（L52–54）：`@triton.autotune` 新增/修改 **归入 `ChangeLaunchConfig`**。
  - `data/annotations.json` L61 `edges_with_autotuner_change: 53`；逐边 reasoning 例（L256）：把硬编码 `BLOCK_SIZE/BLOCK_M` 换成 `+@triton.autotune(configs=..., key=...)`，"kernel 主逻辑未动"，判为 `ChangeLaunchConfig`；L212 备注 "IncreaseWorkPerThread: 每 program 处理量随 autotune 候选一起变"。
- **问题。** 上述即说明：本文词汇表里最"结构"的几类（Tile / ChangeLaunchConfig / IncreaseWorkPerThread / 部分 Vectorize / 隐藏为 `num_stages` 的 SharedMemoryTransform、`num_warps` 的 WarpReduce），在 Triton 里**恰好就是 `triton.Config` 的关键字**。也就是说，"schedule transformation 的搜索单位"与"autotune 枚举的搜索维度"在本文唯一可执行的语言层上**是同一批旋钮**。而 Direct 条件生成的 kernel 在此生态里本身 routinely 带 `@triton.autotune`（这正是语料 53/55 取胜的机制），意味着 **Direct 不是"无搜索"基线，而是"自带运行时枚举搜索"的基线**。§4 全篇把对比框成"structure vs structure-plus-hyperparameter-search"（§4.5 控制 1 的措辞），但在 Triton 里这两者不可分——论文 §3.7(b)（L412–420）自己已说"empirically inseparable"，却没在 §4 把这个事实翻译成对 Direct 基线的重新定义。
- **可执行修法。** 在 §4.1/§4.2 明写：Triton 的 `@triton.autotune` 是**运行时按 key 的枚举搜索**；因此 Structured-vs-Direct 的比较必须把"是否命名变换"与"是否存在枚举搜索"两个轴**正交化**（见 P3-M3 的 2×4 设计），否则 §4 的核心对比把两件事混为一谈，任何正向结果都可被"你只是少给了 Direct 一档候选"解释掉。

---

### 角度二：四条件框架的工程可行性（能否跑 / 成本量级 / 哪条控制最容易走样）

#### **P3-C2 —（CRITICAL）Oracle-action 与 Structured-action 依赖论文自认"从未构造"的 `φ` / realization 语义，按 §4 现文不可自动执行；最危险的后果是把"我们的 oracle 造不出来"误读为"方向已死"（假阴性）。** *community-relative: 否（这是论文内部一致性与实验可执行性问题，任何读者都受影响）*

- **证据锚点。**
  - §7.3 原文（main.md L727–729）："**`phi` is never actually constructed in this paper - we assert that the diffs have labels, not that a mechanical map from diff to label exists**, and the gap between those two is a separate research question (automatic action recognition) that we deliberately did not open here."
  - §4.2：Oracle-action = "**the correct transformation is supplied**; the model only realizes it"。要"supply 正确变换"，必须先有 (diff/parent → ground-truth action) 的 oracle 来源——正是上面那句承认没有的 `φ`。
  - §4.5 控制 3（L512–514）："report, for each selected action, whether the subsequent edit actually **touched the sites the action requires**" —— 需要一个 action→required-sites 的映射（realization obligation 的形式化），论文同样未给。
  - §4.3 读树（L475）："Oracle-action ~ Direct ⇒ **direction is dead**" —— 一个"终止研究方向"的分支，其输入（oracle action）在此 testbed 上无法可靠自动生成。
- **问题。** 终止性判据建立在一个无法自动填充的输入上。若实操中 oracle 由"模型自标 / 词表命中"近似，则读树里 **Structured-action ≈ Oracle-action** 几乎必然成立（同一标注家族，见 §7.1 循环自证），会把"命名无用"错误归因为结构无用；若 oracle 根本填不出，则该臂缺失、读树不闭合。两种情形都使 §4 对系统读者"不可即刻执行"。
- **可执行修法。**
  1. 把 `φ`（diff↔action 的自动识别）显式列为 §4 的**前置工程**，并给一个可运行的最小 recognizer/parser + 其 precision/recall；没有它，Structured 与 Oracle 两臂都只是纸面。
  2. 或将 Oracle-action 降级为**小规模人类专家手填 oracle 动作**的 study（几十个任务量级），并据此缩小读树中"direction is dead"的适用面——把它标注为"依赖 oracle 质量的条件性结论"。
  3. 无论哪种，删除/改写 L475 的绝对化终止语，改为"在 oracle 可信前提下的终止判据"。

#### **P3-M3 —（MAJOR）§4.5 控制 1"固定 autotune 候选集"无法达成它假设的"公平"，且会系统性削弱基线 → 假阴性；这是全套控制里最容易被实现走样的一条。** *community-relative: 否*

- **证据锚点。** §4.5 控制 1（L506–508）："**Autotune candidate sets fixed identically across all four conditions**, or explicitly modelled as a separate action dimension."；§3.7(b) 53/55；`VOCABULARY_FROZEN` L29。
- **问题（三种"实现走样"的具体路径）。**
  - (a) **固定候选列表 ≠ 固定最终配置**：Triton autotuner 在运行时按 `key` 对整张列表实测取优并缓存。把四条件喂同一 `configs=[...]` 只固定了"搜索空间"，最终 config 仍由运行时搜索决定——控制 1 若指望"钉死候选=钉死调参贡献"，对机制理解不足。
  - (b) **为"公平"把列表缩到单档/极小**：等于删除本语料最主要的取胜机制（53/55 共改 autotune），Direct 与 Structured 一起掉到地板（floor effect），真实差异被压到不可见 → 系统性假阴性；且使任务难度远高于原轨迹，held-out 成功率整体塌陷时无法区分"结构无用"与"任务被我阉割"。
  - (c) **各条件各自保留模型自写 autotune 块**：则 Direct（语料显示模型爱写大而全候选）拿到的搜索能力 ≥ Structured（被要求"选一个变换"），两条件 compute/搜索能力不 matched → 反向偏置，Structured 被自己的接口束缚。
- **可执行修法。** 把控制 1 从一句话升级为**可操作协议**：
  1. 明确"matched search"的定义 = 对四条件注入**同一候选池 + 同一 `key`**，并规定 **autotune sweep 的 wall-clock 是否计入 §4.3 的 search cost**（计入则 Direct 也背上 sweep 成本，比较才公平）；
  2. 采用 **2×4 全因子**（{命名变换 / 不命名} × {固定单档 / 固定全池 / 自由候选}），把"是否外部化结构"与"搜索自由度"分离估计，而非"固定候选"一锅端；
  3. 无论何种设计，报告"结构取胜"是否在剔除 autotune 维度后仍成立——否则结论只是"多给候选有用"。

#### **P3-M4 —（MAJOR）计算量级 / 样本量 / 统计功效从未给出，使 §4 无法被评审或复现地"跑一遍"。** *community-relative: 否*

- **证据锚点。** `main.md` 全文 grep `GPU-hour` / `compute budget` / `statistical power` / `trials` / `runs per` → 0 命中；`SPEC.md` grep `budget` / `GPU-hour` / `wall-clock 预算` / `seed` → 仅命中标题行（即设计层无预算条款）。§4.3（L467–468）仅泛泛写"held-out tasks, success rate and search cost (rollouts, tokens, wall-clock)"。
- **问题。** 4 条件 × held-out 任务（KernelBench L1/L2/L3 约 250 题）× 多种子 × 每题到 first-faster 的多 rollout，**每 rollout 含一次完整 Triton 编译 + autotune 运行时 sweep + 正确性核对 + 计时**；再叠加 P3-C2 的前置 recognizer。量级粗估落在 **10⁴–10⁵ GPU-min（数十至数百 GPU-hour）**，且 sweep 主导方差。论文既未给 pilot 规模、停止规则，也未给的功效分析，读者无法判断"零结果"是真无效应还是样本不足。
- **可执行修法。** 给一个 **costed pilot**（例：20 任务 × 5 种子 × {Direct, Structured} 两臂 ≈ N GPU-h，附预期 CI 宽度）与预注册停止规则；或在 §4 开头一句把 §4 明确定位为"接口设计"，从摘要与结论撤下"decidable / protocol that decides it"这类读起来 ready-to-run 的措辞（Abstract L24–26；§8 L765–769）。

---

### 角度三：跨学科可借用性（外推边界；是否把"没做实验"包装成"启示"）

#### **P3-M5 —（MAJOR）§1.1 / §5 的 FORTRAN / FPGA 外推与论文自身 §7.4 的语言依赖结论相互拆台；外推越界。** *community-relative: 部分（"抽象层不同则词表不同"是编译器常识，但对不写编译器的 ML 读者是新信息，值得写进正文而非仅 §7.4）*

- **证据锚点。**
  - §1.1（L52–56）："Substitute FORTRAN or FPGA synthesis and **the recipe transfers untouched**. ... NKI-Agent adapts the CUDA-Agent framework to AWS Trainium ... and reports that GRPO with a binary compilation reward fails to improve on supervised fine-tuning [18]."
  - §7.4（L733–737）："A transformation vocabulary is **not a language-neutral object**. It is **pinned to the abstraction level** of whatever the model is asked to emit, and 'CUDA optimization' spans at least three levels ... **'Any transfer claim must be intra-level unless a mapping between levels is specified.'**"
  - §5 标题即"Implications Beyond CUDA"，但其下 §5.1–§5.6 无一新实验。
- **问题。** 论文最有价值的自身发现之一（§3.1/§7.4：因 Triton 隐藏线程层，`WarpReduce`/`SharedMemoryTransform` 两类在主标签上"消失"）恰恰证明**词表随抽象层漂移**。而 FORTRAN 自动向量化/并行、FPGA HLS（loop pipelining / array partitioning / dataflow fusion，受 FIFO 与面积约束）与 Triton 分块，处在**互不相同的抽象层**，其"Fuse""Tile"语义并不成立为同一动作（HLS 里 fuse 一个 dataflow region 的代价模型与 Triton 里合两趟 kernel 完全不是一回事）。因此 §1.1 的"recipe transfers untouched"与 §5 的跨域"启示"，悄悄依赖一种 §7.4 明令禁止的**跨层可迁移性**。这是"把没做实验包装成启示"的典型：证据仅在单一语言层（Triton schedule 层）成立，外推句却升到"任何代码生成/优化域"。
- **可执行修法。**
  1. 把 §5 各段显式改标为 **"未做实验的假设 / 待检验外推"**（部分段落已近似如此，但摘要/§1.1/§8 的口吻仍是"implications"），并在 §5 首段复述 §7.4 的 intra-level 约束作为外推上界；
  2. 若坚持 FORTRAN/FPGA 例，至少各给一个**该层特有的自然词表 + 一个反例**（说明为何本文 12 类在其上会失效或需重定义），把 §7.4 的语言依赖当作对外推的**约束**而非脚注。

#### **P3-m6 —（MINOR）§5.4 的两条"方法论观察"自称未测量，却承托着"prose 动作空间会失败"的结论；且 FlashRT[28] 类比对象错位。**

- **证据锚点。** §5.4（L586–599）："**Compliance appears to degrade with instruction-file length** ... we treat it as an un-quantified risk rather than a result"；"Note that **we have not measured either observation**"；L580–582 描述 [28] 为"force a model to emit a structured intermediate representation ... before writing code"，但 [28] 标题是 *FlashRT: Agent Harness for ... Real-Time Multimodal Applications*（部署 harness，非动作语义问题）。
- **问题。** 这两条是 §5.4 全部论点（"should implement as a genuinely small typed interface rather than documentation"）的支点后，却是轶事级；把部署 harness 的"先出结构化 IR 再审计"当作动作空间的语义证据，类比偏松。
- **可执行修法。** 降级为"设计约束/假设"；或补一个**极小消融**（同一动作集，prose 目录版 vs DSL 版对模型的遵从率 / 误触发率），哪怕 n 很小也比"we observed"强。

---

### 角度四：实用影响与落地（kernel 团队下周能改什么）

#### **P3-M7 —（MAJOR）诚实回答任务之问：对生产 kernel 团队，本文本周可执行的只有约两条，且均属"基准卫生"，与坐标系主论题无关。**

- **可即用、且我认可其真实价值的两条：**
  1. **§3.7(b)（L412–420）→ 做 Triton 结构改动 A/B 时钉住 autotune 候选，否则测的是重调参不是重构。** 任何在 Triton 上给 kernel 换过 `@triton.autotune` 块的人都被这个坑咬过；这条**值得单独保留并加粗**，是全文对从业者最实的贡献。
  2. **§3.7(a)（L402–411）→ 优化 checklist 缺一项："是否有现成后端（cuBLAS `x.matmul` / `F.scaled_dot_product_attention` / `F.mish`）已比你手写快"。** 5/5 的 `Other` 全指向这条，且样本里最大两次单步取胜（327×、69×，作者已注明父节点病态、不可当能力读）都出自此。对写 kernel 的人这是便宜且常赢的一招。
- **主论题 / §4 对本周工程零即用价值。** §4 是研究协议，§4.6（L521–526）自认 learning 主张需真实训练对比"我们没有基础设施"，search 实验也没跑。因此一个 kernel 团队读完，工程动作 = 上面两条卫生；若想据此改"动作空间/credit 分配"的做法，本文恰恰说"还不知道该不该改，先去做 §4"。
- **so-what（直说，按任务要求）。** 对**工程实践**，本文改变的是两条基准纪律；对**研究 Framing**，它改变"先证明你的抽象打得过它所抽象对象的粗结构，再谈换坐标系"这一自检习惯（§8 L780–783）——这条我认为**确实有用且应更靠前**（放进 §1 的贡献清单）。但"坐标系之问"本身不改变任何工程师下周写 kernel 的方式。
- **可执行修法。** 在 §5 或 §7 增两个小标题：**"What a practitioner should do today"**（放上面两条 + profiler/计时卫生，见 P3-m9）**与 "What this paper is NOT for"**（明确主论题非落地配方），避免读者把 position 论文当 recipe 读。

---

### 角度五：本视角独有的遗漏风险（系统层现实）

#### **P3-M8 —（MAJOR）论文把 profiler 反馈当"噪声"，但 ncu/occupancy/achieved-BW/stall-reasons 恰恰是比 ≤12 类手写词表更结构化、更低熵、硬件接地的观测通道——论文既未把它当观测，也未当竞争坐标系；于是"坐标系之问"漏掉了最强的候选之一。** *community-relative: 是（"cost model 吃 NCU 特征"是自动调优核心方法论；对纯 LLM 读者这是新增视角，故更应写进正文）*

- **证据锚点。**
  - §5.6（L631–634）把 token 级不稳定的成因归为"swallowing compile errors, **profiler dumps**, and odd intermediate program states ... train–inference mismatch and out-of-distribution-context" —— 明确把 profiler 输出归为要被吞的"噪声/OOD 上下文"。
  - 但 §6（L651–655）自述 CudaPerf[15] 的结构奖励来自"memory **coalescing, occupancy, arithmetic intensity, synchronization patterns**" —— 这些正是 NCU/nsight-compute 计数器。
  - §2.1–§2.3 论证动作应"约束实现"；`ncu` 的 stall/occupancy 直接约束 `num_warps`/`num_stages`/tile 决策，是天然接地的结构化信号。
- **问题。** 论文的整套论点是"结构层比 token 层更适合当坐标系"。从系统视角，最"结构"、最信息密集、且**已被自动调优社区当作 cost model 输入**的表示，是硬件计数器向量，而非一个 ≤12 项的散文/DSL 词表（后者粒度还依赖语言层，见 §7.4）。把 profiler 当噪声、又引用吃 profiler 的 CudaPerf，内部有张力；更关键的是，若不做与"结构化数值观测（profiler 向量）"的对照，坐标系之问对系统读者并未被完整提出。
- **可执行修法。**
  1. 把 §5.6 对 profiler 的措辞从"噪声"改为"结构化观测/反馈通道"，正面承认它是候选表示；
  2. 在 §4 增设或讨论一条**以 NCU/occupancy/achieved-BW 为观测的对照**（哪怕是学习型 cost model 臂，见 P3-C1），说明词表动作与之的关系（互补？被其支配？）；
  3. 至少引用一支成熟 GPU 调优/性能预测工作（如 Ou&Sequeira、KernelOpt 类反馈引导优化）以显式定位本文相对"数值反馈坐标系"的立场，而非默认缺席。

#### **P3-m9 —（MINOR，但触及 headline 结论）中心自证伪统计（0.615 vs 0.728 bits）的因变量建立在 `runtime_stats.mean` 单次均值、未复测、且已知 9/55 病态父节点；在 quartile 分箱下计时噪声足以翻转个别点归属、进而可能翻转高低。** *community-relative: 否（测量有效性对任何读者都成立）*

- **证据锚点。** §3.2（L278–280）"the child more than 5% faster (measured **`runtime_stats.mean`**)"；§7.3（L721–725）"Speedup is taken from corpus metadata **without re-measurement**; 9 of 55 edges show >10x gains ... some gains are plausibly **timing artefacts**"；`data/informativeness_calibration.json`（primary bias-corrected 0.6152 vs coarse 0.7283，n=55，quartile H≈2 bits）；`annotations.json` L221（10.21×, parent_fast_p=8.40）、L259（13.27×, parent grid 仅 8 program 欠占用）、L338（5.32×, **parent 比 reference 还慢**）。
- **问题。** 用 `mean` 而非"warmup 后 median-over-reps"，且无方差，是 kernel 基准里公认的脆弱计时法；本又拿它分 quartile（55→每箱约 13 条），个别近箱界的点若计时抖动即换箱，可能改变 0.615/0.728 的差值方向——即"低熵不是论据"这一被作者称为全文最有价值（§1.5 L155–156）的负面结论，其本身建立在含噪测量上，稳健性未证。
- **可执行修法。** 做**敏感性分析**：(a) 剔除 >10× 的病态边后重跑 §3.6；(b) 对一个小子样本用 median-over-reps + 复现方差重测，报告 0.615 vs 0.728 的**符号是否稳健**；若稳健则此结论更硬，若翻转则须弱化"§3.6 是决定性负面"的表述。

#### **P3-m10 —（MINOR）`PrecisionChange` 是第二高频 primary（15/55＝27%），其"correct"依赖输入分布与容差，而语料把"更快且正确"当作 ground-truth 正样本来支撑压缩论。**

- **证据锚点。** `annotations.json` L26 `PrecisionChange: 15`；`VOCABULARY_FROZEN` L32（`tl.float16/input_precision/ALLOW_TF32`）；§5.5（L603–606）自认"numerically wrong yet pass the checker"[20]、oracle 在 mutation 下脆弱[24]、"不 survive 更严测量"[22,23]。
- **问题。** fp16/tf32 的"通过正确性"高度依赖 KernelBench 的随机正态输入与固定 atol/rtol；真实激活分布（outliers、稀疏）下同一 kernel 未必过。于是被论文当作"成功且正确的优化边"里，约 27% 的成员资格由 **oracle 阈值**而非真实数值域决定——这直接冲击 §3.5 的"压缩成立"（分母里这批边的真实性）。（注：纯统计/去重属方法学审稿人领地，我只指出其系统成因：数值 oracle 的分布依赖。）
- **可执行修法。** 对 fp16/tf32 边报告容差与输入分布；对 `PrecisionChange` 单独做"分布敏感性"讨论；把这类边的成功标注为"oracle 依赖"，在 §3.8 verdict 里计入权重折减。

#### **P3-m11 —（MINOR）语料把 kernel 当自包含对象，但 `HostDeviceOverlap`（7/55，第三高频）与 `Fuse`/`EliminateMaterialization` 的**生产合法性**由 host 框架契约决定，"更快且正确"是脱离 host 契约的结论。** *community-relative: 是（生产耦合是系统工程现实，隔离基准的 ML 读者常忽略）*

- **证据锚点。** `VOCABULARY_FROZEN` L25（`EliminateMaterialization` 的 Triton realization＝"去掉 `.contiguous()`/`clone()`"）、L34（`HostDeviceOverlap`＝"减少 torch 端算子往返、避免 device→host 同步点、合并启动"）；§1.3 L109–112（work-per-thread 需 host 端 `N/T→N/(2T)` 协同改动）。
- **问题。** 生产训练循环里，能否去 `.contiguous()`（下游是否要求连续输出）、去某 D2H 同步点（autograd/控制流是否依赖该同步语义）、合并 kernel 启动（是否落在 CUDA-graph capture 约束内）都由 **host 框架契约**决定；KernelBench 单算子隔离评测默认无这些约束。因此语料判为"合法且更快"的 `HostDeviceOverlap`/`EliminateMaterialization` 边，有一批在真实管线里**不可用**——这不是又一条 confound，而是说明"动作"的边界本身被 host 契约截断，正如 §3.7 说动作与 autotune 不可分，这里动作与 host 契约也不可分。
- **可执行修法。** §7 增"生产耦合"一项，明写本文动作单元是**host 契约之外**的理想化对象；对 `HostDeviceOverlap` 边标注其是否在真实训练循环中可执行（哪怕仅作为未来 held-out 的过滤条件）。

---

## 3. 需要正面肯定的地方（具体、非空泛）

为避免"只挑刺"，明确肯定以下**可核验**的优点，它们也是我建议 Major Revision 而非 Reject 的理由：

1. **主动量化自我拆台（§3.6）。** 作者不但报告"语义标签在预测 speedup 上不敌 diff 粗结构（0.615 vs 0.728 bits）"这一**反自家 framing 的**结果，还用 cardinality-matched permutation calibration（20,000 次同基数的随机划分，seed 20261002）来消除"格子多→条件熵自然低"的偏置。`data/informativeness_calibration.json` 的数字与正文一致，方法得当。这在同类 position 论文里罕见。
2. **冻结纪律真实执行。** `VOCABULARY_FROZEN.md` L3–L7 声明冻结**早于任何边 diff 被读取**，且冻结只依赖语料级元数据（目录/JSON 字段/语言普查），红线阈值（单类>40%、`Other`>25%）预承诺、不得回改（§3.4 L316–317）。这是我会在评审中要求、而多数稿件做不到的一条。
3. **诚实的边界感。** §1.1（L58–62）主动声明 CUDA Agent 的数值细节因原文不可得而**不予断言**；§7.1 直面"AI 自标 → 循环自证"并把它列为对 §3 的一阶威胁；§7.5 逐条列"本文未证明什么"。这种口径对我（系统审稿人）建立信任的价值，高于任何实验增量。
4. **描述/搜索/学习三claims 分离（§2、§4.6）。** 论文清楚知道"压缩成立"离"换坐标系有利搜索/学习"很远，并拒绝越界宣称——这与我 P3-M2/P3-C1 的批评并不矛盾：它**没有**假装证明了搜索收益，只是**没把竞争者放进对照组**。

---

## 4. "so what" 总表

| 发现 | 严重度 | 目标读者会因此改变行为吗 |
|---|---|---|
| P3-C1 §5.3 反诘未落成为对照臂 | CRITICAL | **会（研究社区）**：补 Brute-search 臂或降级可判定性主张，否则系统读者视核心问题"未判定" |
| P3-C2 Oracle/Structured 依赖未造的 `φ` | CRITICAL | **会（研究社区）**：把 `φ` 列为前置工程或降级读树终止判据 |
| P3-M2 Triton 里"命名"与"枚举"是同一对象 | MAJOR | **会**：Direct 非"无搜索"基线，须重述 §4.1–§4.2 对比语义 |
| P3-M3 控制 1"固定候选"不可行/削弱基线 | MAJOR | **会（研究者+从业者）**：改 2×4 全因子并规定 sweep 是否计入成本 |
| P3-M4 无计算预算/功效 | MAJOR | **会（复现者）**：给 costed pilot 或撤下"ready-to-run"口吻 |
| P3-M5 FORTRAN/FPGA 外推越界、与 §7.4 拆台 | MAJOR | **会（跨域读者）**：§5 改标"待检验假设"，外推限 intra-level |
| P3-M7 落地仅两条基准卫生 | MAJOR | **部分**：从业者采纳 2 条卫生；主论题不改变写 kernel 的方式 |
| P3-M8 把 profiler 当噪声、漏掉强候选坐标系 | MAJOR | **会（系统读者）**：正面纳入 NCU/结构化观测作对照或讨论 |
| P3-m6/m9/m10/m11 | MINOR | 各自小范围：措辞降级、敏感性分析、容差披露、生产耦合声明 |

---

## 5. 给作者的最优先三条（若只做三件事）

1. **补 §5.3 的对照组（P3-C1）**，或把 §4/摘要/结论里的"可判定/decidable"降级为"可在 LLM 接口内部各变体间判别，不涵盖 vs 工程化搜索"。这是本文从"立场"升到"可证伪立场"的唯一关键缺口。
2. **解决 P3-C2**：把 `φ`（diff↔action）作为 §4 的前置工程并给最小 recognizer + precision/recall；否则 Structured 与 Oracle 两臂都只是设计图，读树的终止判据不可用。
3. **收紧外推与措辞（P3-M5 + P3-M8）**：§5 明确"未做实验/待检验"，并把 profiler 从"噪声"重定位为"结构化观测/竞争坐标系"，与 §6 已引的 CudaPerf 对齐。

---

## 6. 推荐意见

**Major Revision（跨学科 / 系统向）。** 理由：诚实度与自检纪律达到可发表水准，§3.6/§3.7 的负面与意外发现有独立价值；但（i）自称的"最强反诘"未被证伪协议纳入对照（P3-C1），（ii）主交付 §4 按现文不可自动执行（P3-C2/P3-M3），（iii）跨域外推越出自身语言依赖结论（P3-M5）。三者皆可由**补一条对照臂 + 补前置 recognizer + 收紧措辞**在修回内解决，无需新增长周期训练实验。若目标为纯 ML/SE position track 且允许"仅协议设计"体裁，在完成上述措辞降级（尤其撤下 Abstract/§8 的 ready-to-run 口吻）后，可接受为 **Borderline**。

> 本推荐仅代表**第 3 位（跨学科）审稿人**的视角，不构成编辑决定。我对"必须 vs enumerate+cost model 比较""profiler 向量是强候选坐标系"等判据的坚持源于编译器/自动调优社区规范；一个纯 LLM 读者可能认为这些超出本文范围。凡此已在条目内以 *community-relative* 标注，供编辑权衡。

---

*评审完成。未修改稿件；未参考其他审稿人意见。*
