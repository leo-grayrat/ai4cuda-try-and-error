# Devil's Advocate 审稿意见 — 对该论文最致命的反驳

> 角色：Devil's Advocate。职责是构造对本文**最致命**的反驳，专门检测稻草人、确认偏误、精挑数据、逻辑链断裂、过度概括与被忽略的替代解释。
> 独立性：本意见在**未阅读** `stage3_review/` 下其他审稿人文件的前提下独立完成。
> 约束：READ-ONLY，未改动 `paper/main.md`、任何数据或脚本。仅在本目录 `tmp/` 下临时写了两个一次性复算脚本（`attack_c.py` / `attack_a.py`），其输出被引用于下，用完即删。
> 所有被引用的原始数字/片段均来自 `paper/main.md`、`data/annotations.json`、`data/informativeness_calibration.json`、`scripts/*.py` 或我在 `tmp/` 复算脚本对上述真实数据的重算，未编造任何未出现在论文/数据中的内容。

---

## 1. Strongest Counter-Argument（最致命反驳）

这篇论文把自己最珍贵的贡献定义为一次"诚实的自我反驳"：§3.6 与摘要、结论中反复出现的"语义标签的信息量不优于 diff 的表面结构"（Abstract: *"the semantic label yields 0.615 bits of information about realized speedup while a coarse description of the diff's surface structure yields 0.728 bits"*；§3.6: *"surface structure is at least as good as the semantics ... even after cardinality correction"*）。仓库的 `STAGE2_5_INTEGRITY.md` 更把"把 bug 当洞见"列为已通过防线，把 §3.6 列为"关键自查"。**而这条头条负面结论是一个可验证的算术错误产物。** `scripts/calibrate_informativeness.py` 第 107–109 行把 Miller–Madow 偏差项写成 `h + (k_X−1)/2n`（`k_X`＝条件变量基数），这对**互信息**是错的——互信息的正确偏差项应为 `(k_X−1)(k_Y−1)/2n`，代码漏掉了 `(k_Y−1)＝3` 这个因子（`k_Y`＝加速四分位类数）。我在 `tmp/attack_c.py` 中用论文自己的数据复算：采用标准互信息偏差式后，语义标签＝**0.4515** bits、粗结构＝**0.3101** bits，**方向翻转，语义胜出**；采用教科书"占用格"精确 MM 式，数值又变成 0.6243 vs 0.7919，与代码两组都不同，说明代码没有正确实现**任何**一种 MM 变体。而真正不依赖 MM 公式、同时保留两侧边际分布的**公平置换检验**（我补做的、论文对粗结构从未做过的那一步）给出决定性结果：语义标签 p＝**0.0003**（远超同基数随机重标），粗结构 p＝**0.7496**（其观测 MI 甚至**低于**自身粒度的随机基线，纯由 24 格中 11 个单点格造成的机械过拟合）。换言之，论文只给标签做了同基数校准（并引以为豪地报告 98.8 百分位），却拿**未校准的原始 MI**去和标签比大小，这一个不对称加上一个错误的偏差项，恰好把结论钉死成它的"诚实卖点"。一旦这个 bug 被戳穿，被抽掉的不仅是 §3.6 的方向，还有摘要与结论的承重句、以及整篇论文"我们敢于自证其反"的可信度标识——而 `verify_paper_numbers.py` 第 130–132 行恰恰把错误的 0.615/0.728 断言到小数点后三位，因此"69/69 数字全部对账通过"只是把论文与它自己的错误脚本对齐，**结构上无法**发现这类方法学错误。一句话：**本文最被珍视的诚实自查是一个漏乘因子的 bug；修正它，头条结论反向。**

---

## 2. Issue List（问题清单）

维度标记：【统计推断】【测量效度】【论证逻辑】【样本与推断】【可复现机制】【诚实性/循环】【新颖性/差异化】【动机效度】。

### CRITICAL（致命）

**[C1]【统计推断 / 诚实性】§3.6 头条负面结论 = Miller–Madow bug + 单边校准；修正后方向翻转**
- 小节位置：Abstract L21–24；§3.6 L360–398（尤其表 L378–381、句子 L390–391）；结论 L759–763；实现 `scripts/calibrate_informativeness.py` L107–109；对账 `scripts/verify_paper_numbers.py` L130–132。
- 原文片段：*"the semantic label yields 0.615 bits ... while a coarse description of the diff's surface structure yields 0.728 bits"*；*"surface structure is at least as good as the semantics ... even after cardinality correction"*。
- 复算证据（`tmp/attack_c.py`，seed 20261002）：
  - 代码（buggy）：primary 0.6152 / coarse 0.7283 → coarse 胜。
  - 标准 MI 偏差式 `(k_X−1)(k_Y−1)/2n`：primary **0.4515** / coarse **0.3101** → **语义胜**（翻转）。
  - 占用格精确 MM：primary 0.6243 / coarse 0.7919（数值再次不同于代码，坐实代码非任一 MM 变体）。
  - 公平置换检验（保留两边际，无 MM）：label p=**0.0003** SIGNIFICANT；coarse p=**0.7496** not significant；且 coarse 观测 MI 0.937 < 其自身 24 格随机基线均值 0.991。
- 为什么致命：这是论文自我定位的核心美德（"最重要的诚实自查"）与摘要/结论的承重句；它既方向错误，又暴露"预注册纪律 + 数字对账"两套信任机制都对该类错误免疫。审稿人一旦复算即可推翻，且推翻的是论文最引以为傲之处。
- 可否修复：**可**——但代价是改头面：对两个描述符都跑同基数/保留边际的置换检验，撤回"coarse ≥ semantic"的方向断言，改为"在 55 点样本上二者均不可与各自粒度下的随机划分区分/标签显著"。修复后 §3.6 不再是"负面卖点"，论文的身份叙事须重写。

**[C2]【新颖性/差异化】§5.1 三条"区别于 edit-based RL"的差异化指标，实为恒真式或有偏抽样的伪影**
- 小节位置：§5.1 L532–550（*"The distinguishing property is measurable, not rhetorical ... 34 occurrences of ChangeLaunchConfig across 30 tasks with 34 distinct diffs ... the single most concrete piece of evidence"*）；§3.5 表 L339–351；§2.5 L248–249。
- 原文片段：*"No category ever produced the same diff twice, and every frequent category recurs across tens of different tasks."*
- 复算证据（`tmp/attack_a.py`）：
  - **"一动作多 diff"＝恒真**：55 条 diff 指纹**全部唯一**（55/55）。`distinct_diffs` 列对**任何**划分都必等于出现数；连"给每条边贴唯一 edge_id"也能得到 55 distinct diffs。故该列**零信息**。
  - **"高 diff 熵 vs 低标签熵"＝删失上界 vs 天花板**：`H(diff 指纹)=5.7814` 恰等于 `entropy_primary_max_bits=5.7814=log2(55)`（见 `annotations.json`），即指纹近乎单射，其熵被钉在理论上界。"2.67 vs 5.78 的压缩"是拿真实分布跟天花板比，非变换专属证据；问"2.67 是否算低"的正确基线是随机十分组的熵（≈3.3），不是 5.78。
  - **"跨任务复用"＝抽样伪影**：55 边覆盖 45 任务＝1.22 边/任务（每任务上限 3、按 speedup 排序富集）。在"保持标签重数、随机重排落到哪个任务"的零分布下，每个标签的 distinct_tasks 观测值都落在随机 95%CI 内、且≈随机均值：ChangeLaunchConfig 观测 30 vs 零均值 30.1 CI[27,33]；EliminateMaterialization 29 vs 28.5；Fuse 22 vs 22.9；PrecisionChange **18 vs 20.4**（观测反而低于随机，即标签更倾向聚在同任务内）。"复用于约 30 个任务"由抽样设计**强制**，不是迁移信号。
  - **"跨位置协调"＝边级指标错配动作级主张**：`spans_multiple_loci` 定义在**边**级（`ANNOTATION_GUIDE.md` L39，sites⊆hunk_regions）。而 **0/55 边是单标签**（每边带 {2:9, 3:37, 4:9} 个标签），故 54.5% 多区域边**全部**是多动作边，无法区分"一动作散在多位置"与"多动作各占一处"；且多区域边平均 hunk=4.07 vs 单区域 1.12，说明"区域数"机械随 hunk 上升（抽样刻意富集多 hunk）。最直白的反例在论文自己的标注里：`Other` 边 `L2_2_14_86_87` spans [_gemv_kernel, forward] 两区域，其 reasoning 明说这是**两个不同后端替换动作**（matvec→cuBLAS、归约→另一 Triton kernel），恰是"多动作各一处"而非"一动作跨多位置"。
- 为什么致命：§5.1 是对"这不就是 edit-based RL 换皮"的**唯一正面回答**，而它赖以自证的三个"measurable, not rhetorical"量，一个是恒真、一个是抽样伪影、一个是层级错配。差异化随之退回修辞。
- 可否修复：**部分可**。需在**动作级**（非边级）定义 locality 统计、在**无偏且更大**样本上重测、并用 §4.4 的 held-out 迁移实验（而非语料内共现）来真正测"跨任务复用"。当前数据无法测这三个量。

**[C3]【诚实性/循环 / 论证逻辑】§7.1 的循环承认不足以支撑 §3；§3.4 红线在有偏样本上"未触发"是"用声明的纪律伪装客观性"**
- 小节位置：§7.1 L684–695；§3.4 红线表 L307–317；§3.2 抽样偏差声明 L282–285；置信度 §3.7b L415–417 与 `annotations.json`（high 14 / medium 38 / low 3）。
- 原文片段：§3.2 *"This is a biased sample: it can support an existence claim but not a population estimate of the label distribution."*；§3.4 结论"两条红线 **未触发**"。
- 为什么致命（两处内部矛盾）：
  1. 红线（单类 >40%、Other >25%）本质是关于**标签总体分布**的断言，却在一个论文自己声明"**不能估计标签总体分布**"的有偏样本（多 hunk 富集＋每任务≤3＋按 speedup 排序）上判"未触发"。用一个被宣告无效于估计分布的样本去**认证**两条分布性红线未破，是自相矛盾。小样本阈值亦无功效：55 边时 25%＝13.75 边，Other 观测 5，属"几乎不可能触发"的设计。
  2. 论文 §3.7a 自己发现**全部 5 条 Other 是同一个缺失的一级动作（放弃自研、改库后端 BackendChoice）**——这是"词汇表存在系统性缺口＝猜想可能不成立"的**概念性触发**，却因红线量的是"残留占比"而非"缺口是否存在"而被记为"未触发"。红线恰好设成了会让真实反证溜过去的形状。
  3. 循环自证：词汇由本模型家族设计、标签由本模型家族给出、置信度亦由同一模型自评（75% 边它自己只给 medium/low），§3.5–3.6 却把这些 primary 当作真值跑熵与校准。§7.1 只"承认"风险而不"隔离"风险——它自己亦承认"neither of those disciplinary moves removes the circularity"。承认 ≠ 控制。
- 可执行的"足够"标准（用于判定 §7.1 是否合格——目前**不合格**）：(i) ≥2 名**盲于模型标签与假设**的人类专家，在**无偏概率样本**（≥100 边，非多 hunk 富集）上独立标注，报告 Cohen's/Krippendorff's **inter-rater agreement**；(ii) 词汇由**人类初述 bottom-up 聚类**生成、或由**另一模型家族**生成，再检验压缩是否复现；(iii) 报告 human-vs-model 标签一致性 κ；(iv) **§3.6 的整条比较必须在人类共识标签上重跑并存活**。四项缺一，§3.5 最多支持"某一模型家族能用 10 类描述这些 diff"，而非"这个抽象存在于数据中"。
- 可否修复：**可但工作量大**（等于补做人工标注实验，超出本文现有范围）。

### MAJOR（重大）

**[M1]【论证逻辑】全文无干预实验 → 坐标系命题"不可判定"却被当作主贡献**
- 位置：§1.5 L152–156；§3.8 L438–440；§4 整节；§7.5 L739–742（*"No search experiments. No training."*）。
- 片段：§4.4 的操作化检验 *"If this experiment completed perfectly, which judgement about the original research question would it change?"*。
- 为什么重大：论文用 §4.4 这把尺子衡量他人，却应首先量自己。其两大承重型贡献是 (a) 一个测量（§3，其头条是 C1 的 bug）、(b) 一套**未运行**的协议（§4）。Position paper 体裁合法，但当"测量"错、"协议"未做，剩下的是散文。§4.6 自己承认"即便 Structured-action 胜出也只证搜索、不证学习"。
- 可修复：仅能通过实际执行 §4 四条件；本文档内不可。

**[M2]【测量效度】§3.6 的目标变量（实现加速四分位）本身被病态父节点与 autotune 污染**
- 位置：§3.2 定义 L277–280；§3.7b L412–425；`annotations.json` 与标注实例（如 `L2_2_20_60_72` notes：*"parent_fast_p=0.3023 说明 parent ... 严重退化"*，5.49×；`L2_2_24_2_26` parent_fast_p=0.3230，3.22×；`L2_2_18_67_70` 纯加 autotune 即 13.27×）。
- 为什么重大：§3.6 的一切条件熵都以 `H(S)`（加速四分位）为目标。但抽样按 speedup 降序富集、加速取自语料元数据**未重测**、9/55 边 >10×、多条父节点比 PyTorch 参考基线还慢（fast_p<1），53/55 与 autotune 同变。于是 `S` 很大程度编码"父节点有多病态""是否开启 autotune"，而非变换质量。**即使修好 C1，翻转后的正向结论仍继承此污染**——这恰恰说明该比较在本数据上任何方向都不可信，进一步削弱把它当卖点的做法。
- 可修复：重测计时、按 fast_p 门控剔除病态父、把 autotune 显式从目标里剥离后重跑。

**[M3]【动机效度 / 稻草人】对 CUDA Agent 的"错误坐标系"批判建立在读不到正文之上**
- 位置：§1.1 L58–62；§1.2 L67–81；§3.1 L257–262。
- 片段：*"the full text of [1] is not retrievable from the preprint server and the internals are not released"*；摘要式断言系统 *"attach a single execution-level reward at the end of a trajectory of thousands of decisions"*。
- 为什么重大：论文的假想敌是"CUDA Agent 式 token 级坐标系"，却承认其轨迹/逐 token critic/advantage 全部不可得，敌人脸谱是从摘要＋已释出配方**重构**出来的。虽然 §1.1 已 hedge，但"黑洞"叙事与"坐标系之问"的针对性依赖于一个**未经核实的对手刻画**。真正可核查、且更贴近靶心的反例，论文其实已在 §6 引到——CudaPerf [15]"把 CUDA 结构放进 reward 而非 action"——它才是可立为标靶的公开对象，却被降级为 related work。
- 可修复：把批判对象从"读不到正文的 CUDA Agent"改为"可读、且确实把结构塞进 reward 的系统（CudaPerf/CUDA-L1）"，稻草人风险随之下降。

**[M4]【可复现机制】"69/69 对账 + Stage 2.5 PASS"给出的是虚假保证**
- 位置：`verify_paper_numbers.py` L130–132；`STAGE2_5_INTEGRITY.md` L60–69、L104–105（把"把 bug 当洞见"判为 PASS）。
- 为什么重大：对账把论文数字与**生成它的脚本**逐项对齐到 3 位小数，因而**只能保证"论文没抄错"，不能保证"统计方法对"**——C1 恰是脚本内的方法学错误，被这套机制忠实"复现"并背书。声称守住的模式 6（bug-as-insight）在 §3.6 失守。
- 可修复：对账须至少包含一层"独立实现/公式正确性"检验（如用 scikit-learn 或解析式复核 MI/MM），而非只与自身脚本对齐。

### MINOR（次要）

**[m1]【测量效度】** §3.5/§2.5 用"never repeating the same diff"作为叙事卖点（Abstract L29、L248–249），与 C2 的恒真性冲突；应在正文而非仅 `STAGE1_5_REPORT.md` L57 标一句"删失上界"。论文 L327/L329 已近乎承认 5.78 为上界，但仍据此在摘要写"compression is real"，属轻度过度概括。
**[m2]【论证逻辑】** `WarpReduce`/`SharedMemoryTransform` 从不作 primary（L334–335、§3.5 L352–358）被同时用作"词汇表语言依赖性"的证据与"granularity 匹配数据"的证据，方向相反的两个解读并存，宜择一。
**[m3]【诚实性】** §3.7a 把 327×/69× 两条 Other 边的巨大加速既当"最大单步收益来自缺失动作"的证据，又自认"parent 病态，effect size 不可读作优化能力"（L409–410），自我拆台，宜删其一。
**[m4]【动机效度】** §5.4 两条"未测量"的个人经验观察（量化要求压倒质性、指令文件越长越不遵守，L585–599）被放进"Implications"，虽已标注未测量，仍是无证据的断言，宜下沉到附录或删除。

---

## 3. Ignored Alternative Explanations / Paths（被忽略的替代解释）

1. **标签↔加速相关可能是"标注器读了表面线索"的镜像。** 若同一模型在看 diff 时用 hunk 数、"是否删了某个 torch 中间算子"、"是否新增/移除 @triton.autotune" 这些**表面特征**决定 primary，那么 label 与 coarse structure 编码的是**同一批结构线索**——这正好解释 C1 中"二者 MI 相近、coarse 不过是 label 的机械基线"。论文的正面结论（标签非任意）与负面结论（不比结构好）其实同源于此，但都未考虑"标注器 = 结构线索探测器"这一替代解释。
2. **"压缩"可能纯粹是词汇表按定义粗。** 任意 55 项分 10 组，其香农熵天然 ≪ log2(55)；问 2.67 低不低，须对随机十分组基线（≈3.3）而非 5.78。论文从未给出随机十分组的熵基线，把 5.78 当作对照轴（fig1）是一种放大。
3. **赢家可能是 autotuning 而非"程序变换"。** 53/55 同改 autotune、多条纯 autotune 边即 10×+。一个不浪漫的替代叙事是：本语料里"成功边"主要是**超参搜索**成功；若如此，"变换是天然动作单元"塌缩为"跑配置搜索"，新颖性大减。§3.7b 已嗅到但把它当"方法学提醒"，未当成对**主命题**的替代解释来正面处理。
4. **回归均值 / 选择效应。** 边被定义为"child 比 parent 快 >5%"，即对测得的改善做了条件化；慢得离谱的父节点被"打败"是统计必然。这会同时抬高 `H(S)` 极端层与"变换有效"的表象，论文在 §3.7b 只提"计时口径"，未提"以改善为条件→回归均值"这一更基本的目标变量偏倚。
5. **论文真正的、可辩护的贡献其实是那两个"计划外负面"。** §3.7a（BackendChoice 缺口）与 §3.7b（autotune 纠缠），我在 `annotations.json` 中逐条复核为真（Other 5/5 为后端替换；53/55；多条 parent_fast_p<1）。它们**独立于 C1/C2/C3 成立**，是本文最耐久的内容——但它们是关于"这个方向有多难/词汇表会漏掉最关键动作"的论据，**不是**"坐标系该换"的证据。把它作为论文重心（而非 §3.5 的伪压缩）才是诚实的路线，这是一条被忽略的重新定位路径。

---

## 4. Missing Stakeholder Perspectives（缺失的利益相关者视角）

- **编译器/自调优实践者（Ansor/TVM）**：§5.3 承认"若已知空间上的工程化搜索＋代价模型已能找出 schedule，LLM 选变换必须补上代价模型没覆盖的部分"，并称这是"我知道的对该方向最强反驳"——却未给任何实证或机制性回应，仅列为"真有可能是假的假说"。最强对手被写下来又被绕开。
- **分层强化学习实践者**：§5.2 引 option-critic 的 policy necrosis / redundant coverage，恰会**预测**论文想要的正面结果是"不自动"的；但缺一个"固定词汇 vs 学习词汇"如何避开坏死的讨论。
- **基准正确性/Oracle 社区**：论文在 §5.5 大量引用 correctness-illusion [20]、oracle 变异分析 [24]、验证不存活 [22,23,25]，却**不用这些去给自己的数据把关**——即：语料里每条"成功边"的 correctness=true 是否可信？若 oracle 本身可被"数值响但结构错"的写法骗过，则 §3 的整个"成功优化"集合的可信度未审。这是"引了但没用在自己的地基上"。
- **盲标的人类 kernel 专家**：即 §3.4/§7.1 红线最初要求的角色，缺席，是全文最大结构空缺（见 C3）。
- **要真正跑 §4 的工程/资源方**：论文未估 §4 四条件的算力/人力/时间成本，仅说明"§3 无需 GPU"；对"可证伪"的承诺缺少可行性一侧的视角。

---

## 5. Observations (Non-Defects)（确实不是缺陷的观察）

以下不是为平衡而伪造，是我复核后确认的真实优点：

- **两项"计划外负面"是真的、且经我独立复核**：`annotations.json` 中 Other 全部为后端/库替换（cuBLAS / `x.matmul` / `F.scaled_dot_product_attention` / `F.mish`），autotune 53/55 同变，多条 >10× 边的 `parent_fast_p<1`（父节点比参考还慢）。`BackendChoice` 应升为一等动作这一论断，是我见到的一条结实、可迁移的发现。
- **"先冻词汇、后读 diff"且冻结仅依赖语料级元数据的泄漏控制**（`VOCABULARY_FROZEN.md`；23,731/22,007/**0**/1,724 的语言普查）是正当的，而且"公开轨迹全是 Triton、CUDA C++ 为 0"是一个真实、有用、且被如实上报的事实性负面。
- **§4 的四条件设计本身质量高**：Free-plan 控制、Oracle-action 终止分支、会"输且输了也有信息"的阅读树、以及 §4.5 控制 2"结构对齐的非语义基线"，是一套真正**可判定**的协议设计。
- **诚实的姿态（在 bug 之外）值得记功**：§1.5"无训练、无新算法、无搜索实验"、§7.5"标题故意是问句"、拒绝提前造 DSL、§4.4 的"哪项判断会被改变"自检，都是对的方向感；只是这把尺子应先量 §3.6。
- **一个对作者有利、但被写反的事实**：把论文自己的同基数/保留边际检验**对称地**用于两个描述符（而非只用于标签），结果其实是**语义标签显著（p=0.0003）、粗结构不显著（p=0.75）**——即论文的工具在正确使用下，能给出一句**站得住的正面**陈述（标签绝非任意、且优于同粒度随机），只是作者把它写成了反面的头条。修复方向是"改结论"，不是"补实验"。

---

## 附：复算方法与可核验输出

- `tmp/attack_c.py`（seed 20261002）：先逐位复现 `calibrate_informativeness.py`（得 primary 0.6152 / coarse 0.7283，与 `informativeness_calibration.json` 完全一致），证明我读到了同一份数据；随后计算 (a) 标准 MI 偏差式、(b) 占用格精确 MM、(c) 对粗结构补做同基数与保留边际两种零分布。关键输出：correct-MM → 语义 0.4515 vs 粗 0.3101（翻转）；coarse 在自身 k=24 均衡随机零分布中处 **3.40 百分位**（观测 H 高于随机均值）；公平置换检验 label p=0.0003、coarse p=0.7496。
- `tmp/attack_a.py`（seed 20261002）：确认 55/55 指纹唯一（`distinct_diffs` 列恒等）、`H(指纹)=log2(55)=上界`、55 边/45 任务下所有标签 distinct_tasks 落在"随机重排任务"95%CI 内、0/55 单标签、多区域边 hunk 均值 4.07 vs 单区域 1.12、53/55 autotune。
- 两脚本仅读取仓库既有数据文件、不写入任何已跟踪文件；输出用完即随 `tmp/` 一并删除。

---

## 最终判决：这篇论文最容易被一句话打死的地方

**§3.6——被作者与其完整性报告共同封为"最 valuable、最诚实"的那条自我反驳——是一个算术 bug 的产物：`calibrate_informativeness.py` 的 Miller–Madow 项漏乘 `(k_Y−1)=3`，且粗结构从未获得标签所享受的同基数/保留边际校准；补上正确的互信息偏差、或改用不依赖 MM 的公平置换检验，头条结论即从"语义不优于表面结构"翻转为"语义标签显著（p=0.0003）、粗结构与随机重标无异（p=0.75）"。** 而 `verify_paper_numbers.py` 把这个错误数值断言到小数点后三位、Stage 2.5 据此判 PASS，使论文最强的两套信任机制对该错误完全免疫。

一句话杀招：*"你引以为豪的那个负面结果，是一个漏乘 3 算出来的；算对了，它就不负面了——于是摘要、结论、以及'我们敢于自证其反'的整张诚实牌，一起塌。"*

次要但同样足以致命的一句话（若作者回避 C1）：*"§5.1 用来区别 edit-based RL 的三个'可测量而非修辞'的指标，一个恒真（指纹全唯一→distinct_diffs≡出现数）、一个抽样伪影（1.22 边/任务→distinct_tasks≈随机）、一个层级错配（0/55 边是单标签，54.5% 多区域无法把'一动作多位置'与'多动作各一处'分开）。"*
