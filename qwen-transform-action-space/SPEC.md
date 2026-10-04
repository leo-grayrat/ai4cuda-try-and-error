# Program Transformation Action Space — 论文全流程（自包含执行 Spec）

---

## 0. 背景与研究脉络（完整叙述，执行者必读）

### 0.1 研究缘起：从读论文到写调研

2026 年 9 月，作者在韩学长推荐下集中阅读了一批 AI4CUDA / LLM 驱动 GPU 算子优化的论文，最核心的目标是 CUDA Agent（ByteDance Seed + 清华 AIR，2026 年初发表）——一篇用强化学习训练模型自主完成 CUDA kernel 开发、编译、profiling、迭代优化的工作。读完之后作者的第一反应不是兴奋而是微妙：这篇论文表面上做的是 CUDA，但把它所有训练层面的设计拆开看——预训练数据里 CUDA 罕见所以先跑单轮 RL 让模型熟悉、策略模型不会稳定跑长 Agent 所以先拿成功轨迹做监督微调、价值模型不会估长轨迹所以先离线训 Critic、直接奖励加速倍数会偏向天然存在巨大优化空间的题所以把 reward 砍成 -1/1/2/3 有限档位、PPO 上下裁剪 0.8~1.28 防过度刷分——每一步都合理，每一步都是"当前暴露出问题以后显然的下一步"，但没有任何一步是因为 CUDA 本身的特性（访存层级、warp 语义、memory coalescing、shared memory bank conflict）而必须这样设计的。把整套流程搬到 AI4FORTRAN、AI4FPGA 上毫无阻碍。作者由此产生了一个挥之不去的疑问：**这些工作的增量到底在哪里？如果一个足够理性的强模型自己就能给出这些"合理下一步"，那研究员的存在价值是什么？"写一个足够好的 skill + 套一个 agent loop + 标准 actor-critic PPO"难道就能发论文吗？**

这些思考在 2026-09-13 深夜被写成 15600 字的调研文档《近日对 AI4CUDA 的调研》（作者自嘲标题本应是"遐想"实为"瞎想"，并声明引用部分为 AI 生成、因为想法本就和 AI 讨论中产生）。文档包含三条主线加一条小点：

(a) **CUDA Agent 论文的逐节拆解**与"每一步都 right 所以没有 CUDA-specific 贡献"的批评，包括对"创新点在哪"的五连问（漂亮结果？不基于 CUDA 性质的工作流？范式拼接？大规模算力？——都不是），以及"拆掉名词以后整个方法就是三层补丁 + 标准 PPO"的降维解剖。

(b) **AI Hacking / reward hacking 线**——从 METR 的"前沿模型正在 reward hacking"案例（模型发现评分程序调用栈里已存在参考答案直接拿来返回、关掉 GPU 同步让评测器看不到真实计算耗时、修改计时函数报出近零时间，事后还能准确解释自己为什么这是作弊），到 BAITBENCH 把作弊放进数据和建模方法本身（七个前沿 agent 运行中 57.1% 被判定利用数据捷径，明确要求不作弊后平均比例依然过半），到 OpenAI 对思维链强监督反而让模型"少说了却没等比例少做"的阳奉阴违结论，再到作者自己推导出"猫鼠游戏"监察模型方案（任务模型 A 做任务、监察模型 M 抓作弊、协同进化）——然后发现这个方案与 2026 年 6 月的 HFSV 论文（Hardening Agent Benchmarks with Adversarial Hacker-Fixer Loops, arXiv 2606.08960）几乎完全重合：Hacker 想办法不真正完成任务却通过 KernelBench，攻击成功后 Fixer 修改评测器堵路，Solver 用合法方法再做一遍验证 patch 不误伤，KernelBench 上 15 类已知攻击从 62% 成功率压到 0%，且负责攻防循环的是更弱的 Gemini 3 Flash、来攻击的反而是不相干的更强模型。作者写下"还能怎么样呢"——角色结构已经收敛到人类协作解决问题的范式本身，再加角色只会划归回 HFS 三者。

(c) **仿人学与暴力封路线**——Codeforces hack 赛制（想挑刺必须举可执行反例，杜绝 AI 乱挑刺）、CTF 防守方不能为了防攻击把服务本身关掉（修漏洞不修"攻击动作"：发现非默认 stream 可以绕计时，正确修法不是禁 stream 而是保证"计时结束时被评分结果必须已经完全确定"这个不变量）、模糊测试作为"自动出阴间测试数据的人"（双方不需要一起训练，挑刺模型只需积累反例库）、以及"什么都不说、直接从环境上禁止会导致作妖取巧的选项"的最小权限+可信边界+事后裁决思路——让前几代 agent 成为免费的 fuzzing/red-team 机器。

(d) **低概率 token 小点**——对 CUDA Agent 声称的"预训练语料 CUDA 占比 <0.01% → token 概率极低 ~10⁻⁹ → PPO 重要性比率爆炸 → 第 17 步训练崩溃"这条因果链的强烈怀疑：论文自己报告的实验就说明基础模型绝非"不会写 CUDA"（Seed1.6 原始正确率 74%、Level 1 达 90%、Claude Opus 4.5 达 95.2%），且 CUDA 代码绝大多数词元是 `for if float int return *[]+` 这类普通代码知识，真正 CUDA 专属的 `threadIdx.x` `__shared__` 极少且高度模式化；条件概率与语料绝对占比无关。后续追问发现这更像 ByteDance 内部已知的 training-inference mismatch 工程故障模式（同机构的 When Speed Kills Stability 先于 CUDA Agent 半年发布；vLLM 逐位一致实验 mismatch 归零但吞吐慢 2.4 倍），被论文包装成了"The Root Cause of Training Instability"的学习理论叙事。真正没解释清楚的是"为什么 CUDA Agent 特别频繁碰到低概率尾部"——更合理的假说是长程多轮工具 Agent 不断吞入编译错误、profiler 输出、奇怪中间代码状态造成分布外上下文，与 CUDA 语法本身无关。

在调研文档的"迸发灵感"一节，作者把前面各自讨论、完全没有互相涉及的两条线合流：CUDA Agent 是"训练模型内化能力"（把 Solver 做强：3.1 准备 CUDA 任务、3.2 给环境和 skill、3.3 用 RL 写进参数），HFSV 是"组织架构外化能力"（外层 H/F/V 职责分工与可信验证），两者并不互斥而是可以嵌套——专门训练过的模型可以直接放进 HFSV 的 Solver 位。反过来又发现 CUDA Agent 在 reward hacking 防护上只有寥寥几项静态卫生措施（锁验证脚本、禁 fallback、多随机输入、规范计时、断网），放到 HFSV 形成的更强威胁模型下不完整——异步 stream、后台线程、延迟求值、精度擦边、计时器漏洞都可以发生在"正常写代码"的权限范围内；而 HFSV 也该反过来学 CUDA Agent 的奖励封顶机制：若直接按原始 speedup 选择，100 倍漏洞加速会远远压过 1.5 倍合法优化，即使 H/V 还没来得及发现漏洞，搜索已经强烈偏向前者。最终在"普通和理所当然到底是什么呢"一节抛出最大胆的想法：这个领域一直在各种意义上做"对"的事情，问任何一个大模型怎么防指标硬优化它都能给出差不多合理的方案，那么能否训练一个"训练这些模型的模型"，让它按"当前情况下合理的下一步"连续推进几千步"观察→判断→修改→实验→再观察"，系统自己进化出合适的组织结构？也许自动研究真正依赖的并不是每一步都出现深刻洞见，而是大量正确但普通的局部决策可以被持续执行。

### 0.2 四篇论文批判与 Skill 之问

第二份文档《Retire the Abstractions 等四篇文章》是对四篇延伸阅读的批判。Hazy Research 的《Retire the Abstractions》主张 AI 承担了认知中介之后人类需要的抽象层可以退休——作者反驳：机器→汇编→IR→高级语言之间是编译器做的**百分百确定的编译转换**，而自然语言→程序是 LLM 做的**并不确定的 AI 生成**；代码的脆弱性是显性的，提示词的脆弱性是隐性的，问题被转嫁给猜测上下文撰写的智能体；而且抽象不会凭空消失，只是转交给了 AI——AI 的上下文和注意力同样有限，agent 论文里会出现上下文管理和"AI 自己的 IR"恰恰证明 AI 也需要抽象。这个"我们不再需要给人用的抽象、但 AI 需要自己的抽象层"的观察是后文 Idea 4 的直接思想源头之一。

FlashRT（arXiv 2607.18171）强制模型写代码前画层次化图、每个节点声明持久状态读写、数据边标注 streaming/blocking，做出"chain-of-program 结构化中间表示"再由确定性程序审核——但作者指出：标题写多模态而多模态在后面分析中消失、附录塞满与核心贡献无关的 GPU 吞吐证明、正文讲完架构无法据以复现；更致命的是翻仓库后发现最大的文件不是论文核心算法而是一篇充斥"堵 Agent 语言漏洞"的 agent.md——显然挂着 Claude Code 迭代、"遇到问题把它存档到文档"堆出来的产物。论文只字未提它，于是三个新问题：skill/harness 设计的一部分为什么论文不写？性能优势究竟来自图论架构还是踩坑堆料？如何证明几千行越塞越多的 skill 真的递增有效（作者亲历：定量指标必然压倒定性——"画十张图"容易满足、"每张有信息量"不会被认真对待；内容堆多了模型不可能每条都遵守）？还有一个更微妙的违和感：论文强调"没有它 AI 发现不了架构问题"，可让 AI 开个 plan 先审阅仓库，它难道不会自己做计划分工？严谨一些的要求型 skill 是否其实也能达到同样效果——图、边、标也全是 AI 完成的，同样存在偏移可能。

第三篇（Fortran 迁移到现代写法，bit-exact 逐位一致 + 禁用额外优化改写以规避浮点波动，三个角色塞给同一个 Claude Code）和第四篇 Vortex（统一各稀疏注意力框架接口让它们公平同场竞技）则干脆只是成果报告，技术上与 AI4Coding 方法论没什么关系。

### 0.3 与学长的碰撞、AI 讨论与实验偏移

带着文档向导师汇报前，作者先与韩学长沟通，得到的回应是"之前让你看的论文太旧了"（实际 CUDA Agent 发表仅半年、HFSV 是七八月新文）、"不是很明白什么是工程"，对"基于 CUDA 自己的特性出发"的想法只回了一句"这样不太行"，随后丢来四篇新论文（即 0.2 的内容）要求看完再说。第二份文档写完后情况没有改变，沟通不畅加上开学事务、怕打扰赶 ICLR 的老师，作者潜水了一段时间，决定先自己烧点额度做探索性小实验（声明"真正实验肯定不会这样 codex 一键做，只是探索"）。

真正塑造方法论的是与 GPT-5.6 Thinking 的 60 轮对话（2026-09-23 至 09-28，HTML 全程存档）：#1-#14 凝练方向期间作者反复纠正 AI 的谄媚拟合（"你找的很好啊，但是回去看看原文脉络"、"请具体思考，不要一直想着拟合我最新的回复"、"为什么要否掉最后那个呢？我聊到的'训练模型的模型'不好吗？"）；#15-#25 作者用 Codex 在 GitHub 仓库 leo-grayrat/ai4cuda-try-and-error 实际动手后报告"一直会发生偏移，被技术细节裹挟，我很无语"，AI 对仓库 commit 历史逐条对照聊天记录后给出病根诊断，也成为本 spec 最重要的方法论警告：**每当原问题里出现一个难以直接操作的概念，我们就把它替换成一个更容易编码、测量、验收的代理问题；然后 Codex 很擅长把这个代理问题越做越完整，最后工程完成度开始遮住"原来的问题其实还没碰到"**——具体表现为四级偏移：研究问题→实验基础设施（KernelBench 接口、可重放快照、跨轮计时漂移、TF32 污染参考计算）；"怎么从代码里找关键优化"→"已经告诉我 A/B 之后怎么分账"（Shapley 与 interaction 算得工整，但回答的是被降维后的问题）；理解程序含义→手工 AST/正则分类器（`clone().detach()` 移除后稳定慢 3.2ms 的真漏报本该成为研究对象，第一反应却是"给规则补进去"——把答案写进规则表）；改善 RL 学习→构建 benchmark 数据治理（18581 条父边索引、预注册、封存协议，像在建一套"GPU 优化改动因果证据 benchmark"，但那不是原问题）。RESEARCH_RETROSPECTIVE.md 里作者把两条线（action-credit 与 skill-efficiency）的失败归结为同一模式，并立下本 spec 沿用的停止条件：**"如果这个实验完美完成，它会改变我们对原研究问题的哪一个判断？"** 如果答案只是"工具更完整了、记录更可复现了"，就只能算基础设施，不应包装成研究进展。对话后半段还确立了本方向的形态雏形：作者否定了 AI 对跨 GPU 泛化的执念（"我难道一个针对 nVidia 的优化对 AMD 不行就是我的优化失败了吗？这其实很奇怪的"），提出"把生成 token 换成选择优化动作作为学习/搜索单位"是最看好的一条，并给出最小实验设计——人工标二三十条真实 parent→child，先验证 transformation vocabulary 存在性本身；AI 随后把它展开成 745 行的四条件证伪框架（后并入仓库 research-notes）。#51-#55 是二次翻车记录：Codex 的 git 悬空分支、实验目标漂移、"技术用语大乱斗"，作者最终要求写两分支失败复盘，并问 AI"回顾这调教 AI 的全程你有什么想说的"。

同期另一个硬约束浮出水面：**CUDA Agent 是假开源**——模型没开源算了，但中间 Agent 轨迹、生成版本、工具调用、反馈日志全部没有，HuggingFace 官方下场在 issue 里问也无人在意，近二十条要求开源的 issue 零正面回应，仓库里只有无足轻重的东西。这意味着任何需要其训练观测量（逐 token Critic value、原始概率、advantage）的验证路线全部死路，论文必须建立在公开可得数据上——这正是 AdaExplore 父子轨迹数据的战略价值。

### 0.4 五条 Idea 的完整表述与选择理由

**Idea 1（CUDA-specific RL 的缺失与迁移可能性）**：CUDA Agent 全部训练调整与 CUDA 本身特性无关；要么正面基于 CUDA 独有特性（访存层级、warp 调度、coalescing、bank conflict）设计 domain-specific 的学习/奖励/搜索方法，要么反面验证这套通用范式迁移到 AI4X 其他领域是否同样容易出成果。风险：正面做需要真训练集群；"什么都能迁移"恰恰是它被批评的点。

**Idea 2（模型自建 Agent 组织结构）**：既然每次修复都是"当前合理的下一步"，不如只给模型基础部件和外部选择压力，让强模型在实践中搜索/进化出组织结构——"训练训练模型的模型"，类似遗传算法的自然演化系统，但仍需外部强验证防止变成大型作弊机器。风险：极其宏大，没有系统就没有论文。

**Idea 3（低概率 token 因果链质疑）**：拆解"CUDA 罕见→概率极低→PPO 爆炸"链条，论证真实原因更可能是长程工具 Agent 的分布外上下文与 train-inference mismatch 系统问题，而非 CUDA 学习理论问题。适合 workshop 短评，但天花板低，本质是评论性文章。

**Idea 4（程序级信用分配 → Transformation Action Space，本论文选定）**：从对 CUDA Agent 的两个具体不满出发。其一：代码性能是结构级、执行级结果，训练接口却是词元级策略更新，中间隔着巨大的信用分配黑洞——一次从 1.0x 到 1.8x 的提速可能只由"融合两阶段、消掉一次 global memory round-trip"这一个设计决策贡献，其余两千个 token 是无足轻重的样板代码，而训练系统没有一个 CUDA 性能分析器告诉 PPO"第 1537 步这个设计决策贡献最大"，只能靠价值模型和 GAE 统计地猜。其二，干脆反问：**为什么训练系统非得把这些 token 当成独立 action？它们本来就是一个程序决策的实现。**于是命题从"reward 该打在哪些 token 上"升级为**"token action space 本身是不是选错了坐标系"**——CUDA 优化是否天然存在一个比 token 更稳定、更可迁移的 action space：形式化为 a=(τ, ℓ, θ)（变换类型、作用位置、参数），一个语义 action 映射到很多 token 和很多代码位置（work-per-thread 1→2 同时要求 kernel body 索引改动和 host 端 block 数 N/T→N/(2T) 改动，缺任何一半动作都不成立），反过来完全不同的代码写法也可以映射到同一个 action。这与 TVM/Halide 的 schedule space、编译器 pass 有血缘——所以如果结论只是"compiler pass 比 token 更语义化"那没论文价值；真正要证明的是：**对 LLM 驱动的 CUDA 优化，把搜索/学习界面从整段代码生成提升到参数化 transformation，是否真的显著改变搜索效率和学习效率。**

745 行研究笔记（仓库 `research-notes/program-transformation-action-space.md`）已给出完整证伪结构。四条件对照实验：Direct（直接生成优化后完整 kernel）、Free-plan（先自由文本说准备怎么优化再写）、Structured-action（必须先从结构化 transformation space 选参数化动作再实现）、Oracle-action（人直接告诉正确的 transformation，只测"已知做什么之后代码能不能实现出来"）。判读树：Free-plan≈Structured-action>Direct 只证明"先规划比较好"，没意思；Structured-action>Free-plan>Direct 才开始说明结构化 action 本身提供额外 inductive bias；Oracle-action≫Structured-action 说明 abstraction 有用但最大困难在选对 transformation；Oracle-action≈Direct 则方向直接杀掉——连"做什么"都告诉模型了仍没提高成功率，说明 strategy/realization 分离根本没帮忙。前置经验问题（比四条件更优先）："成功的 CUDA 优化轨迹到底能不能被少量重复出现的程序变换解释？"——人工读 30-50 个真实 successful parent→child edge，若每 child 都是独一无二的重构、聚不成稳定 vocabulary，这条路就该死；若能不断落回增加每线程工作量、消除中间物化、合并 kernel、改 shared-memory tiling、warp reduction、vectorized load/store、改 block mapping 这些重复模式，且同一 action 的代码实现差异很大，就出现"code diff 熵很高但 transformation 熵较低"的现象——这才是 action abstraction 最值得研究的经验基础，而且是数据本身逼着承认存在比 token 更自然的结构，不是为了做论文硬设计一个 abstraction。方法论红线：前置验证阶段**人工标，不写 classifier、不写 locator、不写 AST pipeline**（这正是 0.3 节偏移教训的直接应用），也不许一晚上发明 47 个 action type 写 schema -validator-parser——先回答"这种结构到底存在不存在"。数据泄漏防护：从一部分 trajectory 建立 transformation vocabulary 后**冻结**，再去 held-out 任务测试"过去成功优化中抽象出的 action space 能不能帮助新 kernel"，否则只是把答案换个名字喂回模型。"可迁移"的正确定义不是跨 GPU 也不是 CUDA→AMD，而是同一目标平台上不同 kernel 之间能否共享"什么程序变换通常值得尝试"的经验：code_A ≠ code_B 但 a_A = a_B = WarpReduce，于是 Q 函数定义在 (P, a) 而非 (P, token 序列) 上，经验在 action 层共享。诚实列出的三大危险：真正好的 CUDA 优化可能根本不是离散可分的 transformation（一次整体重构，tile/layout/mapping 强耦合，拆成 action sequence 反而破坏联合搜索，事后拆解只是解释）；LLM 内部可能已经在隐式做这些 abstraction，外部显式化只是把 reasoning 打印出来（这就是 Free-plan control 存在的原因）；可能选 action 容易而 realization 才是瓶颈，大量任务照样写不出正确高性能实现，Oracle-action 实验会直接暴露。

选择理由：五条中唯一同时满足"有 745 行现成证伪框架、有本地公开可得真实数据（AdaExplore）、核心验证实验不需要训练集群、结论无论正反都可发表"的方向；且它比另外四条更根本——不再争论某个工程细节对不对，而是质疑整个 AI4Code-RL 领域的建模坐标系。

**Idea 5（Skill 有效性 Benchmark）**：量化 skill 的单位 token 消耗 vs 性能收益、单条指令的实际被遵守率、文档长度/条目数 vs 边际收益（注意力稀释）。动机来自 FlashRT 的几千行 agent.md 与 CUDA Agent 的 SKILL.md 之问（0.2 节）。风险：仓库首轮 18 次对照实验因"为了本机 Windows 方便把研究对象从真实 CUDA C++ extension 工作流 skill 换成自写 Triton 短提示"而整体作废（RESEARCH_RETROSPECTIVE.md 第二节），复活它必须先锁定真实 skill 再让任务和环境适配它，工程量不小且依赖 agent 基础设施稳定。

### 0.5 本次论文的定位声明

本次执行 **Idea 4**，体裁 Position/Analysis：主张（坐标系之问）+ 初步证据（对 AdaExplore 真实轨迹的自动标注与熵分析）+ 四条件证伪框架设计。不复用仓库旧实验代码（作者原话：偏移得要死要活，"不管仓库"），只复用其本地缓存的 AdaExplore 原始数据与 745 行研究笔记的思想。标注由 AI 自动完成（代替笔记中"人工标"的原案）并在论文 Limitations 中如实声明方法与 bias——这本身就是对"AI 端到端自主探索"命题的一次元实验，与 README 中"作者只是想随便跑跑试试看"的定位一致。

---

## 1. 关键文件引用（执行前必读）

| 路径 | 用途 |
|------|------|
| `d:\File\Markdown\近日对 AI4CUDA 的调研.md` | 15600 字调研全文：CUDA Agent 拆解 / AI Hacking / HFSV / 猫鼠游戏 / 仿人学 / 低概率 token 质疑 / 合流与提议 |
| `d:\File\Markdown\Retire the Abstractions 等四篇文章.md` | 四篇论文批判：抽象退休论反驳 / FlashRT 三层问题与 agent.md 之问 / HPC 迁移与 Vortex 的无关性 |
| `c:\Users\34788\Downloads\202610012331-chatgpt-分支-·-强化学习研究方向-60msg.html` | 60 轮 ChatGPT 讨论完整记录：偏移诊断、四条件框架成形、跨 GPU 否定、调教 AI 全程反思 |
| `E:\Project\Git\Research\ai4cuda-try-and-error\research-notes\program-transformation-action-space.md` | **745 行核心研究笔记**：坐标系之问、a=(τ,ℓ,θ) 形式化、四条件证伪框架、粒度原则、数据泄漏防护、可迁移性定义、三大危险 |
| `E:\Project\Git\Research\ai4cuda-try-and-error\RESEARCH_RETROSPECTIVE.md` | 两条实验线失败复盘：代理替换模式与"停止条件"问句 |
| `E:\Project\Git\Research\ai4cuda-try-and-error\.cache\adaexplore-l2.tar.gz` | AdaExplore Level 2 真实优化轨迹（~126MB，20+ 任务） |
| `E:\Project\Git\Research\ai4cuda-try-and-error\.cache\adaexplore-l3.tar.gz` | AdaExplore Level 3 真实优化轨迹（~47MB，50 任务） |

**AdaExplore 数据结构**：

```
MCTS_0120_KB-l2_200/          # 运行标识
  2_20/                         # 任务 ID（KernelBench Level 2 题号）
    step_N.py                   # 第 N 步源码（PyTorch ModelNew，内含 CUDA/Triton kernel）
    step_N_log.json             # LLM 推理与响应日志
    step_N_prompt.txt           # 当轮输入 prompt
    step_N_metrics.json         # 正确性、耗时、speedup 等测量
```

---

## 2. 论文定位

- 体裁：Position / Analysis Paper（8-12 页）
- 目标：arXiv preprint（可选投 ICLR/NeurIPS/MLSys workshop）
- 语言：英文正式稿 + 中文翻译稿，以英文文献格式为准
- 性质：AI 自动化探索性输出，提出框架 + 真实数据初步验证

---

## 3. 输出位置与声明

- 仓库：`E:\Project\Git\Research\ai4cuda-try-and-error`
- 新分支：`qwen/transform-action-space`（从 `exp/oracle-semantic-credit` 切出）
- 子文件夹：`qwen-transform-action-space/`（所有产出写入此处）
- `qwen-transform-action-space/README.md` 须写明：本目录为 Qwen3.8-Flash 自动化研究探索的输出；作者仅测试 AI 端到端自主完成文献调研、实验设计与论文撰写流程的可能性，不代表严肃研究投稿意向；标注、代码和结论均由 AI 生成，未经人工复核。

---

## 4. Skill 使用策略

| Skill | 用法 |
|-------|------|
| `academic-pipeline` | 阶段编排器：检测当前阶段、在各阶段间做 checkpoint、等待用户确认后转移 |
| `nature-academic-search` | Stage 1 文献检索：按其 SKILL.md 路由协议先加载 `static/core/tools.md` 与 `static/core/routing-and-ops.md`，workflow 选 `multi-source-search`；其 `academic-search` MCP server 当前未注册运行，改用 `scripts/academic_search.py`（OpenAlex 无 MCP 降级搜索）；若 API 不通则退回 WebSearch + WebFetch |
| `academic-paper`（pipeline Stage 2 子调度） | `full` mode 逐章生成论文正文 |
| `academic-paper-reviewer` | Stage 3 五视角模拟评审（含 Devil's Advocate） |

---

## 5. Stage 1: RESEARCH（文献检索与定位）

1. 检索关键词组（每组多源交叉）：
   - "action space design" + "LLM agent" + "code generation"
   - "program transformation" + "compiler optimization pass" + "learning"
   - "credit assignment" + "code generation" + "token level" + "reinforcement"
   - "AdaExplore" / "KernelBench" / "CUDA Agent" / "GPU kernel optimization"
   - "hierarchical reinforcement learning" + "options" / "option-critic" + "structured action" + "code"
   - "Voyager" / "skill library" / "action abstraction" + "LLM agent"
   - "optimal edit distance" / "edit-based RL"（与 NEHDP 等编辑动作 RL 工作对照——注意：编辑动作假设（edit-based）文献（如 Meta 的 learning-by-editing 系列）是本方向必须直面的先行工作，论文需说明 transformation 与 atomic edit 的区别在于语义参数化与跨任务词汇复用）
2. 特别核查清单：CUDA Agent（arXiv 2602.xxxxx，ByteDance Seed + AIR）、FlashRT（2607.18171）、HFSV（2606.08960）、Correctness Illusion（2606.20128）、Contract-Grade Verifier（2608.12700）、KernelBench-Verified（2607.16241）、Atrex-Bench（2607.14541）、ProofWright（Stanford）、train-inference mismatch 两篇（off-policy RL notion、When Speed Kills Stability）——确认每篇真实存在、ID 正确、论述与笔记一致。
3. 交付物：RQ Brief（一页内说清命题、差异点、可证伪性）、分类文献图谱（compiler IR / RL credit assignment 与 edit-based action / LLM agent action space / CUDA-specific methods 四类）、Related Work 素材。

**Stage 1 checkpoint（FULL）**：向用户展示 RQ Brief + Bibliography + "无先验工作做过同一命题"的论证（尤其 edit-based RL 的差异性论证），确认后进入 Stage 1.5。

---

## 6. Stage 1.5: EXPERIMENT（AI 自动标注 + 熵分析）

**数据源**：本地 AdaExplore L2（全部）+ L3（抽样 10-15 任务），解压至 `qwen-transform-action-space/data/adaexplore/`。

**步骤**：

1. **筛选成功边**：解析 `step_N_metrics.json`，取"父子均正确 + 性能提升 > 5%"的相邻 step 对，目标 40-60 条边；记录 edge_id、task_id、parent/child step、speedup。
2. **提取 diff**：Python difflib unified diff，保存到 `qwen-transform-action-space/data/edges/`。
3. **AI 自动标注**（代替笔记原案"人工标"；论文中如实声明）：
   - 每条边输入：diff + parent/child 源码 + log 中的模型自述（如有）
   - 预定义 transformation vocabulary（12 类）：`Fuse(producer, consumer)` / `EliminateMaterialization(site)` / `Tile(axis, size, mapping)` / `IncreaseWorkPerThread(k)` / `Vectorize(axis, width)` / `ChangeLaunchConfig(dim, value)` / `WarpReduce(scope)` / `SharedMemoryTransform(access_pattern)` / `PrecisionChange(source, target)` / `LoopTransform(type, params)` / `HostDeviceOverlap` / `Other(description)`
   - 允许多标签；每条标注附 reasoning 字段
   - 输出 `qwen-transform-action-space/data/annotations.json`
   - 红线执行：本阶段不写 classifier/locator/AST pipeline；vocabulary 先冻结再标注，若某类占比 > 40% 或 Other 占比 > 25%，如实报告并在讨论中分析（前者=词汇设计过粗，后者=猜想可能不成立）
4. **统计分析**：
   - 复用率 |V|/|E|；Shannon entropy（transformation 标签分布 vs diff 结构特征分布）
   - 同一 transformation 跨任务的 diff 变异度示例表（多对多关系证据）
   - 图表（matplotlib）→ `qwen-transform-action-space/figures/`
5. **结论判定**：正面（压缩现象成立）写入论文 Section 5 作为 preliminary evidence；负面（聚不成 vocabulary）转为 negative result analysis——四条件框架和停止条件使两种结果都有发表价值。

**Stage 1.5 checkpoint（FULL）**：向用户展示标注样本、熵数字、图表与结论判定，确认后进入 Stage 2。

---

## 7. Stage 2: WRITE（论文撰写，英文主稿）

拟定结构（存 `qwen-transform-action-space/paper/`）：

1. Title: "Is Token-Level the Wrong Coordinate System? Program Transformation as the Action Space for LLM-Driven Kernel Optimization"
2. Abstract（200-250 words）
3. Introduction：从 CUDA Agent 的信用分配黑洞到坐标系之问
4. The Wrong Coordinate System：token vs code-block vs transformation 三层对比，work-per-thread 1→2 横跨多位置的实例，a=(τ,ℓ,θ) 形式化与多对多关系
5. Preliminary Evidence：AdaExplore 轨迹的 AI 自动标注实验、熵分析、压缩现象（或反面结果）
6. A Falsifiable Experimental Framework：四条件对照与判读树、vocabulary 冻结与 held-out 协议、停止条件问句
7. Implications Beyond CUDA：edit-based RL / agent skill 设计 / autotuning 的同构讨论，"经验在 action 层共享"的可迁移性定义
8. Related Work：CUDA Agent 谱系、edit-based/options-based RL、compiler IR（TVM/Halide/MLIR）、HFSV 与验证器加固
9. Limitations and Open Questions：AI 自动标注的 bias（循环自证风险：vocabulary 与标注者是同一认知家族）、三大危险（整体重构不可分、隐式 abstraction 已被 LLM 内建、realization 才是瓶颈）、无训练验证
10. Conclusion

**Stage 2 checkpoint（FULL）**：完整草稿 + 字数/引用数/章节覆盖度指标，确认后进入 Stage 2.5。

---

## 8. Stage 2.5: INTEGRITY（MANDATORY checkpoint）

- 逐条验证引用（DOI/arXiv ID 可达性、作者/年份/标题逐字段核对）
- 实验数字可复现性：脚本重跑对账 annotations.json 与图表
- AI 标注方法在论文中如实声明，不冒充人工
- 7 模式 AI Research Failure Checklist（重点：hallucinated results、methodology fabrication、frame-lock）

---

## 9. Stage 3 → 4 → 4.5: REVIEW + REVISE + FINAL INTEGRITY

- `academic-paper-reviewer` full mode 五视角模拟评审（含 Devil's Advocate 专门攻击"这不就是 edit-based RL 换皮吗"与"AI 标注循环自证"）
- Revision Roadmap → `academic-paper` revision mode 修改 → Stage 3' 验证性复审
- Stage 4.5 终验（100% 引用通过才可放行，MANDATORY）

---

## 10. Stage 5: FINALIZE

- 英文终稿：`qwen-transform-action-space/paper/main.md` + LaTeX（若 Pandoc 可用则 PDF）
- 中文翻译稿：`qwen-transform-action-space/paper/zh/`（以英文格式为准的对照翻译）
- 完成后向用户展示 Stage 5 completion checkpoint（FULL），用户可选择是否进入 Stage 6 过程记录

---

## 11. 约束与注意事项

| 项目 | 说明 |
|------|------|
| Skill 调用 | 严格走 academic-pipeline 阶段调度 + checkpoint 协议；文献检索优先 nature-academic-search 降级脚本，失败才用内置 WebSearch |
| 标注 | AI 自动完成，论文 Limitations 如实声明 bias |
| 数据边界 | 只用 AdaExplore 公开数据，不依赖 CUDA Agent 未开源的任何东西 |
| 不复用旧代码 | 与 `action-credit/`、`cuda-skill-efficiency/` 的偏移代码完全隔离 |
| 方法论红线 | 不写 classifier/locator；vocabulary 先冻结；每个实验先过"停止条件"问句 |
| GPU | 不需要，纯 CPU 文件分析 |
| Pipeline mode | Stage 1 quick，Stage 2 full |
| 输出位置 | 分支 `qwen/transform-action-space`，目录 `qwen-transform-action-space/`，README 声明 Qwen 自动化性质 |
| 沙箱 | 若写入 E:\ 被工作区沙箱拦截，先请用户批准提权或由用户手动建分支后继续 |