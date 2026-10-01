# Stage 3 — Editorial Decision Letter

**稿件**：Is Token-Level the Wrong Coordinate System? Program Transformation as the Action
Space for LLM-Driven Kernel Optimization（`paper/main.md`, v1, 8,825 词, 28 条引用）
**场合**：arXiv preprint；MLSys / ICML / NeurIPS workshop position track
**审稿配置**：Journal-Fit Reviewer + R1 方法学 + R2 领域 + R3 跨学科（编译器/自动调优工程师）+ Devil's Advocate
**独立性**：五份报告互相不可见，各自落盘于 `stage3_review/review_*.md`（未改动稿件，READ-ONLY 全程遵守）

## 评分汇总

| 审稿人 | fit | originality | significance | rigor/method | clarity | overall | 推荐 |
|--------|-----|-------------|--------------|--------------|---------|---------|------|
| Journal-Fit Reviewer | 60 | 58 | 45 | 50 | 62 | **57** | Major Revision |
| R1 方法学 | — | — | — | 统计效度 **38** / 可复现 85 / 设计 60 | — | **48** | Major Revision |
| R2 领域 | — | 新颖性受损 | — | — | — | **52** | Major Revision（主会接收概率≈0.10） |
| R3 跨学科 | — | 58 | 48 | 可行性 **35** | — | **55** | Major Revision（系统向） |
| Devil's Advocate | — | — | — | 核心统计结论判定为 bug 产物 | — | — | 现况可被一句话打死 |

**编辑决定：MAJOR REVISION（一轮）。** 五份意见在「可复现性优秀、统计效度不成立、
新颖性被两篇已存在工作先行、框架不可自动执行」四点上一致或部分一致；
在「是否值得存在」上一致肯定——四条真实优点（自杀式 reading tree、预注册红线被真实执行、
两条计划外负面发现、三 claims 分离）使拒稿不是正确结论。

## 共识问题（≥3 位审稿人独立指出）

### C1 头条统计结论是代码 bug 的产物（R1 + DA 独立复算；EIC 附议）
`scripts/calibrate_informativeness.py` 的 Miller–Madow 项写成 `+(k_X−1)/2n`；
互信息的正确偏差项是 `(k_X−1)(k_Y−1)/2n`（本例 `k_Y=4` 四分位 ⇒ 漏乘 3）。
- 现行（错）：语义 0.615 < 粗结构 0.728 ⇒「语义不优于表面结构」= 论文自诩的诚实自查
- 修正后：语义 0.4515 > 粗结构 0.3101 ⇒ **结论反号**
- 且不依赖任何 MM 公式的公平置换检验（固定标签、置换 Y）：标签 p=**0.0003** 显著；
  粗结构 p=**0.7496** 不显著，观测 MI 甚至低于其自身粒度下的随机基线（24 格中 11 格为单点 ⇒ 过拟合）

**这是本轮最严重的发现**，且它同时暴露第二层问题（见 C5）。

### C2 恒真比较被当作证据（EIC + R1 + DA）
55/55 diff 指纹互异 ⇒ `H(fingerprint) ≡ log₂55 ≡ 最大熵`；`distinct_diffs ≡ occurrences`；
「没有类别产生过两次相同 diff」是**身份式**而非经验式。摘要、§3.5、§5.1、结论四处使用它。
更糟：公平对照（hunk 数熵 2.49、变更行数分桶熵 2.40）**低于**标签熵 2.67 ⇒ 在「压缩」这一维度上语义不占优。

### C3 差异化论证的三条腿全部未真正测量（DA + R2 + R1）
- 跨任务复用：在有偏抽样（多 hunk 富集 + 每任务上限 3 + 按 speedup 截断）下 `distinct_tasks`
  接近随机重排水平；且 PrecisionChange / HostDeviceOverlap / Vectorize 的 distinct_tasks
  **低于**随机（99.4 / 98.4 / 99.7 百分位）⇒ 存在「标注聚在同一任务内」这一反向读法。
- 跨位置协调：**0/55 条边是单标签**，多区域率随标签数上升 ⇒ 54.5% 无法区分
  「一个动作散在多位置」与「多个动作各占一处」。这是边级指标对动作级主张的错配。
- 预注册红线（>40% 单类、>25% Other）在论文自称「不能用于估计总体分布」的样本上判「未触发」⇒ 红线被架空。

### C4 对 CUDA Agent 的批评接近稻草人 + 关键先验工作遗漏（R2 + R3）
1. 论文断言 CUDA Agent 的动作空间是 token 级——其摘要并未如此声明；「skill-augmented environment
   + 工具」的 agent 很可能在环境层已定义了事实上的动作结构。批评对象需重新构造。
2. 「全文不可得」表述不准确：arXiv 无 HTML 版，但 PDF 存在；限制是**我们未解析 PDF**，非服务器不可得。
3. 用 NKI-Agent「GRPO 未超过 SFT」当作「配方原样迁移」的直接证据是**自相矛盾**——那是迁移失败案例。
4. **必须新增的已核实先验工作**（本会话已逐条回源，四条全部真实）：
   - K-Search arXiv:2602.19128（Cao, Mao, Gonzalez, Stoica）——摘要原句
     *"explicitly decouples high-level algorithmic planning from low-level program instantiation"*
     与 *"coordinated, multi-step structural transformations"*。**§4 的 Structured-action 条件已被实现并给出正向结果**
     （平均 2.10×、MoE 最高 14.3×、GPUMode TriMul H100 1030µs SOTA）。
   - CuAsmRL arXiv:2501.08071（He, Yoneki, CGO 2025）——RL agent 在 SASS schedule 上以**显式动作空间**迭代变换，
     并「自动揭示潜在优化动作」；其 §先 autotune 后调度即本文 §4.5 控制 1 的先例。
   - RAS / AEGIS arXiv:2501.18916——把 slow→fast 对分解为可跨程序复用的 **atomic edits**。
   - Kevin arXiv:2507.11948（Baron, Marsella, Pan, Guo, Alberti）——多轮 RL，**turn 级 reward attribution** ⇒
     §2.1 那张「token / code block / transformation」三层表**缺 turn/step 这一层**，是明显漏洞。
   ⇒ 净效应：本文不能声称「四条件协议是首创」；可主张的是「对已有正向结果提供独立证伪工具 + 词汇表的经验测量」。

### C5 验证机制对方法学错误免疫（DA 提出，编辑附议为本文最重要的元发现）
`verify_paper_numbers.py` 69/69 PASS、Stage 2.5 判 PASS，却**结构上无法发现脚本内部一处统计公式错误**——
因为对账的是「论文数字 == 脚本输出」，不是「脚本方法 == 正确方法」。
这与论文自身 §4.4 反对的「内部越来越严谨、但已偏离原问题」是同一失效模式：
**工程验收通过 ≠ 方法有效**。此点应升格为论文正文内容（Appendix 或 §7），而非只在评审记录里。

### C6 §4 协议不可自动执行（R3）
`Oracle-action` 要求「提供正确的 transformation」，§4.5 控制 3 要求判定「动作要求的 site 是否被触碰」，
二者都需要 `φ`；而 §7.3 自认「φ 从未构造」。⇒ 按现文，判读树最后一支
（Oracle ≈ Direct ⇒ 方向死刑）会把「oracle 构造不出来」误判为「方向死了」——最高假阴性风险。

## 分歧问题（编辑裁决）

| 议题 | 分歧 | 编辑裁决 |
|------|------|----------|
| R2 报告「读到了 CUDA Agent HTML 正文并引用原句 */These tools define the action space/*」 | 与 EIC 及本会话的两次抓取结果冲突 | **不采纳该引文**。本会话实测：`arxiv.org/html/2602.24286`、`.../v1`、`.../v2` 均无内容，abs 页只有 PDF / TeX Source，无 HTML 链接 ⇒ 该引文不可复现，判定为审稿子代理幻觉。稻草人批评（C4-1）**仍成立**，但依据改为「摘要未声明动作空间为 token 级，故本文的断言是未经支持的推断」。**记录此事件为论文元发现的一部分。** |
| §5 外推到 AI4X 是否越界 | R3 判为越界并与 §7.4 自相矛盾；EIC 判为可保留 | 保留但收窄：明确 transfer 须 intra-level，删除「recipe transfers untouched」的无条件表述 |
| BackendChoice 是「发现」还是「已知策略性库调用」 | R2 主张后者 | 折中：保留为发现，但改述为「**词汇表设计缺陷**——一个已被其他系统记录的决策被我们的introspective词表漏掉」，并补引支持 |
| 篇幅 8,825 词 | EIC/R3 均判超标；R2 未提 | 采纳：§6 Related Work 与 §5 合并（§6 已退化为指针索引），§5 精简 |

## Revision Roadmap（优先级 P0 为阻断项）

| ID | 优先级 | 问题 | 具体修法 | 验收判据 |
|----|--------|------|----------|----------|
| R-1 | **P0** | C1 MM 公式错误 + 单侧置换 | 修 `bb` 为 `(k_X−1)(k_Y−1)/2n`；改为固定标签置换 Y 的 MI 置换检验；对两个描述符各在其自身基数下对称置换；加 bootstrap 差值 CI | 重跑输出 `info_gain_*`、`p_value_*`、`CI_*`；论文引用新数 |
| R-2 | **P0** | C2 恒真比较 | 摘要/§3.5/§5.1/结论删去 `2.67 vs 5.78` 作为证据；改报归一化熵 `H/log₂k` 并明确「压缩维度上语义不占优」 | `grep` 确认摘要不再出现该比值作为论据 |
| R-3 | **P0** | C3 三条差异化腿未测 | 为 `distinct_tasks` 构造随机子集零模型并报百分位；显式声明 0/55 单标签 ⇒ 跨位置主张不可识别；红线改标「样本描述性，非总体证伪」 | 新增零模型脚本输出 + 正文对应改写 |
| R-4 | **P0** | C4 先验工作与稻草人 | 新增引用 [29]–[32]（已核实）；§2.1 表补 turn/step 层；重写 §5.1/§6 的差异化与新颖性主张；删/改 NKI-Agent 矛盾表述与「全文不可得」表述 | 每条新引用 ID 回源通过；§1.1 不再断言 CUDA Agent 动作空间为 token 级 |
| R-5 | **P1** | C6 φ 不可执行 | §4 增设第 5 条件 **Brute-search**（枚举 + learned cost model）回应 R3-C1；Oracle-action 增「φ 构造前置条件 + 无法构造时的降级判读」 | 判读树补一行「φ 不可构造 ⇒ 本支不判定」 |
| R-6 | **P1** | C5 验证免疫 + 循环自证 | 把「69/69 PASS 却漏掉公式 bug」与「审稿子代理产生不可复现引文」两事件写入 §7/Appendix 作为方法学元发现；给出循环自证的**可执行**消解标准（盲标人类 + κ + 异家族 bottom-up 词表 + §3.6 在共识标签上重跑存活） | 论文含此二事件且标准可验证 |
| R-7 | **P2** | 篇幅与结构 | §6 并入 §5；总字数降至 ≤7,000；§5.6 补 profiler 作为结构化观测通道（R3-M8） | `integrity_scan.py` 字数 + 章节检查 |
| R-8 | **P2** | 测量效度声明不足 | §7.3 明确：oracle 可能假通过（自引 [20][24] 侵蚀 §3 地基）、">5%" 无噪声下界、`fast_p` 口径未审计、未重测计时 | 已部分存在，需补 [20][24] 与地基关系 |

**Early-stopping 判据**：本轮修订后若 R1 的 CRITICAL（R-1/R-2/R-3）全部消除且无新增 P0，
则按流水线规则直接进入 Stage 3′ 验证复审，不再追加修订轮次。
