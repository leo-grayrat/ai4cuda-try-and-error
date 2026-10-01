# Stage 3′ — Verification Review（re-review / 验证复审）

> 角色：Stage 3′ 验证复审执行者（academic-paper-reviewer · re-review 模式）。
> 任务：核查 Stage 4 修订是否**真正落实** Round-1 Revision Roadmap R-1..R-8，而非仅改措辞。
> 权限：**全程 READ-ONLY**。未修改 `paper/main.md`、`data/`、`scripts/` 或任何被版本跟踪的文件；本报告是本阶段唯一新建文件。
> 对照基线：`stage3_review/EDITORIAL_DECISION.md`（R-1..R-8 与验收判据）、五份 Round-1 报告、`stage4_revision/REVISION_NOTES.md`（作者自述，仅作声明、不作为判定依据）、修订稿 `paper/main.md`（v2）、`data/informativeness_calibration.json` 与三个脚本。

---

## 0. 无法核验的部分（前置声明，不假装存在）

1. **v1 稿件从未单独提交**：git 中第一个 paper 提交即为 v2，因此**无法做逐行 diff 式的回归归因**。凡涉及「某句是 v1 残留还是 v2 新增」的判断，一律以 Round-1 报告所引用的 v1 原文片段为**代理基线**，并将新问题的归因标记为 **indeterminate（无法确证）**——但**可见的内容矛盾本身与归因无关**，v2 现状即成立。
2. **未自行运行 `scripts/calibrate_informativeness.py`**：该脚本 `main()` 会 `dest.write_text(...)` 覆写被跟踪文件 `data/informativeness_calibration.json`，为守住 READ-ONLY 而**故意不执行**。R-1 的 p 值/MI/CI 数值由我实际运行的只读脚本 `verify_paper_numbers.py`（112/112 PASS）+ 对 `mi_miller_madow`/`mi_bias_term`/`describe()` 源码的逐行复核 + 与 JSON 的逐格比对背书，**不是我对原始 55 边数据独立重算**。
3. Round-1 五份报告中，`review_methodology.md`、`review_devils_advocate.md`、`EDITORIAL_DECISION.md` 已读；`review_domain.md`/`review_eic.md`/`review_perspective.md` 未逐字读，但 R-1..R-8 的**验收判据与 v1 原文基线**已从 `EDITORIAL_DECISION.md` 完整取得，不阻塞判定。

实际执行（只读）的取证命令：
- `python scripts/verify_paper_numbers.py` → `112/112 numeric assertions reconciled ... REPRODUCIBILITY: PASS`
- `python scripts/integrity_scan.py paper\main.md` → `CJK leakage: 0` / `citations 32, references 32` / `references lacking id/DOI: 0` / `approx words: 12226` / `sections 11/11` / `PASS`
- 4 次 arXiv Atom API 抓取（`export.arxiv.org/api/query?id_list=`）验证 [29]–[32]。

---

## 1. R-1..R-8 判定总表

| ID | 优先级 | 判定 | 一句证据（v2 小节 + 原文） |
|----|--------|------|-----------------------------|
| R-1 | P0 | **MET** | 脚本 MM 公式已改正、置换对全三描述子对称；§3.6 数字与 JSON 逐格一致 |
| R-2 | P0 | **MET** | 摘要把 2.67 vs 5.78 降为「The informative test is not compression」并转向 permutation p |
| R-3 | P0 | **MET** | §3.5 标签置换零模型（Fuse 16 vs 16.03@66.8）撤回跨任务；§3.4 红线「we withdraw it」 |
| R-4 | P0 | **MET** | [29]–[32] 四个 arXiv ID 经 Atom API 全部解析且与正文描述对齐；§2.1 补 Turn 行 |
| R-5 | P1 | **MET** | §4.2 新增 Brute-search 行；§4.3 新增 Brute-search 与「Oracle-action not runnable → indeterminate」两行 + φ 前置段 |
| R-6 | P1 | **MET** | §7.1 四条可执行消解标准 +「We did none of these」；§7.5 两项元发现入文 |
| R-7 | P2 | **PARTIAL** | §6 已目录化，但正文 **12,226 词 >> 7,000**；§5.6 profiler 观测通道未见补入 |
| R-8 | P2 | **PARTIAL** | §7.3 含未重测计时/病态父/autotune 混杂/φ；**缺** [20][24] 侵蚀 §3 地基、>5% 无噪声下界（作者自述「✅ 完成」与稿件不符） |

**P0 阻断项 R-1/R-2/R-3/R-4 全部 MET，无新增 P0**——满足 Roadmap 早停判据（进入 Stage 3′，不再自动追加修订轮）。判定分歧仅出现在 P2（决策惰性项）与下述 1 个 MAJOR 内部矛盾。

---

## 2. 逐条证据 + 6 个「最易声明了没做到」的特别核查

### R-1（P0）Miller–Madow 公式 + 对称置换 —— **MET**

特别核查点 (a)。

- **公式是否正确（我亲自读源码）** `scripts/calibrate_informativeness.py`：
  - `mi_bias_term`：`return (kx - 1) * (ky - 1) / (2 * n)` —— 互信息偏差项含 `k_Y-1` 因子，v1 漏乘的 3 已补回。
  - `mi_miller_madow`：`return mm_entropy(cx, n) + mm_entropy(cy, n) - mm_entropy(cxy, n)` —— 逐项 MM 作用于 H(X)、H(Y)、H(X,Y)，为合法形式（不依赖错误的边际式去校条件熵）。
- **置换是否覆盖「所有」描述子（非只标签）**：`describe(name, X)` 对三描述子统一调用，内部
  `"permutation_test": permutation_p(plugin, X, Y, N_PERM, SEED)`（固定 X、置换 Y），且
  `d_label = describe("primary semantic label", ...)` / `d_coarse = describe("coarse diff structure", ...)` / `d_hunks = describe("hunk count alone", ...)` —— 三个都跑了，非只标签。✓
- **§3.6 数字 vs JSON**：§3.6 表（label 10 cells, MI 0.697, MM 0.624, null 0.397/0.072, **p=0.0005**; coarse 24, 0.937, 0.792, 0.990/0.085, p=0.752; hunk 9, 0.367, 0.276, 0.361/0.071, p=0.436）与 `data/informativeness_calibration.json`（`permutation_test` label p=0.0005 / coarse p=0.752412 / hunk p=0.436178, n_perm=20000；`bootstrap_label_minus_coarse` mean −0.3609, CI[−0.6894,−0.0339]）逐格一致，无漂移。
- 稿件原文（§3.6, L428-434）自陈错误并改正：*"The bias of the plug-in mutual-information estimator is \((k_X-1)(k_Y-1)/2n\); with \(k_Y = 4\) the first draft under-corrected by a factor of three ... The reported conclusion — "semantic labels yield less information than surface structure (0.615 vs 0.728 bits)" — was an artefact of that arithmetic."*
- 独立背书：`verify_paper_numbers.py` 112/112 PASS（含三描述子 p 值、bootstrap CI 上界<0）。
- **结论：MET**（方法、公式、对称性、数值落地四项均达判据）。

### R-2（P0）撤下恒真熵比较作为证据 —— **MET**（附 1 项 MINOR）

特别核查点 (b)：区分「报告数值」与「用它论证」。

- 摘要（L22-27）：*"label entropy (2.67 bits) is well below that of a faithful description of the same diffs (4.26 bits of coarse structure, 5.78 bits of diff identity). **The informative test is not compression, which any coarse summary enjoys**: under symmetric permutation calibration the semantic label is significantly informative ... (mutual information 0.70 bits, p = 0.0005)"*。
  → 摘要**仍列出** 2.67 与 5.78 的数值，但紧接一句把「压缩」明确**排除为论据**（"The informative test is not compression"），并把承重句换成 permutation p。按用户口径，这是「报告数值」而非「用它论证」，满足验收判据「摘要不再出现该比值**作为论据**」。
- §3.5 表（L375）标 `H(diff fingerprint) 5.78 bits = log₂55 exactly → **identity, not evidence**`；§3.5 已补归一化熵 `H/log₂k`（L373-374：0.805 / 0.928）；§8（L946-948）"Both are withdrawn"。
- **MINOR-1**：摘要在列出 5.78 时未就地附「identity / log₂55」限定词，靠下一句去毒；若单看 L23 半句仍可能被误读为压缩证据。属可选打磨，不构成 NOT MET。
- **结论：MET**。

### R-3（P0）跨任务零模型 + 跨位置不可识别 + 红线降级 —— **MET**

特别核查点 (c)：

- §2.5（L278-282）残留检查：**无**「跨数十任务复现」断言；反而写明 *"the evidence this paper originally cited for it **has been withdrawn**. Section 3.5 shows that cross-task recurrence ... sits at chance under a label-permutation null ... **not supported by our data** — it is a hypothesis"*。
- §3.5（L389-394）新增标签置换零模型（4,000 次）：*"Frequent categories recur across tens of different tasks"* 被列为撤回项；Fuse 观测 16 vs 零均值 16.03（66.8 百分位），PrecisionChange 13 vs 14.26（15.5 百分位）。
- §3.5（L410-413）跨位置不可识别：*"**0 of 55 edges carry a single label**, which is why the 54.5% multi-region figure cannot distinguish ... The cross-site property ... is, on this data, *unidentifiable*"。
- §3.4 红线降级（L320-321, L357-359）：*"the red lines of Section 3.4 are descriptive statements about these 55 edges rather than falsification tests"*；*"Reading "not fired" as "the vocabulary is neither too coarse nor fictional" is an over-claim, and we withdraw it."* —— 明确降级为描述性。✓
- 只读核查脚本确认撤回串未泄漏到 §3.5 之外（`verify_paper_numbers.py` confinement check）。
- **结论：MET**。

### R-4（P0）新增 [29]–[32] + 去稻草人 + 修表述 —— **MET（无 CRITICAL）**

特别核查点 (d)：用 arXiv Atom API 独立验证 4 个 ID 是否解析为对应标题。

| 引用 | arXiv ID | Atom API 解析结果 | 与正文描述是否一致 |
|------|----------|-------------------|--------------------|
| [29] K-Search | 2602.19128 | Cao, Mao, Gonzalez, Stoica，"…decouples high-level algorithmic planning from low-level program instantiation" | ✓ §4.2 L558-559 引 2.10x/14.3x MoE |
| [30] CuAsmRL | 2501.08071 | He, Yoneki，SASS schedule via DRL（CGO 2025） | ✓ §4.2 L560-561 |
| [31] RAS/AEGIS | 2501.18916 | "…Retrieval Augmented Search"，atomic edits | ✓ References L1023 |
| [32] Kevin | 2507.11948 | Baronio 等，Multi-Turn RL for CUDA，56%→82% | ✓ §2.1 L192/195 turn 层、§4.2 L566 |

四个 ID **全部解析成功**，无一无法解析 → 无 CRITICAL。
- 去稻草人（§1.1, L60）：*"**We cannot verify what the action space of that system actually is.**"* —— 不再断言 CUDA Agent 动作空间为 token 级。
- §2.1 表（L192）已补 `Turn / tool call` 行（引 [32]），修复 C4「缺 turn/step 层」漏洞。
- 「全文不可得」表述、NKI-Agent 矛盾：由 `verify`/`integrity_scan` 与正文引用闭环佐证（引用 32/32、全部带可验证标识符）。
- **结论：MET**。

### R-5（P1）§4 增设 Brute-search + φ 前置与降级判读 —— **MET**

特别核查点 (e)：

- §4.2 五条件表（L549）新增行：**Brute-search** | *"enumerate a known transformation space and rank candidates with a learned cost model, no LLM in the loop"*。
- §4.3 判读树（L582）新增：*"Structured-action ~ Brute-search | an LLM adds nothing over enumeration with a cost model | **the systems community already owns this problem**"*。
- §4.3 判读树（L583）新增 indeterminate 行：*"Oracle-action not runnable | `phi` cannot be constructed for this task family | **indeterminate — do not read as a null result**"*。
- φ 前置段（L585-591）：*"Oracle-action therefore carries an explicit precondition: a `phi` validated on a held-out set, or a human-supplied oracle standing in for it ... If neither is available, the branch is *not evaluated*"。
- **结论：MET**（Brute-search 行、indeterminate 行、φ 前置三要素齐全）。

### R-6（P1）可执行消解标准 + 两项元发现 —— **MET**

特别核查点 (f)：

- §7.1（L837-850）给出**四条可执行验收标准**而非再仅「声明」：(1) ≥2 名盲标人类 + Cohen's κ/Krippendorff's α；(2) 异家族/异来源 bottom-up 词表 + 收敛效度；(3) **§3.6 permutation 结果必须在共识标签上重跑并存活**（"This is the decisive test and we did not run it"）；(4) 只给类名不给定义。并明确 *"We did none of these. Every number in Section 3 should be read as conditional on them."* —— 标准可验证、且诚实标注未执行。✓
- §7.5（L899-921）**两项元发现入文**：(1) L904-913 "A verification harness can be numerically exhaustive and methodologically blind"（69/69 全绿却对方法错误免疫）；(2) L915-921 "An automated reviewer can fabricate evidence"（审稿子代理产生不可复现引文，被拒并向弱化处理）。
- **结论：MET**。

### R-7（P2）篇幅/结构压缩 —— **PARTIAL**

- 已完成：§6 Related Work 由逐条论证压为「簇—归属—取用」目录式（论证移入 §5）；11/11 节齐全（`integrity_scan`）。
- **未达标**：`integrity_scan` 实测 **approx words: 12226**，远超验收判据「≤7,000」（作者自述亦标 ⚠️ 部分完成，登记为 acknowledged limitation）。
- **未见补入**：Roadmap 要求 §5.6 补 profiler 作为结构化观测通道（R3-M8）；§5.6 未见相应新增内容。
- 归因：篇幅属作者决策项（作者明言压缩会改变论证重心，不由流水线代行）→ 决策惰性，不阻断通过。
- **结论：PARTIAL**。

### R-8（P2）测量效度声明补全 —— **PARTIAL（作者自述与稿件不符）**

- §7.3（L881-889）**已含**：未重测计时（*"Speedup is taken from corpus metadata without re-measurement"*）、病态父（*"at least two of which had parents already near reference speed"*）、autotune 混杂（*"53 of 55 edges change autotuning configuration alongside structure, so per-action attribution ... is confounded by construction"*）、φ 从未构造。
- **缺失（与验收判据不符）**：
  - 「oracle 可能假通过 → 侵蚀 §3 地基」的 [20][24] 交叉引用**不在 §7.3**。`[20]`/`[24]` 实际只出现在 §5.5（L742-743）与 §6（L811），§7.3 未把它们与 §3 数据地基相连。
  - 「">5%" 无测量噪声下界」：`>5%`（实为 L312 §3.2 的入组阈值）在 §7.3 未讨论其缺噪声下界问题。
- **自述 vs 稿件冲突**：`REVISION_NOTES.md` R-8 标 ✅ 完成并声称「§7.3 已含 oracle 可能假通过（自引 [20][24]）、">5%" 无噪声下界、fast_p 口径未审计」；稿件 §7.3 无 [20][24]、无 >5% 噪声下界、无 `fast_p` 字样（`fast_p` 仅现于 L502 §3.7b）。按用户规则**以稿件为准** → 该子项 **NOT MET**，整条 R-8 因其余项已达而判 **PARTIAL**。
- **结论：PARTIAL**（REVISION_NOTES 对 §7.3 的自述失实）。

---

## 3. 新发现问题清单（v2 相对 v1，含回归）

### CRITICAL
- 无。

### MAJOR
- **NEW-1（§5.2, L695，内部自相矛盾 / 回归，归因 indeterminate）**：
  稿件原文：*"Our Section 3.6 finding, that **semantic labels do not out-predict surface structure**, is a mild preview of that class of result and should be read as **prior evidence** that the naive positive outcome is not automatic."*
  这句把**已被 R-1 反转、并被 §3.6/摘要/§8 明确标注为「an artefact of that arithmetic / reversed」**的 v1 头条负面结论，当作 §3.6 的**现行 finding** 与**支持性证据**再次引用。它与下列四处直接矛盾：§3.6 L445（"This reverses the first draft's finding"）、L433-434（该结论是错误产物）、摘要 L27（"this result replaced an opposite conclusion in our first draft"）、§8 L954-956（"The first draft ... reported the opposite conclusion ... it reversed once the arithmetic was fixed"）。§5.2 是全文唯一未随 §3.6 反转同步更新的残留句。
  备注：该句恰是 §7.5 元发现的活例——`verify_paper_numbers.py` 的「被撤回主张 confinement」检查的 stale 串列表未含 "do not out-predict surface structure"，故 112/112 仍 PASS 却对此内容矛盾免疫；这**印证**了稿件自身的论点，但不减免其矛盾。
  严重度：**MAJOR**（涉及论文核心修正结论的内部一致性；一句可外科修复）。

### MINOR
- **NEW-2（§3.6/§7.5/§8 vs 实际 harness，计数漂移）**：稿件三处称对账为「**69** assertions / 69/69」，而当前 `verify_paper_numbers.py` 实测输出 **112/112**（REVISION_NOTES L51 亦称 112/112）。69 疑为 v1 遗留计数未更新到 v2 扩充后的 harness。**MINOR**（不影响结论方向，但属可被审稿人一眼看穿的数字漂移）。
- **NEW-3（摘要，见 R-2 MINOR-1）**：摘要列出 5.78 时未就地附 `identity/log₂55` 限定，靠下句去毒。
- **NEW-4（§7.3，见 R-8）**：`[20][24]`→§3 地基侵蚀、`>5%` 噪声下界两项测量效度声明未落入 §7.3；同时 `REVISION_NOTES` 对 §7.3 的自述与稿件不符（诚实性/可追溯性问题，若按全量口径亦可视作作者声明失准）。

（内部交叉引用小节号、图表编号、§6→索引化、英文正文 CJK 残留：`integrity_scan` 实测 CJK=0、citations/references 32/32 闭环、11/11 节、无缺标识符引用，均**自洽**，未见回归。）

---

## 4. 编辑级决定与推导

**机械推导**：
- must-fix（P0）R-1/R-2/R-3/R-4 → **全部 MET，无新增 P0**；
- should-fix（P1）R-5/R-6 → MET；
- consider（P2）R-7/R-8 → PARTIAL，且均为决策惰性项（篇幅属作者取舍；R-8 缺失为局限声明完备度）；
- 新问题：1× MAJOR（NEW-1，内部矛盾/回归）+ 3× MINOR。

**判定：Minor Revision（定向小修）**，非 Major，非 Accept，无需 user_review_required。

**理由**：
1. Round-1 的全部 P0 阻断项经**独立取证**确认实质落实（公式、对称置换、零模型、引用回源、协议行、可执行标准），而非措辞糊弄——这是从 Major 降级的首要依据。
2. 唯一 MAJOR（NEW-1）是**孤立的单句残留**，与稿件自身已在四处完成的反转直接冲突，修复成本极低（删/改 §5.2 L695 一句），不构成需要再整轮重审的结构性缺陷。
3. R-7/R-8 的 PARTIAL 属 P2「consider」层，且作者已将篇幅明确登记为 acknowledged limitation，按 Roadmap 早停判据不阻断。
4. 不判 Accept：NEW-1 是审稿人必现的自相矛盾，且 R-8 存在作者自述与稿件不符，需一轮定向确认。

**必须落实方可进入下一环节的清单（blocking for final accept）**：
- B1：修复 §5.2 L695（把「do not out-predict surface structure / prior evidence」改为与 §3.6 反转一致的表述）。
- B2：统一 69 → 112 计数（NEW-2）。

**建议（非阻断）**：
- S1：§7.3 补 [20][24]→§3 地基侵蚀 + >5% 噪声下界（R-8/NEW-4），或修订 `REVISION_NOTES` 使自述与稿件一致。
- S2：摘要 5.78 处就地加 `= log₂55 (identity)` 限定（R-2/NEW-3）。
- S3：篇幅与 §5.6 profiler 通道按作者决策处理（R-7）。

---

## 5. Residual Issues（沿 Round-1 遗留、本轮确认仍未解决）

- **人工标注 / κ 完全缺位**（§7.1 四条标准一条未执行）：仍是全文最大真实弱点，流水线无法自解；作者已诚实登记。
- **无任何 §4 干预实验**：§4 仍是协议而非结果（position 体裁合法）。
- **目标变量污染**：加速取自语料元数据未重测、53/55 与 autotune 同变、病态父——即便 R-1 修正方向，§3.6 正向结论仍继承此污染（作者已于 §3.6 L453-458 诚实降级为「an argument for running Section 4, not a substitute」）。

---

## 6. 复审诚实性声明（Judge Record）

- 本报告为**单模型自动化复审**产物，与稿件作者、Stage 4 修订方共享同一底层模型家族，存在与 §7.5(2) 同类的「自产自评」风险；对 R-1 数值的信心依赖只读脚本 `verify_paper_numbers.py` 的实跑 + 源码复核，而非独立重算（见 §0.2）。
- 所有引用句均逐字摘自 `paper/main.md` 当前内容（v2），未编造；判定不采信 `REVISION_NOTES.md` 自述，凡自述与稿件冲突处（R-8）以稿件为准。
- 全程未修改任何被跟踪文件；本 `.md` 为本阶段唯一产出。
