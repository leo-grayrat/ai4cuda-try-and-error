# Stage 1 — 七组关键词执行审计（spec §5.1 逐条对账）

Spec §5.1 列出 7 组关键词，要求「每组多源交叉」。本文件记录每组**实际执行方式、产出、
以及由此新增/否决的引用**，使该要求可被核查而非默认已满足。

方法说明：`nature-academic-search` 的 OpenAlex 降级脚本（`scripts/academic_search.py`）
在本环境对 `search=` 端点连续两次读超时（同一 URL 经 PowerShell 压缩请求约 16 s 可通），
按该 skill 自身的错误处理规则（"script failure 2x → fall back"）改走
**arXiv Atom API + CrossRef API + 内置 WebSearch**。用户亦明确指示文献检索非关键路径。

| # | Spec 关键词组 | 执行 | 主要产出 | 对论文的影响 |
|---|---------------|------|----------|--------------|
| 1 | "action space design" + "LLM agent" + "code generation" | WebSearch + arXiv 核实 | **CodeAct** arXiv:2402.01030（ICML 2024，可执行 Python 作统一动作空间，成功率最高 +20%）；**ReCode** arXiv:2510.23564（plan 作为占位函数递归分解至原语动作，粒度可控） | 新增引用 [33][34]；§5.1 增一条" Against plan/action unification"，明确二者是本文 structured-action 条件的**通用领域近似物**，且都未跑 Free-plan 对照 → §4.3 的 `Free-plan ≈ Structured-action ⇒ no paper` 是鲜活风险 |
| 2 | "program transformation" + "compiler optimization pass" + "learning" | arXiv 核实 | TVM arXiv:1802.04799、Ansor arXiv:2006.06762、Halide doi:10.1145/2499370.2462176 | 已在 [8][9][10]；§5.3 把「枚举 + cost model 已足够」升为第五臂 Brute-search 的动机 |
| 3 | "credit assignment" + "code generation" + "token level" + "reinforcement" | WebSearch + arXiv 核实 | **综述** arXiv:2604.09459《From Reasoning to Agentic: Credit Assignment in RL for LLMs》（69 篇语料、按轨迹/段/步/token 粒度分类、含"文本历史可能使信用连符号都不可辨识"的识别性分析） | 新增 [35]；§1.2 新增一段，把本文主张从「信用分配无人处理」纠正为「现有工作改**信用归属目标**而**动作单元**仍在 token；两者正交，后者才是消除前者的那条路」 |
| 4 | "AdaExplore" / "KernelBench" / "CUDA Agent" / "GPU kernel optimization" | arXiv API | AdaExplore **2604.16625**、KernelBench **2502.10517**、CUDA Agent **2602.24286**（spec 原为 "2602.xxxxx" 未定），另得 CUDA-L1 2507.14111、Dr.Kernel 2602.05885、NKI-Agent 2607.04395、KernelBench-Verified 2607.16241、Atrex 2607.14541、Correctness Illusion 2606.20128、Contract-Grade 2608.12700、Measuring the Checker 2609.22220 | 数据源与批判对象 ID 全部落定；§3.1 明确「只用 AdaExplore 公开数据」的理由是 CUDA Agent 未释出轨迹与训练观测量 |
| 5 | "hierarchical RL" + "options"/"option-critic" + "structured action" + "code" | arXiv/CrossRef | options 框架 doi:10.1016/S0004-3702(99)00052-1、Option-Critic arXiv:1609.05140、**When Do Options Help?** arXiv:2609.05508（policy necrosis / redundant coverage） | [11][12][13]；§5.2 用「高层动作何时无效」为判读树的负面支提供先验；注意：CrossRef 检索修正了我记忆中的 Sutton DOI（00056-5 → **00052-1**） |
| 6 | "Voyager" / "skill library" / "action abstraction" + "LLM agent" | arXiv | Voyager arXiv:2305.16291 | [14]；§5.4 借「技能即可复用程序化动作」连接 skill 长度与遵循度问题 |
| 7 | "optimal edit distance" / "edit-based RL"（Meta learning-by-editing 系） | WebSearch + arXiv | **PIRE** arXiv:2302.07867（Learning Performance-Improving Code Edits，ICLR 2024 Spotlight）、**edit-sequence 建模** arXiv:1904.02818、**RAS/AEGIS** arXiv:2501.18916（atomic edits）、**To Diff or Not to Diff** arXiv:2604.27296、SWE-RL arXiv:2502.18449 | [4][5][6][7][31]；§5.1 成为本文差异化的主战场；R2 审稿后承认「本文新颖性必须相对这些收缩」 |

## Spec §5.2 特别核查清单的处置（摘要）

6 项 VERIFIED（CUDA Agent / FlashRT / HFSV / Correctness Illusion / Contract-Grade /
KernelBench-Verified / Atrex-Bench）；
**2 项 UNVERIFIED 并禁止引用**：`ProofWright (Stanford)`、`When Speed Kills Stability`
（后者是 spec §0.3 叙事用来支撑低概率 token 质疑的关键文献，检索零命中 →
train–inference mismatch 论点改由 2609.32444 / 2609.20807 支撑）。

## 主动否决的引用（不可验证，宁缺勿造）

组 1/3 的检索另出现若干高度相关但**无法解析 ID** 的条目：DelTA、APPO、TRIAGE、CRAFT、
GRAIL、VPR、"rewrite blueprint" 一类。它们多来自二手清单（awesome-list、博客、聚合站），
本会话未能用 arXiv/CrossRef 取得可核对记录。依本流水线既有规则
（「若无法确认某文献真实存在，必须标注为待核实、禁止直接引用」），**一律不写入参考文献**，
在此登记以备作者后续核验。

## 结论

7 组全部有执行记录与实际产出；由此新增 3 条已核实引用（[33]–[35]）并**三处改写论文**
（§1.2 承认信用分配已是独立领域、§5.1 承认 plan/action 统一为通用近似物、§6 增两个集群）。
这直接回应了 R2 审稿给出的最低分维度（文献覆盖 42/100）。

---

## 附：Spec §1「执行前必读」三份外部输入的核对结果

| 输入 | 是否已读 | 对论文的实质影响 |
|------|----------|------------------|
| `research-notes/program-transformation-action-space.md`（745 行） | 已读（全文） | 论文的骨架：三层主张分离、`a=(τ,ℓ,θ)`、四→五条件判读树、粒度原则、泄漏防护、三大危险 |
| `RESEARCH_RETROSPECTIVE.md` | 已读（全文） | 停止条件问句、代理替换阶梯（写入 §4.4）、方向一失败作为 §1.2 的动机 |
| `近日对 AI4CUDA 的调研.md`（26,161 字符） | **已读关键节**（§3.3.1–3.3.4、§3.3 总结、§5.6 相关） | 核对结论：论文 §1.1 对 CUDA Agent 训练链的转述**与原文一致**（裁剪 0.8/1.28、单轮热身、成功轨迹 SFT、离线 Critic 预训练，以及 L203「这些设计与 CUDA 特有部分没有关系」）。原文 L93–95 另有一句已补入 §5.6：稀有性解释对**任何**稀有领域（冷门语言 / HDL / 证明语言）都成立 ⇒ 若处处成立，它就不是关于 GPU 的任何解释 |
| `Retire the Abstractions 等四篇文章.md`（14,094 字符） | 已读（全文） | 两项实质产出：① §2.3 的「契约 vs 标签」粒度判据与 §5.3 的**确定性不对称**（编译器近乎全函数，LLM 的 `a → ΔP` 是随机映射，故反向 `φ` 同样不保证存在）；② §7.5(3) 把「三方审查可能共享同一盲区」反身指向本文自己的五席位评审团。引用 [36] 经 URL 解析 + 页面标题 + 页面内作者姓名三重核验 |
| `202610012331-chatgpt-...-60msg.html`（2.7 MB，60 轮对话） | **未整本读入** | 其方法论产出已由 spec §0.3 固化并被本文直接实现：代理替换阶梯（§4.4）、停止条件问句（§4.4）、四条件框架（§4）、跨 GPU 否定的可迁移性定义（§2.5）。未整本读入的理由：这四项已在 spec 中成形且本文逐条落地，重读 2.7 MB 对话不会改变任何一条论文判断（按停止条件问句自测通过） |

**一处刻意的不一致**：调研文档记录了 CUDA Agent 的具体数值（reward 档位、0.8/1.28 裁剪），
属作者对一手论文的转述。本文仍**不把这些数字写进论文**，因为无法对一手来源核实
（§1.1 已声明）。以作者的二手记录替代一手核验，正是本文 §7.5(2) 拒绝的做法。
