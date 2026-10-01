# Stage 2.5 — Integrity Verification Report

对象：`paper/main.md`（Stage 2 v1，8,825 词，28 条参考文献）。
执行日期：2026-10-02。结论：**PASS（含 4 项已修正的实质性问题）**。

机器检查命令：

```
python scripts/integrity_scan.py paper/main.md      # 结构 + 引用闭环 + 标识符
python scripts/verify_paper_numbers.py              # 69 项数字对账（含语言普查重跑）
```

---

## 1. 参考文献逐条核验（28/28）

方法：arXiv Atom API 按 ID 直查标题/作者/日期；非 arXiv 条目走 CrossRef DOI 解析。
所有条目均在**本会话内**成功解析，无一条仅凭记忆引用。

| # | ID / DOI | 核验方式 | 结果 |
|---|----------|----------|------|
| 1 | arXiv:2602.24286 | arXiv id_list | ✅ 标题/作者（Dai, Wu, Yu…）/2026-02-27 全符 |
| 2 | arXiv:2502.10517 | arXiv id_list | ✅ Ouyang/Guo/Arora/Zhang/Hu/Ré/Mirhoseini |
| 3 | arXiv:2604.16625 | arXiv id_list | ✅ Du/Zhuo/Dong/…/Dettmers/Chen/Yang/Welleck |
| 4 | arXiv:2302.07867 | arXiv id_list | ✅ ICLR 2024 Spotlight 注记来自 arXiv comment 字段 |
| 5 | arXiv:1904.02818 | arXiv id_list | ✅ Zhao/Bieber/Swersky/Tarlow |
| 6 | arXiv:2604.27296 | arXiv id_list | ✅ ACL 2026 Findings 注记来自 arXiv comment |
| 7 | arXiv:2502.18449 | arXiv id_list | ✅ |
| 8 | doi:10.1145/2491956.2462176 | CrossRef | ✅ Halide, PLDI'13, pp.519-530 |
| 9 | arXiv:1802.04799 | arXiv id_list | ✅ |
| 10 | arXiv:2006.06762 | arXiv id_list | ✅ |
| 11 | doi:10.1016/S0004-3702(99)00052-1 | CrossRef | ✅ Sutton/Precup/Singh, *Artificial Intelligence* |
| 12 | arXiv:1609.05140 | arXiv id_list | ✅ **P.-L. Bacon**（初稿误作 D. R. Bacon，已修） |
| 13 | arXiv:2609.05508 | arXiv id_list | ✅ |
| 14 | arXiv:2305.16291 | arXiv id_list | ✅ **Y. Jiang**（初稿误作 T. Jiang，已修） |
| 15 | arXiv:2607.20908 | arXiv id_list | ✅ Mahmud/Ahmed/Jannesari |
| 16 | arXiv:2507.14111 | arXiv id_list | ✅ |
| 17 | arXiv:2602.05885 | arXiv id_list | ✅ |
| 18 | arXiv:2607.04395 | arXiv id_list | ✅ DL4C@ICML 2026 |
| 19 | arXiv:2606.08960 | arXiv id_list | ✅ 数字全部回源（见 §3） |
| 20 | arXiv:2606.20128 | arXiv id_list | ✅ |
| 21 | arXiv:2608.12700 | arXiv id_list | ✅ |
| 22 | arXiv:2607.16241 | arXiv id_list | ✅ |
| 23 | arXiv:2607.14541 | arXiv id_list | ✅ |
| 24 | arXiv:2609.22220 | arXiv id_list | ✅ |
| 25 | arXiv:2609.21058 | arXiv id_list | ✅ |
| 26 | arXiv:2609.32444 | arXiv id_list | ✅ |
| 27 | arXiv:2609.20807 | arXiv id_list | ✅ |
| 28 | arXiv:2607.18171 | arXiv id_list | ✅ |

### 1.1 被拒绝引用的条目（不得出现在论文中）

| Spec 提到的工作 | 处置 | 原因 |
|-----------------|------|------|
| **ProofWright (Stanford)** | **不引用** | arXiv 标题与全文检索均未命中，无法证实存在 |
| **"When Speed Kills Stability"** | **不引用** | `all:"When Speed Kills Stability"` 零命中。该标题来自 spec §0.3 对话叙事，属疑似幻觉；train–inference mismatch 论点改用 #26/#27 两篇真实文献支撑 |

Spec §5.2 特别核查清单中的 8 项，6 项 VERIFIED、2 项 UNVERIFIED 并已剔除。

## 2. 数字可复现性（69/69 断言通过）

`scripts/verify_paper_numbers.py` 从**原始语料**重新推导并断言：

- **语言普查重跑**：23,731 / 22,007 / 0 / 1,724。此组数字原先由临时内联命令产生，
  现已固化为脚本内可重跑函数，消除「不可重跑的数字」。
- 边筛选：2,510 候选（L2 1,906 / L3 604）、145 任务、55 标注边、45 任务、L2 49 / L3 6、33 条多 hunk、55 指纹全互异。
- 熵与分布：2.67 / 5.78 / 3.07 / 2.49 / 2.40 bits；90.9% 覆盖；0.18 复用率；30/55 跨位置；10 类作 primary；
  置信度 14/38/3；autotuner 53/55；suspicious 9/55；10 个 primary 计数逐一；8 个标签的 (出现数, 不同diff数, 任务数) 三元组逐一。
- 校准：H(S)=2.00；原始增益 0.697 / 0.937；校正后 0.615 / 0.728；百分位 98.8；零分布 1.551±0.102；基数 10 vs 24。
- **`Other` 断言逐条验证**：论文声称「5 条 Other 全部是后端/库替换」，脚本对每条 Other 的
  reasoning/notes/why_not_existing 检查是否含后端替换证据 → 5/5 通过。
  （该断言若为编造，本应在此失败；这是本报告唯一一处「论文主张被逐条回源到原始标注文本」的检查，
  因为它是论文里最容易被夸大的定性结论。）

## 3. 本阶段发现并已修正的实质性问题

| # | 问题 | 严重度 | 处置 |
|---|------|--------|------|
| 1 | §1.1 曾把 CUDA Agent 的**内部训练细节**（reward 档位、PPO 非对称裁剪区间、单轮热身原因、预训练语料 CUDA 占比）当作可引述事实 | **高（hallucinated-detail 风险）** | arXiv HTML 全文不可取得（`arxiv.org/html/2602.24286v1` 返回无内容），故全部降级为摘要与已释出配方层面表述，并新增一段**明确声明这些数值无法核验**。同时把「40 points」改为摘要原文的「by about 40%」 |
| 2 | §5.5 引用 [19] 时写了「15 known attack classes」，并把「按原始 speedup 选择会让 100× 漏洞压过 1.5× 合法优化」归为 **HFSV 的设计特征** | **高（错误归因）** | 回源摘要后改写为真实数字（1,968 任务审计 / 323 即 16% 可 hack / KernelBench 62%→0% / 76% 与 61%→0% / 弱模型防强攻击）。奖励上限的推论改为**明确标注为本文自己的推论** |
| 3 | §5.6 曾断言「有公开证据表明基座模型并非不会 CUDA」，隐含依赖中文报道里的 74%/90%/95.2% 等二手数字 | 中 | 改为只用 [1] 摘要自身的表述（通用编程强但对 `torch.compile` 不占优）+ 第一性论证（CUDA kernel 绝大多数 token 是普通代码 token），并新增「刻意不引用无法回源的二手精度数字」的说明 |
| 4 | 参考文献 #12/#14 作者名首字母错误；#2/#3 作者缺失 | 中 | 已按 arXiv 返回修正/补全 |
| 5 | §5.4 讨论 FlashRT 时**无任何引用**，且把两条个人经验写成「该文献的观察」 | 中 | 补引 [28]；两条观察明确改标注为**本文自己的方法学观察、未经测量**，并指向 §4.5 控制 2 作为可测路径 |
| 6 | 英文正文混入中文字符 2 处（`吞`、`程序`） | 低 | 已清除；`integrity_scan.py` 增加 CJK 泄漏检测（当前 0 处） |

## 4. AI 标注方法的如实声明（未冒充人工）

- 论文 §1.5、§3.3、§7.1 三处均声明标注由自动流水线完成、非人工。
- §7.1 把「vocabulary 与标注者同属一个认知家族」列为**对本文核心实证的一阶威胁**，
  并给出最高价值后续（同一批 55 条做独立人工标注 + 报告 inter-rater agreement）。
- 原始 `data/annotations.json` 的 reasoning 字段为中文（subagent 按仓库工作语言产出）。
  论文只对其做**统计聚合**，未逐字翻译引用，故不存在翻译失真风险；已在本文档记录该事实，
  Stage 5 中文版交付时可与英文稿互为校验。

## 5. 七模式 AI 研究失败清单（v3.2，MANDATORY）

| # | 模式 | 判定 | 证据 |
|---|------|------|------|
| 1 | 引用幻觉 | **PASS** | 28/28 回源成功；2 条 spec 提供的条目因无法回源被**主动剔除**；4 处作者名错误已修 |
| 2 | 实现层缺陷 | **PASS** | 三个脚本跑通并产出对账结果；`SequenceMatcher` 的 autojunk 坑已显式处理（`_unified_from_opcodes` 注释说明为何不用 `difflib.unified_diff`） |
| 3 | **结果幻觉** | **PASS（曾经失败并已修）** | 问题 §3#1/#2/#3：三处来自 spec 叙事而非原始论文的量化断言已降级或删除。现所有数字均可由 `verify_paper_numbers.py` 重跑 |
| 4 | 捷径/数据泄漏 | **PASS（有声明的弱点）** | Vocabulary 在任何边 diff 被读前冻结，且冻结仅依赖语料级元数据（§3.3 记录）。弱点如实声明：vocabulary 由全语料审计而来，**未做分折 held-out**，已写入 §7.3 |
| 5 | 方法学造假 | **PASS** | 「人工标」改为「AI 标」在 §3.3 显式声明为偏离原案；`Other` 允许存在且占比 9.1% 照实报告；有偏抽样（多 hunk 富集）在 §3.2 声明「不能用于总体估计」 |
| 6 | **把 bug 当洞见** | **PASS** | 9/55 条 >10× 加速未被当作优化能力证据，而是标注为疑似父节点病态/计时口径问题（§3.7b） |
| 7 | **流水线级 frame-lock** | **PASS（关键自查）** | 本阶段主动做了一件违反「让熵比较好看」的事：§3.6 报告**语义标签不优于粗结构**，并以此削弱论文自己的动机。另有一次实时 frame-lock 拦截：曾开始编写多源文献检索客户端（`openalex_search.py` / `lit_search.py`），按停止条件问句判定其完美工作不改变任何研究判断后**删除**，退回内置检索。方法论红线（不写 classifier/locator/AST）经 `integrity_scan.py` 与代码审阅确认未违反 |

## 6. 遗留项（不阻断，已登记）

1. **无独立人工标注**（§7.1）——本文最大真实弱点，无法在本流水线内解决。
2. **CUDA Agent 全文不可得**，对其训练层面的论述只能停留在摘要级（已在 §1.1 声明）。
3. `phi`（diff → action 的机械映射）从未构造，论文也未声称构造（§7.3 已声明）。
4. 计时数字全部取自语料元数据，未重测；`fast_p` 的基线口径未审计。

---

**Stage 2.5 判定：PASS，进入 Stage 3。**
阻断条件（模式 1/3/5/6 的 INSUFFICIENT EVIDENCE、任何模式的 SUSPECTED）均未触发。

---

## 后记：本阶段的 PASS 是不够的（Stage 4 补记）

本阶段对**引用与数字转写**的核验是充分的（28/28 回源、拦下 3 处来自 spec 叙事的不可验证断言），
但对**统计方法有效性**是免疫的：`verify_paper_numbers.py` 当时 69/69 全绿，而
`calibrate_informativeness.py` 里一处 Miller–Madow 偏差项误用（对条件熵用了边际熵的
`(k_X−1)/2n`，应为 `(k_X−1)(k_Y−1)/2n`）被它完整地保真了下来——因为对账的是
「论文数字 == 脚本输出」，不是「脚本方法 == 有效方法」。

这使上表第 3 行（结果幻觉）与第 7 行（流水线级 frame-lock）的 PASS 只覆盖了它声称的范围：
数据不存在捏造，但存在方法错误。发现者是 Stage 3 的方法学审稿人与 Devil's Advocate（各自独立复算），
不是本阶段自检。教训与两项元发现已写入论文 `paper/main.md` §3.6 与 §7.5。
