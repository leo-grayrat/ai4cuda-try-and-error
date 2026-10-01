# Qwen 自动化研究探索：Program Transformation Action Space

## 性质声明（必读）

本目录是 **Qwen3.8-Flash 自动化研究探索** 的输出。

- 作者仅测试 AI 端到端自主完成「文献调研 → 实验设计 → 论文撰写 → 引用完整性核验」
  这一完整流程的可能性；本目录内容 **不代表严肃研究投稿意向**。
- 文献检索、程序变换标注、统计分析、图表生成、论文正文与结论 **全部由 AI 生成，
  未经人工复核**。
- 其中的实验标注方法（以 AI 自动标注代替研究笔记原案的「人工标注」）已在论文
  Limitations 中如实声明其 bias 与循环自证风险。
- 本目录的方法论约束继承自仓库根目录 `RESEARCH_RETROSPECTIVE.md`：每一项实验在
  执行前都必须回答「如果这个实验完美完成，它会改变我们对原研究问题的哪一个判断？」
  若答案只是「工具更完整了、记录更可复现了」，则视为基础设施，不包装为研究进展。

## 研究命题（Idea 4）

对 LLM 驱动的 CUDA kernel 优化，把搜索 / 学习界面从 *整段代码（token 序列）生成*
提升到 *参数化程序变换* `a = (τ, ℓ, θ)`，是否真的显著改变搜索效率与学习效率？

体裁：Position / Analysis Paper（主张 + 真实数据初步证据 + 可证伪实验框架设计）。

## 目录结构

| 路径 | 内容 |
|------|------|
| `SPEC.md` | 本次执行的研究 spec（自包含，含完整背景脉络） |
| `stage1_research/` | Stage 1 交付：RQ Brief、文献图谱、逐篇核查表、Related Work 素材 |
| `data/adaexplore/` | AdaExplore 原始轨迹解压产物（大文件，不入库） |
| `data/edges/` | 成功 parent→child 边的 unified diff 与元数据 |
| `data/annotations.json` | AI 自动标注结果（冻结 vocabulary + reasoning） |
| `figures/` | 熵分析与复用率图表 |
| `paper/` | 英文主稿（`main.md` / LaTeX） |
| `paper/zh/` | 中文对照稿 |
| `scripts/` | 纯数据管道脚本（**不含** classifier / locator / AST pipeline） |

## 数据边界

只使用公开可得的 AdaExplore 父子轨迹数据（`.cache/adaexplore-l2.tar.gz`、
`.cache/adaexplore-l3.tar.gz`），不依赖 CUDA Agent 未开源的任何观测量
（逐 token Critic value、原始概率、advantage、Agent 轨迹、工具调用日志）。

## 与旧实验代码的隔离

本目录 **不复用** `action-credit/` 与 `cuda-skill-efficiency/` 的任何代码。
方法论红线：不写 classifier、不写 locator、不写 AST pipeline；
transformation vocabulary 先冻结再标注。

## 工作流

按 `academic-pipeline` skill 的阶段编排执行（用户指示：全程自动推进，checkpoint 不等待确认）：
Stage 1 RESEARCH（quick）→ Stage 1.5 EXPERIMENT → Stage 2 WRITE → Stage 2.5 INTEGRITY
→ Stage 3 REVIEW（五视角盲审）→ Stage 4 REVISE → Stage 3′ RE-REVIEW → Stage 4′ RE-REVISE
→ Stage 4.5 FINAL INTEGRITY → Stage 5 FINALIZE。

## 产出状态（全部阶段已完成）

| 阶段 | 产物 | 结果 |
|------|------|------|
| Stage 1 | `stage1_research/` | RQ Brief + 28 源核查账；2 条 spec 文献检索不到 → 禁止引用 |
| Stage 1.5 | `STAGE1_5_REPORT.md`、`data/`、`figures/` | 2,510 候选边 / 55 标注；词表覆盖 90.9% |
| Stage 2 | `paper/main.md` v1 | 8,825 词初稿 |
| Stage 2.5 | `STAGE2_5_INTEGRITY.md` | 28/28 引用回源；69/69 数字；拦下 3 处不可验证断言 |
| Stage 3 | `stage3_review/` | 五份盲审 + 编辑决定：**Major Revision**（overall 48–57） |
| Stage 4 | `stage4_revision/REVISION_NOTES.md` | R-1..R-8 逐项处置；**头条统计结论反号** |
| Stage 3′/4′ | `stage3prime_rereview/` | R-1..R-6 MET、R-7/R-8 PARTIAL；抓出 §5.2 残留矛盾 |
| Stage 4.5 | `STAGE4_5_FINAL_INTEGRITY.md` | 从零复核 **32/32**，新发现并修 3 处引用字段错误 |
| Stage 5 | `paper/latex/`、`paper/zh/full.md` | LaTeX 源 + bib（本机无 TeX 引擎，**PDF 有意未产出**，见 `paper/latex/BUILD.md`）；中文对照稿约 1.95 万字 |

终态关卡：`verify_references.py` 32/32 · `verify_paper_numbers.py` 117/117 ·
`verify_zh_alignment.py` 定量 token 264/264 · `integrity_scan.py` PASS。

### 本目录的重要事实

论文 v2 的核心结论与 v1 **相反**：v1 声称「语义标签不优于 diff 粗结构」，那是
`calibrate_informativeness.py` 误用 Miller–Madow 偏差项产生的假象；修正后语义标签在自己的
置换零模型下显著（p = 0.0005），而粗结构（p = 0.752）与 hunk 数（p = 0.436）都不显著。
同时撤回了另外两项当时当作证据使用的说法（恒等式熵比较、跨任务复现率）。

本流水线自身的两次失败（对账全绿却对方法错误免疫；审稿子代理产出不可复现引文）
已作为**元发现写入论文 §7.5**，而不是当作过程瑕疵隐去。

### 已知未完成（不作表面修补）

1. 正文约 12,500 词，超出 workshop 8–12 页目标；进一步删减会改变论证重心，属作者决策。
2. 人工独立标注与 κ 未执行（§7.1 四条可执行标准全部未完成）——本文最大真实弱点。
3. 无任何干预实验：§4 是协议，L2/L3 主张未被检验。
4. Stage 2 的 v1 稿件当时未单独提交，故 Stage 3′ 无逐行回归 diff（过程缺陷）。
