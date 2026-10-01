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

按 `academic-pipeline` skill 的阶段编排执行：
Stage 1 RESEARCH → Stage 1.5 EXPERIMENT → Stage 2 WRITE → Stage 2.5 INTEGRITY
→ Stage 3 REVIEW → Stage 4 REVISE → Stage 3′/4′ → Stage 4.5 FINAL INTEGRITY
→ Stage 5 FINALIZE。各阶段结束设有 checkpoint。
