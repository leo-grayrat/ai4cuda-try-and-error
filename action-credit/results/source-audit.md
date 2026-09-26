# 公开优化轨迹数据核查（2026-09-24）

为了给性能信用分配实验寻找真实的 Agent 修改，核查了 [Sakana AI CUDA Engineer Archive](https://huggingface.co/datasets/SakanaAI/AI-CUDA-Engineer-Archive)。下载的 `level_1.parquet` 保存在忽略版本控制的 `.cache/`，SHA-256 为 `a74e399809f87b583aedc757a826a47cc84ee53487e6630dae53857ebce565ff`。这一层有 12,157 条记录、91 个任务；每条包含候选 CUDA 代码、正确性、延迟和若干分析字段。

它可以帮我们挑选任务和候选修改，但**不能直接充当已完成的 2×2 因果实验**：表中没有显式的 `parent_id`、修改说明或 A/B 因子；`Kernel_Name` 中的 `base`、`edit_1` 只能提示可能的关系，无法证明四份代码都是同一基线的独立 A、B 及组合 AB。公开数据也没有每个候选的多轮原始计时，不能按本仓库的办法估计轮间波动。

性能数字还要重新验证。Sakana AI [后续说明](https://sakana.ai/ai-cuda-engineer-update/)称早期 KernelBench 评测存在可绕过之处，并发布了 [robust-kbench](https://github.com/SakanaAI/robust-kbench)。因此档案里的延迟只作筛选线索；正式归因实验要对选中的四份代码重新运行正确性检查和计时，并检查是否依赖固定输入、初始化或计时漏洞。本仓库当前的向量加法实验只是验证分析链路；它使用上游 KernelBench，尚未通过 robust-kbench 的更广输入检查。

下一批样本的筛选规则：从逐元素、归约、矩阵/块计算中各选若干任务；读取候选源码与差异，人工确认两项能独立施加的程序变换；构造并审查 base/A/B/AB，再用同一评测器、同一 GPU、多轮乱序评测。没有完整四版本或未通过正确性的任务不进入归因统计。跨 GPU 比较继续使用这些**完全相同的源码和输入**；按当前约定，暂不准备第二台机器或声称已有迁移结果。

## 找到可追溯父版本的真实轨迹

另核查了 [AdaExplore 公开运行轨迹](https://huggingface.co/datasets/VanishD/AdaExplore_Traces)。Level 2 归档 SHA-256 为 `14e31f73af3491cff486ce25f08c48e08f0a042816539abe3bb9541841b39cc5`，Level 3 为 `602732f6132b7a93f9e4a9ce572b303321e60e89542591436003f0dbd0e66ee5`；原始归档留在忽略版本控制的 `.cache/`。它保存 `step_i.py`、提示词、评测结果和带 `parent_node_id` 的搜索日志，可还原实际 Agent 优化树。仓库中的 [`audit_adaexplore.py`](../scripts/audit_adaexplore.py) 只读扫描归档并检查父节点关系，不解包或执行候选代码。

| 归档 | 任务 | 候选及显式父边 | 正确且计时的父子边 | 同编号设备的父子边 | 同编号设备且两步均改代码的链 |
|---|---:|---:|---:|---:|---:|
| Level 2 | 100 | 18,581 | 8,999 | 2,962 | 797 |
| Level 3 | 50 | 5,000 | 2,521 | 762 | 191 |

逐项统计见 [`adaexplore-l2-audit.json`](adaexplore-l2-audit.json) 和 [`adaexplore-l3-audit.json`](adaexplore-l3-audit.json)。原始评测记录标为 NVIDIA RTX A6000，但各候选可能在编号 0–3 的不同设备上测量；表格的“同一编号设备”因此只是筛选条件，仍不等于本机重新测量。记录只有汇总延迟，没有每次原始样本。更关键的是，连续两步 `K₀→K_A→K_AB` **没有自动提供独立的 `K_B`**；还需检查代码差异、构造只做 B 的版本并重新验证。该归档解决了真实父版本来源问题，尚未解决完整 2×2 归因。

归档本身也显示计时不能直接充当修改贡献：Level 2 有 144 条、Level 3 有 35 条正确且同编号设备的父子边，**源码哈希完全相同**。例如 Level 3 任务 42 的节点 32→49 代码完全相同，平均延迟却从 94.0 变为 54.5；节点 32 的记录标准差为 29.2。这个数值差异可能涉及测量状态，不能归因于代码。审计结果单列这些边，筛选真实修改时排除它们。
