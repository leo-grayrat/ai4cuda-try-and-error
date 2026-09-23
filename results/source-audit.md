# 公开优化轨迹数据核查（2026-09-24）

为了给性能信用分配实验寻找真实的 Agent 修改，核查了 [Sakana AI CUDA Engineer Archive](https://huggingface.co/datasets/SakanaAI/AI-CUDA-Engineer-Archive)。下载的 `level_1.parquet` 保存在忽略版本控制的 `.cache/`，SHA-256 为 `a74e399809f87b583aedc757a826a47cc84ee53487e6630dae53857ebce565ff`。这一层有 12,157 条记录、91 个任务；每条包含候选 CUDA 代码、正确性、延迟和若干分析字段。

它可以帮我们挑选任务和候选修改，但**不能直接充当已完成的 2×2 因果实验**：表中没有显式的 `parent_id`、修改说明或 A/B 因子；`Kernel_Name` 中的 `base`、`edit_1` 只能提示可能的关系，无法证明四份代码都是同一基线的独立 A、B 及组合 AB。公开数据也没有每个候选的多轮原始计时，不能按本仓库的办法估计轮间波动。

性能数字还要重新验证。Sakana AI [后续说明](https://sakana.ai/ai-cuda-engineer-update/)称早期 KernelBench 评测存在可绕过之处，并发布了 [robust-kbench](https://github.com/SakanaAI/robust-kbench)。因此档案里的延迟只作筛选线索；正式归因实验要对选中的四份代码重新运行正确性检查和计时，并检查是否依赖固定输入、初始化或计时漏洞。本仓库当前的向量加法实验只是验证分析链路；它使用上游 KernelBench，尚未通过 robust-kbench 的更广输入检查。

下一批样本的筛选规则：从逐元素、归约、矩阵/块计算中各选若干任务；读取候选源码与差异，人工确认两项能独立施加的程序变换；构造并审查 base/A/B/AB，再用同一评测器、同一 GPU、多轮乱序评测。没有完整四版本或未通过正确性的任务不进入归因统计。跨 GPU 比较继续使用这些**完全相同的源码和输入**；按当前约定，暂不准备第二台机器或声称已有迁移结果。
