# 首轮任务与规则材料清单

本轮材料只针对本机可运行的 Triton CUDA kernel 开发工作流。`vector_affine` 与 `row_reduce` 是采集链路的开发任务；封存后的正式任务放在 `tasks/heldout/`。候选接口和独立评测器见各任务目录。

| 材料 | 作用 | 对照范围 |
| --- | --- | --- |
| [`tasks/vector_affine/`](tasks/vector_affine/) | 连续 float32 向量的仿射计算；检查 3 个长度 | 开发试跑，已有原始轨迹 |
| [`tasks/row_reduce/`](tasks/row_reduce/) | 连续 float32 矩阵逐行求和；检查 3 种形状 | 开发试跑 |
| [`tasks/heldout/softmax_rows/`](tasks/heldout/softmax_rows/) | 稳定逐行 softmax；检查大数值与非二次幂宽度 | 正式对照，试验前封存 |
| [`tasks/heldout/layernorm_rows/`](tasks/heldout/layernorm_rows/) | 无仿射参数的逐行 layer norm；检查非二次幂宽度 | 正式对照，试验前封存 |
| [`skills/full.md`](skills/full.md) | 本项目撰写的完整开发提示 | 三种提示条件之一 |
| [`skills/concise.md`](skills/concise.md) | 人工精简提示，保留相同操作要求 | 三种提示条件之一 |
| 无额外 skill | 仍提供任务契约、接口和 Triton kernel 要求 | 三种提示条件之一 |

完整与精简提示的人工核对：

| 操作信息 | 完整版 | 精简版 |
| --- | --- | --- |
| 只编辑候选源码，保护参考与独立评测器 | 第 1 段、末段 | 第 1 段 |
| 自定义 CUDA kernel；PyTorch 只用于分配和元数据 | “Correctness before speed” | 第 1 段 |
| 保持 dtype、shape、device、尾部遮罩与全部元素 | “Correctness before speed” | 第 1 段 |
| 从正确的简单实现开始；每次实质修改后核对大小输入 | “Development sequence” | 第 2 段 |
| 出错时检查索引、遮罩、网格、分配和同步 | “Development sequence” | 第 2 段 |
| 正确后才计时，保留最佳正确版本，报告不确定性 | “Development sequence” | 第 2 段 |
| 元素邻接访问、避免中间张量、逐行规约全列覆盖 | “CUDA reasoning hints” | 第 2 段 |
| 衡量启动开销和访存，异常提速复核正确性 | “CUDA reasoning hints” | 第 2 段 |

这里的 “skill” 是通过附加在同一个 agent 提示中的实验材料实现的，并非 Codex 原生 skill 的安装与自动加载。估算的文档 token 数使用 `o200k_base`，不是所选模型的精确 tokenizer；实际输入、缓存和输出 token 从 CLI 事件单独记录。正式任务参照 KernelBench 的 [softmax](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/23_Softmax.py) 和 [layer norm](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/40_LayerNorm.py) 算子类型，但缩小了输入规模，且 layer norm 去掉了仿射参数，因此不是官方 KernelBench 题目或分数。

[CUDA Agent 公开的 `SKILL.md`](https://github.com/BytedTsinghua-SIA/CUDA-Agent/blob/main/agent_workdir/SKILL.md) 针对其 CUDA C++ 扩展工作目录；[FlashRT 的 `AGENTS.md`](https://github.com/Infini-AI-Lab/FlashRT/blob/main/AGENTS.md) 针对另一套工程。二者可作为后续规则审查的来源，但本轮没有把它们直接施加到上述 Triton 题目，也没有声称精简版是官方材料的等价改写。
