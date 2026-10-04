# Transformation Vocabulary — FROZEN

冻结时间：2026-10-02，**早于任何一条成功边的 diff 被读取**。
冻结依据仅限语料级元数据：目录结构、`step_N_log.json` 的 MCTS 字段、
`step_N_metrics.json` 的字段、以及全仓 23,731 个 `step_*.py` 的语言普查
（22,007 含 Triton、0 含 CUDA C++、1,724 两者皆无＝PyTorch 基线）。
未读取任何 parent→child diff 内容，因此不存在「按某题调 vocabulary」的泄漏路径。

## 与 spec 原案的唯一改动及其理由

Spec §6 给的 12 类带有 CUDA C++ 线程级词（`WarpReduce` / `SharedMemoryTransform` /
`Vectorize(width)` / `ChangeLaunchConfig(blockDim)`）。语言普查显示**公开轨迹全部是
Triton**，Triton 把线程、shared memory、warp shuffle 都藏在 block-level API 之后，
所以这些词在本数据上没有可观察的 realization。

处理方式：**保留全部 12 个类名与数量不变**，但把每一类的定义改为
「调度层语义 + 两种可能的 realization 语言（CUDA-C++ / Triton）」，
使类本身不因语言而失效。改动发生在读取任何边之前，且类数量未增未减。

## 冻结的 12 类

| # | 名称 | 定义（做什么样的程序变换） | CUDA-C++ realization | Triton realization | 参数 θ |
|---|------|---------------------------|----------------------|--------------------|--------|
| 1 | `Fuse(producer, consumer)` | 消掉一个中间张量，把两个 kernel/两趟计算合成一趟 | 两个 `__global__` 合一 | 两个 `@triton.jit` 合一，或去掉一次 `torch` 中间算子 | producer/consumer 名 |
| 2 | `EliminateMaterialization(site)` | 保持 kernel 数不变，但不再物化某个中间量（改为重算/寄存器内联/就地） | 去掉 global scratch | 去掉一个中间 `tl.load`→store 往返、去掉 `.contiguous()`/`clone()` | site |
| 3 | `Tile(axis, size, mapping)` | 引入或改变分块，并改变块到线程/程序层的映射 | shared-memory tile + `threadIdx` 映射 | `BLOCK_*` constexpr + `program_id` 网格映射 | axis, size, mapping |
| 4 | `IncreaseWorkPerThread(k)` | 每个执行单元从 1 个元素改为 k 个元素（grid-stride / register tiling） | `j = i + stride` + host 端 `N/T → N/(kT)` | 内层 `tl.static_range` 或更大的 BLOCK 覆盖更多列 | k |
| 5 | `Vectorize(axis, width)` | 让访存变成宽事务 / 提升 coalescing 或 ILP | `float4` 载入 | 让 `BLOCK` 维与连续内存对齐、`tl.multiple_of`/`max_contiguous` 提示 | width |
| 6 | `ChangeLaunchConfig(dim, value)` | 只改启动配置，不改算法体 | `blockDim`/`gridDim`/stream | `num_warps` / `num_stages` / `BLOCK` / grid 尺寸 | dim, value |
| 7 | `WarpReduce(scope)` | 归约的组织方式变为跨 lane/跨块分级归约 | `__shfl_down_sync` warp reduction | `tl.reduce`/`tl.sum` 在轴上分级、两阶段 atomic vs 分段 | scope |
| 8 | `SharedMemoryTransform(access_pattern)` | 改变片上暂存策略以避免 bank conflict / 复用数据 | `__shared__` 数组、padding | 把重复读取的张量保持在寄存器/一次载入多次使用（Triton 隐式） | access_pattern |
| 9 | `PrecisionChange(source, target)` | 计算或存储精度改变 | `__half`/`tf32` | `tl.float16`/`input_precision`/`ALLOW_TF32` | src, tgt |
| 10 | `LoopTransform(type, params)` | 循环结构变换：unroll / swap / hoist / 边界处理 | `#pragma unroll`、循环交换 | `tl.static_range` vs `range`、循环次序、mask 方式 | type, params |
| 11 | `HostDeviceOverlap` | host 侧并发/同步/分配结构调整（含 multi-stream、异步、减少 CPU-GPU 往返与冗余分配） | `cudaLaunchKernel` on stream, `cudaMemcpyAsync` | 减少 `torch` 端算子往返、避免 device→host 同步点、合并启动 | — |
| 12 | `Other(description)` | 无法归入上述 12 类，必须自由文本描述 | — | — | description |

## 标注协议（冻结，不得中途修改）

1. **多标签**：一条边可标注 1–4 个类；必须区分 `primary`（对性能贡献最大的判断）
   与 `secondary`。
2. 每条标注必须有 `reasoning` 字段，引用 diff 中至少一处具体证据（行内容，非行号）。
3. 每条标注必须有 `sites[]`：本次判定认为该 action 落在哪些代码区域
   （取值来自 diff 的 `hunk_regions`），用于直接检验「一个 action → 多个分散位置」。
4. 若 `primary = Other`，额外记录 `why_not_existing[]`：逐项说明为什么 1–11 都不合适。
5. **报告义务**（spec §6.3 红线）：若任一类占比 > 40% 记为「词汇过粗」；
   若 `Other` 占比 > 25% 记为「猜想可能不成立」；两种情况都必须在论文讨论中如实呈现，
   不得回头修改 vocabulary 使数字变好看。
6. 词汇表在本阶段之后**永久冻结**；held-out 协议（Stage 2 框架部分）要求用一部分轨迹建
   vocabulary、另一部分做测试，本数据集内因 vocabulary 来自全语料级审计而非分折，
   必须在 Limitations 中声明这一弱点。

## 关于「人工标 → AI 自动标」的偏离声明

研究笔记原案为**人工读 30–50 条边**。本执行按 spec §0.5 改为 AI 自动标注，
并在论文 Limitations 中声明循环自证风险：vocabulary 与标注者属同一认知家族。
红线（不写 classifier / locator / AST pipeline）仍然执行——标注由阅读 diff 完成，
脚本只负责产出 diff 与其结构性统计（hunk 数、变更行数、所属顶层构造名）。
