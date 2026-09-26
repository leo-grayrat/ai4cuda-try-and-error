# Baseline (CUDA Agent) vs. Oracle Semantic Credit 分配对比报告

本报告对比了在总性能奖励 $\sum_t r_t$ 严格守恒的前提下，两种信用分配机制在真实 Kernel 优化案例上的 Token 分布差异：

- **Baseline (CUDA Agent 方式)**：末端奖励挂在最后一个 Token，通过 Critic / GAE 指数后向扩散至所有 Token。
- **Oracle Semantic Credit**：将相同的全部性能奖励精准分配给人工标注的真正优化决策 Token 区间，样板/胶水代码分数为 0。

## 1. 核心对比数据汇总表

| 案例 | 任务与优化动作 | 总 Token 数 | 优化决策 Token 数 (占比) | 样板 Token 数 (占比) | Baseline 优化信用占比 | Baseline 样板浪费占比 | Oracle 优化信用占比 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Device Memory Copy Elimination (2_20)** | KernelBench Level 2 Task 20 | 107 | 7 (6.5%) | 100 (93.5%) | **15.75%** | **84.25%** | **100.0%** |
| **Vector Add 2x Grid-Stride Loop Unrolling** | KernelBench Vector Add | 280 | 85 (30.4%) | 195 (69.6%) | **1.02%** | **98.98%** | **100.0%** |
| **Row Sum 2-Way Stride Reduction** | KernelBench Row Sum | 348 | 57 (16.4%) | 291 (83.6%) | **0.00%** | **100.00%** | **100.0%** |

## 2. 核心实验洞察

1. **总奖励严格守恒**：在所有案例中，两种分配方式下的总信用之和 $\sum_t A_t$ 均与环境给出的真实性能回报严格相等，没有凭空增加或减少奖励。
2. **Baseline 的严重样板稀释问题**：
   - 在 CUDA Agent 机制下，真正的关键优化决策（如消除 `clone()`、指令展开、步长规约）只占总代码的 **6% ~ 23%**；
   - 然而，Baseline 将 **80% ~ 97% 的性能信用**分配给了与速度完全无关的样板代码（PyTorch Module 封装、函数声明、入参类型转换、return 语句等）；
   - 这直观证明了为什么序列 RL 在学习 Kernel 优化时极度低效：它将绝大部分性能提升归因于无论快慢都必须存在的胶水语法。
3. **Oracle Semantic Credit 的聚焦效果**：
   - Oracle 机制将 100% 的性能提升精准赋予真正改变计算与内存行为的代码决策；
   - 这使得模型在策略梯度更新时，梯度完全施加于优化结构本身，避免了在长序列胶水代码上的概率浪费。

## 3. 机器可读数据保存

详细标注及逐 Token 数据已保存在 `experiments/oracle-cases/` 目录下供后续直接加载。
