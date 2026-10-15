# lowprob-attribution

低概率 token 归因实验：检验 CUDA Agent 论文声称的"CUDA 语法稀有 → 大量低概率 token → PPO 训练不稳定"因果链起点是否成立。

计划见 [PLAN.md](PLAN.md)。

## 目录结构

| 路径 | 内容 |
|------|------|
| `PLAN.md` | 实验计划（含判读树、守门自检、防偏移护栏） |
| `scripts/record_proxy.py` | 透明记录代理：注入 logprobs 参数、侧录响应 |
| `scripts/mock_api.py` | 模拟上游 API（链路验证用，不依赖 DeepSeek 在线） |
| `scripts/check_logprobs.py` | Phase 0：确认官方 API 返回 logprobs |
| `scripts/check_sentinel.py` | Phase 0：真实生成下的 logprob 分布与哨兵率检查 |
| `scripts/classify.py` | token 分类（类型 + 上下文类型，规则冻结） |
| `scripts/align.py` | 对齐：proxy 记录 ↔ OpenCode events.jsonl |
| `scripts/analyze.py` | 分析：哨兵率按 token 类型 / 上下文类型 / 交互项 / 干预条件 |
| `tasks/` | KernelBench CUDA C++ 任务（kb_relu / kb_sum_reduce / kb_cumsum）+ `evaluate_cuda.py` |
| `runs/` | 轨迹采集输出（每次运行一个子目录） |
| `results/` | 分析结果 CSV + 报告 |

## 当前状态

- Phase 0 链路验证：**通过**（mock 上游 + 记录代理 + 流式/非流式均正常侧录 logprobs，对齐与分类脚本工作正常）
- Phase 0 API 验证：**通过**（2026-10-15 官方 API 恢复后实测）
  - 可用模型为 `deepseek-flash` 与 `deepseek-v4-pro`；推理模型的逐 token logprob 在 `logprobs.reasoning_content`，`deepseek-chat` 别名在 `logprobs.content`
  - **`-9999` 哨兵不存在**：chosen 不在 top-N 内时返回真实 logprob（实测 8/500 个，值域 -2.8~-6.0），`top_logprobs` 取 1/5/10/20 时哨兵率恒为 0%。观测量已改为 `lowprob rate`（logprob 阈值）+ `in_topn` 标记，见 PLAN.md 第九节
  - **`temperature=0` 会让 logprobs 全部退化为 0.000**，调用链不得固定该参数
  - 初步信号（样本小）：低概率 token 全部落在自然语言，无一为 Triton 语法
- 任务适配：3 道 KernelBench CUDA C++ 任务已验证可跑；`evaluate_cuda.py` 已加显存守卫（估算峰值超预算即拒绝执行）

