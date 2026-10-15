# lowprob-attribution

低概率 token 归因实验：检验 CUDA Agent 论文声称的"CUDA 语法稀有 → 大量低概率 token → PPO 训练不稳定"因果链起点是否成立。

计划见 [PLAN.md](PLAN.md)。

## 目录结构

| 路径 | 内容 |
|------|------|
| `PLAN.md` | 实验计划（含判读树、守门自检、防偏移护栏） |
| `scripts/record_proxy.py` | 透明记录代理：注入 logprobs 参数、侧录响应 |
| `scripts/mock_api.py` | 模拟上游 API（链路验证用，不依赖 DeepSeek 在线） |
| `scripts/classify.py` | token 分类（类型 + 上下文类型，规则冻结） |
| `scripts/align.py` | 对齐：proxy 记录 ↔ OpenCode events.jsonl |
| `scripts/analyze.py` | 分析：哨兵率按 token 类型 / 上下文类型 / 交互项 / 干预条件 |
| `runs/` | 轨迹采集输出（每次运行一个子目录） |
| `results/` | 分析结果 CSV + 报告 |

## 当前状态

- Phase 0 链路验证：**通过**（mock 上游 + 记录代理 + 流式/非流式均正常侧录 logprobs，对齐与分类脚本工作正常）
- Phase 0 API 验证：**暂停** — DeepSeek API（官方与墨点中转）当前全线崩溃，POST /chat/completions 挂起超时。等待恢复后重跑 `scripts/check_logprobs.py`。
