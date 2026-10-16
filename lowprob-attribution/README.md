# lowprob-attribution

低概率 token 归因实验：检验 CUDA Agent 论文声称的"CUDA 语法稀有 → 大量低概率 token → PPO 训练不稳定"因果链起点是否成立。

计划见 [PLAN.md](PLAN.md)，关于「闭源 API 拿不到完整逐 token 概率」的公开证据汇总见 [related-work.md](related-work.md)。

## 目录结构

| 路径 | 内容 |
|------|------|
| `PLAN.md` | 实验计划（含判读树、守门自检、防偏移护栏） |
| `scripts/record_proxy.py` | 透明记录代理：注入 logprobs 参数、侧录响应 |
| `scripts/mock_api.py` | 模拟上游 API（链路验证用，不依赖 DeepSeek 在线） |
| `scripts/check_logprobs.py` | Phase 0：确认官方 API 返回 logprobs |
| `scripts/check_sentinel.py` | Phase 0：真实生成下的 logprob 分布与哨兵率检查 |
| `scripts/check_tool_logprobs.py` | 验证工具调用的 token 是否携带 logprobs（含"先写正文"的绕法对比） |
| `scripts/collect.py` | Phase 1 采集器：起代理 → 跑 study.py trial → 按 trial 切分 logprobs |
| `scripts/classify.py` | token 分类（类型 + 上下文类型，规则冻结） |
| `scripts/align.py` | 对齐：按请求顺序配对 proxy 记录与 step，校验 reasoning token 数 |
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
- 代理真实链路：**通过**（`--upstream https://api.deepseek.com`，一次流式请求聚合 304 条 SSE，侧录 300 个 token logprobs，`min=-9.72`、`lowprob(≤-2)=28`）
- OpenCode 端到端：**通过**。用 `OPENCODE_CONFIG=lowprob-attribution/opencode-proxy.jsonc` 把内置 deepseek provider 的 `baseURL` 指向本地代理，`opencode run` 正常产出事件流，代理同步侧录 logprobs。过程中修掉代理丢弃 SSE 空行导致 OpenCode 报 JSON 解析错误的 bug。三点约束见 PLAN.md 第九节第 7 条：
  - 调用须带 `--pure --auto --format json` 与 `OPENCODE_DISABLE_AUTOUPDATE=true`，stdin 给 EOF，否则挂起
  - 代理会额外录到 OpenCode 的"生成会话标题"辅助请求，对齐时须过滤
  - `events.jsonl` 不记录 reasoning 文本（`--thinking` 无效），而推理模型的 logprobs 几乎全在 reasoning 上，对齐单位待定
- **本地 vLLM 路线：通过。** 闭源 API 不返回工具调用参数的概率（厂商策略边界，证据见 [related-work.md](related-work.md)），故改用本机 vLLM + Qwen3-4B-AWQ。绕过的坑：`NO_PROXY` 未设导致回环请求被系统代理拦成 502；OpenCode 需显式声明模型、需精简 agent（`tools.skill=false`）否则 53 个 skill 撑爆 8192 窗口；`subprocess` 探针必须关 stdin 否则挂起；vLLM 需 `--default-chat-template-kwargs '{"enable_thinking": false}'` 否则 4B 把输出预算全烧在 `<think>` 上而不调用工具。
- **Phase 1 首轮采集：完成**（`row_reduce` × 12，15987 token，对齐零失配）。

## ★ 当前结论（2026-10-17，64 run / 49,450 content token）

**判读 1 的两个条件同时成立：CUDA Agent 的因果链起点不成立，而"分布外上下文"这一替代解释成立。**

| 分组 | n | mean | lowprob(≤-2) |
|---|---|---|---|
| CUDA/Triton API 名（宽松表） | 2287 | -0.030 | **0.35%** |
| 其他 | 47163 | -0.069 | 0.95% |
| CUDA/Triton（严格表） | 1198 | -0.014 | **0.17%** |
| 其他 | 48252 | -0.068 | 0.94% |

Fisher p = **0.0016 / 0.0019** —— **CUDA 相关 token 概率显著更高，不是更低**。两种定义边界方向一致。

**上下文效应远强于 token 类型**：

| context | n | lowprob |
|---|---|---|
| **after_failure** | 2142 | **3.22%** |
| after_success | 23399 | 0.81% |
| initial | 23909 | 0.82% |

**干预对照未出现预期效果**：`full` 0.94% vs `truncated` 0.89%——压缩反馈没有压低低概率率。

**低概率的 token 几乎全是自然语言**（`' constants'` -17.75、`' backed'` -11.27），无一为代码 token。

**方法**：不走 agent 框架（工具参数无 logprob）。`scripts/loop_driver.py` 用多轮 plain chat 直接打 chat-completions，驱动自己写文件、跑独立评测器、拼反馈；模型把代码写在正文，logprob 全覆盖（最小验证 448/448）。模型 `deepseek-chat`（非推理，reasoning_tokens=0）。

**射程**：模型 `deepseek-chat`（非 Seed1.6，但已是可用 API 中最接近的一档）；任务 4 道；`after_failure` 占 4.3% 的 token。

### 此前 Qwen3-4B 一轮的结论（已由上述取代）

本地 vLLM + Qwen3-4B：API 名 token 低概率率 1.55% vs 其他 0.95%，Fisher p = 0.211，**不显著**（方向同样是"不更高"）。该轮的**细分结论已全部撤回**——成因是 `classify.py` 在散文段刻意排除 `load`/`zeros`/`store`，使该类别只抓到 20/451 个 token 且恰好含极端值。详见 [RESULTS.md](RESULTS.md)。



