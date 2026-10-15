# 低概率 Token 归因：实验计划

## 总览

用 OpenCode 现有的 agent 循环跑多轮 Triton/CUDA 任务，通过一个透明记录代理捕获逐 token logprobs，统计低概率（意外）token 在"token 类型 × 上下文类型"上的分布，检验 CUDA Agent 论文声称的训练不稳定 root cause（CUDA 语法稀有 → 大量低概率 token）是否成立。

- **周期**：约 3-4 个工作日
- **成本**：API 费用约 $2-5
- **硬件**：仅现有笔记本 RTX 5060，无训练

## 一、架构：OpenCode + 记录代理

三层结构，每层职责单一：

```
study.py（现有，不改核心逻辑）
    ↓ 发起
opencode run --pure --auto --model deepseek/deepseek-flash
    ↓ API 请求
记录代理（localhost，新建 ~200 行 Python）
    ↓ 转发 + 注入 logprobs 参数 + 侧录响应
DeepSeek 官方 API（logprobs=True, top_logprobs=20）
```

- **OpenCode 层**：复用全部现有设施——任务隔离（`isolated_task_workspace`）、评测（`evaluate.py`）、事件流记录（`events.jsonl`）、版本快照（`versions/`）。model 从墨点网关的 deepseek 换成 OpenCode 内置的 `deepseek/deepseek-flash` 分组（直连官方），baseURL 由代理替代（OpenCode 配置里给 deepseek provider 加 `baseURL: http://localhost:8765/v1`，代理再转发到 `https://api.deepseek.com/v1`）。
- **代理层**：接收请求 → 注入 `logprobs: true, top_logprobs: 20` → 转发官方 → 把响应里的 `logprobs.content` / `logprobs.reasoning_content` 侧录到 `logprobs.jsonl`，同时记录完整的 assistant 回复文本作为对齐键。流式和非流式都要处理（OpenCode 用流式，代理需要聚合 SSE 流再侧录完整 logprobs——这是实现上最需要注意的一点）。
- **study.py 层**：只加一个开关，支持"截断反馈"条件（Phase 1 干预用），其余不动。

## 二、Phase 0：可行性验证（半天）

三个验证，全部通过才进入 Phase 1：

1. **官方 API 连通性**：`deepseek/deepseek-flash` 分组是否已配好 API key（OpenCode 的 deepseek 分组需要 `DEEPSEEK_API_KEY` 或 auth 登录）。跑一个最小 chat 请求带 `logprobs: true`，确认响应里有 logprobs 字段和 -9999 哨兵语义符合预期。
2. **代理对齐**：起代理，把 OpenCode 指向它，跑一个最小任务，确认（a）代理侧录的 logprobs 完整、（b）能与 OpenCode events.jsonl 里的 assistant 文本按前缀匹配对齐、（c）流式聚合无丢失。
3. **观测量 sanity check**：原设计的"哨兵率（-9999 占比）"经实测恒为 0，已在第九节修正为 `lowprob rate`（`logprob <= τ`，τ ∈ {-2, -5}）加 `in_topn` 标记。sanity check 相应改为：确认每类样本的 `lowprob rate` 既非恒 0 也非接近 1，且 `in_topn` 在固定 `top_logprobs` 下可比。

**Kill condition**：如果代理对流式 SSE 的 logprobs 聚合在 DeepSeek 官方 API 上无法实现（比如官方流式不返回 logprobs 字段），就改用非流式模式跑 OpenCode（OpenCode 支持配置非流式）或降级到逐轮回放（拿保存的 messages 重新请求）。这条 fallback 在 Phase 0 内验证。

## 三、Phase 1：轨迹采集（1-2 天）

**任务**：4 道现有 Triton 任务（row_reduce / vector_affine / softmax_rows / layernorm_rows）+ 2-3 道 KernelBench CUDA C++ 题（从 Level 1 选简单的：如 vector add、矩阵转置、ReLU——本机已有 `.cache/KernelBench`，评测适配需要把 `kernelbench_adapter.py` 的接口接到新代理流程里，工作量约半天）。

**条件**（干预对照）：
- **完整反馈**：编译错误全文、正确性对比输出、性能数字。现有 study.py 的默认行为。
- **截断反馈**：编译错误只给"compilation failed"一行、正确性只给"output mismatch"、性能只给"correct but slower"。

**规模**：6 道任务 × 2 条件 × 3 重复 = 36 次运行。每次 5-15 轮对话，总量约 15-20 万 token，费用约 $3-6。

**产出**：每次运行一个目录，含 `events.jsonl`（OpenCode 标准）、`logprobs.jsonl`（代理侧录）、`run.json`（复用现有格式）、`work/`（任务文件与版本快照）。

## 四、Phase 2：对齐与分类（1 天）

**对齐脚本**：把 `logprobs.jsonl` 的每条 completion 与 `events.jsonl` 的 `text`/`tool_use` 事件按文本前缀匹配，产出统一的 `tokens.jsonl`：每个 token 一行，字段包括 `logprob`、`logprob_field`（content / reasoning_content）、`in_topn`（chosen 是否在返回的 top_logprobs 内）、`is_lowprob`（`logprob <= τ`）、`token_text`、`event_index`、`turn`、`task`、`condition`。

**分类脚本**（规则冻结后不再改）：
- token 类型：`NL_reasoning` / `generic_code` / `triton_specific`（Triton 关键词表）/ `cuda_specific`（threadIdx、__shared__、__syncthreads 等关键词表）/ `number` / `identifier` / `tool_syntax`
- 上下文类型：`initial` / `after_compile_error` / `after_correctness_fail` / `after_profiling`——由 events.jsonl 的 step 序列推导（上一步 tool_use 的类型决定）

分类规则、关键词表在 Phase 2 开始时写死提交，之后即使发现分类不准也只记录不修补——分类错误率本身进 limitations。

## 五、Phase 3：分析与产出（半天）

四个分析（对应判读树）：

1. **token 类型 × lowprob rate**：`cuda_specific`/`triton_specific` vs `generic_code` 的低概率率比较——低概率是否来自 CUDA/Triton 语法本身。
2. **上下文类型 × lowprob rate**：`after_compile_error`/`after_profiling` vs `initial`——低概率是否来自工具观察后的分布外上下文。
3. **干预对照**：完整反馈 vs 截断反馈的整体 lowprob rate 差异——压缩工具反馈能否消除低概率尾部。
4. **交互项**：`cuda_specific × after_compile_error`——语法本身不低概率、但被奇怪上下文带低，是比"CUDA 语法稀有"精确得多的结论。

每项同时报告该单元的 logprob 均值与低分位数，以及 `in_topn` 比例，避免只由单一阈值承载结论。

**产出**：`lowprob-attribution/` 目录（README、tokens.jsonl 汇总、分析脚本、结果报告），根 README 加一行入口。结果无论正反都写成一篇可独立发布的实证短报告草稿（约 4-6 页）。

## 六、守门自检与边界

**改变哪个判断**：CUDA Agent 论文标题为"The Root Cause of Training Instability"的因果链起点——低概率 token 是否真的来自 CUDA 语法稀有。起点被抽掉，整条链要重新解释；起点被证实，我们的质疑收回。

**不能回答**：低概率是否真的导致 PPO 比值爆炸（需训练）；train-inference mismatch 贡献几何（需双引擎）；Seed1.6 特定权重上的表现（需其权重）。报告里会明确写这三条边界。

**防偏移护栏**：分类规则和判读树在采集开始前冻结；分析中发现的"分类不准"记为结果不修补；代理是通用工具不做研究性扩展；整个计划不包含训练、不包含造 baseline、不包含给 locator 补规则。

## 七、判读树（提前定死）

- `lowprob rate`（τ=-2 与 τ=-5 两档）在 triton/cuda_specific 和 generic_code 之间无显著差异，且 after_compile_error 明显高于 initial → CUDA Agent 的 root cause 叙事需要改写：低概率不来自 CUDA 语法稀有，而来自多轮工具 Agent 的分布外上下文。
- triton/cuda_specific 的 `lowprob rate` 显著高于 generic_code，且各上下文类型下一致 → CUDA Agent 的叙事得到支持，之前的质疑收回。
- 两种模式在不同任务/轮次上混合出现 → 需要更细致的分解，但仍能回答"哪类 token 在什么条件下变低概率"。

## 八、执行顺序

1. Phase 0：验证 deepseek 分组 key → 最小 logprobs 请求 → 写代理 → 代理下跑通 OpenCode → 对齐验证 → 哨兵率 sanity check
2. Phase 1：冻结判读树和分类规则 → 适配 2-3 道 KernelBench CUDA 题 → 36 次运行
3. Phase 2：对齐脚本 → 分类脚本 → tokens.jsonl
4. Phase 3：四项分析 → 结果报告 → README 入口

## 九、Phase 0 实测记录（2026-10-15）

DeepSeek 官方 API 恢复后，用裸 `urllib` 直调 `https://api.deepseek.com/chat/completions` 实测（脚本 `scripts/check_logprobs.py`、`scripts/check_sentinel.py`）。四点与计划假设不符，其中第三点推翻了原观测量定义。

### 1. 模型名

官方 `/models` 只返回 `deepseek-flash` 和 `deepseek-v4-pro`。此前脚本里用的 `deepseek-chat` 是一个可用的别名，但不是官方列表中的模型名。两者生成的 logprobs 语义不同（见下条）。

### 2. 推理模型把 logprobs 放在 `reasoning_content`

`deepseek-flash` / `deepseek-v4-pro` 是推理模型：响应里 `logprobs.content` 为空，逐 token logprob 全部在 `logprobs.reasoning_content`；`deepseek-chat` 别名则把 logprobs 放在 `logprobs.content`。代理必须同时读取两路（`scripts/record_proxy.py` 已按此实现）。分析时必须记录 `logprobs field` 是哪一路，否则两类模型的样本不可比。

### 3. `-9999` 哨兵不存在（推翻原观测量）

计划原假设："生成的 token 不在 top-N 最可能 token 之内时，`logprob` 返回 `-9999.0` 哨兵"。实测不成立。

在同一段 500 token 的 reasoning 里，有 8 个 token 的 chosen 值**不在**返回的 `top_logprobs` 列表内（即严格意义上的"不在 top-5 之内"），它们返回的是真实 logprob 而不是哨兵：

```
chosen=' Trit'      logprob=-3.500   best_alt=' User'@-1.257
chosen=' Need'      logprob=-2.803   best_alt=' Could'@-1.895
chosen=' including' logprob=-4.697   best_alt=' no'@-0.268
chosen=' returns'   logprob=-6.032   best_alt='(x'@-0.857
```

全文 `logprob <= -9998` 的计数为 0。同时 `top_logprobs` 取 1、5、10、20 四档时哨兵率**恒为 0%**，而 `logprob <= -5` 的计数随档位在 0~6 之间波动。

结论：哨兵率这个观测量在当前 API 上恒为 0，没有分辨力。不能作为主指标。

### 4. `temperature=0` 会让 logprobs 退化为全 0

同一 prompt、同一模型，省略 `temperature` 时 `min logprob = -2.118`（真实值）；显式设 `temperature=0.0` 时，全部 420~500 个 token 的 logprob **均为 0.000**，`top_logprobs` 里所有替代 token 也都是 0.000。参数存在但不携带信息。

任何调用链（代理、`study.py`、OpenCode 的 provider 默认值）都不能把 `temperature` 固定为 0，否则采集到的 logprobs 全废且不会报错。

### 5. 观测量修正（采集前冻结，替代原"哨兵率"）

- **主指标**：`lowprob rate` = 每个样本中 `logprob <= τ` 的 token 占比，取 `τ ∈ {-2, -5}` 两档分别报告（不预设单一阈值，避免阈值选择变成自由度）。
- **辅助标记**：`in_topn` = chosen 是否出现在返回的 `top_logprobs` 列表内。固定 `top_logprobs` 取值后它是良定义的一致指标；本次 `top_logprobs=5` 下为 8/500 = 1.6%。
- **分布统计**：每个（token 类型 × 上下文类型）单元的 logprob 均值与低分位数，配合低概率率一起报告。
- 原"哨兵率"表述在本文档其余部分等效替换为 `lowprob rate`。

### 6. 初步信号（样本量小，仅记录不做推断）

两段 `deepseek-flash` reasoning 里，logprob 最低的 token 全部是自然语言和标点——`' Trit'`、`' Need'`、`' including'`、`' Options'`、`'For'`、`' returns'`、`')'`、`' We'`——没有一个是 Triton/CUDA 专属语法。冷启动单轮、`after_error`、`cuda_cpp` 三个场景下 `logprob <= -5` 的比例分别为 1.5%、0%、0%，且 `after_error`（给出编译错误要求修复）的最低 logprob 只有 -0.215，是三者中最"确定"的。

这与计划假设的方向一致，但样本只有几段单轮回复，不构成证据，仅用于确认观测量有力可测。

