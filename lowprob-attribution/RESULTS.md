# 结果

## 当前结论（2026-10-17）

**问题**：CUDA Agent 声称的训练不稳定根因——"CUDA 语法稀有 → 大量低概率 token"——是否成立。

**答案**：**不成立**。而且替代解释成立：**低概率来自多轮 agent 的失败反馈上下文，不是来自 CUDA 语法。**

### 数据

64 run（4 任务 × 2 反馈条件 × 8 重复），**49,450 个 content token**，全部 complete。
模型 `deepseek-chat`（非推理）；任务 `row_reduce` / `vector_affine` / `layernorm_rows` / `softmax_rows`。

### 一、CUDA 语法 token 的概率反而更高

按关键词表在**全部 content token** 上匹配（不从任何分段取子集），并在两种定义边界下各算一次：

| 分组 | n | mean | lowprob(≤-2) |
|---|---|---|---|
| API 名（宽松表） | 2287 | -0.030 | **0.35%** |
| 其他 | 47163 | -0.069 | 0.95% |
| CUDA/Triton（严格表） | 1198 | -0.014 | **0.17%** |
| 其他 | 48252 | -0.068 | 0.94% |

**Fisher p = 0.0016 / 0.0019。两种边界的**方向一致**——CUDA 相关 token 低概率率显著更低。**

### 二、上下文效应远强于 token 类型

| context | n | mean | lowprob | p10 |
|---|---|---|---|---|
| **after_failure** | 2142 | -0.237 | **3.22%** | -0.812 |
| after_success | 23399 | -0.053 | 0.81% | -0.000 |
| initial | 23909 | -0.065 | 0.82% | -0.038 |

**`after_failure` 约为其余上下文的 4 倍。**

按任务：`softmax_rows` 1.59% > `row_reduce` 1.01% > `vector_affine` 0.76% > `layernorm_rows` 0.33%。

### 三、干预对照未出现预期效果

PLAN 预期"压缩工具反馈能消除低概率尾部"。实测：

```
full       0.94%
truncated  0.89%     ← 几乎无差
```

**该预期不成立**：低概率不来自反馈文本的长度或信息量。

### 四、低概率落在自然语言上，不在代码上

全库最低的 token：

```
' constants'   -17.746   row_reduce   full   after_success
' backed'      -11.271   row_reduce   full   after_success
' columns'      -9.527   layernorm    truncated  initial
' wrong'       -10.056   softmax      full   after_success
```

**无一为代码 token。**

## 方法

**不走 agent 框架**：工具调用参数不被 `logprobs` 覆盖（厂商策略边界，证据见 [related-work.md](related-work.md)），而 `message.content` 覆盖。

`scripts/loop_driver.py` 用多轮 plain chat 直接打 chat-completions：模型把代码写在正文，驱动自己写 `candidate.py`、跑独立评测器、拼反馈文本。**最小验证**（`check_plaintext_logprobs.py`）：448/448 个代码 token 带 logprob。

```bash
python scripts/loop_driver.py --tasks row_reduce vector_affine \
    --conditions full truncated --repeats 8 \
    --model deepseek-chat --max-tokens 8192 \
    --runs-root lowprob-attribution/runs-formal
python scripts/analyze_loop.py lowprob-attribution/runs-formal
```

**两个实现坑**：评测器 `argv[1]` 是 candidate **文件**而非目录（传目录 `PermissionError`）；评测器需带 torch 的解释器（`--python` 指向 anaconda）。**模型坑**：`deepseek-flash`（推理）会把预算烧在 reasoning 上、content 返空；必须用 `deepseek-chat`。

## 射程

- 模型 `deepseek-chat`，**不是 Seed1.6**（但已是可用 API 中最接近的一档）
- 任务 4 道，均为元素级 / 归约级，没有融合、tiling、launch 配置类任务
- `after_failure` 占 4.3% 的 token（2142 / 49450）
- 未测：低概率是否真的导致 PPO 比值爆炸（需训练）；train-inference mismatch 的贡献（需双引擎）

## ⚠ 已撤回的产物

2026-10-17 之前基于**本地 vLLM + Qwen3-4B** 的一轮，其细分结论**全部作废**：

| 文件 | 状态 |
|---|---|
| `results/report-triton.html` | **作废** |
| `results/deep-triton.json` | **作废** |
| `results/tokens-triton.jsonl` | 数据有效，但 `token_type` 字段带分段依赖偏差 |

**原因**：`scripts/classify.py` 在散文段刻意排除了 `load`/`zeros`/`store`（怕误判英文散文），使 `triton_specific` 只抓到 20 个 token（全量 451 个），而这 20 个恰好含两个极端采样值 → 假的 p=0.0156。同一个词 `load`：判进 `NL_reasoning` 的 6 个平均 -0.044、零低概率；判进 `triton_specific` 的 4 个平均 -1.034、25% 低概率。**组间差异由分类规则制造。**

**那轮唯一站得住的结论**（现已被新数据覆盖）：API 名 token 低概率率 1.55% vs 其他 0.95%，p = 0.211，**方向同样是"不更高"**。

`deep_analyze.py` / `make_report.py` 消费的 `token_type` 仍带该偏差，**不得据其下结论**。
