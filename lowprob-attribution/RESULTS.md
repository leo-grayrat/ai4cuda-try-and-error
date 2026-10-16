# 结果与撤回状态

## 当前有效结论（2026-10-17）

**判读 1 命中：CUDA Agent 因果链的起点不成立。**

在全部 content token 上按关键词表匹配（**不从分类器划定的分段取子集**）：

| 分组 | n | 平均 logprob | lowprob(≤-2) |
|---|---|---|---|
| CUDA/Triton API 名（`tl`/`load`/`zeros`/`store`/…） | 451 | -0.122 | 1.55% |
| 非 API | 15536 | -0.080 | 0.95% |

**Fisher 精确检验 p = 0.211，不显著。**

逐 token 复核（同一字符串在全部出现位置）：

```
'tl'      n= 7  全部 -0.0
'.store'  n=12  全部 ≈ -0.0
'.load'   n=18  仅 2 次低（-2.634 / -1.490）
'.zeros'  n= 5  2 次低（-2.371 / -1.492）
'.sum'    n= 7  mean -0.045
'.arange' n=10  mean -0.352
```

→ **CUDA/Triton 专属 token 的概率绝大多数时候就是 1.0，不构成低概率尾部。**

**独立于分类的一条观察**：同一 token 跨位置波动极大（`' for'` sd 0.565；`'\n'` sd 0.484、最低 -8.09；`'The'` sd 1.812）。→ **"低概率"是位置属性，不是 token 属性。**

## ⚠ 已撤回的产物

以下文件是 2026-10-17 之前、基于**有偏子集**产出的，**其结论已作废，仅留存以便追溯**：

| 文件 | 状态 |
|---|---|
| `results/report-triton.html` | **作废**——图表基于 20 个 token 的子集 |
| `results/deep-triton.json` | **作废**——同上 |
| `results/tokens-triton.jsonl` | 数据本身有效，但其中 `token_type` 字段的**分段依赖**是偏差来源，使用时须自行按全量重算 |

**为什么作废**：`scripts/classify.py` 在散文段刻意排除了 `load`/`zeros`/`store`（怕误判英文散文），导致 `triton_specific` 只抓到 20 个 token（全量 API 名为 451 个），而这 20 个恰好含两个极端采样值。于是得到 `triton_specific` 低概率率 10%、`p=0.0156` 的假显著。

同一个词 `load` 的两面：

```
判为 NL_reasoning     n=6  mean=-0.044  lowprob= 0.00%
判为 triton_specific  n=4  mean=-1.034  lowprob=25.00%
```

**组间差异由分类规则制造，不是数据本身。**

## 复现方式

```bash
# 采集（需先起 vLLM，见 PLAN.md 第 11 节）
python scripts/collect.py --tasks row_reduce --repeats 12 \
    --runs-root lowprob-attribution/runs-triton --timeout-s 900 \
    --model deepseek/qwen3-4b --upstream http://127.0.0.1:8000 \
    --opencode-config lowprob-attribution/opencode-proxy.jsonc \
    --opencode-agent cuda-min

# 对齐（产出 tokens.jsonl，注意 token_type 的分段依赖）
python scripts/align.py lowprob-attribution/runs-triton \
    --out lowprob-attribution/results/tokens-triton.jsonl

# 全量重算（不依赖分类器的分段，这是当前有效结论的来源）
#   按 . 前缀剥离后在全量 content token 上匹配关键词表，再做 Fisher 检验
```

**注意**：`deep_analyze.py` 与 `make_report.py` 输出的都是**已作废**的口径（它们消费 `token_type` 字段）。在分类器修好之前，不要用它们的结论。

## 结论射程

- **模型**：Qwen3-4B-AWQ（本机 8GB 显存可承载上限；改用本地是环境约束下的既定条件，非方法失误）
- **任务**：1 道（`row_reduce`），原计划 6 道
- **无干预对照**（完整反馈 vs 截断反馈）
- **样本**：12 次 trial / 15987 token

**不能直接外推到 Seed1.6。**
