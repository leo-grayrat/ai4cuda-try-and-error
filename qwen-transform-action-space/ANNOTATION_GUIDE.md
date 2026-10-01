# Annotation Guide — 每条边的记录格式

标注者必须先读 `VOCABULARY_FROZEN.md`（12 类定义 + 标注协议），本文件只规定输出格式。

读取 diff：

```
python scripts/show_edges.py --batch START END --max-bytes 6000
```

**若某条边被标记 TRUNCATED，必须单独完整读取它**：
`python scripts/show_edges.py --diff <edge_id>`，否则标注不可靠。

## 记录 schema（每条边一个 JSON object，必须是合法 JSON）

```json
{
  "edge_id": "L2_2_17_148_150",
  "primary": "ChangeLaunchConfig",
  "primary_args": {"dim": "ROWS|BLOCK", "value": "16->1 | 512->256"},
  "secondary": ["Tile"],
  "sites": ["triton_instance_norm_divide"],
  "spans_multiple_loci": false,
  "reasoning": "diff 中 '-    ROWS = 16' / '-    BLOCK = 512' 被替换为 ROWS=1、BLOCK=256，算法体未改动，属于纯启动/分块常量调整；因此判定 primary=ChangeLaunchConfig 而非 Tile。",
  "confidence": "high",
  "why_not_existing": [],
  "notes": "speedup_vs_parent=1.11；单 hunk、单 region"
}
```

字段要求：

| 字段 | 约束 |
|------|------|
| `primary` | 必须是 12 类之一；对性能贡献最大的那一个判断 |
| `primary_args` | θ 参数，能写就多写；写不出就 `{}` |
| `secondary` | 0–3 个，同样必须来自 12 类 |
| `sites` | 该 action 实际落在哪些代码区域；直接引用 diff 头给出的 `regions=` 列表中的元素（可子集，不可新增） |
| `spans_multiple_loci` | 当 `len(sites) >= 2` 时为 true。**必须**与 `sites` 一致 |
| `reasoning` | 必须引用 diff 里至少一处**具体行内容**（`+`/`-` 行的文本），不得只写行号 |
| `confidence` | `high` / `medium` / `low` |
| `why_not_existing` | 仅当 `primary == "Other"` 时必填：逐项说明 1–11 为何不合适 |
| `notes` | 可选；异常信号（如加速倍数异常大、父节点疑似退化实现）必须写在这里 |

## 判定纪律（重要）

1. **不要为了套进 12 类而硬塞**。`Other` 是被允许且被期待存在的答案；压低估 `Other`
   比例等于制造假阳性。
2. 区分「改了算法语义的正确性重构」与「只改性能参数」。后者才归 `ChangeLaunchConfig`。
3. 一条边同时做了多件事时，`primary` 取**最可能解释大部分加速**的那个；
   无法判断就把 `confidence` 设为 `low` 并在 `notes` 说明。
4. 看到 `@triton.autotune` 被新增/修改，属于 `ChangeLaunchConfig`（autotuning 是启动配置
   搜索的元机制），但**必须**在 `notes` 里写 "introduced autotuner"，
   因为这会让后续加速归因变得可疑。
5. 若 child 相对 parent 的加速倍数 > 10，几乎一定说明 parent 是病态实现。
   这类边要在 `notes` 标 `suspicious_parent_speedup`，论文需单独统计。
