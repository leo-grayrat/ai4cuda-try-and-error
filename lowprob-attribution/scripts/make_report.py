"""Render the deep-analysis JSON into a self-contained HTML report.

No plotting library is available on this machine, so the charts are hand-built
SVG. The output is a single file with no external references.

Usage:
    python make_report.py lowprob-attribution/results/deep-triton.json \
        --out lowprob-attribution/results/report-triton.html
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

PALETTE = {
    "NL_reasoning": "#7aa2f7",
    "generic_code": "#bb9af7",
    "triton_specific": "#f7768e",
    "tool_syntax": "#9ece6a",
    "identifier": "#e0af68",
    "number": "#7dcfff",
    "cuda_specific": "#ff9e64",
}


def bar_chart(rows: list[tuple[str, float, str]], width: int = 640, row_h: int = 30,
              title: str = "", fmt: str = "{:.2f}") -> str:
    """Horizontal bars; `rows` is (label, value, colour)."""
    if not rows:
        return ""
    top = max(abs(v) for _, v, _ in rows) or 1.0
    height = row_h * len(rows) + 46
    label_w, value_w = 170, 70
    track = width - label_w - value_w
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img">']
    parts.append(f'<text x="0" y="16" class="ct">{html.escape(title)}</text>')
    y = 34
    for label, value, colour in rows:
        frac = abs(value) / top
        bw = max(2.0, track * frac)
        parts.append(f'<text x="0" y="{y + 14}" class="cl">{html.escape(label)}</text>')
        parts.append(f'<rect x="{label_w}" y="{y}" width="{track}" height="17" rx="3" '
                     f'fill="#1f2335"/>')
        parts.append(f'<rect x="{label_w}" y="{y}" width="{bw:.1f}" height="17" rx="3" '
                     f'fill="{colour}"/>')
        parts.append(f'<text x="{label_w + track + 8}" y="{y + 14}" class="cv">'
                     f'{html.escape(fmt.format(value))}</text>')
        y += row_h
    parts.append("</svg>")
    return "".join(parts)


def table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>" for r in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def build(data: dict) -> str:
    cats = data["by_category"]
    # lowest mean first, so the interesting category sits on top
    cat_rows = [(k, v["mean"], PALETTE.get(k, "#c0caf5"))
                for k, v in sorted(cats.items(), key=lambda kv: kv[1]["mean"])]
    cat_lowprob = [(k, v["lowprob_rate"], PALETTE.get(k, "#c0caf5"))
                   for k, v in sorted(cats.items(), key=lambda kv: -kv[1]["lowprob_rate"])]

    tools = data["by_tool"]
    tool_rows = [(k.replace("|", " / "), v["lowprob_rate"], "#f7768e" if v["lowprob_rate"] > 1 else "#9ece6a")
                 for k, v in sorted(tools.items(), key=lambda kv: -kv[1]["lowprob_rate"])]

    pos = data["by_position"]
    pos_rows = [(k.replace("_", " "), v["lowprob_rate"], "#7aa2f7") for k, v in pos.items()]

    rel = data["reliability"]
    cl = data["clusters"]
    re_ = data["reuse"]
    oc = data["by_outcome"]

    cat_table = table(
        ["category", "n", "p10", "p50", "mean", "lowprob ≤ -2"],
        [[k, str(v["n"]), f"{v['p10']:.3f}", f"{v['p50']:.3f}", f"{v['mean']:.3f}",
          f"{v['lowprob_rate']:.2f}%"] for k, v in sorted(cats.items(), key=lambda kv: kv[1]["mean"])],
    )
    tool_table = table(
        ["tool / region", "n", "mean", "p10", "lowprob ≤ -2"],
        [[k.replace("|", " / "), str(v["n"]), f"{v['mean']:.3f}", f"{v['p10']:.3f}",
          f"{v['lowprob_rate']:.2f}%"] for k, v in tools.items()],
    )
    oc_table = table(
        ["outcome", "runs", "tokens", "mean", "lowprob ≤ -2"],
        [[k, str(v["runs"]), str(v["tokens"]), f"{v['mean']:.3f}",
          f"{v['lowprob_rate']:.2f}%"] for k, v in oc.items()],
    )
    rep_rows = [[repr(k), str(v["n"]), f"{v['mean']:.3f}", f"{v['sd']:.3f}",
                 f"{v['min']:.2f}"]
                for k, v in sorted(re_.get("repeated_token_stats", {}).items(),
                                   key=lambda kv: -kv[1]["sd"])[:10]]
    rep_table = table(["token", "n", "mean", "sd", "min"], rep_rows)

    return f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>低概率 token 归因 — 采集结果</title>
<style>
 :root {{ color-scheme: dark; }}
 body {{ margin:0; padding:32px; background:#1a1b26; color:#c0caf5;
        font:14px/1.6 -apple-system,"Segoe UI",Roboto,"Helvetica Neue",sans-serif; }}
 h1 {{ font-size:22px; margin:0 0 6px; font-weight:600; }}
 h2 {{ font-size:15px; margin:34px 0 12px; color:#7aa2f7; font-weight:600;
      border-bottom:1px solid #292e42; padding-bottom:6px; }}
 .sub {{ color:#565f89; margin-bottom:8px; }}
 .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:26px; }}
 .card {{ background:#16161e; border:1px solid #292e42; border-radius:8px; padding:18px; }}
 .ct {{ fill:#c0caf5; font-size:12px; font-weight:600; }}
 .cl {{ fill:#a9b1d6; font-size:12px; }}
 .cv {{ fill:#565f89; font-size:11px; }}
 table {{ width:100%; border-collapse:collapse; font-size:12.5px; }}
 th {{ text-align:left; color:#7aa2f7; font-weight:600; padding:6px 10px;
      border-bottom:1px solid #292e42; }}
 td {{ padding:5px 10px; border-bottom:1px solid #1f2335; }}
 tr:last-child td {{ border-bottom:none; }}
 .kpi {{ display:flex; gap:26px; flex-wrap:wrap; margin:18px 0 6px; }}
 .kpi div {{ background:#16161e; border:1px solid #292e42; border-radius:8px;
            padding:12px 18px; min-width:150px; }}
 .kpi b {{ display:block; font-size:22px; color:#9ece6a; font-weight:600; }}
 .kpi span {{ color:#565f89; font-size:12px; }}
 .note {{ background:#16161e; border-left:3px solid #e0af68; padding:12px 16px;
         border-radius:0 6px 6px 0; margin:14px 0; color:#a9b1d6; }}
 code {{ background:#1f2335; padding:1px 5px; border-radius:3px; font-size:12.5px; }}
</style></head><body>

<h1>低概率 token 归因 — 采集结果</h1>
<div class="sub">Qwen3-4B-AWQ · vLLM 本地推理 · {data['tokens']:,} token · 全部对齐零失配</div>

<div class="kpi">
  <div><b>{cats.get('triton_specific',{}).get('lowprob_rate',0):.1f}%</b>
       <span>triton_specific 低概率率</span></div>
  <div><b>{cats.get('tool_syntax',{}).get('lowprob_rate',0):.2f}%</b>
       <span>tool_syntax 低概率率</span></div>
  <div><b>{cl['isolated']}</b><span>孤立低概率 token（成簇仅 {cl['adjacent']}）</span></div>
  <div><b>{re_['lowprob_distinct']}/{re_['lowprob_occurrences']}</b>
       <span>低概率中不重复的字符串</span></div>
</div>

<h2>1 · 按 token 类别</h2>
<div class="grid">
  <div class="card">{bar_chart(cat_rows, title="平均 logprob（越负越不确定）", fmt="{:.3f}")}</div>
  <div class="card">{bar_chart(cat_lowprob, title="低概率率 % (logprob ≤ -2)", fmt="{:.2f}")}</div>
</div>
<div class="card" style="margin-top:18px">{cat_table}</div>
<div class="note"><b>要点</b>：triton_specific 的 p10 为
{cats.get('triton_specific',{}).get('p10',0):.3f}，其余类别为
{min(v['p10'] for k,v in cats.items() if k!='triton_specific'):.3f} ~
{max(v['p10'] for k,v in cats.items() if k!='triton_specific'):.3f}。
差距来自整个分布偏移，不是个别极端值拉动。</div>

<h2>2 · 按工具与区域</h2>
<div class="card">{bar_chart(tool_rows, title="低概率率 % — 工具 × 区域", fmt="{:.2f}")}</div>
<div class="card" style="margin-top:18px">{tool_table}</div>
<div class="note"><b>要点</b>：<code>edit</code>（需精确匹配已有代码）低概率率为
<code>write</code>（自由生成）的 5 倍；<code>read</code> 为 0% —— 不确定性随「候选空间」大小变化。</div>

<h2>3 · token 在单步内的位置</h2>
<div class="card">{bar_chart(pos_rows, title="低概率率 % — 按位置", fmt="{:.2f}")}</div>

<h2>4 · 数据可靠性核查</h2>
<div class="card">{table(
  ["检查项", "结果"],
  [["条目总数", f"{rel['entries']:,}"],
   ["top_logprobs[0] 与被采样值一致", f"{rel['top0_match_rate']:.1f}%"],
   ["后续候选超出被采样值 >1 nat", f"{rel['later_beats_sampled_gt_1nat']} ({rel['rate']:.2f}%)"]],
)}</div>
<div class="note"><b>说明</b>：<code>top_logprobs[0]</code> 是「被采样的 token」而非「概率最高者」，其余候选从
<code>[1]</code> 起。被采样的 logprob 值 100% 自洽，故类别统计与显著性检验不受影响；
但旁支候选与采样分布不完全一致，<b>竞争分析不宜据此下结论</b>。</div>

<h2>5 · 同一 token 跨位置的概率波动</h2>
<div class="card">{rep_table}</div>
<div class="note"><b>要点</b>：标准差最大者接近 2 nat —— 同一个词有时近乎确定、有时低至 -8。
「低概率」是上下文与位置的属性，不是 token 自身属性。</div>

<h2>6 · 失败 vs 完成轨迹</h2>
<div class="card">{oc_table}</div>
<div class="note"><b>要点</b>：两者几乎无差 —— 低概率不是「出错的原因」，而是「决策点的伴随现象」。</div>

<div class="sub" style="margin-top:36px">
  生成自 <code>deep_analyze.py</code> + <code>make_report.py</code> ·
  数据 <code>results/deep-triton.json</code>
</div>
</body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("analysis", type=Path)
    ap.add_argument("--out", type=Path,
                    default=Path("lowprob-attribution/results/report.html"))
    args = ap.parse_args()
    data = json.loads(args.analysis.read_text(encoding="utf-8"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build(data), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
