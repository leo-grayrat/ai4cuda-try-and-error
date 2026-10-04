#!/usr/bin/env python3
"""Hand-decided math conversion for the Chinese parts (mirrors hand_mathify.py).

The Chinese translation uses full-width punctuation and keeps some English relation
names inside its table cells, so the ASCII-anchored rules in hand_mathify.py matched only
part of them. The remaining sites were read from scripts/list_notation.py output per file
and are listed here explicitly.

Left as-is on purpose: prose arrows (「NVIDIA → AMD」, 「慢→fast」, 「父→子」,
「56%→92%」), the code predicate `correctness == true`, and reference annotations.

Usage: python scripts/hand_mathify_zh.py [--dry-run]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P1 = ROOT / "paper" / "zh" / "part1.md"
P2 = ROOT / "paper" / "zh" / "part2.md"

RULES: list[tuple[Path, str, str, str]] = [
    # §3.4 pre-registered red lines
    (P1, r"\| 任一类别占比 > 40% 的边 ⇒ 词表过粗 \|",
     r"| 任一类别占比 $>$40\% 的边 $\Rightarrow$ 词表过粗 |", "红线 1"),
    (P1, r"\| `Other` > 25% ⇒ 该猜想可能为假 \|",
     r"| `Other` $>$25\% $\Rightarrow$ 该猜想可能为假 |", "红线 2"),
    # §3.5 exemplar table header and the joint descriptor
    (P1, r"\| ×相对父节点 \|", r"| $\times$ 相对父节点 |", "样例表头"),
    (P1, r"hunk 数 × 改动行数分桶", r"hunk 数 $\times$ 改动行数分桶", "§3.5 联合描述符"),
    # figure 1 caption states the identity
    (P1, r"恒等于 log2\(n\)", r"恒等于 $\log_{2} n$", "图 1 注"),
    # §2.2 worked example: block count expression is arithmetic, not code
    (P1, r"从 `N/T` 改为 `N/\(2T\)`", r"从 $N/T$ 改为 $N/(2T)$", "§2.2 主机端改动"),
    # §5.1 / §7 K-Search multiples
    (P2, r"平均 2\.10x", r"平均 $2.10\times$", "K-Search 平均"),
    (P2, r"高达 14\.3x", r"高达 $14.3\times$", "K-Search 最高"),
    (P2, r"报告平均 2\.10x", r"报告平均 $2.10\times$", "§7 K-Search"),
    # §4.3 reading-tree row that carries a Chinese conjunction inside the relation
    (P2, r"\| Structured-action > Direct 但 ~ 轮次级分解 \|",
     r"| $ \mathrm{Structured\text{-}action} > \mathrm{Direct} $ 但 $ \sim $ 轮次级分解 |",
     "判读树行"),
]


def main() -> int:
    dry = "--dry-run" in sys.argv
    texts = {p: p.read_text(encoding="utf-8") for p in (P1, P2)}
    misses, applied = [], 0
    for path, pattern, repl, what in RULES:
        text = texts[path]
        hits = len(re.findall(pattern, text))
        if not hits:
            misses.append(f"{path.name} {what}: {pattern}")
            print(f"  MISS  {path.name} {what}")
            continue
        # a lambda keeps the replacement literal: re.sub would otherwise interpret \%
        # and \text in the replacement string as template escapes
        texts[path] = re.sub(pattern, lambda _m, r=repl: r, text)
        applied += 1
        print(f"  ok[{hits}] {path.name} {what}")
    print(f"\napplied {applied}/{len(RULES)}")
    for m in misses:
        print("  PENDING:", m)
    if not dry:
        for path, text in texts.items():
            path.write_text(text, encoding="utf-8")
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
