#!/usr/bin/env python3
"""List every notation site in the manuscript so a human can decide each one.

This is deliberately NOT a fixer. It prints, with line numbers, every line that carries
mathematical notation, tagged by how it is currently typeset:
  MATH   already inside $...$ or $$...$$
  CODE   inside backticks
  BARE   plain text (never typeset as mathematics)
so the decisions can be made by reading a short list rather than by guessing regex rules.

Usage: python scripts/list_notation.py [path]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

NOTATION = set("τℓθφΔ∈≠≈→⇒×₀₁₂₃₄₅…") | {"...", "~"}
WORDS = ("p = ", "p < ", "n = ", "H(", "MI", "log2", "k_X", "k_Y", "Q(", "R(a)",
         "x10", "2.10x", "14.3x", "p-value", "Friedman", "kappa", "alpha")


def tags(line: str) -> str:
    out = []
    if "$" in line:
        out.append("MATH")
    if "`" in line:
        out.append("CODE")
    if not out:
        out.append("BARE")
    return "/".join(out)


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "paper" / "main.md"
    lines = target.read_text(encoding="utf-8").splitlines()
    in_fence = False
    shown = 0
    for i, line in enumerate(lines, 1):
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        stripped = re.sub(r"\$[^$]*\$|\$\$.*?\$\$", "", line)
        has_symbol = any(c in line for c in NOTATION)
        has_word = any(w in line for w in WORDS)
        # report lines that carry notation somewhere, including inside existing math
        if not (has_symbol or has_word):
            continue
        shown += 1
        flag = "FENCED-CODE" if in_fence else tags(line)
        print(f"L{i:>4} [{flag:<11}] {line.strip()[:150]}")
    print(f"\n{target.name}: {shown} notation-bearing lines")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
