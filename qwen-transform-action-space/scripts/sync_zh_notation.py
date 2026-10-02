#!/usr/bin/env python3
"""Bring the Chinese translation back onto the English manuscript's conventions.

The English source switched inline math from \\( ... \\) to $ ... $ (see
scripts/normalise_math.py, which was needed because pandoc 2.12 does not treat the
backslash form as math inside pipe-table cells). Translated passages keep whatever
convention was current when they were written, so the two languages drifted apart.

This script:
  - converts \\( ... \\) to $ ... $ in both Chinese parts
  - fixes the known typo 占踞 -> 占据
  - rebuilds paper/zh/full.md

Usage: python scripts/sync_zh_notation.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]

try:
    from make_zh_full import main as rebuild_full
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(HERE))
    from make_zh_full import main as rebuild_full


def main() -> int:
    for name in ("part1.md", "part2.md"):
        p = ROOT / "paper" / "zh" / name
        if not p.exists():
            print(f"MISSING {p}", file=sys.stderr)
            return 1
        text = p.read_text(encoding="utf-8")
        before = text.count("\\(")
        text = text.replace("\\(", "$").replace("\\)", "$")
        text = text.replace("占踞", "占据")
        p.write_text(text, encoding="utf-8")
        print(f"{name}: converted {before} math span(s); "
              f"remaining backslash-paren = {text.count(chr(92) + '(')}")

    rebuild_full()
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
