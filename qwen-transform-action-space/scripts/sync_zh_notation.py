#!/usr/bin/env python3
"""Propagate two English-manuscript edits into the Chinese translation parts.

Both edits were made to paper/main.md after the translation passes finished, so the
Chinese parts lag behind:

1. notation: the action triple and the mapping were unified in English to the Unicode
   forms `a = (τ, ℓ, θ)` and `φ : ΔP → a`; the parts still carry the earlier ASCII
   spellings in some places.
2. statistics: the abstract and Section 8 changed "mutual information 0.70 bits" to
   the explicit plug-in value with its bias-corrected counterpart, because 0.70 was a
   rounding that no longer matched the Section 3.6 table.

verify_zh_alignment.py catches (2) by multiset comparison but not (1), since notation
carries no digits. This script fixes both and then re-runs the gate.

Written as a file rather than an inline command: backticks are the PowerShell escape
character, and an inline attempt silently corrupted the very strings being replaced.

Usage: python scripts/sync_zh_notation.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARTS = [ROOT / "paper" / "zh" / "part1.md", ROOT / "paper" / "zh" / "part2.md"]

BACKTICK = "`"

REPLACEMENTS = [
    # notation
    (f"{BACKTICK}a = (tau, l, theta){BACKTICK}", f"{BACKTICK}a = (τ, ℓ, θ){BACKTICK}"),
    (f"{BACKTICK}a = (tau, l, theta){BACKTICK}", f"{BACKTICK}a = (τ, ℓ, θ){BACKTICK}"),
    (f"{BACKTICK}phi : Delta P -> a{BACKTICK}", f"{BACKTICK}φ : ΔP → a{BACKTICK}"),
    # statistics: keep the Chinese wording aligned with the English plug-in/BC pair
    (
        "（互信息 0.70 比特，p = 0.0005）",
        "（plug-in 互信息 0.697 比特，偏差校正后为 0.624，p = 0.0005）",
    ),
    (
        "（互信息 0.70 比特，\\(p = 0.0005\\)）",
        "（plug-in 互信息 0.697 比特，偏差校正后 0.624，\\(p = 0.0005\\)）",
    ),
    (
        "互信息 0.70 比特",
        "plug-in 互信息 0.697 比特（偏差校正后 0.624）",
    ),
]


def main() -> int:
    changed = 0
    for path in PARTS:
        text = path.read_text(encoding="utf-8")
        original = text
        for old, new in REPLACEMENTS:
            text = text.replace(old, new)
        if text != original:
            path.write_text(text, encoding="utf-8")
            changed += 1
            print(f"updated {path.name}")
        else:
            print(f"unchanged {path.name}")

    leftovers = 0
    for path in PARTS + [ROOT / "paper" / "main.md"]:
        text = path.read_text(encoding="utf-8")
        ascii_triples = text.count("a = (tau, l, theta)")
        stale_stat = len(re.findall(r"0\.70 (?:比特|bits)", text))
        leftovers += ascii_triples + stale_stat
        print(f"  {path.parent.name}/{path.name}: ascii-triple={ascii_triples} 0.70={stale_stat}")

    if leftovers:
        print("FATAL: desync remains", file=sys.stderr)
        return 1

    print("\nre-assembling full.md and re-running gates")
    for script in ("make_zh_full.py", "verify_zh_alignment.py", "make_latex.py", "make_bib.py"):
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / script)],
                           capture_output=True, text=True)
        tail = [l for l in (r.stdout or "").splitlines() if l.strip()][-2:]
        print(f"[{script}] rc={r.returncode} " + " | ".join(tail))
        if r.returncode != 0:
            print(r.stderr[-400:], file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
