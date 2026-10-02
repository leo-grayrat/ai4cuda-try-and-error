#!/usr/bin/env python3
"""Size Markdown table columns to their content so the LaTeX build stops overprinting.

pandoc derives a pipe table's relative column widths from the number of dashes in its
separator row. Several tables here were written as |---|---|---|, i.e. equal widths, which
in the rendered PDF squeezed "Transformation" into the same box as "typed (tau, l, theta)"
and printed one over the other.

This script recomputes each separator row from the actual cell contents of its table
(proportional to the longest cell per column, with a floor), so widths follow content
rather than authoring accident.

Usage: python scripts/fit_tables.py [--dry-run]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EN = ROOT / "paper" / "main.md"

SEP = re.compile(r"^\s*\|(?:\s*:?-{3,}:?\s*\|)+\s*$")
ROW = re.compile(r"^\s*\|.*\|\s*$")


def cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def main() -> int:
    dry = "--dry-run" in sys.argv
    lines = EN.read_text(encoding="utf-8").splitlines()
    out = list(lines)
    changed = 0

    i = 0
    while i < len(lines):
        if not SEP.match(lines[i]):
            i += 1
            continue
        sep = cells(lines[i])
        ncol = len(sep)

        # header is the line above; body runs until a non-table line
        header = cells(lines[i - 1]) if ROW.match(lines[i - 1]) else [""] * ncol
        j = i + 1
        body: list[list[str]] = []
        while j < len(lines) and ROW.match(lines[j]):
            row = cells(lines[j])
            if len(row) == ncol:
                body.append(row)
            j += 1

        widths = [
            max(4, max((len(header[k]), *(len(r[k]) for r in body)), default=4))
            for k in range(ncol)
        ]
        # A column narrower than its longest unbreakable word overflows into its
        # neighbour and the PDF prints the two on top of each other, so the floor for
        # each column is that word, not a fixed minimum.
        longest_word = [
            max(
                (max((len(w) for w in re.findall(r"[A-Za-z0-9_./+-]{2,}", r[k])), default=0)
                 for r in [header] + body),
                default=4,
            )
            for k in range(ncol)
        ]
        budget = 90
        floors = [min(w, max(5, lw)) for w, lw in zip(widths, longest_word)]
        spare = max(0, budget - sum(floors))
        rest = sum(max(0, w - f) for w, f in zip(widths, floors)) or 1
        dashes = [
            f + int(spare * max(0, w - f) / rest) for w, f in zip(widths, floors)
        ]
        new_sep = "|" + "|".join("-" * max(3, d) for d in dashes) + "|"

        if lines[i] != new_sep:
            out[i] = new_sep
            changed += 1
            print(f"  L{i + 1}: {ncol} cols -> dashes {dashes}")
        i = j

    print(f"\nseparator rows resized: {changed}")
    if not dry:
        EN.write_text("\n".join(out) + "\n", encoding="utf-8")
        print(f"wrote {EN}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
