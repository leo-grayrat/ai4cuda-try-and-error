#!/usr/bin/env python3
"""Normalise inline mathematics in paper/main.md for reliable pandoc -> LaTeX output.

Two defects in the manuscript's math made the PDF abort mid-document:

1. inline formulas were written with \\( ... \\) delimiters. pandoc 2.12 does not
   reliably treat those as math inside pipe-table cells, so `_` got escaped and
   xelatex failed with "Missing $ inserted" on the table in Section 3.5.
2. inside those formulas the Greek letters and operators were written as Unicode
   codepoints (τ, ℓ, θ, φ, Δ, →, ∈, ≠, κ, α). pandoc passes them through unchanged,
   and the text font has no such math glyphs, producing "Missing character" losses.

This script rewrites only the contents of math spans:
  \\( ... \\)            ->  $ ... $
  unicode inside the span -> the corresponding TeX macro

Anything outside math spans is left alone, so prose, tables and code spans keep
their literal characters. Run once, then re-run make_latex.py.

Usage: python scripts/normalise_math.py [--dry-run]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "main.md"

MATH_SPAN = re.compile(r"\\\((.+?)\\\)", re.S)

# Unicode -> TeX macro, applied inside math spans only
GREEK = {
    "τ": r"\tau ", "ℓ": r"\ell ", "θ": r"\theta ", "φ": r"\varphi ",
    "ϕ": r"\phi ", "Δ": r"\Delta ", "δ": r"\delta ", "α": r"\alpha ",
    "β": r"\beta ", "κ": r"\kappa ", "λ": r"\lambda ", "μ": r"\mu ",
    "π": r"\pi ", "σ": r"\sigma ", "Σ": r"\Sigma ", "γ": r"\gamma ",
    "ε": r"\varepsilon ", "Λ": r"\Lambda ",
}
OPERATORS = {
    "→": r"\to ", "←": r"\leftarrow ", "⇒": r"\Rightarrow ",
    "∈": r"\in ", "∉": r"\notin ", "≠": r"\neq ", "=": "=",
    "≈": r"\approx ", "×": r"\times ", "·": r"\cdot ",
    "≥": r"\ge ", "≤": r"\le ", "⊂": r"\subset ",
}

def main() -> int:
    dry = "--dry-run" in sys.argv
    text = PAPER.read_text(encoding="utf-8")
    changes: list[tuple[str, str]] = []

    def fix(match: re.Match[str]) -> str:
        inner = match.group(1)
        out = inner
        for table in (GREEK, OPERATORS):
            for ch, macro in table.items():
                out = out.replace(ch, macro)
        out = re.sub(r"\s+", " ", out).strip()
        # \text{...} contents must keep their literal characters
        out = re.sub(r"\\text\{([^}]*)\}", lambda m: "\\text{" + m.group(1) + "}", out)
        new = f"${out}$"
        if new != match.group(0):
            changes.append((match.group(0)[:90], new[:90]))
        return new

    # protect code spans: `a = (τ, ℓ, θ)` in backticks is prose, not math
    protected: list[str] = []

    def stash(match: re.Match[str]) -> str:
        protected.append(match.group(0))
        return f"\x00CODE{len(protected) - 1}\x00"

    guarded = re.sub(r"`[^`\n]+`", stash, text)
    guarded = MATH_SPAN.sub(fix, guarded)

    # Code spans are restored verbatim. Unicode inside them is a font-coverage problem,
    # not a content problem: make_latex.py selects DejaVu Serif / DejaVu Sans Mono, which
    # carry these glyphs. Rewriting them into \ensuremath{} would have made the Markdown
    # source unreadable to fix a typesetting issue.
    for i, span in enumerate(protected):
        guarded = guarded.replace(f"\x00CODE{i}\x00", span)

    code_uni = sum(1 for s in protected if any(ch in s for ch in GREEK))

    print(f"math spans rewritten: {len(changes)}; code spans kept verbatim: {code_uni}")
    for old, new in changes[:12]:
        print(f"  -  {old}\n  +  {new}")
    if len(changes) > 12:
        print(f"  ... {len(changes) - 12} more")

    if dry:
        print("\n(dry run, nothing written)")
        return 0

    PAPER.write_text(guarded, encoding="utf-8")
    left = re.findall(r"\\\(", guarded)
    print(f"\nwrote {PAPER}; residual \\( delimiters: {len(left)}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
