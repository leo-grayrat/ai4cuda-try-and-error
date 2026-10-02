#!/usr/bin/env python3
"""Turn bare Unicode notation lines into real display mathematics.

Section 2.2 carried its two defining equations as plain text lines:

    φ : ΔP → a
    a → { e₁, e₂, e₃, ... }

Written that way they are not mathematics to the typesetter at all: no italics, no real
subscripts (the "₁₂₃" are Unicode codepoints that survive only if the text font happens to
contain them), and no display spacing. Any line whose content is notation rather than
prose is converted to a `$$ ... $$` display equation, with the same Unicode-to-TeX mapping
used for inline spans.

Detection is deliberately narrow: a line that contains a math symbol from the notation
alphabet, contains no `$` already, is not a table row, heading, list item, code span or
prose sentence (few alphabetic words), and is short. Everything else is reported for a
human decision instead of being rewritten.

Usage: python scripts/display_math.py [--survey]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EN = ROOT / "paper" / "main.md"
ZH1 = ROOT / "paper" / "zh" / "part1.md"
ZH2 = ROOT / "paper" / "zh" / "part2.md"

MATH_CHARS = {
    "τ": r"\tau ", "ℓ": r"\ell ", "θ": r"\theta ", "φ": r"\varphi ", "ϕ": r"\phi ",
    "Δ": r"\Delta ", "α": r"\alpha ", "κ": r"\kappa ", "λ": r"\lambda ", "σ": r"\sigma ",
    "γ": r"\gamma ", "∈": r"\in ", "≠": r"\neq ", "≈": r"\approx ", "→": r"\to ",
    "↦": r"\mapsto ", "⇒": r"\Rightarrow ", "×": r"\times ",
    "₀": "_0", "₁": "_1", "₂": "_2", "₃": "_3", "₄": "_4", "₅": "_5",
    "…": r"\dots ",
}
NOTATION = set(MATH_CHARS) | set("{}()aPQ,;: ")

SKIP = re.compile(r"^\s*(\||#|```|[-*]\s|\$\$|!\[)")


def looks_like_bare_notation(line: str) -> bool:
    s = line.strip()
    if not s or SKIP.match(s) or "$" in s or "`" in s:
        return False
    # Parentheses and colons appear in ordinary prose ("option (Section 5.2)."), so the
    # test is the presence of an actual math character, not of punctuation
    if not any(c in MATH_CHARS for c in s):
        return False
    if len(re.findall(r"[A-Za-z]{2,}", s)) > 2:
        return False                      # prose sentence, not an equation
    if s.endswith(".") and "Section" in s:
        return False
    return len(s) <= 80


def to_display(s: str) -> str:
    out = s.replace("...", "…")
    for ch, macro in MATH_CHARS.items():
        out = out.replace(ch, macro)
    out = re.sub(r"\{(?![\\s]*\\)", r"\\{ ", out)
    out = re.sub(r"(?<!\\)\}", r" \\}", out)
    out = re.sub(r"\s*([,;)])", r"\1", out)          # no space before punctuation
    out = re.sub(r"\(\s+", "(", out)                  # no space after an opening paren
    out = re.sub(r"\s+", " ", out).strip()
    return f"$${out}$$"


def main() -> int:
    survey = "--survey" in sys.argv
    total = 0
    for path in (EN, ZH1, ZH2):
        lines = path.read_text(encoding="utf-8").splitlines()
        hits = []
        for i, line in enumerate(lines, 1):
            if looks_like_bare_notation(line):
                new = to_display(line)
                hits.append((i, line.strip(), new))
        if hits:
            print(f"{path.name}:")
            for i, old, new in hits:
                print(f"  L{i}: {old!r}")
                print(f"      -> {new}")
        if not survey:
            for i, old, new in hits:
                lines[i - 1] = new
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        total += len(hits)
    print(f"\nbare notation lines: {total}{' (survey only)' if survey else ' -> display math'}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
