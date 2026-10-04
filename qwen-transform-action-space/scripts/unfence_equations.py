#!/usr/bin/env python3
"""Un-fence the display equations that were authored as code blocks.

Section 1.2 and Section 2.2 present their defining equations inside triple-backtick
fences:

    ```
    φ : ΔP → a
    ```

so the typesetter treats them as source code: monospace, no italics, Unicode pseudo-
subscripts, and no display spacing. They are mathematics, and belong in `$$ ... $$`
as their own paragraph.

This script walks every fenced block in the manuscript and classifies it:
  * a block whose content is notation (contains a math symbol, at most two alphabetic
    words, no programming punctuation such as `=` chains, `()`, `.` calls) becomes a
    display equation with the Unicode mapped to TeX macros and blank lines around it;
  * anything that looks like code (function calls, assignments, shell, Python) is left
    exactly as it is and reported.

Usage: python scripts/unfence_equations.py [--survey]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [ROOT / "paper" / "main.md",
           ROOT / "paper" / "zh" / "part1.md",
           ROOT / "paper" / "zh" / "part2.md"]

MATH_CHARS = {
    "τ": r"\tau ", "ℓ": r"\ell ", "θ": r"\theta ", "φ": r"\varphi ", "ϕ": r"\phi ",
    "Δ": r"\Delta ", "α": r"\alpha ", "κ": r"\kappa ", "λ": r"\lambda ", "σ": r"\sigma ",
    "γ": r"\gamma ", "∈": r"\in ", "≠": r"\neq ", "≈": r"\approx ", "→": r"\to ",
    "↦": r"\mapsto ", "⇒": r"\Rightarrow ", "×": r"\times ",
    "₀": "_0", "₁": "_1", "₂": "_2", "₃": "_3", "₄": "_4", "₅": "_5",
    "…": r"\dots ", "...": r"\dots ",
}

CODE_TELLS = ("def ", "import ", "python ", "scripts/", "data/", ".py", ".json",
              "torch", "triton", "tl.", "F.", "@", "::", "->", "==", "if ", "for ")

FENCE = re.compile(r"^```[a-zA-Z]*\s*$")


def is_equation(body: str) -> bool:
    s = body.strip()
    if not s or "\n" in s:                 # multi-line blocks stay as-is
        return False
    # already-wrapped display math stranded inside a fence: the fence is the only bug
    if s.startswith("$$") and s.endswith("$$"):
        return True
    if any(t in s for t in CODE_TELLS):
        return False
    if not any(c in MATH_CHARS for c in s):
        return False
    return len(re.findall(r"[A-Za-z]{2,}", s)) <= 2


def to_math(body: str) -> str:
    out = body.strip()
    if out.startswith("$$") and out.endswith("$$"):
        return out
    for ch, macro in MATH_CHARS.items():
        out = out.replace(ch, macro)
    out = out.replace("{", r"\{ ").replace("}", r" \}")
    out = re.sub(r"\s*([,;)])", r"\1", out)
    out = re.sub(r"\(\s+", "(", out)
    out = re.sub(r"\s+", " ", out).strip()
    return f"$${out}$$"


def main() -> int:
    survey = "--survey" in sys.argv
    for path in TARGETS:
        lines = path.read_text(encoding="utf-8").splitlines()
        out: list[str] = []
        i = 0
        converted, kept = 0, 0
        while i < len(lines):
            if FENCE.match(lines[i]):
                j = i + 1
                block: list[str] = []
                while j < len(lines) and not FENCE.match(lines[j]):
                    block.append(lines[j])
                    j += 1
                body = "\n".join(block)
                if j < len(lines) and is_equation(body):
                    out.append(to_math(body))
                    converted += 1
                    print(f"  {path.name}: fenced block -> display math: {body.strip()!r}")
                    print(f"      {to_math(body)}")
                else:
                    out.extend(lines[i : j + 1])
                    kept += 1
                i = j + 1
                continue
            out.append(lines[i])
            i += 1
        print(f"{path.name}: equations un-fenced={converted} code blocks kept={kept}")
        if not survey:
            path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
