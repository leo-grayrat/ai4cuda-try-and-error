#!/usr/bin/env python3
"""Typeset the paper's formal notation in LaTeX math mode instead of monospace.

The notation was originally written as inline code (`a = (tau, l, theta)`) and later
"unified" into Unicode glyphs inside those same code spans, so the PDF renders the
formalism in typewriter font with literal Unicode subscripts: phi : Delta P -> a prints
as ``φ : ΔP → a`` and a -> { e₁, e₂, e₃, ... } as ``a → { e₁, e₂, e₃, . . . }``. That is
code typography applied to mathematics: no italics, no real subscripts, no spacing rules.

Real code identifiers must stay monospace (`@triton.autotune`, `F.mish`,
`torch.nn.functional`, `tl.dot`, `scripts/*.py`), so classification is explicit rather
than pattern-guessed: every span is listed with the decision and the proposed math form,
and anything ambiguous is reported for a human call instead of being rewritten.

Usage:
    python scripts/mathify_notation.py --survey     # print decisions, change nothing
    python scripts/mathify_notation.py              # apply, then report what changed
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [ROOT / "paper" / "main.md",
           ROOT / "paper" / "zh" / "part1.md",
           ROOT / "paper" / "zh" / "part2.md"]

# Unicode that only ever appears in mathematics here -> TeX macro
MATH_CHARS = {
    "τ": r"\tau ", "ℓ": r"\ell ", "θ": r"\theta ", "φ": r"\varphi ", "ϕ": r"\phi ",
    "Δ": r"\Delta ", "δ": r"\delta ", "α": r"\alpha ", "β": r"\beta ", "κ": r"\kappa ",
    "λ": r"\lambda ", "μ": r"\mu ", "σ": r"\sigma ", "Σ": r"\Sigma ", "γ": r"\gamma ",
    "ε": r"\varepsilon ", "∈": r"\in ", "∉": r"\notin ", "≠": r"\neq ",
    "≈": r"\approx ", "→": r"\to ", "⇒": r"\Rightarrow ", "×": r"\times ",
    "∪": r"\cup ", "∩": r"\cap ", "⊂": r"\subset ", "∞": r"\infty ",
    "₀": "_0", "₁": "_1", "₂": "_2", "₃": "_3", "₄": "_4", "₅": "_5",
    "⁴": "^{4}", "⁵": "^{5}", "…": r"\dots ",
}
MATH_SET = set(MATH_CHARS)

# substrings that mark a span as genuine source code, never notation
CODE_MARKERS = (".", "@", "/", "::", "_autotune", "torch", "triton", "cuda", "tl.",
                "F.", "http", ".py", ".json", ".md", ".csv", "()", "->")

SPAN = re.compile(r"`([^`\n]+)`")


def classify(s: str) -> str:
    """Return 'code', 'math' or 'review'."""
    low = s.lower()
    # A bare identifier with no operators is a label, not notation: category names
    # (`Fuse`, `WarpReduce`), edge ids (`L2_2_60_112_114`), field names
    # (`parent_node_id`), class names (`ModelNew`). These stay monospace by design.
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{1,60}", s):
        return "code"
    # A lone letter in this paper is a mathematical object ("Let `P` be a program"), not
    # an identifier, so it belongs in math mode
    if re.fullmatch(r"[A-Za-z]", s):
        return "math"
    # A judgement or an instruction is prose that happens to carry operators:
    # "Free-plan ≈ Structured-action ⇒ no paper", "change line 172 from i to i+stride".
    # The test is the count of alphabetic words, not of whitespace-separated tokens:
    # "a = (τ, ℓ, θ)" has one word and is notation, while the judgement above has five.
    if len(re.findall(r"[A-Za-z]{2,}", s)) > 3:
        return "code"
    if any(m in low for m in CODE_MARKERS):
        # a span may mix both, e.g. `a = (τ, ℓ, θ)` has no code marker, but
        # `F.scaled_dot_product_attention` does; markers win because they are concrete
        if any(c in s for c in MATH_SET):
            return "review"
        return "code"
    if any(c in s for c in MATH_SET):
        return "math"
    if re.fullmatch(r"[A-Za-z][A-Za-z_]{0,3}\s*=\s*[-\d.]+", s):        # n = 55
        return "math"
    if re.fullmatch(r"[pHnk]\s*[=<>≤≥]\s*[\d.]+", s):                    # p = 0.0005
        return "math"
    if re.fullmatch(r"[A-Za-z]\([^)]*\)", s) and s[0].isupper():         # H(label)
        return "math"
    if re.fullmatch(r"log2?\([^)]*\)|log_?\d", s):
        return "math"
    return "review"


def to_math(s: str) -> str:
    # Escape the span's own braces first: doing it later would also escape the braces of
    # the \text{...} wrappers added below and produce \text\{...\}.
    out = s.replace("{", r"\{ ").replace("}", r" \}")
    # snake_case identifiers must not be typeset as subscripts: work_per_thread would
    # render as "work" with "per" below the line. Keep them upright and literal.
    out = re.sub(r"\b([A-Za-z]+(?:_[A-Za-z0-9]+)+)\b",
                 lambda m: "\\text{" + m.group(1).replace("_", "\\_") + "}", out)
    for ch, macro in MATH_CHARS.items():
        out = out.replace(ch, macro)
    out = re.sub(r"\blog2\b", r"\\log_{2}", out)
    out = re.sub(r"\blog_?2\b", r"\\log_{2}", out)
    # words that are not TeX macros become upright text, run together so a phrase reads
    # as one unit rather than as adjacent \text{word} fragments
    def word(m: re.Match[str]) -> str:
        w = m.group(0)
        if w.startswith("\\") or w in {"max", "min", "log", "exp", "text"}:
            return w
        return "\\text{" + w + "}"

    out = re.sub(r"(?<![\\\w_])[a-z]{3,}(?![\w_])", word, out)
    out = re.sub(r"\s*([,;)])", r"\1", out)
    out = re.sub(r"\s+", " ", out).strip()
    # merge adjacent \text{a} \text{b} into one upright phrase so "token string" reads
    # as a phrase rather than as two isolated words
    while re.search(r"\\text\{([^{}]*)\}\s+\\text\{", out):
        out = re.sub(r"\\text\{([^{}]*)\}\s+\\text\{([^{}]*)\}", r"\\text{\1 \2}", out)
    return f"${out}$"


def main() -> int:
    survey = "--survey" in sys.argv
    decisions: dict[str, tuple[str, str]] = {}

    for path in TARGETS:
        text = path.read_text(encoding="utf-8")
        # protect fenced code blocks entirely
        fences: list[str] = []

        def stash(m: re.Match[str]) -> str:
            fences.append(m.group(0))
            return f"\x00FENCE{len(fences) - 1}\x00"

        guarded = re.sub(r"```.*?```", stash, text, flags=re.S)

        counts = {"math": 0, "code": 0, "review": 0}
        new_text = guarded
        for m in list(SPAN.finditer(guarded)):
            span = m.group(1)
            if span in decisions:
                kind, repl = decisions[span]
            else:
                kind = classify(span)
                repl = to_math(span) if kind == "math" else m.group(0)
                decisions[span] = (kind, repl)
            counts[kind] += 1
            if kind == "math":
                new_text = new_text.replace(m.group(0), repl, 1)
            elif kind == "review":
                print(f"  REVIEW  {path.name}: {span!r}")

        for i, fence in enumerate(fences):
            new_text = new_text.replace(f"\x00FENCE{i}\x00", fence)

        print(f"{path.name}: math={counts['math']} code={counts['code']} "
              f"needs-judgement={counts['review']}")
        if not survey:
            path.write_text(new_text, encoding="utf-8")

    print("\nproposed conversions (unique spans):")
    for span, (kind, repl) in sorted(decisions.items()):
        if kind == "math":
            print(f"  {span!r}  ->  {repl}")
    review = sorted({s for s, (k, _) in decisions.items() if k == "review"})
    if review:
        print(f"\nstill needing a judgement call ({len(review)}):")
        for s in review:
            print("  ?", s)
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
