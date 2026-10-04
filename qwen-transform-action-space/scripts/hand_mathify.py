#!/usr/bin/env python3
"""Hand-decided conversion of the manuscript's remaining notation into LaTeX math.

Each entry was chosen by reading scripts/list_notation.py's output line by line; nothing
here is a classifier. Patterns are anchored on delimiters that already exist in the source
(backticks, table pipes, parentheses) and replace only the anchored fragment, so the rest
of the sentence is never rewritten — an earlier version of this script replaced whole
lines using text copied from a truncated listing and would have silently deleted the tail
of every one of those sentences.

Deliberately left alone: prose arrows that are English with a glyph ("NVIDIA → AMD",
"slow→fast", "parent→child", "56%→92%"), and reference annotations that quote sources.

Every pattern must match at least once; a miss aborts with no write, because a drifted
anchor means the decision below no longer applies to the file.

Usage: python scripts/hand_mathify.py [--dry-run]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EN = ROOT / "paper" / "main.md"

# (pattern, replacement, what it is)
RULES: list[tuple[str, str, str]] = [
    # --- statistics written in running prose -------------------------------------
    (r"\(p = 0\.0005\)", r"($p = 0.0005$)", "abstract/§8 p-value"),
    (r"\(p = 0\.752\)", r"($p = 0.752$)", "abstract p-value"),
    (r"\(p = 0\.436\)", r"($p = 0.436$)", "abstract p-value"),
    (r"\(p = 0\.106,", r"($p = 0.106,", "abstract p-value"),
    (r"kappa = 0\.687\)", r"$\\kappa = 0.687$)", "abstract kappa"),
    (r"\bMI 0\.451 bits\b", r"MI 0.451 bits", "already prose, left as-is"),

    # --- speedup multiples --------------------------------------------------------
    (r"2\.10x over state-of-the-art", r"$2.10\\times$ over state-of-the-art", "§5.1 K-Search"),
    (r"14\.3x on complex MoE", r"$14.3\\times$ on complex MoE", "§5.1 K-Search"),
    (r"reports 2\.10x average", r"reports $2.10\\times$ average", "§7 K-Search"),
    (r"up to 14\.3x over evolutionary", r"up to $14.3\\times$ over evolutionary", "§7 K-Search"),
    (r"2\.10x–14\.3x gains", r"$2.10\\times$--$14.3\\times$ gains", "§8 K-Search"),
    (r"show >10× gains", r"show $> 10\\times$ gains", "§7.3"),
    (r"\(327× and 69× versus parent\)", r"($327\\times$ and $69\\times$ versus parent)", "§3.7a"),
    (r"hunk count × changed-line bucket", r"hunk count $\\times$ changed-line bucket", "§3.5"),
    (r"\| ×vs parent \|", r"| $\\times$ vs parent |", "§3.5 exemplar header"),

    # --- the one identity that is a formula, not prose ---------------------------
    (r"identically log2\(n\)", r"identically $\\log_{2} n$", "fig 1 caption"),
    (r"over ~10³–10⁴ steps", r"over $\\sim 10^{3}$--$10^{4}$ steps", "§2.1 table"),

    # --- the reading tree: these relations ARE the formalism ----------------------
    (r"`Free-plan ~ Structured-action > Direct`",
     r"$\\mathrm{Free\\text{-}plan} \\sim \\mathrm{Structured\\text{-}action} > \\mathrm{Direct}$",
     "§7.2 mention"),
    (r"\| Free-plan ~ Structured-action > Direct \|",
     r"| $\\mathrm{Free\\text{-}plan} \\sim \\mathrm{Structured\\text{-}action} > \\mathrm{Direct}$ |",
     "§4.3 row"),
    (r"\| Structured-action > Direct but ~ turn-level decomposition \|",
     r"| $\\mathrm{Structured\\text{-}action} > \\mathrm{Direct}$ but $\\sim$ turn-level decomposition |",
     "§4.3 row"),
    (r"\| Oracle-action >> Structured-action \|",
     r"| $\\mathrm{Oracle\\text{-}action} \\gg \\mathrm{Structured\\text{-}action}$ |",
     "§4.3 row"),
    (r"\| Oracle-action ~ Direct \|",
     r"| $\\mathrm{Oracle\\text{-}action} \\sim \\mathrm{Direct}$ |",
     "§4.3 row"),
    (r"\| Structured-action ~ Brute-search \|",
     r"| $\\mathrm{Structured\\text{-}action} \\sim \\mathrm{Brute\\text{-}search}$ |",
     "§4.3 row"),
    (r"`Free-plan ≈ Structured-action ⇒ no paper`",
     r"$\\mathrm{Free\\text{-}plan} \\approx \\mathrm{Structured\\text{-}action} \\Rightarrow \\text{no paper}$",
     "§5.1 branch"),

    # --- pre-registered thresholds: relations inside table cells ------------------
    (r"\| Any single category > 40% of edges ⇒ vocabulary too coarse \|",
     r"| Any single category $>$40\\% of edges $\\Rightarrow$ vocabulary too coarse |",
     "§3.4 red line"),
    (r"\| `Other` > 25% ⇒ the conjecture may be false \|",
     r"| `Other` $>$25\\% $\\Rightarrow$ the conjecture may be false |",
     "§3.4 red line"),

    # --- notation stranded in code spans ------------------------------------------
    (r"`Q\(P, one particular token string\)`",
     r"$Q(P, \\text{one particular token string})$",
     "§2.3 value function"),
    # the span wraps across a source line break, so the pattern must not assume one line
    (r"`work_per_thread\s*\n?:\s*1 → 2`",
     r"$\\mathit{work\\_per\\_thread} : 1 \\to 2$",
     "§2.2 worked example"),
    # §2.5 states the transfer claim as an equation but it sat in a code fence, so the
    # underscores read as literal underscores rather than as subscripts
    (r"code_A ≠ code_B\s+but\s+a_A = a_B = WarpReduce",
     r"$\\mathit{code}_A \\neq \\mathit{code}_B$ \\quad but \\quad "
     r"$a_A = a_B = \\mathrm{WarpReduce}$",
     "§2.5 transfer equation"),
]


def main() -> int:
    dry = "--dry-run" in sys.argv
    lenient = "--lenient" in sys.argv          # apply what matches; report the rest
    target = EN
    if "--file" in sys.argv:
        target = Path(sys.argv[sys.argv.index("--file") + 1])
    text = target.read_text(encoding="utf-8")

    misses, applied, already = [], 0, 0
    for pattern, repl, what in RULES:
        hits = len(re.findall(pattern, text))
        if hits == 0:
            # "already prose, left as-is" entries are intentional no-ops
            if repl == pattern.strip("`"):
                continue
            # a rule whose output is already in the file was applied by an earlier run;
            # treating that as a miss would make the script un-runnable after a partial pass
            probe = repl.replace("$", "").replace("\\\\", "\\")
            key = re.sub(r"\\[a-zA-Z]+|[{}^_]", "", probe).strip()
            fragment = key.split()[0] if key.split() else ""
            if fragment and fragment in text:
                already += 1
                print(f"  done  {what} (already converted)")
                continue
            misses.append((pattern, what))
            print(f"  MISS  {what}: {pattern}")
            continue
        text = re.sub(pattern, lambda _m, r=repl: r, text)
        applied += 1
        print(f"  ok[{hits}] {what}: {pattern[:56]}")

    if misses and not lenient:
        print(f"\n{len(misses)} anchor(s) did not match; nothing written", file=sys.stderr)
        return 1
    print(f"\napplied {applied} rules; {already} already in place"
          + (f"; {len(misses)} not present in this file" if lenient else ""))
    if not dry:
        target.write_text(text, encoding="utf-8")
        print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
