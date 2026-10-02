#!/usr/bin/env python3
"""Strip pipeline bookkeeping from the manuscript's front matter and audit what is left.

The paper is a deliverable; the run that produced it is not. Three artefacts of the run
were still printed on the first page and in the abstract:

  - a status line "Draft: Stage 4 (v2, revised after five-reviewer simulated peer
    review; revision log in stage4_revision/REVISION_NOTES.md)"
  - an abstract word-count marker "(241 words, measured by scripts/audit_spec_coverage.py)"
  - a date suffix "(Stage 4 v2, post-review)" injected by make_latex.py

Appendix B already carries the production history, and audit_spec_coverage.py measures
the abstract length whether or not the manuscript advertises it.

This script removes the first two from paper/main.md, rewrites make_latex.py's metadata
so the PDF stops printing the third, and then reports every remaining reference to the
producing pipeline inside the body (before ## References), which is the number that
should stay near zero.

Usage: python scripts/clean_frontmatter.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EN = ROOT / "paper" / "main.md"
MKL = ROOT / "scripts" / "make_latex.py"

OLD_FRONT = re.compile(
    r"\*\*Position and analysis paper\.\*\* Draft: Stage 4.*?by that review\.\n",
    re.S,
)
NEW_FRONT = (
    "**Position and analysis.** Every number in Section 3 is computed from the public "
    "AdaExplore archive and is reproducible from the scripts accompanying this paper.\n"
)

OLD_MARKER = re.compile(
    r"\n\*\(\d+ words, measured by `scripts/audit_spec_coverage\.py`\)\*\n"
)

# what counts as pipeline-internal vocabulary in the body
BODY_PATTERNS = [
    r"\bStage \d",
    r"simulated peer review",
    r"revision log",
    r"measured by `?scripts/",
    r"\d+-assertion harness",
    r"our integrity harness",
    r"the review stage",
    r"automated review stage",
    r"five-reviewer",
    r"this pipeline",
    r"our pipeline",
    r"REVISION_NOTES",
    r"post-review",
]


def main() -> int:
    text = EN.read_text(encoding="utf-8")

    n = len(OLD_FRONT.findall(text))
    text = OLD_FRONT.sub(NEW_FRONT, text)
    print(f"front-matter status line replaced: {n}")

    m = len(OLD_MARKER.findall(text))
    text = OLD_MARKER.sub("\n", text)
    print(f"abstract word-count marker removed: {m}")

    if n == 0 or m == 0:
        print("WARNING: an anchor did not match; inspect before trusting this run",
              file=sys.stderr)

    EN.write_text(text, encoding="utf-8")

    mk = MKL.read_text(encoding="utf-8")
    if "(Stage 4 v2, post-review)" in mk:
        mk = mk.replace(
            '"--metadata", "date=2 October 2026 (Stage 4 v2, post-review)"',
            '"--metadata", "date=2 October 2026"',
        ).replace('"date=2 October 2026 (Stage 4 v2, post-review)"', '"date=2 October 2026"')
        MKL.write_text(mk, encoding="utf-8")
        print("make_latex.py date metadata no longer carries the stage suffix")

    body = text[: text.find("## References")]
    print("\nremaining pipeline references in the body:")
    total = 0
    for p in BODY_PATTERNS:
        hits = len(re.findall(p, body, re.I))
        if hits:
            total += hits
            print(f"  {hits:>2}  {p}")
    print(f"  --  total {total}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
