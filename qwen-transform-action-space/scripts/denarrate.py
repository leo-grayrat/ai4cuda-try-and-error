#!/usr/bin/env python3
"""Move the pipeline's production narrative out of the paper body.

The manuscript's argument is about action spaces. Three passages instead narrate how the
pipeline that produced it behaved — a harness that missed a statistical error, a review
stage whose quotation could not be retrieved, and a section listing what the authors did
not do. That material is worth keeping, but as provenance, not as body text: a reader of
a position paper has no use for the run log of one particular agent loop, and its presence
pushes the actual claims off the page.

  1. Section 3.6 loses both incident paragraphs; they are replaced by one paragraph that
     states how to read the table and points to Appendix B.
  2. Section 7.5 ("Three meta-findings about how this paper was produced") is relocated
     verbatim into a new Appendix B, re-headed as production record.
  3. Section 7.1's self-narrating lead-in is rewritten into ordinary methods voice; the
     measurements it reports are unchanged.

Usage: python scripts/denarrate.py [--dry-run]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "main.md"

S36_START = "**The methodological lesson we keep.**"
S36_END = "assigned to attack, and the author must verify its citations with the same discipline applied to any other source."

S36_REPLACEMENT = """**How to read the table.** The quantity that discriminates is the *excess over each descriptor's own null*, not the
raw mutual-information level, because a plug-in estimate of mutual information rises with the number of occupied
cells; both descriptors are computed on the same edges with the same outcomes, and the permutation calibration makes
each null specific to its own granularity. Plug-in and Miller-Madow-corrected levels are both reported so the choice
of correction need not be taken on trust. The production history of this measurement, including the version that
reversed the sign by applying a marginal-entropy correction where a joint one was required, is recorded in Appendix B."""

S75_START = "### 7.5 Three meta-findings"
S75_END = "## 8. Conclusion"

APPENDIX_B = """## Appendix B — Production record

This appendix records how the analysis behind Section 3 came to be stated differently at earlier stages, and what the
automated review stage of this run could and could not establish. It is placed here rather than in the body because it
bears on how much weight to give the numbers, not on the argument about action spaces.

"""


def main() -> int:
    dry = "--dry-run" in sys.argv
    text = PAPER.read_text(encoding="utf-8")

    i = text.find(S36_START)
    j = text.find(S36_END)
    assert i > 0 and j > i, "Section 3.6 incident paragraphs not found"
    block = text[i : j + len(S36_END)]
    text = text.replace(block, S36_REPLACEMENT, 1)
    print(f"removed {len(block.split())} words of run-log narrative from Section 3.6")

    a = text.find(S75_START)
    b = text.find(S75_END)
    assert a > 0 and b > a, "Section 7.5 not found"
    sec75 = text[a:b]
    text = text[:a] + text[b:]
    print(f"relocated Section 7.5 ({len(sec75.split())} words) out of the body")

    # strip its own heading; the relocated content becomes the appendix body
    body = re.sub(r"^### 7\.5[^\n]*\n", "", sec75, flags=re.M).strip()
    text = text.rstrip() + "\n\n---\n\n" + APPENDIX_B + body + "\n"
    print("appended relocated material to Appendix B")

    lead = "We did not merely declare these. Criteria 2, 3 and 4 were executed before releasing this draft, and the result is\nuncomfortable."
    if lead in text:
        text = text.replace(
            lead,
            "Criteria 2, 3 and 4 have been executed. The measurements are reported below; criterion 1 (a genuinely\nindependent, preferably human second annotation) has not, and remains the paper's main open requirement.",
            1,
        )
        print("rewrote Section 7.1 lead-in into methods voice")

    text = text.replace("Section 3.6, Section 7.5", "Section 3.6, Appendix B", 1)
    text = text.replace(
        "our own review stage was five seats filled from one model family,",
        "the review stage that produced these corrections was five seats filled from one model family,",
        1,
    )

    if dry:
        print("\n(dry run, nothing written)")
        return 0
    PAPER.write_text(text, encoding="utf-8")
    print(f"wrote {PAPER}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
