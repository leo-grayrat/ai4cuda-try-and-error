#!/usr/bin/env python3
"""Apply the reader's edit list to the manuscript.

A read-through (not a keyword sweep) found that the body still narrates the run that
produced it: which draft said what, which review stage caught which error, which internal
file stores which artefact. A reader of the paper has no use for any of that, and its
presence pushes the actual claims off the page. Each edit below keeps the substantive
point and drops the narration; the production history stays only in Appendix B.

Also fixes two reading-comprehension defects (an unresolved "L2/L3 claims" label, and an
abstract that cites 12 categories where 10 appear) and the duplicated figure numbering
that made the PDF print "Figure 1: Figure 1. ...".

Every pair is asserted. A miss is reported and the file is still written, so a drifted
anchor shows up instead of silently skipping an edit.

Usage: python scripts/readers_edits.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EN = ROOT / "paper" / "main.md"
ZH1 = ROOT / "paper" / "zh" / "part1.md"
ZH2 = ROOT / "paper" / "zh" / "part2.md"

# (file, old, new)
EDITS: list[tuple[Path, str, str]] = [
    # ---- 1.5 contribution list: drop the self-congratulation ---------------------
    (EN,
     "We consider the negative and self-undermining results in Sections 3.6 and 3.7 to be the most\n"
     "valuable part of this paper, and we have kept them rather than removing them.",
     "Sections 3.6 and 3.7 carry the results that bound the programme, and they are reported there in full."),

    # ---- 3.2 sampling bias: state the consequence, not who missed it -------------
    (EN,
     "the bias has two consequences we did not appreciate in the first draft.",
     "the bias has two consequences for how these numbers may be used."),

    # ---- 3.3 annotation: say what was done and where the risk is handled ---------
    (EN,
     "The design sketch called for a human to read thirty to fifty edges. This execution substituted an automated\n"
     "annotator. The consequence is unavoidable:",
     "Annotation was performed by a model rather than by a human reader; the design sketch called for a human.\n"
     "The consequence is unavoidable:"),

    # ---- 3.5 withdrawn claims: keep the arithmetic, drop the draft history -------
    (EN,
     "**What we withdrew.** The first draft made two claims from the per-category diversity table that are void:",
     "**Two claims that do not survive this table.**"),
    (EN,
     " The corresponding sentence is deleted.",
     ""),

    # ---- 3.6 the reversal: compare readings, not drafts --------------------------
    (EN,
     "Low entropy by itself is uninteresting, and the first draft of this section argued exactly that:",
     "Low entropy by itself is uninteresting, and a surface reading argues exactly that:"),
    (EN,
     "The first draft calibrated it wrongly.",
     "Calibrating only one side gives the opposite answer."),
    (EN,
     "This reverses the first draft's finding, and the reversal does not hinge on a choice of bias formula:",
     "This reverses the uncalibrated comparison, and it does not hinge on a choice of bias formula:"),
    (EN,
     "The production history of this measurement, including the version that\nreversed the sign by applying a marginal-entropy correction where a joint one was required, is recorded in Appendix B.",
     ""),

    # ---- 4.2 brute-search arm: justify it by the objection, not by the review ----
    (EN,
     "Brute-search was added after review exposed that the strongest objection to this direction was raised in the\n"
     "paper and then left unanswered.",
     "Brute-search is the arm that the strongest objection demands: if plain enumeration already solves the\n"
     "problem, no relabelling of the action unit is needed."),

    # ---- 4.3 table: drop the provenance note -------------------------------------
    (EN, " The last row is a correction forced by review.", ""),

    # ---- 4.4 stop-condition: make it a rule, not a diary entry -------------------
    (EN,
     "In the course of producing this paper we deleted two retrieval-tooling scripts on exactly those grounds.",
     "A tool that cannot change any of these judgements should not be built."),

    # ---- 5.1 differentiation: state what fails, not who failed -------------------
    (EN,
     "Our first draft asserted a clean three-way distinction against whole-candidate search, against edit spans, and\n"
     "against plan/action unification. Two of those three legs no longer survive our own review.",
     "The distinction is cleaner than the data supports: two of its three legs do not survive the measurements in\n"
     "Section 3."),

    # ---- 5.2 / 5.3: remove draft references --------------------------------------
    (EN, "which an earlier draft of this paper wrongly read it as one.",
     "which is easy to read as one."),
    (EN, "In the first draft we raised this objection and then left it standing, because none of our four conditions\ncould answer it.",
     "None of the four original conditions could answer it."),

    # ---- 8 conclusion: one pointer at Appendix B, no narration -------------------
    (EN,
     "The answer after one round of adversarial review is:",
     "The answer is:"),
    (EN,
     "That claim is weaker than it first appears: relabelling the same edges without being given the\ncategory definitions leaves the same test insignificant",
     "It is bounded by a control: relabelling the same edges without being given the category definitions leaves\nthe same test insignificant"),
    (EN,
     "Neither withdrawal softens the other. The corrected reading replaces an earlier version of\nSection 3.6 whose conclusion was the opposite; why the error survived every automated check,\nand what that says about self-verification in pipelines like this one, is recorded in Appendix B.",
     "Neither withdrawal softens the other; the calibration behind the corrected reading is recorded in\nAppendix B."),

    # ---- reference note: drop the script name ------------------------------------
    (EN,
     "its URL and page title were resolved programmatically by `scripts/verify_references.py` like any other reference",
     "it is non-archival and its page title was resolved as for any other source"),

    # ---- internal paths are artefacts, not body text -----------------------------
    (EN, "an audit of this identity is emitted in `data/informativeness_calibration.json` (`identity_audit`)",
     "the audit of this identity ships with the data accompanying this paper"),
    (EN, "are in `data/exemplar_Fuse.md`",
     "are in the exemplar table accompanying this paper"),
    (EN, "`data/opencode_result.md`", "the open-coding record accompanying this paper"),

    # ---- comprehension: the claims are named, not numbered L2/L3 -----------------
    (EN, "The L2 and L3 claims of Section 1.4 are untouched.",
     "The search and learning claims of Section 1.4 are untouched."),

    # ---- abstract: 12 defined, 10 observed ---------------------------------------
    (EN,
     "a frozen 12-category vocabulary covers 90.9% of edges using 10\ncategories,",
     "a frozen 12-category vocabulary covers 90.9% of edges, using 10 of its 12\ncategories,"),
]


def main() -> int:
    texts: dict[Path, str] = {}
    for path in (EN, ZH1, ZH2):
        texts[path] = path.read_text(encoding="utf-8")

    applied, missed = 0, []
    for path, old, new in EDITS:
        text = texts[path]
        if old in text:
            text = text.replace(old, new, 1)
            applied += 1
            print(f"  ok    {path.name}: {old[:52]}…")
        else:
            missed.append((path.name, old[:70]))
            print(f"  miss  {path.name}: {old[:70]}…")
        texts[path] = text

    # LaTeX already numbers figures, so a "Figure N." prefix inside the Markdown alt text
    # prints twice ("Figure 1: Figure 1. Description length..."). Strip the hand-written
    # prefix and the bilingual gloss, keeping the caption sentence itself.
    for path in (EN, ZH1, ZH2):
        text = texts[path]
        n1 = len(re.findall(r"!\[Figure \d+\. ", text))
        text = re.sub(r"(!\[)Figure \d+\. ", r"\1", text)
        n2 = len(re.findall(r"!\[图 \d+（Figure \d+）：", text))
        text = re.sub(r"!\[图 (\d+)（Figure [0-9]+）：", r"![图 \1：", text)
        if n1 or n2:
            applied += n1 + n2
            print(f"  ok    {path.name}: de-duplicated {n1 + n2} figure caption(s)")
        texts[path] = text

    for path, text in texts.items():
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path}")
    print(f"\napplied {applied}/{len(EDITS) + 5}; {len(missed)} missed")
    for name, frag in missed:
        print("  PENDING:", name, frag)
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
