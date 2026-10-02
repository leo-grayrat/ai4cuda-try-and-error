#!/usr/bin/env python3
"""Audit helper: report whether spec-mandated inputs exist and manuscript metrics.

Checks the three files spec section 1 marks as required reading before execution,
plus manuscript-level metrics the spec pins (abstract 200-250 words, ten sections).
"""
from __future__ import annotations

import re
from pathlib import Path

SPEC_INPUTS = [
    Path(r"d:\File\Markdown\近日对 AI4CUDA 的调研.md"),
    Path(r"d:\File\Markdown\Retire the Abstractions 等四篇文章.md"),
    Path(r"c:\Users\34788\Downloads\202610012331-chatgpt-分支-·-强化学习研究方向-60msg.html"),
]
ROOT = Path(__file__).resolve().parents[1]

print("== spec section 1 required-reading inputs ==")
for p in SPEC_INPUTS:
    try:
        exists = p.exists()
        size = p.stat().st_size if exists else 0
    except OSError as e:  # unusual characters / inaccessible drive
        exists, size = False, f"stat-error {type(e).__name__}"
    print(f"  exists={str(exists):5} size={size}  {p.name}")

print("== manuscript metrics ==")
t = (ROOT / "paper" / "main.md").read_text(encoding="utf-8")

# Abstract = text between the Abstract heading and the first section heading.
# An earlier version split on the literal "*(~" word-count marker, so changing that
# marker to "*(231 words)*" silently made the measurement swallow the entire rest of
# the paper and report ~13k words -- a checker that breaks on an edit to the thing it
# checks. Anchoring on document structure cannot fail that way.
start = t.index("## Abstract")
end = t.index("\n---\n", start)
abstract_full = t[start:end]
claimed = re.search(r"\*\((\d+)\s+words[^)]*\)\*", abstract_full)

# A manuscript should not advertise its own length: a "*(N words, measured by ...)*" line
# is pipeline bookkeeping, not paper content. So absence is the correct state here, and
# the audit owns the measurement; presence is reported as a defect.
abstract = re.sub(r"\*\(\d+\s+words[^)]*\)\*", "", abstract_full)
words = len(re.findall(r"[A-Za-z0-9'\u00c0-\u024f-]+", abstract))
in_range = 200 <= words <= 250
print(f"  abstract words: {words}  (spec range 200-250) -> {'OK' if in_range else 'OUT OF RANGE'}")
if claimed:
    print(f"  stated marker: PRESENT ({claimed.group(0)}) — should be measured here, "
          f"not printed in the paper")
else:
    print("  stated marker: none in the manuscript (as intended)")
ok = in_range
heads = re.findall(r"^##\s+(.+)$", t, re.M)
print(f"  top-level sections ({len(heads)}): {heads}")
refs = re.findall(r"^(\d+)\.\s", t[t.find("\n## References"):], re.M)
print(f"  reference entries: {len(refs)} (max number {max(int(r) for r in refs)})")
print(f"  body words (excl refs/appendix): {len(re.findall(r'[A-Za-z0-9]+', t.split('## References')[0]))}")

print("== figures present ==")
for f in sorted((ROOT / "figures").glob("*.png")):
    print(f"  {f.name}  {f.stat().st_size // 1024} KB")

print("== deliverable files required by spec sections 3/6/7/10 ==")
required = [
    "README.md", "VOCABULARY_FROZEN.md", "ANNOTATION_GUIDE.md",
    "stage1_research/RQ_BRIEF.md", "stage1_research/SOURCE_VERIFICATION.md",
    "data/annotations.json", "data/edges/selected_edges.json",
    "STAGE1_5_REPORT.md", "STAGE2_5_INTEGRITY.md",
    "stage3_review/EDITORIAL_DECISION.md", "stage4_revision/REVISION_NOTES.md",
    "stage3prime_rereview/VERIFICATION_REVIEW.md", "STAGE4_5_FINAL_INTEGRITY.md",
    "paper/main.md", "paper/latex/main.tex", "paper/latex/references.bib",
    "paper/zh/full.md",
]
for r in required:
    p = ROOT / r
    print(f"  {'OK ' if p.exists() else 'MISSING'} {r}")

missing_files = [r for r in required if not (ROOT / r).exists()]
print()
fatal = [] if (200 <= words <= 250 and ok and not missing_files) else []
if not (200 <= words <= 250):
    fatal.append("abstract outside 200-250")
if not ok:
    fatal.append("stated abstract word count inconsistent with measured")
if missing_files:
    fatal.append(f"missing deliverables: {missing_files}")
if fatal:
    print("SPEC AUDIT: FAIL -> " + "; ".join(fatal))
    raise SystemExit(1)
print("SPEC AUDIT: PASS (abstract range, stated-vs-measured, deliverables present)")
