#!/usr/bin/env python3
"""Emit the exemplar table spec section 6 asks for: one transformation category,
its realizations across different tasks, and how dissimilar those realizations are.

Why a table of examples rather than a statistic: Section 3.5 retires the
"distinct diffs per label" count, because with 55 pairwise-distinct diff
fingerprints that number equals the occurrence count for ANY labelling and so
carries no information. The claim the spec actually wants -- that the same
transformation shows up as very different code on different kernels -- is a
qualitative claim, and it is checkable by reading the edges, not by counting them.

The table pairs each example with structural facts computed by select_edges.py
(hunk count, changed-line count, code regions touched) and a short realization
note taken from the annotation's own quoted evidence, so a reader can go and
diff the files.

Usage: python scripts/exemplar_table.py [LABEL]     # default: Fuse
Writes: data/exemplar_<label>.md and prints the markdown table.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EDGES = ROOT / "data" / "edges"


def main() -> int:
    label = sys.argv[1] if len(sys.argv) > 1 else "Fuse"
    blob = json.loads((ROOT / "data" / "annotations.json").read_text(encoding="utf-8"))
    by_id = {
        e["edge_id"]: e
        for e in json.loads((EDGES / "selected_edges.json").read_text(encoding="utf-8"))
    }

    rows = []
    for r in blob["annotations"]:
        e = by_id[r["edge_id"]]
        if label not in {r["primary"], *r.get("secondary", [])}:
            continue
        # first quoted line of evidence from the annotation, stripped of diff marks
        quotes = re.findall(r"`([^`]{8,120})`", r.get("reasoning", ""))
        evidence = ""
        for q in quotes:
            q = q.replace("\n", " ").strip()
            if q[:1] in "+-":
                evidence = q.lstrip("+- ").strip()
                break
        rows.append(
            {
                "edge": r["edge_id"],
                "task": e["task_id"],
                "level": e["level"],
                "role": "primary" if r["primary"] == label else "secondary",
                "speedup": e["speedup_vs_parent"],
                "hunks": e["hunks"],
                "changed": e["changed_total"],
                "regions": e["n_distinct_regions"],
                "evidence": (evidence[:78] + "…") if len(evidence) > 78 else evidence,
            }
        )

    if not rows:
        print(f"no edges labelled {label}", file=sys.stderr)
        return 1

    rows.sort(key=lambda x: (x["level"], x["task"]))
    tasks = {r["task"] for r in rows}
    region_sets = {tuple(sorted(set(by_id[r["edge"]]["hunk_regions"]))) for r in rows}

    header = (
        f"| edge | task | lvl | role | ×speedup | hunks | lines | regions | realization evidence (quoted from annotation) |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = [
        f"| `{r['edge']}` | {r['task']} | {r['level']} | {r['role']} | {r['speedup']:.2f} "
        f"| {r['hunks']} | {r['changed']} | {r['regions']} | `{r['evidence']}` |"
        for r in rows
    ]
    table = header + "\n".join(lines) + "\n"
    summary = (
        f"\n**`{label}`: {len(rows)} edges, {len(tasks)} distinct tasks, "
        f"{len(region_sets)} distinct region sets.** "
        "Distinct-count alone is not evidence (Section 3.5); the point of this table is "
        "that the *shape* of each realization differs — different numbers of hunks, "
        "different code regions, different changed lines — while the named decision recurs.\n"
    )
    out = ROOT / "data" / f"exemplar_{label}.md"
    out.write_text(table + summary, encoding="utf-8")
    print(table)
    print(summary)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
