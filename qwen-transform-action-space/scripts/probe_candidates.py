#!/usr/bin/env python3
"""Emit diffs for candidate edges that were NOT among the 55 annotated ones.

Needed for the bottom-up open-coding control in Section 7.1: a vocabulary induced from
the same edges that were used to justify the frozen list proves nothing, so this
samples from the remaining eligible edges (out of 2,510) and writes their diffs for a
coder who has not seen the frozen twelve categories.

Usage:
    python scripts/probe_candidates.py --n 24
Writes data/probe_edges/manifest.json and data/probe_edges/<edge_id>.diff, and prints a
compact index so a coding agent can iterate over it.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "adaexplore"
EDGES = ROOT / "data" / "edges"
PROBE = ROOT / "data" / "probe_edges"
SEED = 20261003


def unified(a_text: str, b_text: str) -> list[str]:
    import difflib

    a, b = a_text.splitlines(), b_text.splitlines()
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    out: list[str] = []
    for group in sm.get_grouped_opcodes(3):
        first, last = group[0], group[-1]
        i1, j1, i2, j2 = first[1], first[3], last[2], last[4]
        out.append(f"@@ -{i1 + 1},{i2 - i1} +{j1 + 1},{j2 - j1} @@")
        for tag, x1, x2, y1, y2 in group:
            if tag == "equal":
                out += [" " + a[k] for k in range(x1, x2)]
                continue
            if tag in ("delete", "replace"):
                out += ["-" + a[k] for k in range(x1, x2)]
            if tag in ("insert", "replace"):
                out += ["+" + b[k] for k in range(y1, y2)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--max-bytes", type=int, default=9000)
    args = ap.parse_args()

    cands = json.loads((EDGES / "edge_index.json").read_text(encoding="utf-8"))
    used = {
        e["edge_id"]
        for e in json.loads((EDGES / "selected_edges.json").read_text(encoding="utf-8"))
    }
    # exclude the annotated sample and any edge sharing a task with it, so induced
    # categories cannot be a re-derivation of the ones already reported
    used_tasks = {e["task_id"] for e in json.loads(
        (EDGES / "selected_edges.json").read_text(encoding="utf-8"))}
    pool = [c for c in cands if c["edge_id"] not in used and c["task_id"] not in used_tasks]
    if len(pool) < args.n:
        pool = [c for c in cands if c["edge_id"] not in used]

    # spread across tasks, then across improvement size
    by_task: dict[str, list[dict]] = {}
    for c in pool:
        by_task.setdefault(c["task_id"], []).append(c)
    rng = random.Random(SEED)
    tasks = sorted(by_task)
    rng.shuffle(tasks)
    chosen: list[dict] = []
    for t in tasks:
        chosen.append(max(by_task[t], key=lambda e: e["speedup_vs_parent"]))
        if len(chosen) >= args.n:
            break

    PROBE.mkdir(parents=True, exist_ok=True)
    manifest = []
    for c in chosen:
        run = next(p for p in DATA.iterdir() if p.is_dir() and p.name == c["run"])
        tdir = run / c["task_id"]
        a = (tdir / f"step_{c['parent_node']}.py").read_text(encoding="utf-8", errors="replace")
        b = (tdir / f"step_{c['child_node']}.py").read_text(encoding="utf-8", errors="replace")
        d = unified(a, b)
        text = "\n".join(d)
        trunc = len(text) > args.max_bytes
        (PROBE / f"{c['edge_id']}.diff").write_text(
            (text[: args.max_bytes] + "\n...[TRUNCATED]" if trunc else text),
            encoding="utf-8",
        )
        plus = sum(1 for l in d if l.startswith("+") and not l.startswith("+++"))
        minus = sum(1 for l in d if l.startswith("-") and not l.startswith("---"))
        hunks = len([l for l in d if l.startswith("@@")])
        manifest.append(
            {
                "edge_id": c["edge_id"],
                "task_id": c["task_id"],
                "level": c["level"],
                "speedup_vs_parent": c["speedup_vs_parent"],
                "hunks": hunks,
                "added": plus,
                "removed": minus,
                "diff_file": f"data/probe_edges/{c['edge_id']}.diff",
                "truncated": trunc,
            }
        )

    (PROBE / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"probe set: {len(manifest)} edges across {len({m['task_id'] for m in manifest})} tasks")
    print(f"tasks overlapping the annotated sample: "
          f"{len({m['task_id'] for m in manifest} & used_tasks)} (want 0)")
    for m in manifest:
        flag = " [TRUNCATED: re-read fully]" if m["truncated"] else ""
        print(f"  {m['edge_id']:<24} {m['level']} hunks={m['hunks']:>2} "
              f"+{m['added']}/-{m['removed']} x{m['speedup_vs_parent']:.2f}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
