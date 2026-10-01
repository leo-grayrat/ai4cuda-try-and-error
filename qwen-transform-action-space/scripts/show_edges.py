#!/usr/bin/env python3
"""Print the selected-edge table, or the full diff of specific edges, for reading.

Viewing aid for the annotation pass. No classification happens here.

  python scripts/show_edges.py --summary
  python scripts/show_edges.py --diff L2_2_20_100_101 [more ids...]
  python scripts/show_edges.py --batch 0 10     # diffs for edges [0,10) of the table
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EDGES = ROOT / "data" / "edges"


def load() -> list[dict]:
    return json.loads((EDGES / "selected_edges.json").read_text(encoding="utf-8"))


def summary(sel: list[dict]) -> None:
    print(f"{'idx':>3} {'edge_id':<26} {'lv':<3} {'hunk':>4} {'reg':>3} "
          f"{'chg':>5} {'spd':>6}  regions")
    for i, e in enumerate(sel):
        print(f"{i:>3} {e['edge_id']:<26} {e['level']:<3} {e['hunks']:>4} "
              f"{e['n_distinct_regions']:>3} {e['changed_total']:>5} "
              f"{e['speedup_vs_parent']:>6.2f}  "
              f"{','.join(sorted(set(e['hunk_regions'])))[:70]}")


def show(e: dict) -> None:
    p = EDGES / f"{e['edge_id']}.diff"
    print("=" * 100)
    print(f"EDGE {e['edge_id']}  task={e['task_id']} {e['level']} "
          f"parent={e['parent_node']} child={e['child_node']} "
          f"speedup_vs_parent={e['speedup_vs_parent']} "
          f"parent_fast_p={e['parent_fast_p']} child_fast_p={e['child_fast_p']} "
          f"hunks={e['hunks']} regions={e['hunk_regions']}")
    print("-" * 100)
    print(p.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--diff", nargs="+", default=[])
    ap.add_argument("--batch", nargs=2, type=int, metavar=("START", "END"))
    ap.add_argument("--max-bytes", type=int, default=14000)
    args = ap.parse_args()
    sel = load()

    if args.summary:
        summary(sel)
    for eid in args.diff:
        hit = [e for e in sel if e["edge_id"] == eid]
        if not hit:
            print(f"!! unknown edge {eid}", file=__import__("sys").stderr)
            continue
        show(hit[0])
    if args.batch:
        s, t = args.batch
        for e in sel[s:t]:
            txt = (EDGES / f"{e['edge_id']}.diff").read_text(encoding="utf-8")
            if len(txt) > args.max_bytes:
                txt = txt[: args.max_bytes] + f"\n...[TRUNCATED {len(txt)} bytes total]"
            print("=" * 100)
            print(f"EDGE {e['edge_id']} task={e['task_id']} {e['level']} "
                  f"parent={e['parent_node']} child={e['child_node']} "
                  f"spd_vs_parent={e['speedup_vs_parent']} "
                  f"parent_fast_p={e['parent_fast_p']} child_fast_p={e['child_fast_p']} "
                  f"hunks={e['hunks']} regions={e['hunk_regions']}")
            print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
