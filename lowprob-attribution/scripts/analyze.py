"""Analysis: compute sentinel (low-probability) rates by token type and context type.

Reads tokens.jsonl (from align.py), produces:
    - by_token_type.csv:  sentinel_rate, count, total per token_type
    - by_context_type.csv: sentinel_rate, count, total per context_type
    - by_interaction.csv:  sentinel_rate for token_type × context_type
    - by_condition.csv:    overall sentinel_rate per feedback condition (full/truncated)

Also prints a summary table.
"""

from __future__ import annotations

import json
import csv
import sys
from collections import defaultdict
from pathlib import Path


def load_tokens(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rate(rows: list[dict]) -> tuple[float, int, int]:
    total = len(rows)
    sent = sum(1 for r in rows if r.get("is_sentinel"))
    rate = sent / total if total else 0.0
    return rate, sent, total


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("tokens_jsonl")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    tokens = load_tokens(Path(args.tokens_jsonl))
    out = Path(args.out_dir)

    # --- by token type ---
    by_type = defaultdict(list)
    for t in tokens:
        by_type[t["token_type"]].append(t)

    print("\n=== Sentinel rate by token type ===")
    print(f"{'type':<22} {'sentinel_rate':>14} {'sentinel':>10} {'total':>10}")
    with (out / "by_token_type.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["token_type", "sentinel_rate", "sentinel_count", "total_count"])
        for tp in sorted(by_type):
            r, s, n = rate(by_type[tp])
            print(f"{tp:<22} {r:>14.4f} {s:>10} {n:>10}")
            w.writerow([tp, f"{r:.4f}", s, n])

    # --- by context type ---
    by_ctx = defaultdict(list)
    for t in tokens:
        by_ctx[t["context_type"]].append(t)

    print("\n=== Sentinel rate by context type ===")
    print(f"{'context':<28} {'sentinel_rate':>14} {'sentinel':>10} {'total':>10}")
    with (out / "by_context_type.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["context_type", "sentinel_rate", "sentinel_count", "total_count"])
        for ctx in sorted(by_ctx):
            r, s, n = rate(by_ctx[ctx])
            print(f"{ctx:<28} {r:>14.4f} {s:>10} {n:>10}")
            w.writerow([ctx, f"{r:.4f}", s, n])

    # --- interaction: token_type × context_type ---
    by_inter = defaultdict(list)
    for t in tokens:
        key = (t["token_type"], t["context_type"])
        by_inter[key].append(t)

    print("\n=== Sentinel rate: token_type × context_type (only cells ≥20 tokens) ===")
    with (out / "by_interaction.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["token_type", "context_type", "sentinel_rate", "sentinel_count", "total_count"])
        for (tp, ctx) in sorted(by_inter):
            r, s, n = rate(by_inter[(tp, ctx)])
            if n >= 20:
                print(f"  {tp:<20} × {ctx:<24} = {r:.4f}  ({s}/{n})")
                w.writerow([tp, ctx, f"{r:.4f}", s, n])

    # --- by run condition (if present in tokens) ---
    by_cond = defaultdict(list)
    for t in tokens:
        cond = t.get("condition", "unknown")
        by_cond[cond].append(t)

    if len(by_cond) > 1:
        print("\n=== Sentinel rate by feedback condition ===")
        with (out / "by_condition.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["condition", "sentinel_rate", "sentinel_count", "total_count"])
            for cond in sorted(by_cond):
                r, s, n = rate(by_cond[cond])
                print(f"  {cond:<12} = {r:.4f}  ({s}/{n})")
                w.writerow([cond, f"{r:.4f}", s, n])


if __name__ == "__main__":
    main()
