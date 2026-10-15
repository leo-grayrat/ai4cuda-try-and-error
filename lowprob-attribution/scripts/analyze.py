"""Analysis: lowprob rates by token type, context type, and their interaction.

Reads tokens.jsonl (from align.py) and writes one CSV per grouping plus a
printed summary.

Observables per PLAN.md section 9: lowprob(<=-2), lowprob(<=-5),
not_in_topn (chosen token absent from the returned top_logprobs) and the mean
logprob. The -9999 sentinel is NOT used - DeepSeek returns the real logprob for
a token outside top-N, and the sentinel rate is 0% at every top_logprobs setting
tested.

Usage:
    python lowprob-attribution/scripts/analyze.py lowprob-attribution/results/tokens.jsonl \
        --out-dir lowprob-attribution/results
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

COLUMNS = [
    "group", "total", "lowprob_2", "lowprob_2_rate", "lowprob_5",
    "lowprob_5_rate", "not_in_topn", "not_in_topn_rate", "mean_logprob",
]


def load_tokens(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize(rows: list[dict]) -> dict:
    total = len(rows)
    if not total:
        return {}
    logprobs = [r["logprob"] for r in rows if r.get("logprob") is not None]
    low2 = sum(1 for r in rows if r.get("lowprob_2"))
    low5 = sum(1 for r in rows if r.get("lowprob_5"))
    outside = sum(1 for r in rows if r.get("in_topn") is False)
    return {
        "total": total,
        "lowprob_2": low2,
        "lowprob_2_rate": low2 / total,
        "lowprob_5": low5,
        "lowprob_5_rate": low5 / total,
        "not_in_topn": outside,
        "not_in_topn_rate": outside / total,
        "mean_logprob": (sum(logprobs) / len(logprobs)) if logprobs else None,
    }


def write_group(out_dir: Path, name: str, header: str, grouped: dict) -> None:
    path = out_dir / f"by_{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([header] + COLUMNS[1:])
        print(f"\n=== by {name} ===")
        print(f"{header:<24}" + "".join(f"{c:>16}" for c in COLUMNS[1:]))
        for key in sorted(grouped):
            stats = summarize(grouped[key])
            if not stats:
                continue
            writer.writerow([
                key, stats["total"], stats["lowprob_2"], f"{stats['lowprob_2_rate']:.4f}",
                stats["lowprob_5"], f"{stats['lowprob_5_rate']:.4f}",
                stats["not_in_topn"], f"{stats['not_in_topn_rate']:.4f}",
                f"{stats['mean_logprob']:.4f}" if stats["mean_logprob"] is not None else "",
            ])
            print(f"{str(key):<24}" + "".join(f"{v:>16}" for v in (
                stats["total"], stats["lowprob_2"], f"{stats['lowprob_2_rate']:.4f}",
                stats["lowprob_5"], f"{stats['lowprob_5_rate']:.4f}",
                stats["not_in_topn"], f"{stats['not_in_topn_rate']:.4f}",
                f"{stats['mean_logprob']:.4f}" if stats["mean_logprob"] is not None else "-",
            )))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tokens_jsonl")
    ap.add_argument("--out-dir", default="lowprob-attribution/results")
    ap.add_argument("--min-cell", type=int, default=20,
                    help="smallest cell reported in the interaction table")
    args = ap.parse_args()

    tokens = load_tokens(Path(args.tokens_jsonl))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overall = summarize(tokens)
    print(f"tokens: {overall.get('total')}  "
          f"lowprob(<=-2)={overall.get('lowprob_2')} ({overall.get('lowprob_2_rate', 0):.1%})  "
          f"lowprob(<=-5)={overall.get('lowprob_5')} ({overall.get('lowprob_5_rate', 0):.1%})  "
          f"outside_topn={overall.get('not_in_topn')}  "
          f"mean_logprob={overall.get('mean_logprob')}")

    for name, key in (("token_type", "token_type"), ("context_type", "context_type")):
        grouped = defaultdict(list)
        for token in tokens:
            grouped[token.get(key, "unknown")].append(token)
        write_group(out_dir, name, key, grouped)

    grouped = defaultdict(list)
    for token in tokens:
        grouped[(token.get("token_type"), token.get("context_type"))].append(token)
    interaction = {k: v for k, v in grouped.items() if len(v) >= args.min_cell}
    write_group(out_dir, "interaction", "token_type x context_type", interaction)

    conditions = defaultdict(list)
    for token in tokens:
        conditions[token.get("condition", "unknown")].append(token)
    if len(conditions) > 1:
        write_group(out_dir, "condition", "condition", conditions)


if __name__ == "__main__":
    main()
