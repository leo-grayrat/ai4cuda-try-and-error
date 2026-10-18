"""Analyze the plain-chat runs.

Guards against the failure mode that bit the earlier vLLM analysis: token types
are computed by matching the keyword tables against every content token in the
stream, NOT by trusting a segment assignment made elsewhere. Every headline
number is also reported under a second, deliberately different definition
(a stricter keyword set) so that a conclusion that only holds under one
boundary is visible as such.

Usage:
    python analyze_loop.py lowprob-attribution/runs-formal
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
from math import comb
from pathlib import Path

LOWPROB = -2.0

# Triton/CUDA vocabulary. These are the tokens the CUDA Agent thesis is about.
TRITON_API = {
    "tl", "load", "store", "dot", "sum", "max", "min", "reduce", "arange",
    "zeros", "full", "expand_dims", "broadcast_to", "where", "view", "reshape",
    "trans", "ravel", "atomic_add", "atomic_cas", "atomic_max", "atomic_min",
    "cast", "to", "num_warps", "num_stages", "constexpr", "program_id",
}
CUDA_NATIVE = {
    "threadidx", "blockidx", "blockdim", "griddim", "__shared__", "__syncthreads",
    "__global__", "__device__", "atomicadd", "warpsize", "__shfl_sync",
    "__ballot_sync", "smem", "mma", "wmma",
}
# a stricter boundary: only the unambiguous spellings
STRICT = {"tl", "num_warps", "num_stages", "constexpr", "program_id"} | CUDA_NATIVE

# structural tokens: no alternative spelling exists, so a low probability here
# would have a different meaning than at a choice point.
STRUCTURAL = {
    "for", "if", "else", "elif", "while", "def", "class", "import", "from",
    "return", "in", "not", "and", "or", "is", "none", "true", "false", "pass",
}


def norm(token: str) -> str:
    return token.lstrip(".@#`").strip().lower()


def fisher(a: int, n1: int, c: int, n2: int) -> float:
    n = n1 + n2
    k = a + c

    def hyp(x: int) -> float:
        return comb(n1, x) * comb(n2, k - x) / comb(n, k)

    observed = hyp(a)
    return sum(hyp(x) for x in range(0, min(n1, k) + 1) if hyp(x) <= observed + 1e-12)


def load_tokens(runs_dir: Path) -> list[dict]:
    """One row per content token, across every turn of every run."""
    out = []
    for path in sorted(glob.glob(str(runs_dir / "*" / "turns.jsonl"))):
        run = Path(path).parent.name
        run_dir = Path(path).parent
        condition = "truncated" if "-truncated-" in run else "full"
        task = run.rsplit("-", 2)[0]
        for line in open(path, encoding="utf-8"):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            turn = record.get("turn", 0)
            # context: first turn has no prior observation; later turns follow
            # an evaluator result that is either a success or a failure.
            if turn == 0:
                context = "initial"
            else:
                prev_correct = record.get("result", {}).get("correct")
                context = "after_success" if prev_correct else "after_failure"
            for entry in record.get("content_logprobs") or []:
                if entry.get("logprob") is None:
                    continue
                out.append({
                    "token": entry.get("token") or "",
                    "logprob": entry["logprob"],
                    "run": run, "task": task, "condition": condition,
                    "turn": turn, "context": context,
                })
    return out


def group_stats(rows: list[dict], key) -> dict:
    buckets = collections.defaultdict(list)
    for r in rows:
        buckets[key(r)].append(r["logprob"])
    out = {}
    for name, values in buckets.items():
        low = sum(1 for v in values if v <= LOWPROB)
        out[name] = {
            "n": len(values), "mean": sum(values) / len(values),
            "lowprob": low, "lowprob_rate": 100 * low / len(values),
            "p10": sorted(values)[min(len(values) - 1, int(len(values) * 0.10))],
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs_dir", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    rows = load_tokens(args.runs_dir)
    print(f"content tokens: {len(rows):,}  runs: {len({r['run'] for r in rows})}\n")

    # --- headline comparison, two boundary definitions ---
    for label, vocab in (("API 名 (宽松表)", TRITON_API),
                         ("严格表 (含 CUDA 原生)", STRICT)):
        in_v = [r for r in rows if norm(r["token"]) in vocab]
        out_v = [r for r in rows if norm(r["token"]) not in vocab]
        a = sum(1 for r in in_v if r["logprob"] <= LOWPROB)
        c = sum(1 for r in out_v if r["logprob"] <= LOWPROB)
        p = fisher(a, len(in_v), c, len(out_v)) if in_v and out_v else float("nan")
        print(f"{label}")
        print(f"  命中   n={len(in_v):<6} mean={sum(r['logprob'] for r in in_v)/max(1,len(in_v)):7.3f} "
              f"lowprob={100*a/max(1,len(in_v)):5.2f}%")
        print(f"  未命中 n={len(out_v):<6} mean={sum(r['logprob'] for r in out_v)/max(1,len(out_v)):7.3f} "
              f"lowprob={100*c/max(1,len(out_v)):5.2f}%")
        print(f"  Fisher p = {p:.4f}\n")

    # --- structural tokens as a third reference class ---
    print("结构 token（无替代写法）")
    st = group_stats(rows, lambda r: "structural" if norm(r["token"]) in STRUCTURAL else "other")
    for name, s in st.items():
        print(f"  {name:<12} n={s['n']:<6} mean={s['mean']:7.3f} lowprob={s['lowprob_rate']:5.2f}%")
    print()

    # --- by context ---
    print("按上下文")
    for name, s in sorted(group_stats(rows, lambda r: r["context"]).items()):
        print(f"  {name:<14} n={s['n']:<6} mean={s['mean']:7.3f} lowprob={s['lowprob_rate']:5.2f}%  p10={s['p10']:7.3f}")
    print()

    # --- by condition (the intervention arm) ---
    print("按反馈条件")
    for name, s in sorted(group_stats(rows, lambda r: r["condition"]).items()):
        print(f"  {name:<12} n={s['n']:<6} mean={s['mean']:7.3f} lowprob={s['lowprob_rate']:5.2f}%")
    print()

    # --- by task ---
    print("按任务")
    for name, s in sorted(group_stats(rows, lambda r: r["task"]).items()):
        print(f"  {name:<16} n={s['n']:<6} mean={s['mean']:7.3f} lowprob={s['lowprob_rate']:5.2f}%")
    print()

    # --- the lowest tokens overall ---
    print("最低的 20 个 token")
    for r in sorted(rows, key=lambda x: x["logprob"])[:20]:
        print(f"  {r['token']!r:<18} {r['logprob']:8.3f}  {r['task']:<16} {r['condition']:<10} {r['context']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "tokens": len(rows),
            "by_context": group_stats(rows, lambda r: r["context"]),
            "by_condition": group_stats(rows, lambda r: r["condition"]),
            "by_task": group_stats(rows, lambda r: r["task"]),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
