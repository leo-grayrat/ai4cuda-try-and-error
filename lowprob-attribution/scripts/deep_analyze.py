"""Deep analysis of the collected token data.

Complements analyze.py (which reports per-category means) with the breakdowns
that turned out to carry the signal:

  quantiles   per-category distribution, not just the mean
  tools       per-tool and per-region (envelope vs arguments) statistics
  position    where inside its own step a token sits
  clusters    whether low-probability tokens are adjacent or isolated
  reuse       how much the same token string varies across occurrences
  reliability sanity check of the returned top_logprobs
  outcomes    failed vs complete runs

Usage:
    python deep_analyze.py lowprob-attribution/results/tokens-triton.jsonl \
        --runs-dir lowprob-attribution/runs-triton \
        --out lowprob-attribution/results/deep-triton.json
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import re
from pathlib import Path

LOWPROB_AT = -2.0
LOWPROB_DIR = Path(__file__).resolve().parents[1]
TOOL_CALL_RE = re.compile(r"<tool_call>\s*\{\"name\":\s*\"([a-zA-Z_]+)\"")
CATEGORIES = ("NL_reasoning", "generic_code", "tool_syntax", "identifier",
              "number", "triton_specific", "cuda_specific")


def load_tokens(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def quantiles(values: list[float], points=(0.01, 0.10, 0.50, 0.90)) -> dict:
    v = sorted(values)
    if not v:
        return {}
    q = {"n": len(v), "mean": sum(v) / len(v), "min": v[0], "max": v[-1],
         "lowprob_rate": 100 * sum(1 for x in v if x <= LOWPROB_AT) / len(v)}
    for p in points:
        q[f"p{int(p*100)}"] = v[min(len(v) - 1, int(len(v) * p))]
    return q


def by_category(rows: list[dict]) -> dict:
    out = {}
    for cat in CATEGORIES:
        vals = [r["logprob"] for r in rows
                if r["token_type"] == cat and r.get("logprob") is not None]
        if vals:
            out[cat] = quantiles(vals)
    return out


def by_position(rows: list[dict]) -> dict:
    buckets = collections.defaultdict(list)
    per_run = collections.defaultdict(list)
    for r in rows:
        per_run[r["run_id"]].append(r)
    for rs in per_run.values():
        by_turn = collections.defaultdict(list)
        for r in sorted(rs, key=lambda x: (x["turn"],
                                           0 if x["logprob_field"] == "reasoning_content" else 1)):
            by_turn[r["turn"]].append(r)
        for ts in by_turn.values():
            n = len(ts)
            for i, r in enumerate(ts):
                if r.get("logprob") is None:
                    continue
                frac = i / n if n else 0
                key = ("first_quarter" if frac < 0.25
                       else "middle" if frac < 0.75 else "last_quarter")
                buckets[key].append(r["logprob"])
    return {k: quantiles(v) for k, v in buckets.items()}


def clusters(rows: list[dict]) -> dict:
    per_run = collections.defaultdict(list)
    for r in rows:
        per_run[r["run_id"]].append(r)
    adj = iso = 0
    runs_with = 0
    for rs in per_run.values():
        flags = [1 if (r.get("logprob") is not None and r["logprob"] <= LOWPROB_AT) else 0 for r in rs]
        if sum(flags):
            runs_with += 1
        for i, f in enumerate(flags):
            if not f:
                continue
            if (i > 0 and flags[i - 1]) or (i + 1 < len(flags) and flags[i + 1]):
                adj += 1
            else:
                iso += 1
    return {"runs_with_lowprob": runs_with, "runs_total": len(per_run),
            "adjacent": adj, "isolated": iso}


def reuse(rows: list[dict], min_seen: int = 8) -> dict:
    by_tok = collections.defaultdict(list)
    for r in rows:
        if r.get("logprob") is not None and r["logprob_field"] == "content":
            by_tok[r["token"]].append(r["logprob"])
    stats = {}
    for tok, vals in by_tok.items():
        if len(vals) < min_seen:
            continue
        mean = sum(vals) / len(vals)
        sd = (sum((x - mean) ** 2 for x in vals) / len(vals)) ** 0.5
        stats[tok] = {"n": len(vals), "mean": mean, "sd": sd, "min": min(vals)}
    lows = [r["token"] for r in rows
            if r.get("logprob") is not None and r["logprob"] <= LOWPROB_AT
            and r["logprob_field"] == "content"]
    counts = collections.Counter(lows)
    return {
        "tokens_seen_at_least": min_seen,
        "repeated_token_stats": dict(sorted(stats.items(),
                                            key=lambda kv: kv[1]["sd"], reverse=True)[:20]),
        "lowprob_occurrences": len(lows),
        "lowprob_distinct": len(counts),
        "lowprob_repeated": {k: v for k, v in counts.most_common(10) if v > 1},
    }


def per_tool(runs_dir: Path) -> dict:
    groups = collections.defaultdict(list)
    for path in sorted(glob.glob(str(runs_dir / "*" / "logprobs.jsonl"))):
        for line in open(path, encoding="utf-8"):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            clp = record.get("content_logprobs") or []
            if not clp:
                continue
            text = "".join(t.get("token", "") for t in clp)
            match = TOOL_CALL_RE.search(text)
            tool = match.group(1) if match else "none"
            arg_at = text.find('"arguments"')
            # find which stream index the arguments region starts at
            cursor = 0
            arg_index = None
            for i, t in enumerate(clp):
                idx = text.find(t.get("token", ""), cursor)
                cursor = idx + len(t.get("token", "")) if idx >= 0 else cursor
                if arg_index is None and arg_at >= 0 and idx >= arg_at:
                    arg_index = i
            for i, t in enumerate(clp):
                if t.get("logprob") is None:
                    continue
                region = ("arguments" if (arg_index is not None and i >= arg_index)
                          else "envelope")
                groups[(tool, region)].append(t["logprob"])
    return {f"{k[0]}|{k[1]}": quantiles(v) for k, v in
            sorted(groups.items(), key=lambda kv: -len(kv[1]))}


def reliability(runs_dir: Path) -> dict:
    total = consistent = beats = 0
    worst = None
    for path in sorted(glob.glob(str(runs_dir / "*" / "logprobs.jsonl"))):
        for line in open(path, encoding="utf-8"):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            for t in (record.get("content_logprobs") or []):
                tops = t.get("top_logprobs") or []
                lp = t.get("logprob")
                if lp is None or len(tops) < 2:
                    continue
                total += 1
                if abs((tops[0].get("logprob") or 0) - lp) < 1e-6:
                    consistent += 1
                best = max((a.get("logprob") for a in tops[1:]
                            if a.get("logprob") is not None), default=None)
                if best is not None and best - lp > 1.0:
                    beats += 1
                    if worst is None or (best - lp) > worst["gap"]:
                        worst = {"gap": best - lp, "sampled": t.get("token"),
                                 "sampled_lp": lp, "best_alt_lp": best,
                                 "alts": [a.get("token") for a in tops[1:4]]}
    return {"entries": total,
            "top0_matches_sampled": consistent,
            "top0_match_rate": 100 * consistent / total if total else 0,
            "later_beats_sampled_gt_1nat": beats,
            "rate": 100 * beats / total if total else 0,
            "worst": worst}


def by_outcome(rows: list[dict], failed: set[str]) -> dict:
    out = {}
    for label, want_failed in (("failed", True), ("complete", False)):
        sel = [r for r in rows if (r["run_id"] in failed) == want_failed]
        vals = [r["logprob"] for r in sel if r.get("logprob") is not None]
        runs = {r["run_id"] for r in sel}
        if vals:
            out[label] = {"runs": len(runs), "tokens": len(vals),
                          **quantiles(vals)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tokens", type=Path)
    ap.add_argument("--runs-dir", type=Path, default=LOWPROB_DIR / "runs-triton")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    rows = load_tokens(args.tokens)
    result = {
        "tokens": len(rows),
        "by_category": by_category(rows),
        "by_position": by_position(rows),
        "clusters": clusters(rows),
        "reuse": reuse(rows),
        "by_tool": per_tool(args.runs_dir),
        "reliability": reliability(args.runs_dir),
        "by_outcome": by_outcome(rows, {"row_reduce-full-07", "row_reduce-full-08"}),
    }

    print(f"tokens: {result['tokens']}\n")

    print("=== category quantiles ===")
    print(f"  {'category':<18}{'n':>7}{'p10':>9}{'p50':>9}{'mean':>9}{'lowprob%':>11}")
    for cat, q in sorted(result["by_category"].items(), key=lambda kv: kv[1]["mean"]):
        print(f"  {cat:<18}{q['n']:>7}{q['p10']:>9.3f}{q['p50']:>9.3f}"
              f"{q['mean']:>9.3f}{q['lowprob_rate']:>10.2f}%")

    print("\n=== by tool | region ===")
    print(f"  {'tool|region':<22}{'n':>7}{'mean':>9}{'p10':>9}{'lowprob%':>11}")
    for key, q in result["by_tool"].items():
        print(f"  {key:<22}{q['n']:>7}{q['mean']:>9.3f}{q['p10']:>9.3f}"
              f"{q['lowprob_rate']:>10.2f}%")

    print("\n=== position within step ===")
    for key, q in result["by_position"].items():
        print(f"  {key:<16}n={q['n']:<6} mean={q['mean']:>8.3f}  lowprob={q['lowprob_rate']:.2f}%")

    c = result["clusters"]
    print(f"\n=== clusters === runs with lowprob {c['runs_with_lowprob']}/{c['runs_total']}, "
          f"adjacent {c['adjacent']}, isolated {c['isolated']}")

    r = result["reuse"]
    print(f"=== reuse === lowprob occurrences {r['lowprob_occurrences']}, "
          f"distinct {r['lowprob_distinct']}, repeated {r['lowprob_repeated']}")

    rel = result["reliability"]
    print(f"\n=== reliability === entries {rel['entries']}, "
          f"top0 matches sampled {rel['top0_match_rate']:.1f}%, "
          f"later candidate beats sampled by >1nat {rel['later_beats_sampled_gt_1nat']} "
          f"({rel['rate']:.2f}%)")

    print("\n=== failed vs complete ===")
    for label, q in result["by_outcome"].items():
        print(f"  {label:<10}runs={q['runs']:<3} tokens={q['tokens']:<6} "
              f"lowprob={q['lowprob_rate']:.2f}%  mean={q['mean']:.3f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
