"""Streaming statistics over collected runs.

analyze_loop.py keeps every token row in memory, which is fine for the reduction
tasks (98k tokens) but not for the level-2 runs (559 MB of logprob arrays, 113k+
tokens and 20 candidates each). This walks run by run and only keeps counters.

Reports, per task and overall:
  - token counts inside vs outside fenced code blocks
  - low-probability counts and density in each
  - context split (initial / after_success / after_failure)

Usage:
    python stream_stats.py runs-kb2
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path

LOWPROB = -2.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs_dir", type=Path)
    args = ap.parse_args()

    # counters[task][region] = [tokens, lowprob]
    counters = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    ctx = collections.defaultdict(lambda: [0, 0])
    per_run = {}

    for path in sorted(glob.glob(str(args.runs_dir / "*" / "turns.jsonl"))):
        run = Path(path).parent.name
        task = run.rsplit("-", 2)[0]
        rtok = rlow = 0
        for line in open(path, encoding="utf-8"):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            clp = rec.get("content_logprobs") or []
            if not clp:
                continue
            turn = rec.get("turn", 0)
            if turn == 0:
                c = "initial"
            else:
                c = ("after_success" if rec.get("result", {}).get("correct")
                     else "after_failure")

            in_code = False
            for entry in clp:
                tok = entry.get("token") or ""
                if "```" in tok:
                    in_code = not in_code
                lp = entry.get("logprob")
                if lp is None:
                    continue
                region = "code" if in_code else "prose"
                low = lp <= LOWPROB
                counters[task][region][0] += 1
                rtok += 1
                if low:
                    counters[task][region][1] += 1
                    rlow += 1
                ctx[c][0] += 1
                if low:
                    ctx[c][1] += 1
        per_run[run] = (rtok, rlow)

    print(f"runs: {len(per_run)}  tokens: {sum(v[0] for v in per_run.values()):,}\n")

    print(f"{'task':<38}{'code tok':>10}{'code low%':>11}{'prose tok':>11}{'prose low%':>12}")
    totals = collections.defaultdict(lambda: [0, 0])
    for task in sorted(counters):
        code = counters[task]["code"]
        prose = counters[task]["prose"]
        for k in ("code", "prose"):
            totals[k][0] += counters[task][k][0]
            totals[k][1] += counters[task][k][1]
        print(f"{task:<38}{code[0]:>10,}{100*code[1]/max(1,code[0]):>10.2f}%"
              f"{prose[0]:>11,}{100*prose[1]/max(1,prose[0]):>11.2f}%")

    code, prose = totals["code"], totals["prose"]
    print(f"\n{'TOTAL':<38}{code[0]:>10,}{100*code[1]/max(1,code[0]):>10.2f}%"
          f"{prose[0]:>11,}{100*prose[1]/max(1,prose[0]):>11.2f}%")
    print(f"density ratio prose/code: "
          f"{(prose[1]/max(1,prose[0]))/max(1e-12, code[1]/max(1,code[0])):.1f}x")

    print(f"\n{'context':<16}{'tokens':>10}{'low%':>8}")
    for c in ("initial", "after_success", "after_failure"):
        n, low = ctx[c]
        print(f"{c:<16}{n:>10,}{100*low/max(1,n):>7.2f}%")


if __name__ == "__main__":
    main()
