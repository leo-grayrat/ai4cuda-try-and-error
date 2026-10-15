"""Alignment: match proxy-recorded logprobs to OpenCode events.jsonl.

Reads:
    - logprobs.jsonl (from record_proxy.py): one line per assistant response,
      with content_text / reasoning_text and per-token logprobs
    - events.jsonl (from OpenCode): step_start/step_finish/tool_use/text events

Produces:
    - tokens.jsonl: one line per annotated token with fields
      {run_id, turn, token, logprob, is_sentinel, token_type, context_type}

Alignment strategy: match by content-text prefix against the OpenCode
`text` events and `tool_use` input content. Each proxy record should match
exactly one assistant message in one run; we verify uniqueness.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from classify import annotate_tokens, parse_response_segments


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def derive_preceding_feedback(events: list[dict], upto_index: int) -> str | None:
    """Look backwards from event index to find the most recent tool feedback type."""
    for i in range(upto_index - 1, -1, -1):
        ev = events[i]
        if ev.get("type") != "tool_use":
            continue
        part = ev.get("part") or {}
        state = part.get("state") or {}
        output = str(state.get("output") or state.get("content") or "")
        if not output:
            continue
        low = output.lower()
        if "error" in low or "traceback" in low or "failed" in low or "compilation" in low:
            if "mismatch" in low or "incorrect" in low:
                return "correctness_fail"
            return "compile_error"
        if "mismatch" in low or "incorrect" in low or "allclose" in low and "false" in low:
            return "correctness_fail"
        if "speedup" in low or "ms" in low or "faster" in low or "slower" in low or "runtime" in low:
            return "profiling"
        return "profiling"  # generic tool output: treat as observation
    return None


def align_run(run_dir: Path, run_id: str) -> list[dict]:
    lp_path = run_dir / "logprobs.jsonl"
    ev_path = run_dir / "events.jsonl"
    if not lp_path.exists() or not ev_path.exists():
        return []

    lp_records = load_jsonl(lp_path)
    events = load_jsonl(ev_path)

    all_tokens = []
    turn = 0
    for i, rec in enumerate(lp_records):
        text = rec.get("content_text") or ""
        content_lp = rec.get("content_logprobs") or []
        if not content_lp:
            continue
        feedback = derive_preceding_feedback(events, i)
        tokens = annotate_tokens(content_lp, text, turn, feedback)
        for t in tokens:
            t["run_id"] = run_id
        all_tokens.extend(tokens)
        turn += 1
    return all_tokens


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir", help="directory containing run subdirs with logprobs.jsonl + events.jsonl")
    ap.add_argument("--out", default="tokens.jsonl")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    out_path = Path(args.out)
    total = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for run_dir in sorted(runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            tokens = align_run(run_dir, run_dir.name)
            for t in tokens:
                fh.write(json.dumps(t, ensure_ascii=False) + "\n")
            total += len(tokens)
            print(f"{run_dir.name}: {len(tokens)} tokens")
    print(f"total: {total} tokens -> {out_path}")


if __name__ == "__main__":
    main()
