"""Alignment: pair proxy-recorded logprobs with OpenCode steps.

The unit of alignment is the model request, not a text prefix. The proxy writes
one record per chat completion and events.jsonl writes one step_start /
step_finish pair per chat completion, so ordering is the natural key. Reasoning
dominates the token count and has no text carrier in events.jsonl (`--thinking`
does not change that), which rules out matching on text. Each pair is instead
verified against the step's own accounting: the step's reasoning token count
must equal len(reasoning_logprobs).

Pipeline:
  1. drop OpenCode's own session-title request
  2. pair the remaining proxy records with step_finish events in order
  3. report per-run pairing statistics, flagging token-count mismatches
  4. annotate every token and write tokens.jsonl

Usage:
    python lowprob-attribution/scripts/align.py lowprob-attribution/runs \
        --out lowprob-attribution/results/tokens.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from classify import annotate_tokens  # noqa: E402

# OpenCode asks the same model for a session title; those requests are not part
# of the task trajectory. Matched against the head of the first user message.
TITLE_MARKERS = (
    "generate a title",
    "title for this",
    "session title",
    "conversation title",
)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_title_request(record: dict) -> bool:
    head = (record.get("first_user_message") or "").lower()
    return any(marker in head for marker in TITLE_MARKERS)


def classify_tool_output(output: str) -> str:
    """Context type implied by a tool result; this becomes the context of the
    next step, since the model sees it only on the following request."""
    low = (output or "").lower()
    if not low.strip():
        return "after_tool"
    if "mismatch" in low or "not close" in low or ("correct" in low and "false" in low):
        return "after_correctness_fail"
    if "error" in low or "traceback" in low or "exception" in low or "failed" in low:
        return "after_compile_error"
    if "speedup" in low or "median" in low or "faster" in low or "slower" in low or " ms" in low:
        return "after_profiling"
    return "after_tool"


def step_contexts(events: list[dict]) -> list[str]:
    """Context type for each step, taken from the tool result inside the step
    that precedes it."""
    contexts = []
    pending = None
    for event in events:
        kind = event.get("type")
        if kind == "tool_use":
            part = event.get("part") or {}
            state = part.get("state") or {}
            output = str(state.get("output") or state.get("error") or "")
            pending = classify_tool_output(output)
        elif kind == "step_finish":
            contexts.append(pending or "initial")
            pending = None
    return contexts


def step_reasoning_tokens(events: list[dict]) -> list[int | None]:
    counts = []
    for event in events:
        if event.get("type") != "step_finish":
            continue
        tokens = (event.get("part") or {}).get("tokens") or {}
        counts.append(tokens.get("reasoning"))
    return counts


def align_run(run_dir: Path, run_id: str, strict: bool = False) -> tuple[list[dict], dict]:
    records = load_jsonl(run_dir / "logprobs.jsonl")
    events = load_jsonl(run_dir / "events.jsonl")
    if not records or not events:
        return [], {"run": run_id, "error": "missing logprobs.jsonl or events.jsonl"}

    task_records = [record for record in records if not is_title_request(record)]
    dropped = len(records) - len(task_records)

    contexts = step_contexts(events)
    expected_counts = step_reasoning_tokens(events)

    pairs = []
    mismatches = []
    for index, record in enumerate(task_records):
        context = contexts[index] if index < len(contexts) else "unknown"
        expected = expected_counts[index] if index < len(expected_counts) else None
        actual = len(record.get("reasoning_logprobs") or [])
        if expected is not None and expected != actual:
            mismatches.append({"turn": index, "expected": expected, "actual": actual})
        pairs.append((record, index, context))

    stats = {
        "run": run_id,
        "proxy_records": len(records),
        "title_requests_dropped": dropped,
        "steps": len(expected_counts),
        "paired": len(pairs),
        "reasoning_token_mismatches": mismatches,
    }
    if strict and mismatches:
        raise SystemExit(f"{run_id}: {len(mismatches)} reasoning token mismatches")
    if len(task_records) != len(expected_counts):
        stats["pairing_warning"] = (
            f"{len(task_records)} task records vs {len(expected_counts)} steps"
        )

    tokens = []
    for record, turn, context in pairs:
        for logprobs, text, field in (
            (record.get("reasoning_logprobs") or [], record.get("reasoning_text") or "", "reasoning_content"),
            (record.get("content_logprobs") or [], record.get("content_text") or "", "content"),
        ):
            if not logprobs:
                continue
            for token in annotate_tokens(logprobs, text, field, turn, context):
                token["run_id"] = run_id
                token["temperature"] = record.get("temperature")
                tokens.append(token)
    stats["tokens"] = len(tokens)
    return tokens, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs_dir", help="directory containing run subdirectories")
    ap.add_argument("--out", default="lowprob-attribution/results/tokens.jsonl")
    ap.add_argument("--strict", action="store_true",
                    help="fail instead of reporting when token counts disagree")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    all_stats = []
    with out_path.open("w", encoding="utf-8") as handle:
        for run_dir in sorted(runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            tokens, stats = align_run(run_dir, run_dir.name, strict=args.strict)
            all_stats.append(stats)
            for token in tokens:
                handle.write(json.dumps(token, ensure_ascii=False) + "\n")
            total += len(tokens)

    for stats in all_stats:
        print(json.dumps(stats, ensure_ascii=False))
    print(f"total: {total} tokens -> {out_path}")


if __name__ == "__main__":
    main()
