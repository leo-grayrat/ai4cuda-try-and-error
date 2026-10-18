"""Multi-turn loop driver: plain chat, code in the reply body, logprobs captured.

Why not the agent framework: tool-call arguments are not covered by `logprobs`
while `message.content` is. Measured in check_plaintext_logprobs.py -- 448/448
code tokens inside the reply body carry a logprob. So the loop below keeps the
model's natural behaviour (write the kernel out) and does the file writing and
executing itself, on this side of the API boundary.

Two feedback conditions, decided before collection and not changed after:
  full       compile errors / correctness output / timing, verbatim
  truncated  one line each: "compilation failed" / "output mismatch" / ...

Usage:
    python loop_driver.py --tasks row_reduce vector_affine \
        --conditions full truncated --repeats 3 --max-turns 5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.deepseek.com/v1/chat/completions"
LOWPROB = Path(__file__).resolve().parents[1]
REPO = LOWPROB.parent
DEFAULT_TASKS = REPO / "cuda-skill-efficiency" / "tasks"

PROMPT_TEMPLATE = """You are a CUDA/Triton kernel engineer.

Implement `run(x)` in `candidate.py` for the computation in `reference.py`.

--- reference.py ---
{reference}

--- requirement ---
Use a Triton GPU kernel (`@triton.jit` with `tl.load` / `tl.store`). A plain
Python loop, a list comprehension, or a torch/numpy implementation is NOT
acceptable.

Reply with the complete contents of `candidate.py` inside a single fenced code
block. Do not use any tools; put the code in your reply.
"""

FENCE_RE = re.compile(r"```(?:python)?\n([\s\S]*?)```")


def api_key() -> str:
    path = Path(os.path.expanduser("~/.local/share/opencode/auth.json"))
    return json.loads(path.read_text(encoding="utf-8"))["deepseek"]["key"]


def call_api(messages: list[dict], model: str, max_tokens: int) -> dict:
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "logprobs": True,
        "top_logprobs": 20,
    }).encode()
    request = urllib.request.Request(
        API, data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key()}"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=900) as response:
        return json.loads(response.read())


def run_evaluator(work: Path, evaluator: Path, python: str) -> dict:
    """Run the independent evaluator against the candidate in `work`.

    `python` must be the interpreter that has torch/triton installed; the driver
    itself can run under a bare interpreter, but the evaluator cannot.
    """
    started = time.time()
    env = {**os.environ, "PYTHONPATH": str(work), "NO_PROXY": "127.0.0.1,localhost,::1"}
    # argv[1] is the candidate FILE, not the directory -- eval_common hashes it
    # directly (study.py calls it the same way). Passing the directory makes it
    # try to read_bytes() a folder and die with PermissionError.
    proc = subprocess.run(
        [python, str(evaluator), str(work / "candidate.py")],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=600, cwd=str(work), env=env,
    )
    try:
        parsed = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        parsed = {"correct": False, "error": (proc.stderr or proc.stdout)[-1200:]}
    parsed["_stdout"] = proc.stdout[-3000:]
    parsed["_stderr"] = proc.stderr[-3000:]
    parsed["_seconds"] = round(time.time() - started, 2)
    return parsed


def feedback_text(result: dict, condition: str) -> str:
    if condition == "full":
        body = (result.get("_stderr") or result.get("_stdout") or
                json.dumps(result, ensure_ascii=False))
        return f"The evaluator returned:\n\n{body[-2500:]}"
    # truncated: deliberately strips the diagnostic detail
    if result.get("correct") is True:
        return "The evaluator reported: correct."
    err = (result.get("error") or "").lower()
    if "triton" in err or "compile" in err or "syntax" in err:
        return "The evaluator reported: compilation failed."
    return "The evaluator reported: output mismatch."


def one_run(task_dir: Path, condition: str, out_dir: Path, model: str,
            max_turns: int, max_tokens: int, python: str) -> dict:
    reference = (task_dir / "reference.py").read_text(encoding="utf-8")
    candidate_template = (task_dir / "candidate.py").read_text(encoding="utf-8")
    # absolute: the evaluator runs with cwd set to a temp dir, so a relative
    # path resolved against the original cwd would not be found
    evaluator = (task_dir / "evaluate.py").resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="loop-"))
    for name in ("reference.py", "evaluate.py"):
        shutil.copy2(task_dir / name, work / name)
    # evaluate.py resolves the shared evaluator through parents[2], which breaks
    # once it is copied out of the task tree. Copy the module next to it instead
    # and let the import find it locally.
    shutil.copy2(REPO / "cuda-skill-efficiency" / "eval_common.py", work / "eval_common.py")

    messages = [{"role": "user",
                 "content": PROMPT_TEMPLATE.format(reference=reference)}]
    turns: list[dict] = []
    transcript: list[dict] = []

    for turn in range(max_turns):
        try:
            data = call_api(messages, model, max_tokens)
        except urllib.error.HTTPError as exc:
            transcript.append({"turn": turn, "error": f"HTTP {exc.code}",
                               "detail": exc.read().decode("utf-8", "replace")[:300]})
            break
        except Exception as exc:  # noqa: BLE001
            transcript.append({"turn": turn, "error": f"{type(exc).__name__}: {exc}"})
            break

        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        logprobs = choice.get("logprobs") or {}

        fenced = FENCE_RE.findall(content)
        code = fenced[0] if fenced else ""

        record = {
            "turn": turn,
            "finish_reason": choice.get("finish_reason"),
            "content": content,
            "reasoning": message.get("reasoning_content") or "",
            "content_logprobs": logprobs.get("content") or [],
            "reasoning_logprobs": logprobs.get("reasoning_content") or [],
            "usage": data.get("usage"),
            "code_found": bool(code),
            "code": code,
        }

        if not code:
            record["result"] = {"correct": False, "error": "no fenced code block"}
            turns.append(record)
            transcript.append({"turn": turn, "status": "no_code"})
            break

        (work / "candidate.py").write_text(code, encoding="utf-8")
        result = run_evaluator(work, evaluator, python)
        record["result"] = result
        turns.append(record)
        transcript.append({"turn": turn, "correct": result.get("correct"),
                           "seconds": result.get("_seconds")})

        if result.get("correct") is True and turn >= 1:
            break

        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user",
                         "content": feedback_text(result, condition)})

    with (out_dir / "turns.jsonl").open("w", encoding="utf-8") as handle:
        for record in turns:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (out_dir / "transcript.json").write_text(
        json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "final_candidate.py").write_text(
        (turns[-1].get("code") or candidate_template) if turns else candidate_template,
        encoding="utf-8")

    summary = {
        "task": task_dir.name, "condition": condition,
        "turns": len(turns),
        "correct_any": any(t.get("result", {}).get("correct") for t in turns),
        "correct_last": bool(turns and turns[-1].get("result", {}).get("correct")),
        "content_tokens": sum(len(t.get("content_logprobs") or []) for t in turns),
        "reasoning_tokens": sum(len(t.get("reasoning_logprobs") or []) for t in turns),
    }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", nargs="+",
                    default=["row_reduce", "vector_affine"])
    ap.add_argument("--heldout", action="store_true",
                    help="tasks live under tasks/heldout/")
    ap.add_argument("--tasks-root", type=Path, default=None,
                    help="override the task root (e.g. tasks-kb2 for KernelBench level 2)")
    ap.add_argument("--conditions", nargs="+", default=["full", "truncated"])
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--max-turns", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--python", default=r"C:\Users\34788\anaconda3\python.exe",
                    help="interpreter used for the evaluator (needs torch/triton)")
    ap.add_argument("--runs-root", type=Path, default=LOWPROB / "runs-loop")
    args = ap.parse_args()

    root = args.tasks_root or (DEFAULT_TASKS / ("heldout" if args.heldout else ""))
    root = root.resolve()
    summaries = []
    for task in args.tasks:
        for condition in args.conditions:
            for index in range(args.repeats):
                out = args.runs_root / f"{task}-{condition}-{index:02d}"
                if out.exists():
                    print(f"[skip] {out.name}")
                    continue
                print(f"[run ] {out.name}", flush=True)
                try:
                    summary = one_run(root / task, condition, out, args.model,
                                      args.max_turns, args.max_tokens, args.python)
                except Exception as exc:  # noqa: BLE001
                    summary = {"task": task, "condition": condition,
                               "error": f"{type(exc).__name__}: {exc}"}
                summaries.append(summary)
                print("       " + json.dumps(summary, ensure_ascii=False))

    (args.runs_root / "summaries.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {args.runs_root / 'summaries.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
