"""Phase 1 collector: run OpenCode trials through the recording proxy.

Starts the proxy once, runs each (task, repeat) trial through study.run_trial
with OPENCODE_CONFIG pointing at the proxy overlay, and slices the proxy's JSONL
record after every trial so each run directory carries its own logprobs.jsonl.

Run it with the interpreter that has torch/triton on PATH, because study.py
prepends sys.executable's directory to the child PATH: the agent needs a python
that can actually import torch when it checks its own kernel.

Usage:
    python lowprob-attribution/scripts/collect.py --tasks row_reduce --repeats 1
"""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOWPROB = REPO / "lowprob-attribution"
STUDY_DIR = REPO / "cuda-skill-efficiency"
sys.path.insert(0, str(STUDY_DIR))

import study  # noqa: E402

PROXY = LOWPROB / "scripts" / "record_proxy.py"
PROXY_CONFIG = LOWPROB / "opencode-proxy.jsonc"
PROXY_OUT = LOWPROB / "proxy-records"
PROXY_RECORDS = PROXY_OUT / "logprobs.jsonl"


def wait_for_port(host: str, port: int, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks-root", type=Path, default=STUDY_DIR / "tasks")
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--model", default="deepseek/deepseek-flash")
    ap.add_argument("--timeout-s", type=int, default=300)
    ap.add_argument("--runs-root", type=Path, default=LOWPROB / "runs")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--prompt-suffix", default=None,
                    help="extra instruction appended to the task prompt; used to make the "
                         "model print code in the reply body so it carries logprobs")
    args = ap.parse_args()

    args.runs_root.mkdir(parents=True, exist_ok=True)
    PROXY_OUT.mkdir(parents=True, exist_ok=True)

    # The proxy's output must go to a file, not an unread PIPE: once the pipe
    # buffer fills, the proxy blocks on write and stops serving requests, which
    # shows up as a trial that stalls after its first model call.
    log_path = PROXY_OUT / "proxy.log"
    log_handle = log_path.open("w", encoding="utf-8")
    proxy = subprocess.Popen(
        [sys.executable, "-u", str(PROXY),
         "--upstream", "https://api.deepseek.com",
         "--listen", f"127.0.0.1:{args.port}",
         "--out", str(PROXY_OUT),
         "--top-logprobs", "20",
         "--temperature-override", "1.0"],
        stdout=log_handle, stderr=subprocess.STDOUT,
    )
    if not wait_for_port("127.0.0.1", args.port):
        proxy.terminate()
        log_handle.close()
        raise SystemExit("proxy did not come up")

    seen = 0
    if PROXY_RECORDS.exists():
        seen = len(PROXY_RECORDS.read_text(encoding="utf-8").splitlines())

    env_extra = {"OPENCODE_CONFIG": str(PROXY_CONFIG.resolve())}
    results = []
    try:
        for task_name in args.tasks:
            for index in range(args.repeats):
                run_dir = args.runs_root / f"{task_name}-full-{index:02d}"
                if run_dir.exists():
                    print(f"[skip] {run_dir.name} already exists")
                    continue
                print(f"[trial] {run_dir.name}")
                summary = study.run_trial(
                    (args.tasks_root / task_name).resolve(), None, args.model,
                    run_dir, args.timeout_s, "opencode", env_extra=env_extra,
                    prompt_suffix=args.prompt_suffix,
                )
                lines = (PROXY_RECORDS.read_text(encoding="utf-8").splitlines()
                         if PROXY_RECORDS.exists() else [])
                chunk = lines[seen:]
                seen = len(lines)
                (run_dir / "logprobs.jsonl").write_text(
                    "\n".join(chunk) + ("\n" if chunk else ""), encoding="utf-8",
                )
                results.append({
                    "run": run_dir.name,
                    "status": summary.get("status"),
                    "proxy_records": len(chunk),
                    "usage": summary.get("usage"),
                })
                print(f"    status={summary.get('status')} proxy_records={len(chunk)}")
    finally:
        proxy.terminate()
        try:
            proxy.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proxy.kill()
        log_handle.close()

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
