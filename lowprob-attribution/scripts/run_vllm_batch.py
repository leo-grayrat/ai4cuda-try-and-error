"""Wait for the GPU to free up, then serve vLLM locally and run the collector.

The machine is shared: a batch of whisper transcriptions was running on the GPU
when this was written. Rather than competing for VRAM, this script polls until
enough is free, starts the vLLM server inside WSL, waits for it to answer, runs
the requested number of trials, then shuts the server back down.

Everything here is local inference - no API spend, no external calls.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

LOWPROB = Path(__file__).resolve().parents[1]
REPO = LOWPROB.parent

VLLM = "/root/vllm-env/venv/bin/vllm"
MODEL_DIR = "/root/models/Qwen3-4B-AWQ"
PORT = 8000
# 4B-AWQ needs roughly 6 GiB of the 7.93 GiB card; Windows keeps ~1 GiB for itself.
NEEDED_FREE_MIB = 6500


def gpu_free_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=60,
    )
    return int(out.stdout.strip().splitlines()[0])


def wait_for_gpu(poll_seconds: int, timeout_s: int) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        free = gpu_free_mib()
        print(f"[gpu] free={free} MiB (need {NEEDED_FREE_MIB})", flush=True)
        if free >= NEEDED_FREE_MIB:
            return True
        time.sleep(poll_seconds)
    return False


def start_vllm() -> subprocess.Popen:
    cmd = [
        "wsl", "-d", "Ubuntu", "--",
        "env", "VLLM_USE_FLASHINFER_SAMPLER=0", VLLM, "serve", MODEL_DIR,
        "--port", str(PORT),
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.86",
        "--served-model-name", "qwen3-4b",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",
    ]
    log = open(REPO / "lowprob-attribution" / "vllm-serve.log", "a", encoding="utf-8")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)


def wait_ready(timeout_s: int) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            subprocess.run(
                ["curl", "-sS", "-m", "5", f"http://127.0.0.1:{PORT}/health"],
                capture_output=True, timeout=15,
            )
            return True
        except Exception:
            time.sleep(10)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="row_reduce")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--timeout-s", type=int, default=1800)
    ap.add_argument("--wait-timeout-s", type=int, default=6 * 3600)
    ap.add_argument("--poll-seconds", type=int, default=120)
    args = ap.parse_args()

    print("[batch] waiting for the GPU to free up", flush=True)
    if not wait_for_gpu(args.poll_seconds, args.wait_timeout_s):
        print("[batch] GPU never freed up; stopping without disturbing anything.")
        return 1

    print("[batch] starting vLLM", flush=True)
    proc = start_vllm()
    try:
        if not wait_ready(900):
            print("[batch] server did not become ready; see lowprob-attribution/vllm-serve.log")
            return 1
        print("[batch] server ready; running collector", flush=True)
        cmd = [
            sys.executable, str(LOWPROB / "scripts" / "collect.py"),
            "--tasks", args.tasks,
            "--repeats", str(args.repeats),
            "--timeout-s", str(args.timeout_s),
            "--model", "vllm/qwen3-4b",
            "--upstream", f"http://127.0.0.1:{PORT}",
            "--opencode-config", str(LOWPROB / "opencode-vllm-proxy.jsonc"),
        ]
        subprocess.run(cmd, cwd=str(REPO), timeout=args.timeout_s * args.repeats + 600)
    finally:
        print("[batch] shutting vLLM down", flush=True)
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
