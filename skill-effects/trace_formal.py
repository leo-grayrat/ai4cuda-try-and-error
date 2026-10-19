"""Independent post-campaign native activity evidence; no primary retiming.

Official CLI options: https://docs.nvidia.com/nsight-compute/NsightComputeCli/
--profile-from-start off honors cudaProfilerStart/Stop (local torch wrapper
inspected). Raw metric rows identify launches; their sum is not full-call time.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from environment_probe import msvc_environment
from formal_campaign import verify
from gpu_lock import gpu_lock
from runtime import digest

ROOT = Path(__file__).resolve().parent
METRIC = "gpu__time_duration.sum"
DEFAULT_NCU = Path("C:/Program Files/NVIDIA Corporation/Nsight Compute 2025.2.1/target/windows-desktop-win7-x64/ncu.exe")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_native_csv(text):
    """Read raw CLI output including the profiler's non-CSV prelude.

    Empty, partial or invalid observations fail closed, never mean zero work.
    Launch identity includes process/context/device so repeated IDs in another
    process are not silently collapsed.
    """
    header = None
    launches = {}
    for row in csv.reader(io.StringIO(text)):
        if "Kernel Name" in row and "Metric Name" in row and "Metric Value" in row:
            header = row
            continue
        if not row or row[0].startswith("==") or header is None:
            continue
        if len(row) != len(header):
            raise ValueError("Malformed row after Nsight CSV header")
        item = dict(zip(header, row))
        if item["Metric Name"] != METRIC:
            continue
        scale = {"ns": 1e-3, "us": 1, "µs": 1, "ms": 1e3, "s": 1e6}.get(item["Metric Unit"])
        if scale is None:
            raise ValueError("Unknown native duration unit")
        duration = float(item["Metric Value"].replace(",", "")) * scale
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Nonpositive/nonfinite native duration")
        key = tuple(item.get(field, "") for field in ["Process ID", "Context", "Device", "ID"])
        launch = {"kernel_name": item["Kernel Name"], "duration_us": duration,
                  **{field: item.get(field) for field in ["ID", "Process ID", "Context", "Stream", "Device", "CC", "Block Size", "Grid Size"]}}
        if key in launches and launches[key] != launch:
            raise ValueError("Conflicting duplicate native launch measurement")
        launches[key] = launch
    if not launches:
        raise ValueError("No positive native launch observations")
    rows = list(launches.values())
    return {"positive_activity_observed": True, "launch_count": len(rows),
            "kernel_name_counts": dict(Counter(row["kernel_name"] for row in rows)),
            "kernel_duration_sum_us": sum(row["duration_us"] for row in rows), "launches": rows,
            "scope": "One warmed nominal forward call; summed profiled kernel duration is not full-call latency"}


def snapshot(run):
    return {path.relative_to(run).as_posix(): digest(path)
            for directory in [run / "work", run / "control"] for path in sorted(directory.rglob("*"))
            if path.is_file() and path.suffix in {".py", ".cpp", ".cu", ".h", ".cuh", ".pyd", ".so"}}


def require_finished(campaign, collection):
    if campaign["status"] != "complete" or len(campaign["trials"]) != collection["scheduled"]:
        raise ValueError("Formal campaign is still active/incomplete; do not interfere with measurements")
    if collection["terminal"] != collection["scheduled"]:
        raise ValueError("Refresh complete terminal collection before profiling")


def trace_role(run, role, output, ncu, base_env, timeout):
    destination = output / role
    destination.mkdir(exist_ok=False)
    env = base_env.copy()
    for key, folder in [("TEMP", "temp"), ("TMP", "temp"),
                        ("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("TRITON_CACHE_DIR", "triton")]:
        path = destination / folder
        path.mkdir(exist_ok=True)
        env[key] = str(path)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["NV_COMPUTE_PROFILER_DISABLE_STOCK_FILE_DEPLOYMENT"] = "1"
    command = [str(ncu), "--config-file", "off", "--profile-from-start", "off",
               "--clock-control", "none", "--cache-control", "none", "--csv", "--page", "raw",
               "--print-metric-name", "name", "--metrics", METRIC,
               "--log-file", str(destination / "native.csv"),
               sys.executable, str(ROOT / "native_trace_worker.py"), "--work", str(run / "work"),
               "--control", str(run / "control"), "--role", role]
    started = time.monotonic()
    with (destination / "worker.log").open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=run / "work", env=env, stdout=stream, stderr=subprocess.STDOUT)
        timed_out = False
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], stdout=stream, stderr=subprocess.STDOUT, check=False)
            else:
                process.kill()
            code = process.wait()
    row = {"role": role, "command": command, "returncode": code, "timeout": timed_out,
           "seconds": time.monotonic() - started, "status": "trace_failed"}
    if code == 0 and not timed_out:
        try:
            log = (destination / "worker.log").read_text(encoding="utf-8", errors="replace")
            markers = [json.loads(line) for line in log.splitlines() if line.startswith('{"trace_worker":')]
            if len(markers) != 1 or markers[0].get("trace_worker") != "complete" or markers[0].get("role") != role:
                raise ValueError("Worker did not confirm the selected call completed")
            native_path = destination / "native.csv"
            row.update(parse_native_csv(native_path.read_text(encoding="utf-8", errors="replace")))
            row.update(status="observed", raw_sha256=digest(native_path), worker=markers[0])
        except (ValueError, OSError) as exc:
            row["error"] = str(exc)
    write(destination / "result.json", row)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, default=ROOT / "runs/formal-series-01/campaign.json")
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol/v1/manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ncu", type=Path, default=DEFAULT_NCU)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    campaign, collection, protocol = read(args.campaign), read(args.collection), read(args.protocol)
    require_finished(campaign, collection)
    verify(protocol)
    if not (digest(args.protocol) == collection["protocol_sha256"] == campaign["protocol_sha256"]):
        raise ValueError("Protocol identity mismatch")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    env = msvc_environment()
    rows = []
    write(args.output / "provenance.json", {"protocol_sha256": digest(args.protocol),
          "collection_sha256": digest(args.collection), "campaign_sha256": digest(args.campaign),
          "worker_sha256": digest(ROOT / "native_trace_worker.py"), "controller_sha256": digest(Path(__file__)),
          "ncu_path": str(args.ncu), "ncu_sha256": digest(args.ncu),
          "started_utc": datetime.now(timezone.utc).isoformat(),
          "purpose": "Post-campaign native activity check; never replaces paired CUDA-event timings"})
    for trial in collection["trials"]:
        row = {"index": trial["index"], "run": trial.get("run"), "status": "not_executable_or_not_independently_correct"}
        checks = trial.get("independent_checks", {})
        if checks.get("compile") and checks.get("verify"):
            run = ROOT / "runs" / trial["run"]
            if digest(run / "result.json") != trial["result_sha256"]:
                raise ValueError("Raw trial changed since collection")
            destination = args.output / trial["run"]
            destination.mkdir()
            before = snapshot(run)
            with gpu_lock(ROOT / "runs/gpu.lock") as wait:
                # Re-read status under the lock before using the GPU.
                require_finished(read(args.campaign), collection)
                roles = [trace_role(run, role, destination, args.ncu, env, args.timeout)
                         for role in ["eager", "compiled", "candidate"]]
            after = snapshot(run)
            write(destination / "artifact_hashes.json", {"before": before, "after": after})
            if before != after:
                raise ValueError("Profiling changed candidate/control source or native artifact")
            row.update(status="observed" if all(role["status"] == "observed" for role in roles) else "incomplete_trace",
                       roles=roles, gpu_wait_seconds=wait)
        rows.append(row)
        write(args.output / "results.json", {"planned": collection["scheduled"], "processed": len(rows), "trials": rows})
        print(json.dumps({"index": row["index"], "status": row["status"]}), flush=True)


if __name__ == "__main__":
    main()
