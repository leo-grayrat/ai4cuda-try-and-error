"""Queue read-only collection and native checks after the formal campaign.

Does not restart formal trials, retry failed phases or claim research completion.
The live exec handle is the authoritative observation handle for this process.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from runtime import digest

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def ready(campaign):
    if campaign["status"] == "infrastructure_error":
        raise RuntimeError("Formal controller stopped with infrastructure_error; do not restart or profile partial data")
    return campaign["status"] == "complete"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, default=ROOT / "runs/formal-series-01/campaign.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60:
        parser.error("poll-seconds must be between 1 and 60")
    args.campaign = args.campaign.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    state_path = args.output / "state.json"
    state = {"pid": os.getpid(), "status": "waiting_for_formal_completion",
             "campaign": str(args.campaign), "controller_sha256": digest(Path(__file__)),
             "started_utc": datetime.now(timezone.utc).isoformat(), "phases": []}
    save(state_path, state)
    print(json.dumps({"pid": os.getpid(), "status": state["status"]}), flush=True)
    try:
        while True:
            try:
                campaign = read(args.campaign)
            except (OSError, json.JSONDecodeError) as exc:
                # The formal controller writes its own state file. A read can
                # overlap that write; observation failure is not termination.
                state["last_observation_error"] = str(exc)
                save(state_path, state)
                time.sleep(args.poll_seconds)
                continue
            if ready(campaign):
                break
            time.sleep(args.poll_seconds)
        base = args.campaign.parent
        collection = base / "collection.json"
        commands = [
            ("collect", "collect_formal.py", ["--campaign", str(args.campaign), "--output", str(collection)]),
            ("review_materials", "prepare_blind_review.py", ["--collection", str(collection), "--output", str(base / "blind-review-schema2")]),
            ("native_activity", "trace_formal.py", ["--campaign", str(args.campaign), "--collection", str(collection), "--output", str(base / "native-activity")]),
            ("paired_analysis", "analyze_formal.py", ["--collection", str(collection), "--output", str(base / "analysis.json")]),
        ]
        for name, filename, arguments in commands:
            command = [sys.executable, str(ROOT / filename), *arguments]
            row = {"phase": name, "command": command, "script_sha256": digest(ROOT / filename),
                   "status": "running", "started_utc": datetime.now(timezone.utc).isoformat()}
            started = time.monotonic()
            with (args.output / (name + ".log")).open("w", encoding="utf-8") as stream:
                process = subprocess.Popen(command, cwd=ROOT.parent, stdout=stream, stderr=subprocess.STDOUT)
                state.update(status="running", current_phase=name, live_child_pid=process.pid)
                save(state_path, state)
                print(json.dumps({"phase": name, "pid": process.pid, "status": "running"}), flush=True)
                code = process.wait()
            row.update(returncode=code, seconds=time.monotonic() - started,
                       completed_utc=datetime.now(timezone.utc).isoformat(),
                       status="complete" if code == 0 else "failed")
            state["phases"].append(row)
            state.pop("live_child_pid", None)
            save(state_path, state)
            if code != 0:
                raise RuntimeError(f"Postprocessing phase {name} failed; raw log retained, no automatic retry")
        state.update(status="postprocessing_complete_audits_and_report_pending",
                     completed_utc=datetime.now(timezone.utc).isoformat())
        save(state_path, state)
        print(json.dumps({"status": state["status"]}), flush=True)
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        state.update(status="stopped", error=str(exc), completed_utc=datetime.now(timezone.utc).isoformat())
        save(state_path, state)
        print(json.dumps({"status": state["status"], "error": str(exc)}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
