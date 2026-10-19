"""Run a frozen protocol serially; never silently retry or discard a trial."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
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


def verify(protocol):
    for relative, expected in protocol["frozen_files"].items():
        path = ROOT / relative
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Frozen input changed or disappeared: {relative}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--after-development", help="Wait for this existing development campaign to finish")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in {".", ".."}:
        parser.error("name must be one directory name")
    protocol_path = args.protocol.resolve()
    protocol = read(protocol_path)
    assert protocol["status"] == "frozen"
    verify(protocol)
    directory = ROOT / "runs" / args.name
    if args.resume:
        metadata = read(directory / "campaign.json")
        if metadata["protocol_sha256"] != digest(protocol_path):
            raise ValueError("Cannot resume with a different protocol")
        if metadata["status"] not in {"infrastructure_error", "complete"}:
            raise ValueError("Resume requires a recorded terminal controller state and a confirmed terminal tool handle")
        if metadata["status"] == "complete":
            print("This frozen campaign is already complete.")
            return 0
    else:
        directory.mkdir(parents=True, exist_ok=False)
        metadata = {"protocol": str(protocol_path), "protocol_sha256": digest(protocol_path),
                    "started_utc": datetime.now(timezone.utc).isoformat(),
                    "status": "starting", "trials": []}
    if args.after_development:
        development = ROOT / "runs" / args.after_development / "campaign.json"
        metadata["status"] = "waiting_for_development"
        metadata["development_campaign"] = str(development)
        save(directory / "campaign.json", metadata)
        print(f"Waiting for existing development campaign {args.after_development}", flush=True)
        while True:
            state = read(development)["status"]
            if state == "complete":
                break
            if state not in {"running"}:
                metadata.update(status="infrastructure_error", error=f"Development ended with {state}")
                save(directory / "campaign.json", metadata)
                return 1
            time.sleep(20)
    completed = {trial["index"] for trial in metadata["trials"]}
    for trial in protocol["schedule"]:
        if trial["index"] in completed:
            continue
        verify(protocol)
        condition = protocol["conditions"][trial["condition"]]
        name = f"{protocol['run_prefix']}-{trial['index']:03}"
        if (ROOT / "runs" / name).exists():
            raise ValueError("Existing trial folder is not a recorded terminal trial; inspect its actual process before continuing")
        budget = protocol["budget"]
        command = [sys.executable, str(ROOT / "agent_trial.py"), "--name", name,
                   "--purpose", "formal", "--task", trial["task"],
                   "--steps", str(budget["requests"]), "--output-budget", str(budget["generated_tokens"]),
                   "--max-response-tokens", str(budget["per_response_tokens"])]
        if condition["document"]:
            command += ["--skill", str(ROOT / condition["document"])]
        for flag in ["permissions", "checker"]:
            if condition[flag]:
                command.append("--" + flag)
        print(f"trial {trial['index'] + 1}/{len(protocol['schedule'])}: {trial['task']}, {trial['condition']}, repeat {trial['repeat']}", flush=True)
        metadata.update(status="running", current_trial={**trial, "run": name})
        with (directory / f"trial-{trial['index']:03}.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT.parent, stdout=log, stderr=subprocess.STDOUT)
            metadata["live_child_pid"] = process.pid
            save(directory / "campaign.json", metadata)
            exit_code = process.wait()
        metadata.pop("live_child_pid", None)
        record = {**trial, "run": name, "exit_code": exit_code}
        result_path = ROOT / "runs" / name / "result.json"
        if result_path.exists():
            result = read(result_path)
            record.update(state=result["state"], result_sha256=digest(result_path))
            metadata["trials"].append(record)
            save(directory / "campaign.json", metadata)
            print(json.dumps(record), flush=True)
        else:
            record["state"] = "runner_missing_result"
            metadata["trials"].append(record)
            metadata.update(status="infrastructure_error", error="Terminal trial has no result.json; retained as incomplete, not retried")
            save(directory / "campaign.json", metadata)
            return 1
    metadata.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat())
    metadata.pop("current_trial", None)
    save(directory / "campaign.json", metadata)
    print(json.dumps({"status": "complete", "trials": len(metadata["trials"])}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
