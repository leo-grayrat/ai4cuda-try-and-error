"""Run serial development episodes and preserve each append-only snapshot.

No held-out task is used or read. A complete trial record may be unsuccessful;
it remains part of the sequence. Missing records stop the campaign for diagnosis.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from agent_trial import ROOT, check_task_split, save
from runtime import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--start-document", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--tasks", nargs="+", default=["dev_cumsum", "dev_rmsnorm", "axpby"])
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--output-budget", type=int, default=96000)
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in {".", ".."}:
        parser.error("name must be a directory name")
    if args.rounds < 1 or args.steps < 1 or args.output_budget < 256:
        parser.error("positive rounds, steps and output budget are required")
    for task in args.tasks:
        check_task_split(task, "development")
    directory = ROOT / "runs" / args.name
    directory.mkdir(parents=True, exist_ok=False)
    document = args.start_document.resolve()
    metadata = {"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(), "rounds": args.rounds, "tasks": args.tasks, "steps": args.steps, "output_budget": args.output_budget, "start_document": str(document), "start_sha256": digest(document), "purpose": "development corpus construction only", "status": "running", "episodes": []}
    save(directory / "campaign.json", metadata)

    def execute(command, log_path):
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT.parent)
            metadata["live_child_pid"] = process.pid
            metadata["live_child_command"] = command
            save(directory / "campaign.json", metadata)
            code = process.wait()
        metadata.pop("live_child_pid", None)
        metadata.pop("live_child_command", None)
        return code

    for index in range(args.rounds):
        task = args.tasks[index % len(args.tasks)]
        trial_name = f"{args.name}-episode-{index + 1:02d}-{task}"
        print(f"episode {index + 1}/{args.rounds}: {task}, document {document}", flush=True)
        episode = {"index": index + 1, "task": task, "document_sha256": digest(document), "trial": trial_name, "status": "agent_running"}
        metadata["episodes"].append(episode)
        command = [sys.executable, str(ROOT / "agent_trial.py"), "--name", trial_name, "--task", task, "--skill", str(document), "--steps", str(args.steps), "--output-budget", str(args.output_budget), "--max-response-tokens", "32768"]
        episode["agent_exit_code"] = execute(command, directory / f"episode-{index + 1:02d}-agent.log")
        result_path = ROOT / "runs" / trial_name / "result.json"
        if not result_path.exists():
            metadata["status"] = "infrastructure_error"
            episode["status"] = "missing_trial_result"
            save(directory / "campaign.json", metadata)
            return 1
        result = json.loads(result_path.read_text(encoding="utf-8"))
        episode["trial_state"] = result["state"]
        episode["trial_usage"] = result["usage"]
        if result["state"] == "api_or_trace_error" or result.get("usage_error"):
            episode["status"] = "no_addition_due_to_incomplete_trace"
            save(directory / "campaign.json", metadata)
            continue
        additions = directory / f"append-{index + 1:02d}"
        episode["status"] = "rule_writing"
        command = [sys.executable, str(ROOT / "append_rules.py"), "--trial", str(result_path.parent), "--existing", str(document), "--output", str(additions), "--max-tokens", "32768"]
        code = execute(command, directory / f"episode-{index + 1:02d}-append.log")
        if code != 0 or not (additions / "cumulative.md").exists():
            episode["status"] = "rule_writing_incomplete"
            metadata["status"] = "rule_writing_error"
            save(directory / "campaign.json", metadata)
            return 1
        document = additions / "cumulative.md"
        episode.update(status="complete", cumulative_sha256=digest(document), cumulative_lines=len(document.read_text(encoding="utf-8").splitlines()), addition_lines=len((additions / "addition.md").read_text(encoding="utf-8").splitlines()))
        save(directory / "campaign.json", metadata)
        print(json.dumps(episode), flush=True)
    metadata.update(status="complete", final_document=str(document), final_sha256=digest(document), completed_utc=datetime.now(timezone.utc).isoformat())
    save(directory / "campaign.json", metadata)
    print(json.dumps({"status": "complete", "final_document": str(document)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
