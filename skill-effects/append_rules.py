"""Capture failure-driven append-only instruction writing, without padding.

The input trial must be a development trial. Held-out task records are never
fed to the rule writer. Maintenance API usage is saved separately.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import requests

from agent_trial import ROOT, save
from runtime import compiler_feedback, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--existing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=32768)
    args = parser.parse_args()
    trial = args.trial.resolve()
    config = json.loads((trial / "config.json").read_text(encoding="utf-8"))
    if config["purpose"] != "development":
        raise ValueError("Only development trials can supply rule-writing feedback")
    outcome = json.loads((trial / "result.json").read_text(encoding="utf-8"))
    first_request = json.loads((trial / "request-000.json").read_text(encoding="utf-8"))
    initial_user = next(message["content"] for message in first_request["messages"] if message["role"] == "user")
    task_request = initial_user.split("\n\nAdditional development guidance:\n", 1)[0]
    dialogue = json.loads((trial / "messages.json").read_text(encoding="utf-8"))
    visible_explanations = [{"message_index": index, "content": message["content"]} for index, message in enumerate(dialogue) if message["role"] == "assistant" and message.get("content")]
    report_path = trial / "submitted-report.json"
    last_report_attempt = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
    events = [json.loads(line) for line in (trial / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    timeline = []
    for event in events:
        if event["type"] == "tool":
            function = event["call"]["function"]
            try:
                arguments = json.loads(function["arguments"])
            except ValueError:
                arguments = {}
            if not isinstance(arguments, dict):
                arguments = {"invalid_argument_type": type(arguments).__name__}
            result = event["result"]
            timeline.append({"request": event["index"], "function": function["name"], "operation": arguments.get("operation"), "path": arguments.get("path"), "argument_fields": sorted(arguments) if isinstance(arguments, dict) else [], "ok": result.get("ok"), "error": result.get("error"), "log": result.get("log")})
    final_logs = {}
    for name, result in outcome["final_evaluation"].items():
        if result.get("log"):
            log = trial / result["log"]
            final_logs[name] = compiler_feedback(log.read_text(encoding="utf-8", errors="replace"))
    failure_logs = {}
    for item in timeline:
        if item.get("ok") is False and item.get("log"):
            path = trial / item["log"]
            failure_logs[item["log"]] = compiler_feedback(path.read_text(encoding="utf-8", errors="replace"))
    evidence = {"input_policy": "actual-task-and-visible-dialogue-v2", "task": config["task"], "actual_task_request": task_request, "visible_assistant_explanations": visible_explanations, "last_report_attempt": last_report_attempt, "report_was_submitted": outcome["state"] == "submitted", "state": outcome["state"], "source_findings": outcome["source_findings"], "changed_infrastructure": outcome["changed_infrastructure"], "timeline": timeline, "final_logs": final_logs, "observed_failure_logs": failure_logs, "candidate": {str(p.relative_to(trial / "work")): p.read_text(encoding="utf-8") for p in (trial / "work").rglob("*") if p.is_file() and (p.name == "model_new.py" or p.parent.name == "kernels") and p.suffix in {".py", ".cu", ".cpp", ".h"}}}
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    save(directory / "evidence.json", evidence)
    existing = args.existing.read_text(encoding="utf-8")
    prompt = """You maintain development instructions for a CUDA coding agent. After the development episode below, write the guidance you would append to the current instruction document to prevent the observed failures from recurring. Keep the existing document unchanged. Base additions on actual observations; do not treat a tooling transport/encoding failure as a CUDA coding mistake. Do not claim your additions have already improved behavior. Return only the Markdown material to append, with any scope or exception needed for the rules to be accurate. Choose the amount of explanation you consider useful; no line-count quota is imposed.

CURRENT DOCUMENT:
""" + existing + "\n\nOBSERVED DEVELOPMENT EPISODE:\n" + json.dumps(evidence, ensure_ascii=False)
    body = {"model": "deepseek-flash", "messages": [{"role": "user", "content": prompt}], "thinking": {"type": "enabled"}, "reasoning_effort": "high", "max_tokens": args.max_tokens, "temperature": 1.0}
    save(directory / "request.json", body)
    save(directory / "provenance.json", {"created_utc": datetime.now(timezone.utc).isoformat(), "trial": str(trial.relative_to(ROOT)), "existing": str(args.existing), "existing_sha256": digest(args.existing), "input_policy": evidence["input_policy"], "writer_code_sha256": digest(Path(__file__)), "purpose": "constructed append-only development history, not recovered FlashRT author history", "status": "requested"})
    key = json.loads((Path.home() / ".local/share/opencode/auth.json").read_text(encoding="utf-8"))["deepseek"]["key"]
    response = requests.post("https://api.deepseek.com/chat/completions", json=body, headers={"Authorization": "Bearer " + key}, timeout=(20, 180))
    data = response.json()
    save(directory / "response.json", data)
    response.raise_for_status()
    choice = data["choices"][0]
    if choice["finish_reason"] != "stop" or not choice["message"].get("content"):
        provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
        provenance.update(status="incomplete", finish_reason=choice["finish_reason"], usage=data["usage"])
        save(directory / "provenance.json", provenance)
        raise ValueError("Rule-writing output is incomplete; raw response retained")
    addition = choice["message"]["content"]
    (directory / "addition.md").write_text(addition, encoding="utf-8")
    (directory / "cumulative.md").write_text(existing.rstrip() + "\n\n" + addition.rstrip() + "\n", encoding="utf-8")
    provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
    provenance.update(status="complete", addition_sha256=digest(directory / "addition.md"), cumulative_sha256=digest(directory / "cumulative.md"), usage=data["usage"])
    save(directory / "provenance.json", provenance)
    print(json.dumps({"addition_lines": len(addition.splitlines()), "cumulative_lines": len((directory / "cumulative.md").read_text(encoding="utf-8").splitlines()), "usage": data["usage"]}))


if __name__ == "__main__":
    main()
