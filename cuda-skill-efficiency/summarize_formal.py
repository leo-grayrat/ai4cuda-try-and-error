"""Audit the frozen 2 x 3 x 3 CUDA skill comparison and export a compact result."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TASKS = {"softmax": "softmax_rows", "layernorm": "layernorm_rows"}
SKILLS = {"none": None, "full": ROOT / "skills" / "full.md", "concise": ROOT / "skills" / "concise.md"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_digest(path: Path) -> str:
    """Git may check text files out with CRLF on Windows; trials used LF bytes."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def collect() -> dict:
    trials = []
    for short_task, task_name in TASKS.items():
        task = ROOT / "tasks" / "heldout" / task_name
        for skill_name, skill_path in SKILLS.items():
            for repeat in (1, 2, 3):
                name = f"formal-r{repeat}-{short_task}-{skill_name}"
                folder = ROOT / "results" / "traces" / name
                run_path = folder / "run.json"
                evaluation_path = folder / "evaluation.json"
                events_path = folder / "events.jsonl"
                run = json.loads(run_path.read_text(encoding="utf-8"))
                evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
                events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
                assert run["task"] == task_name and run["skill"] == (skill_path.name if skill_path else "none"), name
                assert run["model_requested"] == "deepseek/deepseek-flash" and run["runner_version"] == "1.18.12", name
                assert run["usage"] is not None, name
                assert run["tool_events"] == run["tool_feedback_events"], name
                assert run["usage"]["steps"] == sum(event.get("type") == "step_finish" for event in events), name
                for field, filename in (
                    ("task_sha256", "task.md"), ("reference_sha256", "reference.py"),
                    ("candidate_sha256", "candidate.py"), ("evaluator_sha256", "evaluate.py"),
                ):
                    assert run["task_hashes"][field] == source_digest(task / filename), name
                assert run["task_hashes"]["evaluator_common_sha256"] == source_digest(ROOT / "eval_common.py"), name
                assert run["skill_sha256"] == (source_digest(skill_path) if skill_path else None), name
                assert evaluation["evaluator_common_sha256"] == run["task_hashes"]["evaluator_common_sha256"], name
                assert evaluation["candidate_sha256"] == run["final_candidate_sha256"], name
                complete = run["status"] == "complete"
                valid = complete and evaluation["correct"] is True
                trials.append({
                    "name": name,
                    "task": task_name,
                    "skill": skill_name,
                    "repeat": repeat,
                    "status": run["status"],
                    "agent_complete": complete,
                    "candidate_correct": evaluation["correct"],
                    "valid": valid,
                    "fast_1": valid and evaluation.get("speedup", 0) > 1,
                    "speedup": evaluation.get("speedup") if evaluation["correct"] else None,
                    "evaluation_error": evaluation.get("error"),
                    "elapsed_s": run["elapsed_s"],
                    "steps": run["usage"]["steps"],
                    "tool_events": run["tool_events"],
                    "skill_tokens_estimate": run["skill_tokens_estimate"],
                    "usage": run["usage"],
                    "candidate_sha256": run["final_candidate_sha256"],
                    "events_sha256": digest(events_path),
                    "run_sha256": digest(run_path),
                    "evaluation_sha256": digest(evaluation_path),
                })
    cells = []
    for short_task, task_name in TASKS.items():
        for skill_name in SKILLS:
            rows = [row for row in trials if row["task"] == task_name and row["skill"] == skill_name]
            costs = [row["usage"]["total_tokens"] for row in rows]
            cells.append({
                "task": task_name,
                "skill": skill_name,
                "attempts": len(rows),
                "agent_complete": sum(row["agent_complete"] for row in rows),
                "valid": sum(row["valid"] for row in rows),
                "fast_1": sum(row["fast_1"] for row in rows),
                "total_tokens_median": statistics.median(costs),
                "total_tokens_min": min(costs),
                "total_tokens_max": max(costs),
                "valid_speedups": [row["speedup"] for row in rows if row["valid"]],
            })
    return {
        "frozen_commit": "0ecd96c",
        "model_requested": "deepseek/deepseek-flash",
        "runner_version": "1.18.12",
        "trials": trials,
        "cells": cells,
    }


if __name__ == "__main__":
    destination = ROOT / "results" / "formal-summary.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(collect(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(destination)
