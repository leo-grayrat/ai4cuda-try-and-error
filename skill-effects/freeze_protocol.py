"""Freeze v1 documents, task inputs, runtime hashes, budgets and random schedule."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil
import subprocess

from runtime import digest


ROOT = Path(__file__).resolve().parent


def main():
    directory = ROOT / "protocol/v1"
    directory.mkdir(parents=True, exist_ok=False)
    docs = directory / "documents"
    docs.mkdir()
    choices = {
        "original": "sources/CUDA-Agent.SKILL.md",
        "formatted": "materials/original-conditions/formatted.md",
        "optimized": "materials/original-conditions/optimized.md",
        "core": "materials/original-conditions/core.md",
        "workflow": "materials/original-conditions/workflow.md",
        "length_control": "materials/original-conditions/length_control.md",
        "original_permissions": "sources/CUDA-Agent.SKILL.md",
        "original_permissions_checker": "sources/CUDA-Agent.SKILL.md",
        "optimized_permissions": "materials/original-conditions/optimized.md",
        "optimized_permissions_checker": "materials/original-conditions/optimized.md",
        "residual_permissions_checker": "materials/original-conditions/optimized_residual.md",
    }
    assert all(path == "sources/CUDA-Agent.SKILL.md" or path.startswith("materials/original-conditions/") for path in choices.values())
    conditions = {}
    for name, source in choices.items():
        target = docs / (name + ".md")
        if source:
            shutil.copyfile(ROOT / source, target)
        conditions[name] = {"document": target.relative_to(ROOT).as_posix() if source else None,
                            "source": source, "document_sha256": digest(target) if source else None,
                            "permissions": "permissions" in name, "checker": "checker" in name}
    metadata = directory / "material-audit"
    metadata.mkdir()
    for source in ["materials/original-conditions/manifest.json", "materials/cuda_rules.json",
                   "materials/report_rubric.json", "materials/cuda_rules.tsv", "materials/SOURCE_AUDIT.zh.md"]:
        target = metadata / (source.replace("/", "--"))
        shutil.copyfile(ROOT / source, target)
    tasks = ["sigmoid", "softmax", "layernorm", "gemm_chain"]
    seed = 20261019
    rng = random.Random(seed)
    blocks = [(task, repeat) for task in tasks for repeat in [1, 2]]
    rng.shuffle(blocks)
    schedule = []
    for task, repeat in blocks:
        names = list(conditions)
        rng.shuffle(names)
        for name in names:
            schedule.append({"index": len(schedule), "task": task, "repeat": repeat, "condition": name})
    assert len(schedule) == 88
    assert [row["index"] for row in schedule] == list(range(88))
    assert len({(r["task"], r["repeat"], r["condition"]) for r in schedule}) == 88
    frozen = [ROOT / name for name in ["agent_trial.py", "runtime.py", "environment_probe.py", "event_timing_probe.py",
        "verify_candidate.py", "gpu_lock.py", "prepare_tasks.py", "task_catalog.json", "formal_campaign.py", "freeze_protocol.py",
        "prepare_original_conditions.py", "build_materials.py", "materials/cuda_rules.tsv", "materials/cuda_rules.json",
        "materials/report_contract.md", "sources/deepseek-pricing-2026-10-19.json", "FORMAL_PROTOCOL.zh.md"]]
    frozen += [path for path in (ROOT / "sources/CUDA-Agent").rglob("*") if path.is_file() and "__pycache__" not in path.parts]
    frozen += [ROOT / "tasks" / task / name for task in tasks for name in ["model.py", "contract.md"]]
    frozen += [path for path in directory.rglob("*") if path.is_file()]
    protocol = {"status": "frozen", "version": "v1", "run_prefix": "f01",
                "frozen_utc": datetime.now(timezone.utc).isoformat(),
                "repository_commit_before_freeze": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, text=True).strip(),
                "seed": seed, "tasks": tasks, "repeats": 2, "conditions": conditions, "schedule": schedule,
                "budget": {"requests": 64, "generated_tokens": 96000, "per_response_tokens": 32768,
                           "tool_timeout_seconds": 180, "input_tokens": "measured, not capped independently"},
                "model": {"alias": "deepseek-flash", "temperature": 1.0, "thinking": "enabled", "reasoning_effort": "high", "api_sampling_seed": "not requested; no deterministic repeat guarantee"},
                "ordering": "Random task/repeat blocks; condition order randomized within each block; GPU and formal trials serial",
                "construction_cutoff": "All conditions derive from the pinned original CUDA Agent skill. Constructed failure-driven appendices and synthetic repetition-pressure documents are excluded from v1.",
                "failures": "Retain every scheduled trial. No outcome-based retries. Report unsubmitted, budget, truncation, API/trace and infrastructure failures separately; all-run delivery denominator remains 88. Sensitivity analysis for measurement failures is additional, not a replacement.",
                "analysis_protocol": "FORMAL_PROTOCOL.zh.md",
                "frozen_files": {path.relative_to(ROOT).as_posix(): digest(path) for path in sorted(set(frozen))}}
    path = directory / "manifest.json"
    path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"protocol": str(path), "conditions": len(conditions), "trials": len(schedule), "first_trial": schedule[0], "sha256": digest(path)}))


if __name__ == "__main__":
    main()
