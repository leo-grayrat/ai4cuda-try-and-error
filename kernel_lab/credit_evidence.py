"""Join code-first action hypotheses with existing KernelBench run records."""

from __future__ import annotations

import math
from statistics import median


def summarize_case(trace: dict, records: list[dict]) -> dict:
    """Report conditional leave-one-action-out effects, not unique causality."""
    if not records:
        raise ValueError("At least one measurement record is required")
    expected_hashes = trace["candidate_sha256"]
    comparison_fields = ("experiment_id", "workload_hash", "evaluator_source_sha256",
                         "reference_sha256", "upstream_revision")
    first = records[0]
    reference_hash = (trace.get("source_sha256") or {}).get("reference")
    if reference_hash is not None and first.get("reference_sha256") != reference_hash:
        raise ValueError("Reference source hash differs from prepared trace")
    for record in records:
        if record["experiment_id"] != trace["experiment_id"]:
            raise ValueError("Experiment IDs differ")
        if any(record.get(field) != first.get(field) for field in comparison_fields):
            raise ValueError("Measurement settings differ across runs")
        if record["hardware"]["uuid"] != first["hardware"]["uuid"]:
            raise ValueError("Measurements came from different GPUs")
        candidates = {item["id"]: item for item in record["candidates"]}
        if len(candidates) != len(record["candidates"]):
            raise ValueError("Duplicate candidate IDs in a measurement record")
        if set(candidates) != set(expected_hashes):
            raise ValueError("Candidate sets differ from prepared case")
        for candidate_id, expected in expected_hashes.items():
            if candidates[candidate_id]["code_hash"] != expected:
                raise ValueError(f"Candidate source hash differs: {candidate_id}")

    endpoint_speedups = []
    for record in records:
        candidates = {item["id"]: item for item in record["candidates"]}
        parent, child = candidates["parent"], candidates["child"]
        if not _timed_and_correct(parent) or not _timed_and_correct(child):
            endpoint_speedups = []
            break
        endpoint_speedups.append(parent["median_ms"] / child["median_ms"])
    endpoint = {
        "status": "measured" if endpoint_speedups else "invalid_parent_or_child",
        "speedups": endpoint_speedups,
        "median_speedup": median(endpoint_speedups) if endpoint_speedups else None,
    }

    action_results = []
    for action in trace["actions"]:
        candidate_id = action.get("candidate_id")
        mapping = (trace.get("source_to_response") or {}).get("actions", {}).get(action["id"])
        result = {
            "id": action["id"], "category": action["category"],
            "candidate_id": candidate_id,
            "source_edits": [
                {"parent_lines": edit["parent_lines"], "child_lines": edit["child_lines"]}
                for edit in action.get("edits", [])
            ],
            "generation_position": mapping,
            "comparison": "child versus child_without_action",
            "status": "not_probed", "log_effects": [],
            "median_log_effect": None, "sign_consistent": None,
        }
        if candidate_id is None:
            action_results.append(result)
            continue
        effects = []
        invalid = False
        for record in records:
            candidates = {item["id"]: item for item in record["candidates"]}
            child = candidates["child"]
            without = candidates[candidate_id]
            if not _timed_and_correct(child) or not _timed_and_correct(without):
                invalid = True
                break
            effects.append(math.log(without["median_ms"] / child["median_ms"]))
        if invalid:
            result["status"] = "invalid_reversion"
        else:
            result["status"] = "measured"
            result["log_effects"] = effects
            result["median_log_effect"] = median(effects)
            result["sign_consistent"] = all(value > 0 for value in effects) or all(value < 0 for value in effects)
        action_results.append(result)
    extra_per_run = sum(candidate_id not in ("parent", "child") for candidate_id in expected_hashes)
    return {
        "experiment_id": trace["experiment_id"],
        "provenance": trace.get("provenance"),
        "source_sha256": trace.get("source_sha256"),
        "candidate_sha256": expected_hashes,
        "reference_sha256": first.get("reference_sha256"),
        "evaluator_source_sha256": first.get("evaluator_source_sha256"),
        "workload_hash": first.get("workload_hash"),
        "upstream_revision": first.get("upstream_revision"),
        "hardware": first["hardware"],
        "rounds": len(records),
        "endpoint": endpoint,
        "score": "log(without_action_median_ms / child_median_ms); positive means action helped in child context",
        "actions": action_results,
        "extra_candidate_evaluations": extra_per_run * len(records),
    }


def _timed_and_correct(candidate: dict) -> bool:
    value = candidate.get("median_ms")
    return candidate.get("correct") is True and isinstance(value, (int, float)) and math.isfinite(value) and value > 0
