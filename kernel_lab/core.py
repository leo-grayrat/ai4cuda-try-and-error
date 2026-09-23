"""Run records and analyses. Performance measurements are delegated to evaluators."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev


def stable_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_hash(source: bytes) -> str:
    """Hash source text consistently across Windows and Unix checkouts."""
    return hashlib.sha256(source.replace(b"\r\n", b"\n")).hexdigest()


def hardware_info() -> dict:
    command = [
        "nvidia-smi",
        "--query-gpu=uuid,name,driver_version,compute_cap",
        "--format=csv,noheader",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError("Select exactly one visible GPU before collecting a run")
    uuid, name, driver, capability = [part.strip() for part in rows[0].split(",", 3)]
    return {"uuid": uuid, "name": name, "driver": driver, "compute_capability": capability}


def validate_manifest(manifest: dict) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported manifest schema")
    if not manifest.get("experiment_id") or not isinstance(manifest.get("candidates"), list):
        raise ValueError("Manifest needs an experiment_id and candidates")
    ids = [item["id"] for item in manifest["candidates"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Candidate IDs must be unique")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", candidate_id) for candidate_id in ids):
        raise ValueError("Candidate IDs must use letters, digits, hyphens or underscores")
    if any(not isinstance(item.get("changes"), list) for item in manifest["candidates"]):
        raise ValueError("Each candidate needs a changes list")


def make_record(
    manifest: dict,
    measurements: list[dict],
    hardware: dict,
    evaluator_source: bytes,
    reference_sha256: str | None = None,
    upstream_revision: str | None = None,
    runtime: dict | None = None,
) -> dict:
    validate_manifest(manifest)
    by_id = {item["id"]: item for item in measurements}
    if set(by_id) != {item["id"] for item in manifest["candidates"]}:
        raise ValueError("Evaluator did not return every candidate exactly once")
    evaluator_hash = source_hash(evaluator_source)
    candidates = []
    for item in manifest["candidates"]:
        measurement = by_id[item["id"]]
        candidate_config = {key: value for key, value in item.items() if key not in ("id", "changes")}
        code_hash = measurement.get("code_hash") or stable_hash({"source": evaluator_hash, "config": candidate_config})
        candidates.append(
            {
                **measurement,
                "changes": item["changes"],
                "code_hash": code_hash,
            }
        )
    return {
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluator": manifest["evaluator"],
        "factors": manifest.get("factors", []),
        "workload": manifest["workload"],
        "workload_hash": stable_hash(manifest["workload"]),
        "evaluator_source_sha256": evaluator_hash,
        "reference_sha256": reference_sha256,
        "upstream_revision": upstream_revision,
        "runtime": runtime or {},
        "hardware": hardware,
        "candidates": candidates,
    }


def _valid_candidates(record: dict) -> dict[str, dict]:
    candidates = {item["id"]: item for item in record["candidates"]}
    if len(candidates) != len(record["candidates"]):
        raise ValueError("Duplicate candidate IDs")
    for item in candidates.values():
        if not item["correct"] or not math.isfinite(item["median_ms"]) or item["median_ms"] <= 0:
            raise ValueError(f"Candidate {item['id']} is incorrect or has invalid timing")
    return candidates


def credit_analysis(record: dict) -> dict:
    candidates = _valid_candidates(record)
    factors = record.get("factors") or sorted(set().union(*(set(item["changes"]) for item in candidates.values())))
    if len(factors) != 2 or len(set(factors)) != 2:
        raise ValueError("Credit analysis requires exactly two intervention factors")
    a, b = factors
    lookup: dict[frozenset, dict] = {}
    for item in candidates.values():
        changes = frozenset(item["changes"])
        if changes in lookup:
            raise ValueError("Only one measured candidate per factor combination is supported")
        lookup[changes] = item
    needed = [frozenset(), frozenset([a]), frozenset([b]), frozenset([a, b])]
    if set(lookup) != set(needed):
        raise ValueError("Need base, A, B and AB measured candidates")
    base, only_a, only_b, both = (lookup[key] for key in needed)
    score = lambda item: -math.log(item["median_ms"])
    return {
        "experiment_id": record["experiment_id"],
        "hardware": record["hardware"],
        "score": "negative log median latency in ms; higher is better",
        "factors": [a, b],
        "candidate_ids": {"base": base["id"], "A": only_a["id"], "B": only_b["id"], "AB": both["id"]},
        "median_ms": {"base": base["median_ms"], "A": only_a["median_ms"], "B": only_b["median_ms"], "AB": both["median_ms"]},
        "speedup_vs_base": {
            "A": base["median_ms"] / only_a["median_ms"],
            "B": base["median_ms"] / only_b["median_ms"],
            "AB": base["median_ms"] / both["median_ms"],
        },
        "log_contribution": {
            a: score(only_a) - score(base),
            b: score(only_b) - score(base),
            "interaction": score(both) - score(only_a) - score(only_b) + score(base),
        },
    }


def compare_hardware(first: dict, second: dict) -> dict:
    if first["experiment_id"] != second["experiment_id"] or first["workload_hash"] != second["workload_hash"]:
        raise ValueError("Runs must use the same experiment and workload")
    if first.get("evaluator") != second.get("evaluator") or first.get("evaluator_source_sha256") != second.get("evaluator_source_sha256"):
        raise ValueError("Evaluator implementations differ")
    if first["hardware"]["uuid"] == second["hardware"]["uuid"]:
        raise ValueError("Runs came from the same GPU; cross-hardware comparison needs another device")
    if first.get("reference_sha256") != second.get("reference_sha256"):
        raise ValueError("Reference implementations differ")
    if first.get("upstream_revision") != second.get("upstream_revision"):
        raise ValueError("Evaluator revisions differ")
    left, right = _valid_candidates(first), _valid_candidates(second)
    if set(left) != set(right):
        raise ValueError("Candidate sets differ")
    for candidate_id in left:
        if left[candidate_id]["code_hash"] != right[candidate_id]["code_hash"]:
            raise ValueError(f"Candidate {candidate_id} has different code/configuration")
    ids = sorted(left)
    rank_left = _average_ranks(left)
    rank_right = _average_ranks(right)
    n = len(ids)
    left_mean = mean(rank_left.values())
    right_mean = mean(rank_right.values())
    covariance = sum((rank_left[key] - left_mean) * (rank_right[key] - right_mean) for key in ids)
    left_variance = sum((rank_left[key] - left_mean) ** 2 for key in ids)
    right_variance = sum((rank_right[key] - right_mean) ** 2 for key in ids)
    rho = covariance / math.sqrt(left_variance * right_variance) if left_variance and right_variance else None
    flips = 0
    pairs = 0
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            left_diff = left[a]["median_ms"] - left[b]["median_ms"]
            right_diff = right[a]["median_ms"] - right[b]["median_ms"]
            if left_diff == 0 or right_diff == 0:
                continue
            pairs += 1
            flips += left_diff * right_diff < 0
    first_winner = min(ids, key=lambda key: left[key]["median_ms"])
    return {
        "experiment_id": first["experiment_id"],
        "gpu_a": first["hardware"],
        "gpu_b": second["hardware"],
        "candidate_count": n,
        "rank_correlation_spearman": rho,
        "pairwise_flips": flips,
        "comparable_pairs": pairs,
        "pairwise_flip_rate": None if not pairs else flips / pairs,
        "gpu_a_winner": first_winner,
        "gpu_a_winner_rank_on_b": rank_right[first_winner],
        "ranks_a": rank_left,
        "ranks_b": rank_right,
    }


def _average_ranks(candidates: dict[str, dict]) -> dict[str, float]:
    ordered = sorted(candidates, key=lambda key: candidates[key]["median_ms"])
    ranks = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and candidates[ordered[end]]["median_ms"] == candidates[ordered[start]]["median_ms"]:
            end += 1
        average_rank = (start + 1 + end) / 2
        for key in ordered[start:end]:
            ranks[key] = average_rank
        start = end
    return ranks


def credit_series(records: list[dict]) -> dict:
    if len(records) < 2:
        raise ValueError("Use at least two independent runs for a credit series")
    first = records[0]
    first_candidates = {item["id"]: item["code_hash"] for item in first["candidates"]}
    for record in records[1:]:
        if record["experiment_id"] != first["experiment_id"] or record["workload_hash"] != first["workload_hash"]:
            raise ValueError("Runs use different experiment/workload definitions")
        if record["hardware"]["uuid"] != first["hardware"]["uuid"]:
            raise ValueError("Credit series runs must use the same GPU")
        if {item["id"]: item["code_hash"] for item in record["candidates"]} != first_candidates:
            raise ValueError("Candidate code differs across runs")
        if record.get("reference_sha256") != first.get("reference_sha256") or record.get("upstream_revision") != first.get("upstream_revision"):
            raise ValueError("Reference or upstream evaluator differs across runs")
        if record.get("evaluator_source_sha256") != first.get("evaluator_source_sha256") or record.get("runtime") != first.get("runtime"):
            raise ValueError("Evaluator implementation or runtime differs across runs")
    analyses = [credit_analysis(record) for record in records]
    terms = analyses[0]["log_contribution"]
    summary = {}
    for term in terms:
        values = [analysis["log_contribution"][term] for analysis in analyses]
        summary[term] = {"mean": mean(values), "sample_std": stdev(values), "min": min(values), "max": max(values)}
    return {
        "experiment_id": first["experiment_id"],
        "hardware": first["hardware"],
        "run_count": len(records),
        "per_run": [analysis["log_contribution"] for analysis in analyses],
        "summary": summary,
    }


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
