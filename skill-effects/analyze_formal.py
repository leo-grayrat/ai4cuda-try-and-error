"""All-slot condition summaries and task/repeat pairs for the frozen protocol.

Only runs on a complete collection. Semantic and native-activity audits remain
separate outcomes; numerical success does not imply all requirements satisfied.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from formal_campaign import verify
from runtime import digest

ROOT = Path(__file__).resolve().parent
CONTRASTS = [
    ("core", "workflow"), ("workflow", "optimized"),
    ("original", "formatted"), ("original", "optimized"), ("optimized", "length_control"),
    ("original", "original_permissions"), ("original_permissions", "original_permissions_checker"),
    ("optimized", "optimized_permissions"), ("optimized_permissions", "optimized_permissions_checker"),
    ("optimized_permissions_checker", "residual_permissions_checker"),
]


def numeric_success(row):
    return bool(row.get("submitted_correct_and_fast"))


def independently_correct(row):
    checks = row.get("independent_checks", {})
    return bool(checks.get("compile") and checks.get("verify"))


def summarize(rows):
    successes = sum(numeric_success(row) for row in rows)
    known = [row for row in rows if row.get("usage") is not None]
    usage = {key: sum(row["usage"][key] for row in known)
             for key in ["input_tokens", "uncached_input_tokens", "cached_input_tokens",
                         "output_tokens", "reasoning_tokens", "total_tokens"]}
    cost = {period: sum(row["cost_for_observed_usage"][period] for row in known)
            for period in ["off_peak_usd", "peak_usd"]}
    latencies = [row["candidate_over_compiled"] for row in rows
                 if independently_correct(row) and row.get("candidate_over_compiled") is not None]
    walls = [row["actor_loop_wall_seconds"] for row in rows if row.get("actor_loop_wall_seconds") is not None]
    return {"scheduled": len(rows), "states": dict(Counter(row["state"] for row in rows)),
            "submitted": sum(bool(row.get("submitted")) for row in rows),
            "independently_correct_last_artifact": sum(independently_correct(row) for row in rows),
            "submitted_correct_and_fast": successes,
            "all_applicable_requirements": "pending separate source/behavior audit",
            "report_available": sum(bool(row.get("report_available")) for row in rows),
            "report_quality": "pending masked review; missing reports are not assigned fabricated prose scores",
            "observed_usage": usage, "trials_with_usage": len(known),
            "trials_without_usage": len(rows) - len(known),
            "attempts_without_usage": sum(row.get("attempts_without_usage", 0) for row in rows),
            "published_cost_for_observed_usage": cost,
            "observed_cost_per_numerical_delivery": {key: value / successes if successes else None for key, value in cost.items()},
            "cost_interpretation": "All observed trial usage, including unsuccessful trials, divided by submitted/correct/fast count. Null at zero successes. Unreported usage remains unknown; these are published-price estimates, not invoices.",
            "correct_artifact_timing_count": len(latencies),
            "median_candidate_over_compiled_on_correct_subset": statistics.median(latencies) if latencies else None,
            "actor_wall_count": len(walls), "median_actor_loop_wall_seconds": statistics.median(walls) if walls else None}


def difference(left, right, key):
    a, b = left.get(key), right.get(key)
    return b - a if a is not None and b is not None else None


def pair(left, right):
    def identity(row):
        return {key: row.get(key) for key in ["index", "run", "state", "submitted", "report_available",
                "independent_checks", "candidate_over_compiled", "submitted_correct_and_fast",
                "usage", "attempts_without_usage", "cost_for_observed_usage"]}
    both_correct_timed = all(independently_correct(row) and row.get("candidate_over_compiled") is not None
                             for row in [left, right])
    return {"task": left["task"], "repeat": left["repeat"], "left": identity(left), "right": identity(right),
            "numerical_delivery_difference_right_minus_left": int(numeric_success(right)) - int(numeric_success(left)),
            "report_availability_difference_right_minus_left": int(bool(right.get("report_available"))) - int(bool(left.get("report_available"))),
            "both_independently_correct_and_timed": both_correct_timed,
            "candidate_over_compiled_difference_right_minus_left": difference(left, right, "candidate_over_compiled") if both_correct_timed else None,
            "output_token_difference_right_minus_left": difference(left.get("usage") or {}, right.get("usage") or {}, "output_tokens"),
            "observed_price_difference_right_minus_left": {
                key: difference(left.get("cost_for_observed_usage") or {}, right.get("cost_for_observed_usage") or {}, key)
                for key in ["off_peak_usd", "peak_usd"]}}


def analyze(collection, protocol):
    if collection["terminal"] != collection["scheduled"] or collection["campaign_status"] != "complete":
        raise ValueError("Wait for all scheduled slots; partial data are not a condition comparison")
    rows = collection["trials"]
    scheduled = {row["index"]: row for row in protocol["schedule"]}
    if len(rows) != len(scheduled) or len({row["index"] for row in rows}) != len(rows):
        raise ValueError("Missing or duplicated scheduled slot")
    for row in rows:
        expected = scheduled[row["index"]]
        if any(row[key] != expected[key] for key in ["condition", "task", "repeat"]):
            raise ValueError("Trial does not match the frozen schedule")
    conditions = {name: summarize([row for row in rows if row["condition"] == name]) for name in protocol["conditions"]}
    by_key = {(row["condition"], row["task"], row["repeat"]): row for row in rows}
    pairs = []
    for left, right in CONTRASTS:
        if left not in conditions or right not in conditions:
            continue
        paired = [pair(by_key[left, task, repeat], by_key[right, task, repeat])
                  for task in protocol["tasks"] for repeat in range(1, protocol["repeats"] + 1)]
        pairs.append({"left_condition": left, "right_condition": right, "pairs": paired,
                      "paired_slots": len(paired),
                      "timing_pair_count": sum(row["both_independently_correct_and_timed"] for row in paired)})
    return {"scope": "Four task types and two generations per task/condition; no claim of universal significance. Repeat pairs do not create new independent task types.",
            "quality_scope": "Numerical delivery, artifact correctness, report availability and qualitative compliance are distinct outcomes.",
            "conditions": conditions, "paired_comparisons": pairs, "all_trials": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol/v1/manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collection = json.loads(args.collection.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    verify(protocol)
    if collection["protocol_sha256"] != digest(args.protocol):
        raise ValueError("Protocol identity mismatch")
    data = analyze(collection, protocol)
    data.update(collection_sha256=digest(args.collection), protocol_sha256=digest(args.protocol),
                analysis_sha256=digest(Path(__file__)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"conditions": len(data["conditions"]), "trials": len(data["all_trials"]),
                      "comparisons": len(data["paired_comparisons"])}))


if __name__ == "__main__":
    main()
