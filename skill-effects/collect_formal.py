"""Collect terminal frozen-protocol trials without dropping pending/failing slots.

Read-only with respect to runtime inputs and raw trials. A previous tuning timing
file is never used when the final independent benchmark did not pass.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from formal_campaign import verify
from runtime import digest, summarize_usage


ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def price_bounds(usage, price):
    if usage is None:
        return None
    return {period + "_usd": sum(usage[key] * price[period][rate] for key, rate in [
        ("uncached_input_tokens", "uncached_input"), ("cached_input_tokens", "cached_input"),
        ("output_tokens", "output")]) / price["per_tokens"] for period in ["off_peak", "peak"]}


def collect_trial(run, record, condition, protocol, price):
    result = read(run / "result.json")
    assert digest(run / "result.json") == record["result_sha256"]
    config = read(run / "config.json")
    assert config["purpose"] == "formal" and config["task"] == record["task"]
    for flag in ["permissions", "checker"]:
        assert config[flag] == condition[flag]
    assert config["skill_sha256"] == condition["document_sha256"]
    for filename, expected in config["runner_hashes"].items():
        assert protocol["frozen_files"][filename] == expected
    responses = [read(path) for path in sorted(run.glob("response-*.json"))]
    usage_rows = [row["usage"] for row in responses if row.get("usage")]
    known_usage = summarize_usage(usage_rows) if usage_rows else None
    if result.get("usage") is not None:
        assert known_usage == result["usage"]
    events = [json.loads(line) for line in (run / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    tools = [event for event in events if event["type"] == "tool"]
    operations = []
    accepted = False
    after_finish = 0
    permission_denials = 0
    for event in tools:
        if accepted:
            after_finish += 1
        try:
            arguments = json.loads(event["call"]["function"]["arguments"])
        except (KeyError, ValueError):
            arguments = None
        operation = arguments.get("operation") if isinstance(arguments, dict) else None
        operations.append(operation if isinstance(operation, str) else "invalid")
        outcome = event["result"]
        permission_denials += "PermissionError" in outcome.get("error", "")
        accepted |= bool(outcome.get("submitted"))
    final = result["final_evaluation"]
    checks = {name: bool(final.get(name, {}).get("ok")) for name in ["compile", "verify", "benchmark"]}
    timing, timing_hash = None, None
    if all(checks.values()):
        path = run / "cuda_event_timing.json"
        timing = read(path)
        timing_hash = digest(path)
        assert all(value > 0 for value in timing["median_us"].values())
    ratio = timing["median_us"]["candidate"] / timing["median_us"]["compiled"] if timing else None
    submitted = result["state"] == "submitted"
    report_path = run / "submitted-report.json"
    report = read(report_path).get("report", "") if submitted and report_path.exists() else ""
    final_seconds = sum(row.get("seconds", 0) for row in final.values())
    final_wait = sum(row.get("gpu_wait_seconds", 0) for row in final.values())
    actor_wait = sum(event["result"].get("gpu_wait_seconds", 0) for event in tools)
    return {**record, "submitted": submitted, "independent_checks": checks,
        "source_findings": result["source_findings"], "changed_infrastructure": result["changed_infrastructure"],
        "report_available": bool(report.strip()), "report_path": str(report_path.relative_to(ROOT)) if report.strip() else None,
        "median_us": timing["median_us"] if timing else None, "candidate_over_compiled": ratio,
        "performance_target_met": ratio is not None and ratio <= 0.95,
        "submitted_correct_and_fast": submitted and all(checks.values()) and ratio is not None and ratio <= 0.95,
        "semantic_compliance": "pending independent rule and explanation audit",
        "kernel_activity": "pending independent native-kernel trace",
        "timing_sha256": timing_hash, "usage": known_usage,
        "cost_for_observed_usage": price_bounds(known_usage, price),
        "requests_attempted": result["requests_attempted"], "responses_with_usage": len(usage_rows),
        "attempts_without_usage": result["requests_attempted"] - len(usage_rows),
        "tool_operation_counts": dict(Counter(operations)),
        "failed_tool_calls": sum(event["result"].get("ok") is False for event in tools),
        "permission_denials": permission_denials,
        "finish_rejections": sum(bool(event["result"].get("finish_rejected")) for event in tools),
        "calls_after_accepted_finish": after_finish,
        "actor_executable_tool_seconds": sum(event["result"].get("seconds", 0) for event in tools),
        "actor_checker_seconds": sum(event["result"].get("checker_seconds", 0) for event in tools),
        "actor_gpu_wait_seconds": actor_wait,
        "actor_loop_wall_seconds": max(0, result["seconds"] - final_seconds - final_wait),
        "independent_final_tool_seconds": final_seconds, "independent_final_gpu_wait_seconds": final_wait,
        "loop_plus_final_seconds": result["seconds"],
        "wall_scope": "Driver loop, excluding workspace initialization and models metadata GET; final independent phase separated; GPU wait recorded separately"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol/v1/manifest.json")
    parser.add_argument("--campaign", type=Path, default=ROOT / "runs/formal-series-01/campaign.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol, campaign = read(args.protocol), read(args.campaign)
    verify(protocol)
    assert campaign["protocol_sha256"] == digest(args.protocol)
    price = read(ROOT / "sources/deepseek-pricing-2026-10-19.json")
    records = {record["index"]: record for record in campaign["trials"]}
    rows = []
    for scheduled in protocol["schedule"]:
        record = records.get(scheduled["index"])
        if record is None:
            rows.append({**scheduled, "state": "pending_or_running", "submitted": False})
        elif record["state"] == "runner_missing_result":
            rows.append({**record, "submitted": False, "usage": None, "cost_for_observed_usage": None})
        else:
            rows.append(collect_trial(ROOT / "runs" / record["run"], record,
                                      protocol["conditions"][record["condition"]], protocol, price))
    result = {"collected_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": digest(args.protocol),
              "campaign_status": campaign["status"], "scheduled": len(rows), "terminal": len(records),
              "state_counts": dict(Counter(row["state"] for row in rows)),
              "status": "complete data collection" if len(records) == len(rows) else "partial collection; no comparative conclusion",
              "cost_note": "Published price bounds for observed usage; unanswered/error attempts may have unknown billed usage; not an invoice",
              "trials": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "trials"}))


if __name__ == "__main__":
    main()
