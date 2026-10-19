"""Analysis integrity checks; fixtures are not experimental observations."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collect_formal import collect_trial
from runtime import digest


def fixture(tmp_path, monkeypatch, attempts=1):
    monkeypatch.setattr("collect_formal.ROOT", tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    usage = {"prompt_tokens": 90, "completion_tokens": 10, "total_tokens": 100,
             "prompt_cache_hit_tokens": 70, "prompt_cache_miss_tokens": 20,
             "completion_tokens_details": {"reasoning_tokens": 4}}
    totals = {"input_tokens": 90, "output_tokens": 10, "total_tokens": 100,
              "cached_input_tokens": 70, "uncached_input_tokens": 20, "reasoning_tokens": 4}
    files = {
        "response-000.json": {"usage": usage},
        "config.json": {"purpose": "formal", "task": "fixture", "permissions": False, "checker": False,
                        "skill_sha256": "fixture-document", "runner_hashes": {}},
        "result.json": {"state": "step_budget", "seconds": 5, "requests_attempted": attempts, "usage": totals,
                        "source_findings": {}, "changed_infrastructure": [],
                        "final_evaluation": {"compile": {"ok": True, "seconds": 2}, "verify": {"ok": False, "seconds": 1}}},
        # A much faster tuning measurement exists, but final correctness failed.
        "cuda_event_timing.json": {"median_us": {"candidate": 0.1, "compiled": 100, "eager": 200}},
    }
    for name, data in files.items():
        (run / name).write_text(json.dumps(data), encoding="utf-8")
    (run / "events.jsonl").write_text("", encoding="utf-8")
    record = {"task": "fixture", "result_sha256": digest(run / "result.json")}
    condition = {"permissions": False, "checker": False, "document_sha256": "fixture-document"}
    price = {"per_tokens": 1000000, "off_peak": {"uncached_input": .15, "cached_input": .003, "output": .6},
             "peak": {"uncached_input": .3, "cached_input": .006, "output": 1.2}}
    return collect_trial(run, record, condition, {"frozen_files": {}}, price)


def test_previous_fast_tuning_result_does_not_survive_failed_final_verification(tmp_path, monkeypatch):
    row = fixture(tmp_path, monkeypatch)
    assert row["median_us"] is None
    assert row["candidate_over_compiled"] is None
    assert not row["performance_target_met"]
    assert not row["submitted_correct_and_fast"]


def test_unknown_request_usage_stays_visible_and_reasoning_is_not_charged_twice(tmp_path, monkeypatch):
    row = fixture(tmp_path, monkeypatch, attempts=2)
    assert row["attempts_without_usage"] == 1
    assert row["usage"]["output_tokens"] == 10
    assert row["cost_for_observed_usage"]["off_peak_usd"] == pytest.approx(9.21e-6)
    assert row["actor_loop_wall_seconds"] == 2
