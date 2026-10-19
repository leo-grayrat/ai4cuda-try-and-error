"""Prevent success-only cost estimates and silent loss of failed pairs."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_formal import analyze, pair, summarize


def row(success, output, cost):
    return {"task": "fixture", "repeat": 1, "state": "submitted" if success else "generation_truncated",
            "submitted": success, "submitted_correct_and_fast": success, "report_available": success,
            "independent_checks": {"compile": True, "verify": success},
            "candidate_over_compiled": .5 if success else None,
            "usage": {"input_tokens": 100, "uncached_input_tokens": 100, "cached_input_tokens": 0,
                      "output_tokens": output, "reasoning_tokens": output // 2, "total_tokens": 100 + output},
            "cost_for_observed_usage": {"off_peak_usd": cost, "peak_usd": cost * 2}}


def test_failed_trials_still_contribute_cost_and_delivery_denominator():
    data = summarize([row(True, 100, 1), row(False, 900, 9)])
    assert data["scheduled"] == 2
    assert data["submitted_correct_and_fast"] == 1
    assert data["observed_usage"]["output_tokens"] == 1000
    assert data["observed_cost_per_numerical_delivery"]["off_peak_usd"] == 10
    assert data["correct_artifact_timing_count"] == 1
    assert summarize([row(False, 900, 9)])["observed_cost_per_numerical_delivery"]["off_peak_usd"] is None


def test_failed_pair_keeps_both_states_and_costs_without_fabricating_speed():
    data = pair(row(True, 100, 1), row(False, 900, 9))
    assert data["right"]["state"] == "generation_truncated"
    assert data["numerical_delivery_difference_right_minus_left"] == -1
    assert data["candidate_over_compiled_difference_right_minus_left"] is None
    assert data["observed_price_difference_right_minus_left"]["off_peak_usd"] == 8


def test_partial_collection_cannot_produce_comparative_results():
    with pytest.raises(ValueError, match="all scheduled"):
        analyze({"terminal": 1, "scheduled": 88, "campaign_status": "running"}, {})
