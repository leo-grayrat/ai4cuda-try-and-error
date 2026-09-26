"""Measured edit effects require matching code and a valid comparison."""

import math

import pytest

from kernel_lab.credit_evidence import summarize_case


def _trace() -> dict:
    return {
        "experiment_id": "case-1",
        "candidate_sha256": {"parent": "hash-parent", "child": "hash-child"},
        "provenance": {"task": "2_1", "parent_node": 1, "child_node": 2},
        "actions": [{"id": "action-1", "category": "kernel_compute", "candidate_id": "parent",
                     "edits": [{"parent_lines": [4, 4], "child_lines": [5, 5]}]}],
    }


def _record(parent_ms: float, child_ms: float) -> dict:
    return {
        "experiment_id": "case-1", "hardware": {"uuid": "gpu-1"},
        "workload_hash": "workload", "evaluator_source_sha256": "eval",
        "reference_sha256": "reference", "upstream_revision": "revision",
        "candidates": [
            {"id": "parent", "code_hash": "hash-parent", "correct": True, "median_ms": parent_ms},
            {"id": "child", "code_hash": "hash-child", "correct": True, "median_ms": child_ms},
        ],
    }


def test_undo_effect_is_measured_in_child_context_across_repeats() -> None:
    result = summarize_case(_trace(), [_record(10.0, 8.0), _record(11.0, 8.5)])
    assert result["endpoint"]["status"] == "measured"
    assert result["endpoint"]["median_speedup"] == pytest.approx(
        sorted([10.0 / 8.0, 11.0 / 8.5])[0] / 2 + sorted([10.0 / 8.0, 11.0 / 8.5])[1] / 2
    )
    action = result["actions"][0]
    assert result["provenance"]["parent_node"] == 1
    assert result["candidate_sha256"]["child"] == "hash-child"
    assert result["workload_hash"] == "workload"
    assert result["upstream_revision"] == "revision"
    assert action["source_edits"] == [{"parent_lines": [4, 4], "child_lines": [5, 5]}]
    assert action["generation_position"] is None
    assert action["status"] == "measured"
    assert action["comparison"] == "child versus child_without_action"
    assert action["log_effects"][0] == pytest.approx(math.log(10.0 / 8.0))
    assert action["sign_consistent"] is True


def test_rejects_run_for_different_candidate_source() -> None:
    wrong = _record(10.0, 8.0)
    wrong["candidates"][1]["code_hash"] = "other"
    with pytest.raises(ValueError, match="hash"):
        summarize_case(_trace(), [wrong])


def test_rejects_run_for_different_reference_source() -> None:
    trace = _trace()
    trace["source_sha256"] = {"reference": "another-reference"}
    with pytest.raises(ValueError, match="Reference"):
        summarize_case(trace, [_record(10.0, 8.0)])


def test_incorrect_reversion_is_unattributable() -> None:
    record = _record(10.0, 8.0)
    record["candidates"][0]["correct"] = False
    record["candidates"][0]["median_ms"] = None
    result = summarize_case(_trace(), [record])
    assert result["actions"][0]["status"] == "invalid_reversion"
