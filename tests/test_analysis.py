import math

import pytest

from kernel_lab.core import compare_hardware, credit_analysis, credit_series, source_hash


def sample_record(uuid, times):
    return {
        "experiment_id": "one-problem",
        "workload_hash": "fixed-workload",
        "hardware": {"uuid": uuid, "name": uuid},
        "factors": ["A", "B"],
        "candidates": [
            {"id": key, "changes": changes, "correct": True, "median_ms": times[key], "code_hash": f"hash-{key}"}
            for key, changes in [("base", []), ("a", ["A"]), ("b", ["B"]), ("ab", ["A", "B"])]
        ],
    }


def test_credit_uses_measured_counterfactuals():
    result = credit_analysis(sample_record("gpu-a", {"base": 10, "a": 8, "b": 9, "ab": 6}))
    assert result["speedup_vs_base"]["AB"] == pytest.approx(10 / 6)
    assert result["log_contribution"]["A"] == pytest.approx(math.log(10 / 8))
    assert result["log_contribution"]["interaction"] == pytest.approx(math.log(8 * 9 / (6 * 10)))
    assert sum(result["shapley_log_credit"].values()) == pytest.approx(math.log(10 / 6))
    assert result["shapley_log_credit"]["A"] == pytest.approx(
        (math.log(10 / 8) + math.log(9 / 6)) / 2
    )


def test_hardware_comparison_counts_rank_flips():
    a = sample_record("gpu-a", {"base": 10, "a": 8, "b": 9, "ab": 6})
    b = sample_record("gpu-b", {"base": 10, "a": 5, "b": 9, "ab": 6})
    result = compare_hardware(a, b)
    assert result["pairwise_flips"] == 1
    assert result["comparable_pairs"] == 6
    assert result["gpu_a_winner_rank_on_b"] == 2


def test_cross_hardware_rejects_same_device_or_changed_code():
    a = sample_record("gpu-a", {"base": 10, "a": 8, "b": 9, "ab": 6})
    with pytest.raises(ValueError, match="same GPU"):
        compare_hardware(a, a)
    b = sample_record("gpu-b", {"base": 10, "a": 5, "b": 9, "ab": 6})
    b["candidates"][0]["code_hash"] = "different"
    with pytest.raises(ValueError, match="different code"):
        compare_hardware(a, b)


def test_hardware_comparison_handles_tied_timings():
    a = sample_record("gpu-a", {"base": 10, "a": 8, "b": 8, "ab": 6})
    b = sample_record("gpu-b", {"base": 10, "a": 8, "b": 8, "ab": 6})
    result = compare_hardware(a, b)
    assert result["rank_correlation_spearman"] == pytest.approx(1)
    assert result["ranks_a"]["a"] == result["ranks_a"]["b"] == 2.5
    assert result["comparable_pairs"] == 5


def test_credit_rejects_incorrect_candidate():
    record = sample_record("gpu-a", {"base": 10, "a": 8, "b": 9, "ab": 6})
    record["candidates"][3]["correct"] = False
    with pytest.raises(ValueError, match="incorrect"):
        credit_analysis(record)


def test_credit_series_requires_same_code_and_device():
    a = sample_record("gpu-a", {"base": 10, "a": 8, "b": 9, "ab": 6})
    b = sample_record("gpu-a", {"base": 10, "a": 9, "b": 8, "ab": 7})
    result = credit_series([a, b])
    assert result["run_count"] == 2
    assert result["summary"]["A"]["sample_std"] > 0
    assert result["shapley_summary"]["A"]["sample_std"] > 0
    b["hardware"]["uuid"] = "gpu-b"
    with pytest.raises(ValueError, match="same GPU"):
        credit_series([a, b])


def test_credit_series_rejects_relabelled_interventions():
    a = sample_record("gpu-a", {"base": 10, "a": 8, "b": 9, "ab": 6})
    b = sample_record("gpu-a", {"base": 10, "a": 8, "b": 9, "ab": 6})
    b["candidates"][1]["changes"] = ["B"]
    with pytest.raises(ValueError, match="different intervention factors"):
        credit_series([a, b])


def test_source_hash_ignores_only_line_endings():
    assert source_hash(b"a\r\nb\r\n") == source_hash(b"a\nb\n")
    assert source_hash(b"a\r\nb\r\n") != source_hash(b"a\nc\n")
