"""Native evidence parser tests, using a real archived environment trace."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trace_formal import parse_native_csv, require_finished

ROOT = Path(__file__).resolve().parents[1]


def test_archived_actual_kernel_name_grid_and_comma_duration_are_retained():
    text = (ROOT / "evidence/environment-02/nsight-kernel-launch.log").read_text(encoding="utf-8")
    data = parse_native_csv(text)
    assert data["positive_activity_observed"]
    assert data["launch_count"] == 1
    assert data["kernel_duration_sum_us"] == pytest.approx(2.624)
    assert "axpby_kernel<256>" in data["launches"][0]["kernel_name"]
    assert data["launches"][0]["Grid Size"] == "(1, 1, 1)"
    assert data["launches"][0]["Stream"] == "7"


@pytest.mark.parametrize("text", [
    "==ERROR== ERR_NVGPUCTRPERM\n",
    '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n"0","f","gpu__time_duration.sum","ns","0"\n',
    '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n"0","f","gpu__time_duration.sum","ns","nan"\n',
    '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n"0","f","gpu__time_duration.sum","cycles","2"\n',
])
def test_missing_or_invalid_measurements_never_become_zero_work_or_success(text):
    with pytest.raises(ValueError):
        parse_native_csv(text)


def test_duplicate_metric_rows_are_not_counted_as_extra_launches():
    text = (ROOT / "evidence/environment-02/nsight-kernel-launch.log").read_text(encoding="utf-8")
    text += text.splitlines()[-1] + "\n"
    assert parse_native_csv(text)["launch_count"] == 1


def test_profiling_refuses_an_active_campaign_even_with_stale_complete_collection():
    collection = {"scheduled": 88, "terminal": 88}
    with pytest.raises(ValueError, match="active/incomplete"):
        require_finished({"status": "running", "trials": [None] * 88}, collection)
    with pytest.raises(ValueError, match="Refresh"):
        require_finished({"status": "complete", "trials": [None] * 88}, {"scheduled": 88, "terminal": 87})
    require_finished({"status": "complete", "trials": [None] * 88}, collection)
