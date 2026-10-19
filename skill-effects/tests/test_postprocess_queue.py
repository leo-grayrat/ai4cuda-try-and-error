"""A stopped or active experiment never becomes a completed dataset."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from postprocess_when_ready import ready


def test_waits_for_actual_completion_and_preserves_infrastructure_stop():
    assert not ready({"status": "running"})
    assert ready({"status": "complete"})
    with pytest.raises(RuntimeError, match="do not restart"):
        ready({"status": "infrastructure_error"})
