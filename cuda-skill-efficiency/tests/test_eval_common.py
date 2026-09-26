import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_common import _paired_graph_benchmark, evaluate


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_reference_copy_is_not_a_valid_custom_kernel(tmp_path):
    candidate = tmp_path / "candidate.py"
    candidate.write_text(
        "def run(x):\n    return x * 1.75 + 0.25\n", encoding="utf-8",
    )
    def reference(x):
        return x * 1.75 + 0.25
    result = evaluate(
        candidate, reference,
        lambda shape: torch.randn(shape, device="cuda", dtype=torch.float32),
        [(16,), (1024,)],
    )
    assert result["correct"] is False
    assert "Triton" in result["error"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_graph_benchmark_records_interleaved_pairs():
    value = torch.randn((1024,), device="cuda")
    result = _paired_graph_benchmark(lambda x: x + 1, lambda x: x * 2, value, repeats=8, pairs=6)
    assert len(result["reference_samples_ms"]) == 6
    assert len(result["candidate_samples_ms"]) == 6
    assert set(result["pair_order"]) == {"reference_first", "candidate_first"}
    assert min(result["reference_samples_ms"] + result["candidate_samples_ms"]) > 0
