import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_compute_guard_allows_allocation_and_metadata():
    from verify_candidate import NativeComputeOnly
    x = torch.ones(7)
    with NativeComputeOnly():
        result = torch.empty_like(x)
        assert result.shape == x.shape
        assert result.dtype == x.dtype


def test_compute_guard_detects_pytorch_tensor_computation():
    from verify_candidate import NativeComputeOnly
    x = torch.ones(7)
    with pytest.raises(RuntimeError, match="PyTorch computation"):
        with NativeComputeOnly():
            x + 1


def test_compute_guard_detects_nn_module_computation():
    from verify_candidate import NativeComputeOnly
    model = torch.nn.LayerNorm(7)
    x = torch.ones(2, 7)
    with pytest.raises(RuntimeError, match="PyTorch computation"):
        with NativeComputeOnly():
            model(x)


@pytest.mark.parametrize("fake", ["zeros", "sparse"])
def test_wide_softmax_cannot_pass_by_exploiting_elementwise_tolerance(fake):
    from verify_candidate import compare_output
    expected = torch.full((2, 10000), 1e-4)
    actual = torch.zeros_like(expected)
    if fake == "sparse":
        actual[:, :100] = 0.01
    # The original absolute tolerance accepts these incorrect distributions.
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
    with pytest.raises(AssertionError):
        compare_output(actual, expected, "softmax")


def test_wide_softmax_distribution_check_accepts_genuine_probabilities():
    from verify_candidate import compare_output
    expected = torch.softmax(torch.linspace(-3, 3, 10000).repeat(2, 1), dim=1)
    compare_output(expected.clone(), expected, "softmax")
