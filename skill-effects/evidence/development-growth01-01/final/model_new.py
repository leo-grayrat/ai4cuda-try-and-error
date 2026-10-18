import torch
import torch.nn as nn
import cuda_extension

_axpy = cuda_extension.axpy_forward


class ModelNew(nn.Module):
    """alpha * a + b via a single fused, vectorized CUDA kernel."""

    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, a, b):
        return _axpy(a, b, self.alpha, 0)
