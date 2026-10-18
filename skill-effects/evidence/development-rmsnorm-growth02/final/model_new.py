import torch
import torch.nn as nn

import cuda_extension

_forward = cuda_extension.rmsnorm_forward


class ModelNew(nn.Module):
    """RMSNorm over dim=1, computed by a single fused CUDA kernel."""

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _forward(x, self.eps)
