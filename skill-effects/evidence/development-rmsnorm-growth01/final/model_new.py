import torch.nn as nn

import cuda_extension

_op = cuda_extension.rmsnorm_forward
_CONFIG = 11  # single pass, 512 threads


class ModelNew(nn.Module):
    """RMSNorm over dim=1 implemented with a custom fused CUDA kernel."""

    def __init__(self, num_features: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x):
        return _op(x, self.eps, _CONFIG)
