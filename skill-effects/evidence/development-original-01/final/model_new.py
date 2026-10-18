import torch
import torch.nn as nn
import cuda_extension

_axpby = cuda_extension.axpby


class ModelNew(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self._alpha = float(alpha)

    def forward(self, a, b):
        return _axpby(a, b, self._alpha)

    # Bypass nn.Module.__call__ hook machinery on the hot path.
    __call__ = forward
