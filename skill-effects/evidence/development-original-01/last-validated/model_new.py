import torch
import torch.nn as nn
import cuda_extension

_axpby = cuda_extension.axpby


class ModelNew(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self._alpha_f = float(alpha)

    def forward(self, a, b):
        return _axpby(a, b, self._alpha_f)
