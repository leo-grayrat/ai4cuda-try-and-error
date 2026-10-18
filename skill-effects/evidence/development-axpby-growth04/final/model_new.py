import torch.nn as nn

import cuda_extension


class ModelNew(nn.Module):
    """AXPBY: out = alpha * a + b, computed by a custom CUDA kernel."""

    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha
        # Bind the extension entry point once so the hot path stays lean.
        self._axpby = cuda_extension.axpby

    def forward(self, a, b):
        return self._axpby(a, b, self.alpha)
