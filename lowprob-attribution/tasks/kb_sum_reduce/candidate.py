import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs sum reduction over a specified dimension.
    Implement the forward pass using a custom CUDA kernel.
    """
    def __init__(self, dim: int):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x, dim=self.dim, keepdim=True)

# Scaled down from the KernelBench original (128, 4096, 4095 = 8.59 GB fp32),
# which does not fit an 8 GB laptop GPU. Reduction width and inner dim kept.
batch_size = 16
dim1 = 4096
dim2 = 4095
reduce_dim = 1

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return [reduce_dim]
