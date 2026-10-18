import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, *inputs):
        raise NotImplementedError('Implement CUDA extension')
