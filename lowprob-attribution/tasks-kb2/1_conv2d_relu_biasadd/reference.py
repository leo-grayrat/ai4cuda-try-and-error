"""Reference for 1_Conv2D_ReLU_BiasAdd.

Auto-generated from KernelBench level 2. The model is instantiated once with the
declared init inputs; `run` is the single-tensor entry point the driver uses.
"""
import torch
import torch.nn as nn


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a convolution, applies ReLU, and adds a bias term.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        x = x + self.bias
        return x

batch_size = 128
in_channels  = 64  
out_channels = 128  
height = width = 128
kernel_size = 3
bias_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]


torch.manual_seed(0)

_INIT = get_init_inputs()
_MODEL = Model(*_INIT).cuda().eval()
MODEL = _MODEL          # exported so a candidate can read the same weights


def run(x):
    with torch.no_grad():
        return _MODEL(x)
