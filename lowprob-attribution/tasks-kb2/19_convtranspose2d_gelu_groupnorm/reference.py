"""Reference for 19_ConvTranspose2d_GELU_GroupNorm.

Auto-generated from KernelBench level 2. The model is instantiated once with the
declared init inputs; `run` is the single-tensor entry point the driver uses.
"""
import torch
import torch.nn as nn


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a transposed convolution, applies GELU, and normalizes with GroupNorm.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = torch.nn.functional.gelu(x)
        x = self.group_norm(x)
        return x

batch_size   = 128  
in_channels  = 64  
out_channels = 64  
height = width = 256  
kernel_size  = 3
stride       = 1
groups = 8
num_groups = 8

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, groups, num_groups]


torch.manual_seed(0)

_INIT = get_init_inputs()
_MODEL = Model(*_INIT).cuda().eval()
MODEL = _MODEL          # exported so a candidate can read the same weights


def run(x):
    with torch.no_grad():
        return _MODEL(x)
