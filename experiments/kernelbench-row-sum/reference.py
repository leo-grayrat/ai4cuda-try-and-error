"""KernelBench-format row reduction; shape stays fixed within this pilot."""

import torch
from torch import nn


class Model(nn.Module):
    def forward(self, x):
        return x.sum(dim=1)


def get_inputs():
    return [torch.randn(4096, 1024)]


def get_init_inputs():
    return []
