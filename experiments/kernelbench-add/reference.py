"""A local KernelBench-format vector addition problem for an integration pilot."""

import torch
from torch import nn


class Model(nn.Module):
    def forward(self, a, b):
        return a + b


def get_inputs():
    return [torch.randn(1 << 22), torch.randn(1 << 22)]


def get_init_inputs():
    return []
