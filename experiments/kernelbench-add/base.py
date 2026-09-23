import cupy as cp
import numpy as np
import torch
from torch import nn


SOURCE = r"""
extern "C" __global__ void add_kernel(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}
"""
KERNEL = cp.RawKernel(SOURCE, "add_kernel")


class ModelNew(nn.Module):
    def forward(self, a, b):
        out = torch.empty_like(a)
        threads = 128
        blocks = (a.numel() + threads - 1) // threads
        with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
            KERNEL((blocks,), (threads,), (cp.asarray(a), cp.asarray(b), cp.asarray(out), np.int32(a.numel())))
        return out
