import cupy as cp
import numpy as np
import torch
from torch import nn


SOURCE = r"""
extern "C" __global__ void add_kernel(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    if (i < n) out[i] = a[i] + b[i];
    int j = i + stride;
    if (j < n) out[j] = a[j] + b[j];
}
"""
KERNEL = cp.RawKernel(SOURCE, "add_kernel")


class ModelNew(nn.Module):
    def forward(self, a, b):
        out = torch.empty_like(a)
        threads = 256
        blocks = (a.numel() + 2 * threads - 1) // (2 * threads)
        with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
            KERNEL((blocks,), (threads,), (cp.asarray(a), cp.asarray(b), cp.asarray(out), np.int32(a.numel())))
        return out
