import cupy as cp
import numpy as np
import torch
from torch import nn


SOURCE = r"""
extern "C" __global__ void row_sum(const float* x, float* out, int columns) {
    __shared__ float values[256];
    int row = blockIdx.x;
    int lane = threadIdx.x;
    float sum = 0.0f;
    for (int col = lane; col < columns; col += 256) {
        sum += x[row * columns + col];
    }
    values[lane] = sum;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) values[lane] += values[lane + stride];
        __syncthreads();
    }
    if (lane == 0) out[row] = values[0];
}
"""
KERNEL = cp.RawKernel(SOURCE, "row_sum")


class ModelNew(nn.Module):
    def forward(self, x):
        out = torch.empty(x.shape[0], device=x.device, dtype=x.dtype)
        with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
            KERNEL((x.shape[0],), (256,), (cp.asarray(x), cp.asarray(out), np.int32(x.shape[1])))
        return out
