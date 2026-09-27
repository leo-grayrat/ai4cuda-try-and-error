import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_kernel(X, Y, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(Y + row * stride + cols, xc * rstd, mask=mask)


def run(x):
    x = x.contiguous()
    M, N = x.shape
    y = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(N)
    num_warps = max(1, min(16, BLOCK // 64))
    _layer_norm_kernel[(M,)](x, y, x.stride(0), N, 1e-5, BLOCK=BLOCK, num_warps=num_warps)
    return y
