import torch
import triton
import triton.language as tl


@triton.jit
def _ln_row_kernel(X, Y, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    X += row * stride
    Y += row * stride
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(Y + offs, d * rstd, mask=mask)


@triton.jit
def _ln_loop_kernel(X, Y, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    X += row * stride
    Y += row * stride
    offs = tl.arange(0, BLOCK)
    mean = 0.0
    for start in range(0, N, BLOCK):
        idx = start + offs
        x = tl.load(X + idx, mask=idx < N, other=0.0)
        mean += tl.sum(x, axis=0)
    mean = mean / N
    var = 0.0
    for start in range(0, N, BLOCK):
        idx = start + offs
        mask = idx < N
        x = tl.load(X + idx, mask=mask, other=0.0)
        d = tl.where(mask, x - mean, 0.0)
        var += tl.sum(d * d, axis=0)
    var = var / N
    rstd = 1.0 / tl.sqrt(var + eps)
    for start in range(0, N, BLOCK):
        idx = start + offs
        mask = idx < N
        x = tl.load(X + idx, mask=mask, other=0.0)
        tl.store(Y + idx, (x - mean) * rstd, mask=mask)


def run(x):
    x = x.contiguous()
    M, N = x.shape
    y = torch.empty_like(x)
    if x.numel() == 0:
        return y
    if N <= 4096:
        BLOCK = triton.next_power_of_2(N)
        _ln_row_kernel[(M,)](x, y, x.stride(0), N, 1e-5,
                             BLOCK=BLOCK, num_warps=max(1, BLOCK // 128))
    else:
        BLOCK = 4096
        _ln_loop_kernel[(M,)](x, y, x.stride(0), N, 1e-5,
                              BLOCK=BLOCK, num_warps=8)
    return y
