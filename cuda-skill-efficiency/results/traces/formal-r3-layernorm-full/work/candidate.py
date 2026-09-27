import torch
import triton
import triton.language as tl


@triton.jit
def _ln_single(X, Y, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    X += row * N
    Y += row * N

    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    xm = tl.where(mask, x, 0.0)
    mean = tl.sum(xm, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(Y + cols, ((x - mean) * rstd).to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _ln_loop(X, Y, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    X += row * N
    Y += row * N

    _sum = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        _sum += tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(_sum, axis=0) / N

    _var = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        d = tl.where(mask, tl.load(X + cols, mask=mask, other=0.0) - mean, 0.0)
        _var += d * d
    rstd = 1.0 / tl.sqrt(tl.sum(_var, axis=0) / N + eps)

    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        d = tl.where(mask, tl.load(X + cols, mask=mask, other=0.0) - mean, 0.0)
        tl.store(Y + cols, (d * rstd).to(Y.dtype.element_ty), mask=mask)


_MAX_SINGLE = 8192


def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32
    x = x.contiguous()
    N = x.shape[-1]
    M = x.numel() // N if N > 0 else 0

    y = torch.empty_like(x)
    if M == 0 or N == 0:
        return y

    if N <= _MAX_SINGLE:
        BLOCK = triton.next_power_of_2(N)
        num_warps = max(1, min(16, BLOCK // 128))
        _ln_single[(M,)](x, y, N, 1e-5, BLOCK=BLOCK, num_warps=num_warps)
    else:
        BLOCK = 4096
        _ln_loop[(M,)](x, y, N, 1e-5, BLOCK=BLOCK, num_warps=8)
    return y
