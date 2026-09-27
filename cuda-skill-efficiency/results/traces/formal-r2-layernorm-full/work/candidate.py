import torch
import triton
import triton.language as tl

_EPS = 1e-5
_MAX_BLOCK = 4096


@triton.jit
def _ln_single(X, Y, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * N
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + base + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(Y + base + cols, (x - mean) * rstd, mask=mask)


@triton.jit
def _ln_loop(X, Y, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * N
    mean = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + base + cols, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x, axis=0)
    mean = mean / N
    var = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + base + cols, mask=mask, other=0.0).to(tl.float32)
        d = tl.where(mask, x - mean, 0.0)
        var += tl.sum(d * d, axis=0)
    rstd = 1.0 / tl.sqrt(var / N + eps)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + base + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(Y + base + cols, (x - mean) * rstd, mask=mask)


def run(x):
    if not x.is_contiguous():
        x = x.contiguous()
    n = x.shape[1]
    m = x.numel() // n
    y = torch.empty_like(x)
    if n == 0 or m == 0:
        return y
    block = triton.next_power_of_2(n)
    if block <= _MAX_BLOCK:
        _ln_single[(m,)](x, y, n, _EPS, BLOCK=block)
    else:
        _ln_loop[(m,)](x, y, n, _EPS, BLOCK=2048)
    return y
