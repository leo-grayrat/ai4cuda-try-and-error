import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_kernel(X, Y, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_ptr = X + row * N
    y_ptr = Y + row * N

    mean = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x, axis=0)
    mean = mean / N

    var = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        d = x - mean
        var += tl.sum(d * d, axis=0)
    var = var / N
    rstd = 1.0 / tl.sqrt(var + eps)

    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd
        tl.store(y_ptr + cols, y, mask=mask)


def run(x):
    if not x.is_contiguous():
        x = x.contiguous()
    n = x.shape[1]
    y = torch.empty_like(x)
    m = x.numel() // n
    if n == 0 or m == 0:
        return y
    block = min(1024, triton.next_power_of_2(n))
    _layernorm_kernel[(m,)](x, y, n, 1e-5, BLOCK=block)
    return y
