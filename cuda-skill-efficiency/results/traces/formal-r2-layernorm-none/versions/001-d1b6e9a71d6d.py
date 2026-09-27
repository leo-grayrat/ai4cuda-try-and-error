import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_kernel(X, Y, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    X += row * stride
    Y += row * stride

    mean = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0)
        mean += tl.sum(x, axis=0)
    mean = mean / N

    var = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0)
        d = tl.where(mask, x - mean, 0.0)
        var += tl.sum(d * d, axis=0)
    var = var / N
    rstd = 1.0 / tl.sqrt(var + eps)

    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0)
        y = (x - mean) * rstd
        tl.store(Y + cols, y, mask=mask)


def run(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    M, N = x.shape
    y = torch.empty_like(x)
    if M == 0 or N == 0:
        return y
    BLOCK = min(triton.next_power_of_2(N), 4096)
    _layer_norm_kernel[(M,)](x, y, x.stride(0), N, 1e-5, BLOCK=BLOCK)
    return y
