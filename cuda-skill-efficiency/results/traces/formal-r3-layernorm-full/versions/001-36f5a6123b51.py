import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_kernel(
    X, Y, N, eps, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * N
    Y += row * N

    # Pass 1: row sum (masked lanes contribute 0).
    _sum = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        _sum += x
    mean = tl.sum(_sum, axis=0) / N

    # Pass 2: sum of squared deviations (masked lanes contribute 0).
    _var = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        d = tl.where(mask, x - mean, 0.0)
        _var += d * d
    var = tl.sum(_var, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # Pass 3: normalize and store.
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        d = tl.where(mask, x - mean, 0.0)
        tl.store(Y + cols, (d * rstd).to(Y.dtype.element_ty), mask=mask)


def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32
    x = x.contiguous()
    N = x.shape[1] if x.dim() > 1 else x.shape[0]
    M = x.numel() // N if N > 0 else 0

    y = torch.empty_like(x)
    if M == 0 or N == 0:
        return y

    BLOCK = min(4096, triton.next_power_of_2(N))
    grid = (M,)
    _layer_norm_kernel[grid](
        x, y, N, 1e-5, BLOCK=BLOCK, num_warps=8,
    )
    return y
