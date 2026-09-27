import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_kernel(
    X, Y, stride, N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride
    Y += row * stride
    offs = tl.arange(0, BLOCK)

    mean = 0.0
    for start in range(0, N, BLOCK):
        idx = start + offs
        mask = idx < N
        x = tl.load(X + idx, mask=mask, other=0.0)
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
    BLOCK = min(4096, triton.next_power_of_2(N))
    num_warps = 4 if BLOCK < 2048 else 8
    _layer_norm_kernel[(M,)](x, y, x.stride(0), N, 1e-5, BLOCK=BLOCK, num_warps=num_warps)
    return y
