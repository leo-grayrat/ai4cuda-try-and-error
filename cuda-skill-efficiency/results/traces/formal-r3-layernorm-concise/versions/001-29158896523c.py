import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_kernel(
    X,
    Y,
    N,
    stride,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    Xrow = X + row * stride
    Yrow = Y + row * stride

    n_blocks = tl.cdiv(N, BLOCK)

    total = 0.0
    total_sq = 0.0
    for i in range(0, n_blocks):
        offs = i * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        vals = tl.load(Xrow + offs, mask=mask, other=0.0)
        total += tl.sum(vals, axis=0)
        total_sq += tl.sum(vals * vals, axis=0)

    mean = total / N
    var = total_sq / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    for i in range(0, n_blocks):
        offs = i * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        vals = tl.load(Xrow + offs, mask=mask, other=0.0)
        out = (vals - mean) * rstd
        tl.store(Yrow + offs, out, mask=mask)


def run(x):
    x = x.contiguous()
    M, N = x.shape
    y = torch.empty_like(x)

    BLOCK = min(triton.next_power_of_2(N), 4096)
    num_warps = 4
    if BLOCK >= 2048:
        num_warps = 8

    _layer_norm_kernel[(M,)](
        x,
        y,
        N,
        x.stride(0),
        1e-5,
        BLOCK=BLOCK,
        num_warps=num_warps,
    )
    return y
