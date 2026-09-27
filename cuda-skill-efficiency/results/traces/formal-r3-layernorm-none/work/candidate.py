import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_kernel(X, Y, stride_x, stride_y, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    offs = row * stride_x + cols
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(Y + row * stride_y + cols, d * rstd, mask=mask)


def run(x):
    x = x.contiguous()
    M, N = x.shape
    y = torch.empty_like(x)
    BLOCK = max(triton.next_power_of_2(N), 16)
    _layernorm_kernel[(M,)](x, y, x.stride(0), y.stride(0), N, 1e-5, BLOCK=BLOCK)
    return y
