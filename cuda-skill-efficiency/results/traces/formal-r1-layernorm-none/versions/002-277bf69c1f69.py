import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_kernel(X, Y, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    Xr = X + row * stride
    Yr = Y + row * stride
    offs = tl.arange(0, BLOCK)

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in tl.range(0, N, BLOCK):
        c = start + offs
        m = c < N
        acc += tl.load(Xr + c, mask=m, other=0.0)
    mean = tl.sum(acc, axis=0) / N

    var_acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in tl.range(0, N, BLOCK):
        c = start + offs
        m = c < N
        v = tl.load(Xr + c, mask=m, other=0.0) - mean
        v = tl.where(m, v, 0.0)
        var_acc += v * v
    var = tl.sum(var_acc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    for start in tl.range(0, N, BLOCK):
        c = start + offs
        m = c < N
        v = tl.load(Xr + c, mask=m, other=0.0) - mean
        tl.store(Yr + c, v * rstd, mask=m)


def run(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    M, N = x.shape
    y = torch.empty_like(x)
    if x.numel() == 0:
        return y
    BLOCK = min(triton.next_power_of_2(N), 4096)
    _layer_norm_kernel[(M,)](x, y, x.stride(0), N, 1e-5, BLOCK=BLOCK)
    return y
