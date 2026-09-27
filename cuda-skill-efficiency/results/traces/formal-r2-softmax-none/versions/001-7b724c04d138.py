import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(x_ptr, out_ptr, n_cols, stride_row, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    base = row * stride_row
    x = tl.load(x_ptr + base + offs, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    e = e / tl.sum(e, axis=0)
    tl.store(out_ptr + base + offs, e, mask=mask)


def run(x):
    if not x.is_cuda:
        raise ValueError("run(x) expects a CUDA tensor")
    x = x.contiguous()
    if x.dtype != torch.float32:
        x = x.float()
    n_rows = x.shape[0]
    n_cols = x.numel() // n_rows
    out = torch.empty_like(x)
    block = max(16, triton.next_power_of_2(n_cols))
    _softmax_kernel[(n_rows,)](x, out, n_cols, x.stride(0), BLOCK=block)
    return out
