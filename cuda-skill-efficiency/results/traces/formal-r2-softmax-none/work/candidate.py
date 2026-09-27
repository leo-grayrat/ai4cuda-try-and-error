import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(x_ptr, out_ptr, n_cols, stride_row, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * stride_row
    m = float("-inf")
    s = 0.0
    for start in range(0, n_cols, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < n_cols
        x = tl.load(x_ptr + base + offs, mask=mask, other=float("-inf"))
        mb = tl.max(x, axis=0)
        m_new = tl.maximum(m, mb)
        s = s * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new
    for start in range(0, n_cols, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < n_cols
        x = tl.load(x_ptr + base + offs, mask=mask, other=float("-inf"))
        tl.store(out_ptr + base + offs, tl.exp(x - m) / s, mask=mask)


def run(x):
    if not x.is_cuda:
        raise ValueError("run(x) expects a CUDA tensor")
    x = x.contiguous()
    if x.dtype != torch.float32:
        x = x.float()
    n_rows = x.shape[0]
    n_cols = x.numel() // n_rows
    out = torch.empty_like(x)
    if n_rows == 0 or n_cols == 0:
        return out
    block = min(2048, max(16, triton.next_power_of_2(n_cols)))
    _softmax_kernel[(n_rows,)](x, out, n_cols, x.stride(0), BLOCK=block)
    return out
