import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(x_ptr, y_ptr, n_cols, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * stride
    cols = tl.arange(0, BLOCK)

    m = -float("inf")
    d = 0.0
    for off in range(0, n_cols, BLOCK):
        idx = off + cols
        x = tl.load(x_ptr + base + idx, mask=idx < n_cols, other=-float("inf"))
        m_new = tl.maximum(m, tl.max(x, axis=0))
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new

    for off in range(0, n_cols, BLOCK):
        idx = off + cols
        x = tl.load(x_ptr + base + idx, mask=idx < n_cols, other=-float("inf"))
        y = tl.exp(x - m) / d
        tl.store(y_ptr + base + idx, y, mask=idx < n_cols)


def run(x):
    x = x.contiguous()
    y = torch.empty_like(x)
    n_cols = x.shape[1]
    if n_cols == 0 or x.numel() == 0:
        return y
    rows = x.numel() // n_cols
    BLOCK = min(triton.next_power_of_2(n_cols), 4096)
    num_warps = 4 if BLOCK <= 1024 else 8
    _softmax_kernel[(rows,)](x, y, n_cols, x.stride(0), BLOCK=BLOCK, num_warps=num_warps)
    return y
