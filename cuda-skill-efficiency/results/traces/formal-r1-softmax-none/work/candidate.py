import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    x_ptr,
    y_ptr,
    row_stride,
    n_cols,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x_ptr += row * row_stride + cols
    y_ptr += row * row_stride + cols

    x = tl.load(x_ptr, mask=mask, other=float("-inf")).to(tl.float32)

    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(y_ptr, y.to(tl.float32), mask=mask)


def run(x):
    x = x.contiguous()
    n_rows, n_cols = x.shape
    out = torch.empty_like(x, dtype=torch.float32)

    BLOCK = triton.next_power_of_2(n_cols)
    BLOCK = max(16, BLOCK)

    _softmax_kernel[(n_rows,)](
        x,
        out,
        n_cols,
        n_cols,
        BLOCK=BLOCK,
        num_warps=4,
    )
    return out
