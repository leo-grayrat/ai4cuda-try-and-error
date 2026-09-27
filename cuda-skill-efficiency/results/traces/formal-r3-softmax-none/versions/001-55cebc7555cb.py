import torch
import triton
import triton.language as tl


_MAX_DENSE_BLOCK = 8192


@triton.jit
def _softmax_dense_kernel(
    x_ptr,
    y_ptr,
    stride_x,
    stride_y,
    n_cols,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    off = row * stride_x + cols
    x = tl.load(x_ptr + off, mask=mask, other=-float("inf"))
    m = tl.max(x, axis=0)
    num = tl.exp(x - m)
    den = tl.sum(num, axis=0)
    y = num / den
    tl.store(y_ptr + row * stride_y + cols, y, mask=mask)


@triton.jit
def _softmax_loop_kernel(
    x_ptr,
    y_ptr,
    stride_x,
    stride_y,
    n_cols,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row * stride_x
    y_row = y_ptr + row * stride_y

    m = -float("inf")
    for off in tl.range(0, n_cols, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(x_row + cols, mask=mask, other=-float("inf"))
        m = tl.maximum(m, tl.max(x, axis=0))

    s = 0.0
    for off in tl.range(0, n_cols, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(x_row + cols, mask=mask, other=0.0)
        s += tl.sum(tl.exp(x - m), axis=0)

    r = 1.0 / s
    for off in tl.range(0, n_cols, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(x_row + cols, mask=mask, other=0.0)
        tl.store(y_row + cols, tl.exp(x - m) * r, mask=mask)


def _num_warps(block: int) -> int:
    if block <= 1024:
        return 4
    if block <= 2048:
        return 8
    if block <= 4096:
        return 16
    return 32


def run(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    if x.dtype != torch.float32:
        raise TypeError("expected float32 input")
    if x.dim() != 2:
        raise ValueError("expected a 2D matrix")

    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    if n_rows == 0 or n_cols == 0:
        return y

    if n_cols <= _MAX_DENSE_BLOCK:
        block = max(16, triton.next_power_of_2(n_cols))
        _softmax_dense_kernel[(n_rows,)](
            x,
            y,
            x.stride(0),
            y.stride(0),
            n_cols,
            BLOCK=block,
            num_warps=_num_warps(block),
        )
    else:
        block = 1024
        _softmax_loop_kernel[(n_rows,)](
            x,
            y,
            x.stride(0),
            y.stride(0),
            n_cols,
            BLOCK=block,
            num_warps=4,
        )
    return y
