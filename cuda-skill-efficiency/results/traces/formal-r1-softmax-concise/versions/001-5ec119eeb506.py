import torch
import triton
import triton.language as tl

_BLOCK = 1024


@triton.jit
def _softmax_kernel(
    x_ptr,
    y_ptr,
    n_cols,
    stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr += row * stride
    y_ptr += row * stride

    row_max = -float("inf")
    for off in range(0, n_cols, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < n_cols
        v = tl.load(x_ptr + cols, mask=mask, other=-float("inf"))
        row_max = tl.maximum(row_max, tl.max(v, axis=0))

    row_sum = 0.0
    for off in range(0, n_cols, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < n_cols
        v = tl.load(x_ptr + cols, mask=mask, other=-float("inf"))
        e = tl.exp(v - row_max)
        e = tl.where(mask, e, 0.0)
        row_sum += tl.sum(e, axis=0)
        tl.store(y_ptr + cols, e, mask=mask)

    for off in range(0, n_cols, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < n_cols
        e = tl.load(y_ptr + cols, mask=mask, other=0.0)
        tl.store(y_ptr + cols, e / row_sum, mask=mask)


def run(x):
    if not x.is_cuda:
        raise RuntimeError("input must be a CUDA tensor")
    if x.dim() < 2:
        raise RuntimeError("input must have at least 2 dimensions")

    x = x.contiguous()
    m = x.shape[0]
    n = x.shape[1]
    if x.dim() > 2:
        x = x.reshape(m, n, -1)
        rows = m * (x.shape[2] if x.dim() == 3 else 1)
        out = torch.empty_like(x)
        nc = x.shape[1]
        flat_x = x.reshape(-1, nc)
        flat_out = out.reshape(-1, nc)
        r = flat_x.shape[0]
        if nc > 0 and r > 0:
            _launch(flat_x, flat_out, r, nc)
        return out

    out = torch.empty_like(x)
    if m > 0 and n > 0:
        _launch(x, out, m, n)
    return out


def _launch(x, out, rows, n):
    grid = (rows,)
    _softmax_kernel[grid](x, out, n, x.stride(0), BLOCK=_BLOCK, num_warps=4)
