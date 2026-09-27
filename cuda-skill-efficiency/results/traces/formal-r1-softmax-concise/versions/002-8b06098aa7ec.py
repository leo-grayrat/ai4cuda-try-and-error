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
        e = tl.where(mask, tl.exp(v - row_max), 0.0)
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
    if x.dtype != torch.float32:
        x = x.float()

    x = x.contiguous()
    move = x.dim() > 2
    if move:
        x = x.movedim(1, -1).contiguous()

    n = x.shape[-1]
    flat = x.reshape(-1, n)
    out = torch.empty_like(flat)

    rows = flat.shape[0]
    if rows > 0 and n > 0:
        _softmax_kernel[(rows,)](flat, out, n, flat.stride(0), BLOCK=_BLOCK, num_warps=4)

    out = out.reshape(x.shape)
    if move:
        out = out.movedim(-1, 1).contiguous()
    return out
