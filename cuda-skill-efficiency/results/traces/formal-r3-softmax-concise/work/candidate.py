import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    x_ptr,
    out_ptr,
    n_cols,
    stride_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    base = x_ptr + row * stride_row
    out_base = out_ptr + row * stride_row

    # Pass 1: row maximum (for numerical stability with large-magnitude inputs).
    m = -float("inf")
    for off in range(0, n_cols, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(base + cols, mask=mask, other=-float("inf"))
        m = tl.maximum(m, tl.max(x, axis=0))

    # Pass 2: sum of exponentials.
    den = 0.0
    for off in range(0, n_cols, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(base + cols, mask=mask, other=-float("inf"))
        den += tl.sum(tl.exp(x - m), axis=0)

    # Pass 3: normalize and store.
    inv_den = 1.0 / den
    for off in range(0, n_cols, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(base + cols, mask=mask, other=0.0)
        y = tl.exp(x - m) * inv_den
        tl.store(out_base + cols, y, mask=mask)


def _block_for(n_cols):
    return min(triton.next_power_of_2(max(n_cols, 1)), 8192)


def run(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    if x.ndim < 2:
        x2 = x.reshape(1, -1)
        return run(x2).reshape(x.shape)

    if x.ndim == 2:
        x2 = x
        out2 = torch.empty_like(x2)
        n_rows, n_cols = x2.shape
        if n_rows == 0:
            return out2
        _softmax_kernel[(n_rows,)](
            x2,
            out2,
            n_cols,
            x2.stride(0),
            BLOCK=_block_for(n_cols),
        )
        return out2

    # Generic fallback for >2D: move dim 1 to the last axis, flatten rows.
    x2 = x.movedim(1, -1).contiguous().reshape(-1, x.shape[1])
    n_rows, n_cols = x2.shape
    out2 = torch.empty_like(x2)
    if n_rows:
        _softmax_kernel[(n_rows,)](
            x2,
            out2,
            n_cols,
            x2.stride(0),
            BLOCK=_block_for(n_cols),
        )
    out = out2.reshape(x.shape[0], *x.shape[2:], x.shape[1]).movedim(-1, 1)
    return out.contiguous()
