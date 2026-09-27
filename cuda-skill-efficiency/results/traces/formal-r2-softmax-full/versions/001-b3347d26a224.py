import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    x_ptr,
    y_ptr,
    n_rows,
    n_cols,
    stride_row,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < n_rows
    offs_n = tl.arange(0, BLOCK_N)

    m = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    s = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for start in range(0, n_cols, BLOCK_N):
        cols = start + offs_n
        mask = row_mask[:, None] & (cols < n_cols)[None, :]
        v = tl.load(x_ptr + rows[:, None] * stride_row + cols[None, :],
                    mask=mask, other=float("-inf"))
        tile_max = tl.max(v, axis=1)
        m_new = tl.maximum(m, tile_max)
        s = s * tl.exp(m - m_new) + tl.sum(tl.exp(v - m_new[:, None]), axis=1)
        m = m_new

    inv_s = 1.0 / s
    for start in range(0, n_cols, BLOCK_N):
        cols = start + offs_n
        mask = row_mask[:, None] & (cols < n_cols)[None, :]
        v = tl.load(x_ptr + rows[:, None] * stride_row + cols[None, :],
                    mask=mask, other=float("-inf"))
        p = tl.exp(v - m[:, None]) * inv_s[:, None]
        tl.store(y_ptr + rows[:, None] * stride_row + cols[None, :], p, mask=mask)


def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def run(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 2:
        raise ValueError("expected a 2D matrix")
    x = x.contiguous()
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    if n_rows == 0 or n_cols == 0:
        return y

    BLOCK_N = min(_next_pow2(n_cols), 4096)
    BLOCK_M = max(1, min(8, 16384 // BLOCK_N))
    num_warps = max(1, min(8, (BLOCK_M * BLOCK_N) // 256))

    grid = (triton.cdiv(n_rows, BLOCK_M),)
    _softmax_kernel[grid](
        x, y, n_rows, n_cols, x.stride(0),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )
    return y
