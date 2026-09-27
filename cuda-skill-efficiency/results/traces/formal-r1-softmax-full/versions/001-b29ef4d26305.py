import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    x_ptr,
    out_ptr,
    n_cols,
    stride_row,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    x_row_ptr = x_ptr + row * stride_row
    x = tl.load(x_row_ptr + col_offsets, mask=mask, other=float("-inf"))

    x_max = tl.max(x, axis=0)
    x = x - x_max
    e = tl.exp(x)
    denom = tl.sum(e, axis=0)
    out = e / denom

    out_row_ptr = out_ptr + row * stride_row
    tl.store(out_row_ptr + col_offsets, out, mask=mask)


def run(x: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda:
        x = x.cuda()
    x = x.contiguous()
    if x.dtype != torch.float32:
        x = x.to(torch.float32)

    if x.dim() != 2:
        x = x.reshape(-1, x.shape[-1])

    n_rows, n_cols = x.shape
    out = torch.empty_like(x)

    if n_rows == 0 or n_cols == 0:
        return out

    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows,)
    _softmax_kernel[grid](
        x,
        out,
        n_cols,
        x.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
