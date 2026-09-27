import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    x_ptr,
    out_ptr,
    n_cols,
    n_inner,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    outer = pid // n_inner
    inner = pid % n_inner

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    base = outer * n_cols * n_inner + inner
    x = tl.load(x_ptr + base + col_offsets * n_inner, mask=mask, other=float("-inf"))

    x_max = tl.max(x, axis=0)
    x = x - x_max
    e = tl.exp(x)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(out_ptr + base + col_offsets * n_inner, out, mask=mask)


def run(x: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda:
        x = x.cuda()
    if x.dtype != torch.float32:
        x = x.to(torch.float32)
    x = x.contiguous()

    if x.dim() == 1:
        x = x.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    orig_shape = x.shape
    n_cols = orig_shape[1]
    n_outer = 1
    for d in orig_shape[:1]:
        n_outer *= d
    n_inner = 1
    for d in orig_shape[2:]:
        n_inner *= d

    out = torch.empty_like(x)

    if n_cols == 0 or n_outer == 0 or n_inner == 0:
        return out.squeeze(0) if squeeze else out

    grid = (n_outer * n_inner,)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    _softmax_kernel[grid](
        x,
        out,
        n_cols,
        n_inner,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out.squeeze(0) if squeeze else out
