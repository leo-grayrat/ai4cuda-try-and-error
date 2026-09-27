import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(x_ptr, y_ptr, n_cols, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_ptr += row * stride
    y_ptr += row * stride
    offs = tl.arange(0, BLOCK)

    m = float("-inf")
    for start in range(0, n_cols, BLOCK):
        idx = start + offs
        mask = idx < n_cols
        x = tl.load(x_ptr + idx, mask=mask, other=float("-inf"))
        m = tl.maximum(m, tl.max(x, axis=0))

    s = 0.0
    for start in range(0, n_cols, BLOCK):
        idx = start + offs
        mask = idx < n_cols
        x = tl.load(x_ptr + idx, mask=mask, other=float("-inf"))
        e = tl.where(mask, tl.exp(x - m), 0.0)
        s += tl.sum(e, axis=0)

    inv = 1.0 / s
    for start in range(0, n_cols, BLOCK):
        idx = start + offs
        mask = idx < n_cols
        x = tl.load(x_ptr + idx, mask=mask, other=0.0)
        e = tl.where(mask, tl.exp(x - m), 0.0)
        tl.store(y_ptr + idx, e * inv, mask=mask)


def run(x: torch.Tensor) -> torch.Tensor:
    if not x.is_contiguous():
        x = x.contiguous()
    n_cols = x.shape[1] if x.dim() > 1 else x.shape[0]
    n_rows = x.numel() // n_cols if n_cols else 0

    y = torch.empty_like(x, dtype=torch.float32)

    if n_rows == 0 or n_cols == 0:
        return y

    BLOCK = min(triton.next_power_of_2(n_cols), 2048)
    grid = (n_rows,)
    _softmax_kernel[grid](x, y, n_cols, n_cols, BLOCK=BLOCK)
    return y
