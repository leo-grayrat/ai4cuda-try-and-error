# CUDA kernel development guidance for this study

The task specification defines the required output. Read its reference computation and shapes before selecting a kernel layout. Work only in `candidate.py`; the provided reference is explanatory, and the independent evaluator is outside the agent workspace.

## Correctness before speed

The candidate must launch a custom CUDA kernel. Triton is available and is suitable for these tasks. PyTorch may allocate output tensors and expose tensor metadata, but PyTorch arithmetic or reductions must not compute the answer. Preserve dtype, shape, device and all elements, including tails where the size is not a full block. Do not assume that random test values or a single example shape exhaust the evaluator. Check your implementation against the reference on at least a small and a large input after each substantive edit. A faster incorrect candidate is a failure.

## Development sequence

1. Inspect `reference.py`, the task text and the current candidate.
2. Implement one simple, correct kernel before tuning block sizes or memory access.
3. Run a local comparison against the reference. If it fails, inspect indexing, masks, launch grid, output allocation and synchronization before trying a performance change.
4. After every substantive kernel change, run the comparison again. Only compare speed after correctness passes. Keep the best correct source rather than the most recent untested source.
5. Report what you checked and any remaining uncertainty in the final status.

## CUDA reasoning hints

For elementwise work, map adjacent elements to neighboring lanes and mask the tail. Avoid creating unnecessary intermediate GPU tensors. For a row reduction, assign work by row, reduce within the program, and check that every column contributes exactly once. Consider launch overhead and memory traffic before adding more complicated tiling. A configuration that helps one shape can hurt another; test the stated shape range before keeping it.

The evaluator, reference and task contract must remain untouched. A suspiciously large speedup is a reason to recheck correctness and input coverage, not evidence by itself.
