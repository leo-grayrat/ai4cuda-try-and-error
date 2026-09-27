Implement `run(x)` in `candidate.py` for the computation in `reference.py`: stable softmax across each row of a contiguous CUDA float32 matrix. Return a float32 matrix of the same shape. Inputs include non-power-of-two row widths and values with large magnitude; use relative and absolute tolerance `1e-4`.

Use one or more Triton GPU kernels. PyTorch may allocate output tensors and expose metadata, but may not compute the softmax. Edit only `candidate.py`. Check correctness on a small and a large CUDA input before reporting performance. The independent evaluator is outside this workspace.
