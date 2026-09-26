Implement `run(x)` in `candidate.py` for the computation in `reference.py`: `x * 1.75 + 0.25`. `x` is a contiguous CUDA float32 vector. Return a CUDA tensor with the same shape and values, within relative and absolute tolerance `1e-4`.

Use a Triton GPU kernel. Do not use PyTorch arithmetic to perform the computation. You may edit `candidate.py`; `reference.py` is supplied for understanding. The independent evaluator is outside this workspace. Correctness precedes any speed comparison.
