Implement `run(x)` in `candidate.py` for the computation in `reference.py`: sum each row of a contiguous CUDA float32 matrix. Return one float32 value per row, within relative and absolute tolerance `1e-4`.

Use a custom GPU kernel (Triton is available). Do not use PyTorch reduction operations to perform the computation. You may edit `candidate.py`; `reference.py` is supplied for understanding. The independent evaluator is outside this workspace. Correctness precedes any speed comparison.
