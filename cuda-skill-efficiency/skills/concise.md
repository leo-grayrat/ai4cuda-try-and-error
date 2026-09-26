# CUDA kernel development guidance for this study

Read the task and reference. Edit only `candidate.py`; leave the reference and evaluator untouched. Use a custom CUDA kernel (Triton is available). PyTorch may allocate output and provide metadata, but must not compute the answer. Preserve dtype, shape, device, all elements and masked tails across the stated shapes.

Implement a simple correct kernel first. Compare small and large inputs to the reference after each substantive edit; if wrong, check indexing, masks, grid, allocation and synchronization. Measure speed only after correctness passes, retain the best correct source, and report checks and uncertainty. For elementwise work use adjacent lanes and avoid intermediate tensors; for row reductions cover each column once. Weigh launch overhead and memory traffic before complex tiling, and recheck correctness if speedup looks suspicious.
