Implement `run(x)` in `candidate.py` for the computation in `reference.py`.

The reference is a chain of 3 operators: Conv2D + ReLU + BiasAdd. It is currently implemented as
a sequence of PyTorch ops, so each intermediate result is materialised to global
memory. A single fused Triton kernel, or a small number of them, should be able
to do the same work without those round trips.

The weights are available as `MODEL` (import it from `reference`); the seed is
fixed, so both sides see the same parameters.

Use a Triton GPU kernel (`@triton.jit` with `tl.load` / `tl.store`). Calling the
PyTorch ops directly, or any torch/numpy equivalent, is not acceptable.
