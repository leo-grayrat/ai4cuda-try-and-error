# CUDA development guidance

- [C01] Requirement: Produce correct CUDA C++ extension output and achieve candidate latency <= 0.95 times torch.compile, then seek the best attainable performance.
- [C02] Requirement: Use raw CUDA computation rather than torch C++ compute operators; tensor types, binding metadata, and allocation such as torch::empty_like are allowed.
- [C03] Requirement: In model_new.py forward, use custom extension ops, allocation and tensor properties; do not implement tensor computation with PyTorch operations.
- [C04] Requirement: Use no third-party compute libraries except cuBLAS for GEMM and cuDNN for convolution; use cuDNN for Conv and ConvTranspose.
- [C05] Requirement: Do not modify utils/ infrastructure.
- [C06] Requirement: Do not modify root binding.cpp or binding_registry.h.
- [C07] Requirement: Keep the original model.py unchanged.
- [C08] Requirement: Implement the candidate in model_new.py and kernels/; kernel fixes belong in kernels/*.cu and *_binding.cpp.
- [C44] Requirement: Pass numerical comparison with atol=1e-2 and rtol=1e-2.
