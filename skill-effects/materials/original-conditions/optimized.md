# CUDA development guidance

- [C01] Requirement: Produce correct CUDA C++ extension output and achieve candidate latency <= 0.95 times torch.compile, then seek the best attainable performance.

- [C02] Requirement: Use raw CUDA computation rather than torch C++ compute operators; tensor types, binding metadata, and allocation such as torch::empty_like are allowed.

- [C03] Requirement: In model_new.py forward, use custom extension ops, allocation and tensor properties; do not implement tensor computation with PyTorch operations.

- [C04] Requirement: Use no third-party compute libraries except cuBLAS for GEMM and cuDNN for convolution; use cuDNN for Conv and ConvTranspose.

- [C05] Requirement: Do not modify utils/ infrastructure.

- [C06] Requirement: Do not modify root binding.cpp or binding_registry.h.

- [C07] Requirement: Keep the original model.py unchanged.

- [C08] Requirement: Implement the candidate in model_new.py and kernels/; kernel fixes belong in kernels/*.cu and *_binding.cpp.

- [C09] Requirement: Keep custom CUDA kernels in .cu files and tensor/Python bindings in separate *_binding.cpp files; .cpp compute wrappers are for cuBLAS/cuDNN rather than custom CUDA kernels.

- [C10] Example practice: Prefer torch/types.h and torch/csrc/utils/pybind.h instead of torch/extension.h for faster binding compilation.

- [C11] Example practice: Validate CUDA placement, contiguity and supported dtype in the binding.

- [C12] Example practice: Launch on the current PyTorch CUDA stream.

- [C13] Example practice: Register custom binding functions through REGISTER_BINDING.

- [C14] Requirement: Match the original Model constructor signature in ModelNew.

- [C15] Requirement: Preserve parameter structure so the original state_dict loads into ModelNew.

- [C16] Requirement: Compile for the actual GPU architecture, then execute verification and profiling using the supplied platform-equivalent commands.

- [C17] Conditional advice: Prioritize algorithmic changes such as fusion, shared-memory reuse and coalesced memory access when applicable.

- [C18] Conditional advice: Next consider vectorized loads, warp primitives and occupancy tuning when useful.

- [C19] Conditional advice: Consider instruction-level parallelism, appropriate mixed precision, prefetching and double buffering as finer optimizations.

- [C20] Requirement: Use parameter tuning as a last resort after algorithmic options are exhausted and performance is within 1.2 times the target.

- [C21] Requirement: Expose configuration parameters through bindings for runtime tuning without recompilation.

- [C22] Requirement: When correctness fails, debug the failing kernel, fix it, recompile and retest until it passes within the external experiment budget.

- [C23] Requirement: Before implementing each optimization, document the expected effect and why it should help.

- [C24] Requirement: Debug the optimized version instead of reverting to a slower version merely to avoid fixing it.

- [C25] Requirement: Measure the optimization and explain why performance changed.

- [C26] Requirement: Continue exploring improvements after the minimum target is met until no further improvement or diminishing returns; the source suggests 1-2 initial and 3-5 further iterations.

- [C27] Requirement: Before completion, retain only the final optimized kernels and bindings; remove intermediate versions, old/test attempts and backups.

- [C28] Conditional advice: Check thread bounds and handle non-divisible input sizes.

- [C29] Conditional advice: Place synchronization correctly before shared-memory reuse.

- [C30] Conditional advice: Use correct data types and conversions.

- [C31] Conditional advice: Respect memory alignment and avoid out-of-bounds access.

- [C32] Conditional advice: Use numerically stable algorithms and investigate NaN/Inf, division by zero and missing bounds checks.

- [C33] Conditional advice: Consider grid-stride loops for data larger than one grid.

- [C34] Conditional advice: Avoid shared-memory bank conflicts, for example with padding.

- [C35] Conditional advice: Consider loop unrolling for instruction-level parallelism.

- [C36] Conditional advice: Consider tensor cores for eligible GEMM operations.

- [C37] Conditional advice: Consider persistent kernels to retain data in registers when useful.

- [C38] Conditional advice: Consider CUDA graphs to reduce launch overhead.

- [C39] Conditional advice: For undefined-symbol errors, check matching extern C declarations.

- [C40] Conditional advice: For missing GPU kernel images, check that the compilation architecture matches the GPU.

- [C41] Conditional advice: For incorrect values, inspect math and indexing and test simple inputs.

- [C42] Conditional advice: For shape mismatches, inspect tensor shapes, dimension calculations and reduction logic.

- [C43] Conditional advice: For poor performance, investigate missing fusion, occupancy, coalescing and excessive kernel count as relevant to the observed symptom.

- [C44] Requirement: Pass numerical comparison with atol=1e-2 and rtol=1e-2.

- [C45] Requirement: Give tests descriptive names that show the optimizations being tested.

## Original workspace and code examples

```
.
├── binding_registry.h    # Do NOT modify - registration system
├── binding.cpp           # Do NOT modify - main module binding
├── kernels/              # YOUR WORK: Implement all kernels here
├── utils/                # DO NOT modify - Compilation, verification and profiling tools 
├── model.py              # DO NOT modify - Original PyTorch model
└── model_new.py          # YOUR WORK: Your optimized model using custom ops.
```

```cuda
#include <cuda_runtime.h>

// Template kernel for performance tuning
template<int BLOCK_SIZE, int TILE_SIZE>
__global__ void my_kernel_impl(float* output, const float* input, int size) {
    // Shared memory for tiling
    extern __shared__ float smem[];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop for large data
    for (int i = tid; i < size; i += stride) {
        // Kernel logic with optimizations
        output[i] = /* computation */;
    }
}

// C-interface launcher (no PyTorch dependencies)
extern "C" void my_kernel_launcher(
    float* output,
    const float* input,
    int size,
    int config,
    cudaStream_t stream
) {
    // Dynamic configuration selection
    int blocks = (size + 255) / 256;
    int shared_mem_size = 0;
    
    switch(config) {
        case 0: 
            shared_mem_size = 256 * sizeof(float);
            my_kernel_impl<256, 16><<<blocks, 256, shared_mem_size, stream>>>(
                output, input, size);
            break;
        case 1: 
            shared_mem_size = 128 * sizeof(float);
            my_kernel_impl<128, 32><<<blocks, 128, shared_mem_size, stream>>>(
                output, input, size);
            break;
        default:
            my_kernel_impl<256, 16><<<blocks, 256, 0, stream>>>(
                output, input, size);
    }
}
```

```cpp
// Use this two headers to replace torch/extension.h for faster compilation
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from .cu file
extern "C" void my_kernel_launcher(
    float* output,
    const float* input,
    int size,
    int config,
    cudaStream_t stream
);

// PyTorch wrapper with config parameter
torch::Tensor my_kernel_forward(torch::Tensor input, int config = 0) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    
    auto output = torch::empty_like(input);
    
    // Get current CUDA stream (correct way)
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    // Call CUDA launcher with config
    my_kernel_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        input.numel(),
        config,
        stream
    );
    
    return output;
}

// Registration function
void register_my_kernel(pybind11::module& m) {
    m.def("my_kernel_forward", &my_kernel_forward, 
          "My kernel forward",
          py::arg("input"),
          py::arg("config") = 0);
}

// Auto-register
REGISTER_BINDING(my_kernel, register_my_kernel);
```

```python
import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self, ...):  # MUST match Model signature exactly
        super().__init__()
        # Initialize parameters - preserve original structure for state_dict compatibility
        self.weight = nn.Parameter(torch.randn(...))
        self.bias = nn.Parameter(torch.zeros(...))
        
    def forward(self, x):
        # Use custom ops only - NO torch operations
        x = cuda_extension.my_kernel_forward(x, config=0)
        x = cuda_extension.gemm_forward(x, self.weight, self.bias)
        return x
```

```bash
# Compile with architecture-specific optimizations
TORCH_CUDA_ARCH_LIST=9.0 bash utils/compile.sh

# Test in sandbox 
sudo python3 -m utils.verification
sudo python3 -m utils.profiling
```

```python
# tune_kernel.py - NO recompilation needed
import time, torch, cuda_extension

configs = [
    (0, "256_threads_16_tile"),
    (1, "128_threads_32_tile"),
    (2, "512_threads_8_tile")
]

# Test input
x = torch.randn(batch_size, features).cuda()

# Benchmark each config
best_config, best_time = 0, float('inf')
for config_id, name in configs:
    # Warmup
    for _ in range(10):
        cuda_extension.my_kernel_forward(x, config=config_id)
    torch.cuda.synchronize()
    
    # Measure
    start = time.perf_counter()
    for _ in range(100):
        cuda_extension.my_kernel_forward(x, config=config_id)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    print(f"Config {name}: {elapsed:.4f}s")
    if elapsed < best_time:
        best_time, best_config = elapsed, config_id

print(f"Best: config {best_config} ({best_time:.4f}s)")
# Update model_new.py with best_config
```

```bash
# Remove version files, old attempts, test versions
rm kernels/*_v[0-9].cu kernels/*_old.cu kernels/*_test.cu kernels/*.bak

# Keep only the final optimized implementation
# Example final structure:
# kernels/
#   ├── fused_kernel.cu           # Final implementation
#   └── fused_kernel_binding.cpp  # Final binding
```

```bash
# Use sudo to run sandbox utilities
sudo python3 -m utils.verification
sudo python3 -m utils.profiling
```

```bash
TORCH_CUDA_ARCH_LIST=9.0 bash utils/compile.sh
```
