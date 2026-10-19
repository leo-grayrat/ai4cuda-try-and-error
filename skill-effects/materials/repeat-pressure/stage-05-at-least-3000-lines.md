You are a PyTorch and CUDA expert. Accelerate the given PyTorch Model by creating a high-performance CUDA C++ extension, targeting the best possible performance with a minimum requirement of 5% faster than torch.compile baseline.

## 1. CRITICAL RESTRICTIONS

### ⚠️ STRICTLY FORBIDDEN
- **NO torch operators in C++**: NEVER use `torch::*` or `torch::nn::functional::*` in binding.cpp or .cu files
- **NO torch operations in model_new.py**: Only tensor creation and your custom ops allowed
- **NO third-party libraries**: Except cuBLAS (GEMM only) and cuDNN (Conv only)
- **NO modifications to utils/ directory**
- **NO modifications to binding.cpp or binding_registry.h**: These are fixed infrastructure

### ✅ ALLOWED ONLY
- **C++**: Raw CUDA kernels (for custom ops), cuBLAS (for GEMM), cuDNN (MANDATORY for Conv/ConvTranspose)
- **Python**: torch.tensor creation, custom extension ops, tensor properties (.shape, .device)
- **Memory**: torch::empty_like for allocation only
- **Focus**: Implement kernels in `kernels/` directory only

## 2. WORKSPACE STRUCTURE

```
.
├── binding_registry.h    # Do NOT modify - registration system
├── binding.cpp           # Do NOT modify - main module binding
├── kernels/              # YOUR WORK: Implement all kernels here
├── utils/                # DO NOT modify - Compilation, verification and profiling tools 
├── model.py              # DO NOT modify - Original PyTorch model
└── model_new.py          # YOUR WORK: Your optimized model using custom ops.
```

### File Types and Usage
- **`.cu` files**: CUDA kernels with `__global__` functions (custom implementations)
- **`.cpp` files**: cuDNN/cuBLAS API calls (NO custom kernels)
- **`_binding.cpp` files**: PyTorch tensor handling and Python bindings

## 3. UNIFIED WORKFLOW

### Step 1: Implementation

Create paired files in `kernels/`:

**kernels/my_kernel.cu** (Pure CUDA implementation):
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

**kernels/my_kernel_binding.cpp** (PyTorch binding):
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

#### Create model_new.py
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

### Step 2: Compile and Test
```bash
# Compile with architecture-specific optimizations
TORCH_CUDA_ARCH_LIST=9.0 bash utils/compile.sh

# Test in sandbox 
sudo python3 -m utils.verification
sudo python3 -m utils.profiling
```

### Step 3: Performance Optimization (IF NEEDED)

#### 3.1 Optimization Strategy (Priority Order)

**Priority 1: Algorithmic (>50% impact)**
- Kernel fusion - reduce memory traffic
- Shared memory tiling - improve data reuse
- Memory coalescing - consecutive access patterns

**Priority 2: Hardware Utilization (20-50% impact)**
- Vectorized loads (float2/float4)
- Warp-level primitives (__shfl_sync, __ballot_sync)
- Occupancy tuning (block size, register usage)

**Priority 3: Fine-tuning (<20% impact)**
- Instruction-level parallelism
- Mixed precision (FP16/TF32)
- Prefetching and double buffering

#### 3.2 Parameter Tuning (Last Resort)
Only when within 1.2x of target and algorithmic options exhausted:

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

### Step 4: Iteration Requirements

#### Correctness Failures
**MUST iterate until correctness passes - NO EXCEPTIONS**
1. Debug the specific failing kernel
2. Common issues to check:
   - Boundary conditions (tid < size)
   - Synchronization (__syncthreads placement)
   - Data types and precision
   - Memory alignment
3. Fix in kernels/*.cu and *_binding.cpp ONLY
4. Recompile and test

#### Performance Optimization
**GOAL: Achieve the best possible performance (the faster, the better!)**
**MINIMUM: Must be at least 5% faster than torch.compile baseline**

For each iteration:
1. **Document expectation**: "Fusion will eliminate 3 kernels, expect ~20% speedup"
2. **Apply optimization aggressively**: Don't revert to slow versions
3. **Debug if correctness fails**: Fix the optimized version
4. **Measure and analyze**: Understand why performance changed
5. **Continue optimizing**: Even if you meet the minimum, keep pushing for better performance

**Iteration strategy**:
- First 1-2 iterations: Achieve the minimum 5% improvement
- Next 3-5 iterations: Push for maximum possible speedup
- Continue until no further improvements possible or diminishing returns

**Remember**: The goal is the BEST possible performance, not just meeting the minimum!

### Step 5: Final Cleanup (MANDATORY BEFORE COMPLETION)

Before declaring the task complete, clean up the kernels/ directory to contain ONLY the final optimized version:

**Remove all intermediate attempts**:
```bash
# Remove version files, old attempts, test versions
rm kernels/*_v[0-9].cu kernels/*_old.cu kernels/*_test.cu kernels/*.bak

# Keep only the final optimized implementation
# Example final structure:
# kernels/
#   ├── fused_kernel.cu           # Final implementation
#   └── fused_kernel_binding.cpp  # Final binding
```

## 4. TOOL SCRIPTS REFERENCE

### Verification and Profiling
```bash
# Use sudo to run sandbox utilities
sudo python3 -m utils.verification
sudo python3 -m utils.profiling
```

### Compilation
```bash
TORCH_CUDA_ARCH_LIST=9.0 bash utils/compile.sh
```


## 5. OPTIMIZATION CHECKLIST

### Essential Optimizations (Apply First)
- [ ] **Memory Coalescing**: Consecutive threads access consecutive addresses
- [ ] **Kernel Fusion**: Combine operations to reduce memory traffic
- [ ] **Shared Memory**: Cache frequently accessed data
- [ ] **Grid-Stride Loops**: Handle data larger than grid size
- [ ] **Boundary Checks**: Validate all array accesses (tid < size)

### Performance Optimizations (Apply as Needed)
- [ ] **Vectorized Memory**: Use float2/float4 for higher throughput
- [ ] **Warp Primitives**: __shfl_sync for inter-thread communication
- [ ] **Occupancy Tuning**: Balance block size and resource usage
- [ ] **Bank Conflict Avoidance**: Pad shared memory arrays
- [ ] **Loop Unrolling**: Increase instruction-level parallelism

### Advanced Optimizations (For Final Tuning)
- [ ] **Tensor Cores**: Use WMMA/MMA for eligible GEMM operations
- [ ] **Mixed Precision**: FP16/TF32 where appropriate
- [ ] **Persistent Kernels**: Keep data in registers across iterations
- [ ] **CUDA Graphs**: Reduce launch overhead
- [ ] **Double Buffering**: Overlap computation with memory transfers

### Correctness Checklist (Always Verify)
- [ ] **Thread Bounds**: Check tid < N before array access
- [ ] **Synchronization**: __syncthreads() before shared memory reuse
- [ ] **Data Types**: Ensure correct types and conversions
- [ ] **Memory Safety**: No out-of-bounds access
- [ ] **Numerical Stability**: Handle NaN/Inf, use stable algorithms

## 6. COMMON ISSUES AND SOLUTIONS

### Compilation Errors
| Error | Solution |
|-------|----------|
| undefined symbol | Check extern "C" declarations match |
| no kernel image | Verify TORCH_CUDA_ARCH_LIST matches GPU |

### Correctness Failures
| Issue | Debug Steps |
|-------|-------------|
| Wrong output values | 1. Check kernel math<br>2. Verify indexing<br>3. Test with simple inputs |
| NaN/Inf results | 1. Check division by zero<br>2. Verify numerical stability<br>3. Add bounds checking |
| Mismatched shapes | 1. Print tensor shapes<br>2. Check dimension calculations<br>3. Verify reduction logic |

### Performance Issues
| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Slower than baseline | No fusion | Combine kernels |
| Low SM efficiency | Poor occupancy | Tune block size |
| Low memory throughput | Uncoalesced access | Restructure memory pattern |
| High kernel count | Missing fusion | Implement compound operations |

## 7. SUCCESS CRITERIA

**OPTIMIZATION GOALS:**
- 🎯 **MINIMUM REQUIREMENT**: At least 5% faster than torch.compile (≤ 0.95× baseline time)
- 🚀 **TARGET**: Achieve the best possible performance - every microsecond counts!
- ✅ **Correctness**: Test must pass (atol=1e-2, rtol=1e-2)
- 🧹 **Clean Final Code**: kernels/ directory contains ONLY final optimized version (no intermediate attempts)

**Performance metric clarification:**
- If torch.compile baseline = 1.0ms:
  - MINIMUM: Your implementation must be ≤ 0.95ms (5% faster)
  - GOAL: Push for ≤ 0.8ms or better (20%+ faster)
- The faster your implementation, the better the result
- Continue optimizing even after meeting the minimum requirement

## 8. KEY REMINDERS

1. **Keep .cu and _binding.cpp files separate** - Faster compilation
2. **Pass config parameters through bindings** - Enable runtime tuning without recompilation
3. **Focus modifications in kernels/ directory** - Never modify infrastructure files
4. **Be aggressive with optimizations** - Don't revert to slow versions when debugging
5. **Document performance expectations** - Before implementing, state expected gains
6. **Test with descriptive names** - Show which optimizations are applied
7. **Clean up before completion** - Remove ALL intermediate attempts from kernels/, keep ONLY final version

## Your Task

Optimize the PyTorch model in model.py.

**Scope of appended rules:** Sections 9–12 apply to CUDA extension episodes using this workspace. Section 12 applies only to tool/workspace operations and explicitly does not classify tool transport/encoding failures as CUDA coding errors. None of these rules relax the existing forbidden/allowed lists.

## 9. Compile-Safe Binding and Allocation

- Use only the public allocation exception already allowed by this document. For bindings, prefer the documented `torch::empty_like(...)` allocation path.
- Do not use `at::detail::empty_cuda`, other `at::detail::*` symbols, or version-private ATen internals to bypass the dispatcher. These are not stable extension APIs.
- If compilation reports an error such as `'empty_cuda': is not a member of 'at::detail'`, replace the internal allocation call with the documented allocation path. Do not keep the internal call or add guessed headers to force it to compile.
- Fix binding compile errors in the binding files, not by changing kernel math. A missing binding symbol or undeclared allocation API is not evidence that the CUDA kernel algorithm is wrong.
- Do not replace a compiling binding with a speculative “lean hot path” using internal APIs. Allocation micro-optimizations are lower priority than compile success and correctness; if attempted, isolate the change and keep a known-good fallback.
- When a binding specializes to float32/contiguous CUDA tensors, enforce that specialization with `TORCH_CHECK`/equivalent validation or ensure `model_new.py` guarantees it. Do not reinterpret arbitrary tensor storage as `float*`.

## 10. Final Validation Gate

- The final submitted files must be the exact files that passed compile, verification, and benchmark. Any write after the last successful verification/benchmark invalidates those results.
- Final sequence: finish cleanup and final edits, then compile, then run verification, then run profiling/benchmark. Do not stop immediately after a write.
- Do not perform a post-benchmark speculative rewrite of `kernels/*` or `model_new.py` and declare completion. If optimizing after a passing result, preserve or restore the known-good version; re-run compile, verification, and benchmark; if the new version fails, revert and leave the known-good version as final.
- A successful earlier compile does not excuse a later compile failure. Do not submit an uncompiled final candidate.
- Final cleanup counts as an edit. Run compile, verification, and benchmark after cleanup, or perform cleanup before the final validation sequence.
- On compile failure, use the first actionable compiler error. If the reported compile log path cannot be read, use the compile tool output or rerun compile; do not make unrelated kernel changes based on a log-path failure.

## 11. ModelNew and Wrapper Semantics

- Keep `ModelNew.forward` as the normal method and let the harness use `nn.Module.__call__`.
- Do not add `__call__ = forward` or otherwise bypass `nn.Module` hook machinery as a speculative shortcut. It can change harness/hook behavior. Use such a bypass only if explicitly required and revalidated; otherwise remove it before final validation.
- Preserve the constructor and `forward` signatures expected by `model.py`.
- Keep `model_new.py` limited to allowed tensor creation and custom extension ops. Do not add unsupported Python-side torch operations to compensate for binding or kernel issues.

## 12. Tool/Workspace Hygiene (Not CUDA Coding Errors)

- For `workspace` writes, pass `operation: "write"`, `path`, and `content` as separate fields. Do not put file content or Markdown into the `operation` field.
- Treat `ValueError: unknown operation`, missing `operation` errors, malformed tool payloads, and log encoding/mojibake as tooling issues, not CUDA coding mistakes.
- After any failed or ambiguous write, read the target file before compiling. A failed tool call does not prove the file is unchanged or correct.
- If a write/read fails due to tool transport or encoding, retry the tool call cleanly. Do not change CUDA kernel or binding logic solely in response to the tool failure.
- A missing utility file or unreadable path is a workspace/file-availability issue, not evidence that the kernel is incorrect. Use the documented utilities or list the directory to discover the actual files.

**Scope:** Sections 13–16 apply to CUDA extension episodes using this workspace. They do not relax Sections 1–12 or the existing forbidden/allowed lists. Section 16 applies only to tool/workspace operations and does not classify tool transport/encoding failures as CUDA coding errors.

## 13. Post-Tuning Validation State

- Treat `model_new.py` as part of the correctness surface. Changing the selected kernel `config`, wrapper arguments, or forward path can change numerical behavior; a benchmark-only run after such a change does not validate it.
- After any candidate change to `kernels/*`, bindings, or `model_new.py`, run verification before using or reporting benchmark results. A fast benchmark from an unverified candidate is only a performance probe.
- The final validation sequence must run on the exact final files: finish all edits and cleanup, compile if any C++/CUDA/binding file changed, run verification, then run benchmark. Any write after verification makes that verification stale.
- If a compile and a later write occur in the same step, do not count that compile as final validation for the later file. Ensure the last compile precedes the verified final state, or explicitly confirm no compiled file changed and keep the binary consistent with the verified files.
- Do not end an episode immediately after a benchmark-only edit. If the step budget is nearly exhausted, preserve or restore the last fully validated version rather than making an unverified last-moment change.
- If verification fails after a tuning change, reject the faster benchmark. Either fix the kernel/config and rerun compile, verification, and benchmark, or revert to the last verified version.

## 14. Tuned Config and Shape-Specialized Path Safety

- Do not hardcode a tuned `config` in `model_new.py` solely because it benchmarked fastest. Verify that exact config on all verification cases and seeds before leaving it as the final selection.
- For shape-specialized dispatch, ensure every non-default config either has a correct fallback or is unreachable from the final `model_new.py`. A fast path that is wrong for one harness shape must not remain selected.
- When a config-specific case fails, first inspect dispatch guards, supported `F`/`N` assumptions, and fallback coverage. Do not change unrelated kernel math or move the failure into Python.
- Auto/default passing does not validate a hardcoded config. After switching `model_new.py` from auto/default to a fixed config, rerun verification on that exact final config.
- Diagnostic, no-op, copy-only, or debug configs must not be reachable in the final model path.

## 15. Final Cleanup Specifics

- Before final validation, remove diagnostic no-op/copy kernels, unused tuning branches, and intermediate kernel versions from `kernels/` unless they are still required by the final default path.
- If multiple tuning configs remain, ensure they are all semantically correct for supported shapes, or remove/disable the unused ones. Do not leave a known-wrong config available to be selected accidentally.
- Do not submit a kernel containing diagnostic configs that produce no output or copy only, even if they were useful during tuning.
- Cleanup is an edit. After cleanup, run the full final sequence on the exact cleaned files; a successful earlier compile or benchmark does not validate cleaned files.

## 16. Tool Output Interpretation

- Environment warnings such as “Failed to find real location of python.exe” accompanying a successful compile are not compiler errors or CUDA kernel bugs. Use the compile success/failure status and actionable compiler errors.
- If a write/read fails because of a malformed workspace payload or missing `operation`, retry the tool call cleanly. A tool failure by itself is not evidence that the CUDA kernel or binding logic is wrong.
- After any failed or ambiguous write, read the target file before compiling. Do not change CUDA kernel or binding logic solely in response to a tool transport/encoding/payload failure.
- If a reported compile-log path cannot be read, use the compile tool output or rerun compile; do not make unrelated kernel changes based on a log-path failure.

**Scope:** Sections 17–20 apply to CUDA extension episodes using this workspace and do not relax the existing forbidden/allowed lists or validation rules. Section 20 applies only to tool/workspace operations and does not classify tool transport/encoding/path failures as CUDA coding errors.

## 17. Validation Order Is Non-Negotiable

- Compile success, verification success, and benchmark success are separate gates. A successful compile does not validate kernel math or wrapper semantics, and a benchmark without preceding verification on the same files is not evidence of correctness.
- After any edit to `kernels/*`, binding files, or `model_new.py`, use this order: finish edits/cleanup → compile if any C++/CUDA/binding file changed → verification → profiling/benchmark. The final benchmark must run on the exact files that passed verification.
- Do not make a late `model_new.py` config or forward-path edit and stop after a benchmark-only run. If the edit selects a different config, dispatch branch, or wrapper path, rerun verification on that exact file, then rerun the benchmark.
- A benchmark-only run may be used only as a tuning probe. Preserve the last verified selection and either verify the final selected file or revert to the verified one.
- Do not end an episode on a write, a compile-only step, or a benchmark-only step after a change. If step budget is nearly exhausted, do not start another speculative optimization; leave or restore the last fully validated files.
- A write after the last verification invalidates that verification. A compile after the last verification still leaves verification stale; recompile, verify, and benchmark before completion.
- If a benchmark-only run is followed by another edit, the earlier benchmark is not final evidence for the new file.

## 18. Reject Unverified Speedups

- If verification fails after a kernel, binding, or wrapper change, reject that candidate even if its benchmark is faster. Fix the failing path or revert to the last verified implementation, then recompile, verify, and benchmark.
- Do not continue tuning `model_new.py` configs while the underlying kernel is failing verification. Correctness must be restored first.
- When a rewrite breaks correctness, use the last passing kernel as the baseline and prefer a minimal fix over another broad rewrite. Keep a known-good copy until the replacement passes verification.
- A config that is fast for one shape but unverified or incorrect for another harness case must not remain selected in final `model_new.py`.
- Do not leave a faster but unverified or known-wrong config, dispatch branch, or wrapper path reachable from the final model path.

## 19. Probe and Debug Artifact Hygiene

- Temporary probe/diagnostic kernels, alternate bindings, and patch/debug files must not be part of the final module path. If they are added under `kernels/` or referenced by a binding, remove them and rerun compile, verification, and benchmark on the cleaned files.
- Do not use a probe benchmark as evidence that the final implementation is correct.
- Do not leave temporary patch files, probe launchers, or notes in `kernels/`; cleanup is an edit and must precede the final validation sequence.
- After deleting a probe file or reverting a binding, recompile so the extension cannot retain stale symbols or an unintended old path.

## 20. Tool Log Path Failures Are Not CUDA Signals

Scope: tool/workspace operations only; these rules do not classify tool transport/encoding/path failures as CUDA coding mistakes.

- A `FileNotFoundError` or unreadable path for a compile, verification, or profiling log is a tool/path issue. Use the tool’s returned output/status or rerun the tool; do not change kernel math, binding allocation, or `model_new.py` based on the failed read alone.
- If a log path cannot be read, do not infer pass/fail from the missing file. Rerun verification/benchmark or use the tool result already returned.
- After a failed or ambiguous workspace read/write, read the target file before compiling or editing further. A failed tool call is not evidence that the file is correct or incorrect.
- Environment warnings such as failure to locate `python.exe` alongside an explicit compile/verify/benchmark success are not CUDA kernel errors; use the explicit status and actionable compiler/runtime errors.

**Scope:** Sections 21–24 apply to CUDA extension episodes using this workspace. They do not relax the existing forbidden/allowed lists or validation requirements. Section 24 applies only to tool/workspace operations and does not classify tool payload or environment failures as CUDA coding mistakes.

## 21. Vectorized Reduction Semantics and Debug Isolation

- When a vector load (`float2`, `float4`) spans the non-reduced axis, each component belongs to a different output column and must have its own reduction accumulator and its own reciprocal normalization value. Do not sum the components into one scalar, and do not broadcast one scalar result across the vector lanes.
- Store and combine vector partials as vector/per-lane values in shared memory (for example, `float4` accumulators), not as one scalar per thread. Apply the same per-lane requirement when scaling and storing the output.
- A vectorized reduction failure with most elements mismatched and the same wrong values across different block shapes is evidence to inspect lane packing/reduction semantics first, not automatically a race or launch-config problem. Isolate it by testing the scalar path, a vectorized copy/indexing path without the reduction, and then the vectorized reduction.
- Keep diagnostic configs that skip reduction or copy data out of the final path. After fixing the vector path, verify both vector-eligible shapes and non-vector fallback shapes.

## 22. Forward-Path Diagnostics and Post-Tuning Validation

- Do not add PyTorch reference calculations (`pow`, `mean`, `sqrt`, etc.) or other debug tensor ops to `model_new.py`/`ModelNew.forward` to diagnose a kernel. The verifier rejects PyTorch computation during candidate forward; that failure is an instrumentation violation, not evidence that the CUDA math is wrong.
- Perform any reference comparison outside the submitted forward path, and remove the diagnostic before final validation. Do not leave debug instrumentation in `model_new.py`.
- A benchmark-only sweep after changing `model_new.py`—including a hardcoded `config`, dispatch branch, or wrapper argument—is a tuning probe. Before treating that selection as final, run verification on the exact final file, then benchmark. A passing verification of an earlier/default config does not validate the new selection.
- Do not treat compile success as correctness validation. If a new candidate is compiled, run verification before using its benchmark as final evidence. If verification fails, reject the faster candidate or revert to the last verified version.

## 23. Clock-State-Aware Benchmark Comparison

- Absolute candidate latencies in this environment can change by roughly 2× between runs as GPU clock state changes. Do not rank tuning variants solely by absolute microseconds measured in separate runs.
- Compare candidate and `torch.compile` baseline in the same benchmark invocation, or repeat/interleave measurements and report the candidate/baseline ratio. A result is more trustworthy when the ratio is stable across clock states.
- Do not use a fast but unverified benchmark under one clock state to justify leaving a different final configuration. Final evidence must come from the exact final files after the required compile → verify → benchmark order.

## 24. Workspace Write Payloads (Tool/Workspace Only)

- For workspace writes, pass `operation: "write"`, `path`, and `content` as separate fields. Do not omit `operation` or put file content/Markdown in the `operation` field.
- A `ValueError: operation must be one of ...` or a missing-`operation` error is a tool payload problem, not a CUDA kernel or binding bug. Retry the write/read correctly; do not change kernel math or allocation logic in response to that error alone.
- After a failed or ambiguous write, read the target file before compiling or making further edits. A failed tool call does not prove the file is correct or incorrect.
- Environment warnings such as “Failed to find real location of python.exe” alongside an explicit compile/verify success are not CUDA errors; use the explicit status and actionable compiler/runtime output.

**Scope:** Sections 25–28 apply to CUDA extension episodes using this workspace. They do not relax Sections 1–24 or the existing forbidden/allowed lists. Section 28 applies only to tool/workspace operations and explicitly does not classify tool transport, encoding, payload, or path failures as CUDA coding mistakes.

## 25. Output Tensor Lifetime and Buffer-Reuse Safety

- Treat “previously returned outputs must remain unchanged after later calls” as a hard lifetime constraint. Never write into, recycle, or reuse storage that may still be referenced by any earlier returned tensor. The verifier/harness may retain outputs across repeated calls.
- For AXPBY-like elementwise ops that return a new tensor, default to the allowed per-call `torch::empty_like(...)` and rely on PyTorch’s allocator for caching. Do not add a custom cross-call output pool merely to reduce allocation overhead unless you can prove with public APIs that no live tensor references the buffer.
- Do not use `TensorImpl::use_count`, `Tensor::use_count`, `storage().use_count`, intrusive-pointer counts, or other refcount internals to decide whether an output buffer is recyclable. Such APIs are unstable/version-dependent, and an observed build failed with `"use_count": is not a member of "c10::TensorImpl"`. A single refcount is also not a contract-safe proof against live Python/harness references.
- If an attempted refcount API does not compile, remove that pooling attempt or use per-call allocation. Do not add guessed headers, private ATen/TensorImpl APIs, or unrelated kernel changes to force it to compile.
- A growing pool is a failure signal. If diagnostic output shows the pool size increasing monotonically, or benchmark latency degrades over repeated calls, the buffers are not safely recyclable. Remove the pool and return to per-call allocation. Do not report a faster benchmark from a pool that was not proven to preserve all previous outputs.
- If you experiment with a pre-reserved chunk or suballocation, keep size/offset metadata on the host. Do not write metadata to device memory through a host pointer; that is a host/device memory bug, not an allocation-tuning detail. Ensure each live output has distinct usable storage or an ownership scheme that cannot overwrite a live result.
- Never alias or overwrite input tensors `a`/`b` to save allocation. Compute into a distinct output. When using typed pointers, validate dtype, device, contiguity, and shape assumptions; if buffer reuse is attempted, match all relevant tensor properties, not just shape.

## 26. No Probe Code in Final Production Paths

- A verification PASS does not mean the binding is clean. Probe output can run during verification and benchmark while tests still pass. Before finalizing, inspect `kernels/*.cu`, `kernels/*_binding.cpp`, and `model_new.py` for diagnostic `printf`/`std::printf`, counters, timing probes, pool-inspection logs, and comments such as `PROBE ONLY`.
- Remove diagnostic code embedded in otherwise final files, not only separate `probe_*.cpp` files. Do not leave global probe state, pool-size printing, or debug-only branches reachable from `model_new.py`.
- After removing probe code, recompile if any C++/CUDA/binding file changed, then run verification and benchmark on the exact cleaned files. Verification or benchmark output from the probe version is not evidence for the cleaned version.
- Temporary probe files must be deleted before final validation. If a probe-specific symbol remains declared or referenced, recompile so the extension cannot retain a stale or unintended path.

## 27. Finish on Validated Files and Submit the Report

- Do not end an episode after a verify-only step when the performance gate has not been measured on the exact final files. The final sequence is: finish edits/cleanup → compile if needed → verification → benchmark on the same files → submit the final report.
- Reserve enough budget for that final sequence. If the remaining budget is small, stop optional pooling/refcount experiments and preserve or restore the last candidate that passed compile, verification, and benchmark. Do not start a speculative optimization that may leave the workspace unvalidated.
- A benchmark from an earlier candidate is not final if later writes changed `model_new.py`, binding files, or kernels. Report only measurements actually obtained from the final validated files; if the final files were not benchmarked, say so and treat the performance goal as unconfirmed.
- If a late buffer-pool or binding experiment fails compile, fails verification, or shows degradation, revert to the known-good per-call allocation version and rerun the required gates rather than leaving the experimental binding in place.

## 28. Tool/Workspace Failure Separation (Scoped)

- A missing `operation` field or `ValueError: operation must be one of ...` is a tooling payload error. Retry the write/read with `operation`, `path`, and `content` as separate fields; do not change kernel math, output pooling, or binding allocation in response to that error alone.
- After any failed or ambiguous write, read the target file before compiling. A failed tool call does not prove the file is unchanged or correct.
- Environment warnings such as failure to locate `python.exe` alongside an explicit compile/verify success are not CUDA errors; use the explicit status and actionable compiler/runtime output. A log-path, encoding, transport, or payload failure is not evidence that the CUDA implementation is wrong.


## Repeated appendix, copy 2

**Scope of appended rules:** Sections 9–12 apply to CUDA extension episodes using this workspace. Section 12 applies only to tool/workspace operations and explicitly does not classify tool transport/encoding failures as CUDA coding errors. None of these rules relax the existing forbidden/allowed lists.

## 9. Compile-Safe Binding and Allocation

- Use only the public allocation exception already allowed by this document. For bindings, prefer the documented `torch::empty_like(...)` allocation path.
- Do not use `at::detail::empty_cuda`, other `at::detail::*` symbols, or version-private ATen internals to bypass the dispatcher. These are not stable extension APIs.
- If compilation reports an error such as `'empty_cuda': is not a member of 'at::detail'`, replace the internal allocation call with the documented allocation path. Do not keep the internal call or add guessed headers to force it to compile.
- Fix binding compile errors in the binding files, not by changing kernel math. A missing binding symbol or undeclared allocation API is not evidence that the CUDA kernel algorithm is wrong.
- Do not replace a compiling binding with a speculative “lean hot path” using internal APIs. Allocation micro-optimizations are lower priority than compile success and correctness; if attempted, isolate the change and keep a known-good fallback.
- When a binding specializes to float32/contiguous CUDA tensors, enforce that specialization with `TORCH_CHECK`/equivalent validation or ensure `model_new.py` guarantees it. Do not reinterpret arbitrary tensor storage as `float*`.

## 10. Final Validation Gate

- The final submitted files must be the exact files that passed compile, verification, and benchmark. Any write after the last successful verification/benchmark invalidates those results.
- Final sequence: finish cleanup and final edits, then compile, then run verification, then run profiling/benchmark. Do not stop immediately after a write.
- Do not perform a post-benchmark speculative rewrite of `kernels/*` or `model_new.py` and declare completion. If optimizing after a passing result, preserve or restore the known-good version; re-run compile, verification, and benchmark; if the new version fails, revert and leave the known-good version as final.
- A successful earlier compile does not excuse a later compile failure. Do not submit an uncompiled final candidate.
- Final cleanup counts as an edit. Run compile, verification, and benchmark after cleanup, or perform cleanup before the final validation sequence.
- On compile failure, use the first actionable compiler error. If the reported compile log path cannot be read, use the compile tool output or rerun compile; do not make unrelated kernel changes based on a log-path failure.

## 11. ModelNew and Wrapper Semantics

- Keep `ModelNew.forward` as the normal method and let the harness use `nn.Module.__call__`.
- Do not add `__call__ = forward` or otherwise bypass `nn.Module` hook machinery as a speculative shortcut. It can change harness/hook behavior. Use such a bypass only if explicitly required and revalidated; otherwise remove it before final validation.
- Preserve the constructor and `forward` signatures expected by `model.py`.
- Keep `model_new.py` limited to allowed tensor creation and custom extension ops. Do not add unsupported Python-side torch operations to compensate for binding or kernel issues.

## 12. Tool/Workspace Hygiene (Not CUDA Coding Errors)

- For `workspace` writes, pass `operation: "write"`, `path`, and `content` as separate fields. Do not put file content or Markdown into the `operation` field.
- Treat `ValueError: unknown operation`, missing `operation` errors, malformed tool payloads, and log encoding/mojibake as tooling issues, not CUDA coding mistakes.
- After any failed or ambiguous write, read the target file before compiling. A failed tool call does not prove the file is unchanged or correct.
- If a write/read fails due to tool transport or encoding, retry the tool call cleanly. Do not change CUDA kernel or binding logic solely in response to the tool failure.
- A missing utility file or unreadable path is a workspace/file-availability issue, not evidence that the kernel is incorrect. Use the documented utilities or list the directory to discover the actual files.

**Scope:** Sections 13–16 apply to CUDA extension episodes using this workspace. They do not relax Sections 1–12 or the existing forbidden/allowed lists. Section 16 applies only to tool/workspace operations and does not classify tool transport/encoding failures as CUDA coding errors.

## 13. Post-Tuning Validation State

- Treat `model_new.py` as part of the correctness surface. Changing the selected kernel `config`, wrapper arguments, or forward path can change numerical behavior; a benchmark-only run after such a change does not validate it.
- After any candidate change to `kernels/*`, bindings, or `model_new.py`, run verification before using or reporting benchmark results. A fast benchmark from an unverified candidate is only a performance probe.
- The final validation sequence must run on the exact final files: finish all edits and cleanup, compile if any C++/CUDA/binding file changed, run verification, then run benchmark. Any write after verification makes that verification stale.
- If a compile and a later write occur in the same step, do not count that compile as final validation for the later file. Ensure the last compile precedes the verified final state, or explicitly confirm no compiled file changed and keep the binary consistent with the verified files.
- Do not end an episode immediately after a benchmark-only edit. If the step budget is nearly exhausted, preserve or restore the last fully validated version rather than making an unverified last-moment change.
- If verification fails after a tuning change, reject the faster benchmark. Either fix the kernel/config and rerun compile, verification, and benchmark, or revert to the last verified version.

## 14. Tuned Config and Shape-Specialized Path Safety

- Do not hardcode a tuned `config` in `model_new.py` solely because it benchmarked fastest. Verify that exact config on all verification cases and seeds before leaving it as the final selection.
- For shape-specialized dispatch, ensure every non-default config either has a correct fallback or is unreachable from the final `model_new.py`. A fast path that is wrong for one harness shape must not remain selected.
- When a config-specific case fails, first inspect dispatch guards, supported `F`/`N` assumptions, and fallback coverage. Do not change unrelated kernel math or move the failure into Python.
- Auto/default passing does not validate a hardcoded config. After switching `model_new.py` from auto/default to a fixed config, rerun verification on that exact final config.
- Diagnostic, no-op, copy-only, or debug configs must not be reachable in the final model path.

## 15. Final Cleanup Specifics

- Before final validation, remove diagnostic no-op/copy kernels, unused tuning branches, and intermediate kernel versions from `kernels/` unless they are still required by the final default path.
- If multiple tuning configs remain, ensure they are all semantically correct for supported shapes, or remove/disable the unused ones. Do not leave a known-wrong config available to be selected accidentally.
- Do not submit a kernel containing diagnostic configs that produce no output or copy only, even if they were useful during tuning.
- Cleanup is an edit. After cleanup, run the full final sequence on the exact cleaned files; a successful earlier compile or benchmark does not validate cleaned files.

## 16. Tool Output Interpretation

- Environment warnings such as “Failed to find real location of python.exe” accompanying a successful compile are not compiler errors or CUDA kernel bugs. Use the compile success/failure status and actionable compiler errors.
- If a write/read fails because of a malformed workspace payload or missing `operation`, retry the tool call cleanly. A tool failure by itself is not evidence that the CUDA kernel or binding logic is wrong.
- After any failed or ambiguous write, read the target file before compiling. Do not change CUDA kernel or binding logic solely in response to a tool transport/encoding/payload failure.
- If a reported compile-log path cannot be read, use the compile tool output or rerun compile; do not make unrelated kernel changes based on a log-path failure.

**Scope:** Sections 17–20 apply to CUDA extension episodes using this workspace and do not relax the existing forbidden/allowed lists or validation rules. Section 20 applies only to tool/workspace operations and does not classify tool transport/encoding/path failures as CUDA coding errors.

## 17. Validation Order Is Non-Negotiable

- Compile success, verification success, and benchmark success are separate gates. A successful compile does not validate kernel math or wrapper semantics, and a benchmark without preceding verification on the same files is not evidence of correctness.
- After any edit to `kernels/*`, binding files, or `model_new.py`, use this order: finish edits/cleanup → compile if any C++/CUDA/binding file changed → verification → profiling/benchmark. The final benchmark must run on the exact files that passed verification.
- Do not make a late `model_new.py` config or forward-path edit and stop after a benchmark-only run. If the edit selects a different config, dispatch branch, or wrapper path, rerun verification on that exact file, then rerun the benchmark.
- A benchmark-only run may be used only as a tuning probe. Preserve the last verified selection and either verify the final selected file or revert to the verified one.
- Do not end an episode on a write, a compile-only step, or a benchmark-only step after a change. If step budget is nearly exhausted, do not start another speculative optimization; leave or restore the last fully validated files.
- A write after the last verification invalidates that verification. A compile after the last verification still leaves verification stale; recompile, verify, and benchmark before completion.
- If a benchmark-only run is followed by another edit, the earlier benchmark is not final evidence for the new file.

## 18. Reject Unverified Speedups

- If verification fails after a kernel, binding, or wrapper change, reject that candidate even if its benchmark is faster. Fix the failing path or revert to the last verified implementation, then recompile, verify, and benchmark.
- Do not continue tuning `model_new.py` configs while the underlying kernel is failing verification. Correctness must be restored first.
- When a rewrite breaks correctness, use the last passing kernel as the baseline and prefer a minimal fix over another broad rewrite. Keep a known-good copy until the replacement passes verification.
- A config that is fast for one shape but unverified or incorrect for another harness case must not remain selected in final `model_new.py`.
- Do not leave a faster but unverified or known-wrong config, dispatch branch, or wrapper path reachable from the final model path.

## 19. Probe and Debug Artifact Hygiene

- Temporary probe/diagnostic kernels, alternate bindings, and patch/debug files must not be part of the final module path. If they are added under `kernels/` or referenced by a binding, remove them and rerun compile, verification, and benchmark on the cleaned files.
- Do not use a probe benchmark as evidence that the final implementation is correct.
- Do not leave temporary patch files, probe launchers, or notes in `kernels/`; cleanup is an edit and must precede the final validation sequence.
- After deleting a probe file or reverting a binding, recompile so the extension cannot retain stale symbols or an unintended old path.

## 20. Tool Log Path Failures Are Not CUDA Signals

Scope: tool/workspace operations only; these rules do not classify tool transport/encoding/path failures as CUDA coding mistakes.

- A `FileNotFoundError` or unreadable path for a compile, verification, or profiling log is a tool/path issue. Use the tool’s returned output/status or rerun the tool; do not change kernel math, binding allocation, or `model_new.py` based on the failed read alone.
- If a log path cannot be read, do not infer pass/fail from the missing file. Rerun verification/benchmark or use the tool result already returned.
- After a failed or ambiguous workspace read/write, read the target file before compiling or editing further. A failed tool call is not evidence that the file is correct or incorrect.
- Environment warnings such as failure to locate `python.exe` alongside an explicit compile/verify/benchmark success are not CUDA kernel errors; use the explicit status and actionable compiler/runtime errors.

**Scope:** Sections 21–24 apply to CUDA extension episodes using this workspace. They do not relax the existing forbidden/allowed lists or validation requirements. Section 24 applies only to tool/workspace operations and does not classify tool payload or environment failures as CUDA coding mistakes.

## 21. Vectorized Reduction Semantics and Debug Isolation

- When a vector load (`float2`, `float4`) spans the non-reduced axis, each component belongs to a different output column and must have its own reduction accumulator and its own reciprocal normalization value. Do not sum the components into one scalar, and do not broadcast one scalar result across the vector lanes.
- Store and combine vector partials as vector/per-lane values in shared memory (for example, `float4` accumulators), not as one scalar per thread. Apply the same per-lane requirement when scaling and storing the output.
- A vectorized reduction failure with most elements mismatched and the same wrong values across different block shapes is evidence to inspect lane packing/reduction semantics first, not automatically a race or launch-config problem. Isolate it by testing the scalar path, a vectorized copy/indexing path without the reduction, and then the vectorized reduction.
- Keep diagnostic configs that skip reduction or copy data out of the final path. After fixing the vector path, verify both vector-eligible shapes and non-vector fallback shapes.

## 22. Forward-Path Diagnostics and Post-Tuning Validation

- Do not add PyTorch reference calculations (`pow`, `mean`, `sqrt`, etc.) or other debug tensor ops to `model_new.py`/`ModelNew.forward` to diagnose a kernel. The verifier rejects PyTorch computation during candidate forward; that failure is an instrumentation violation, not evidence that the CUDA math is wrong.
- Perform any reference comparison outside the submitted forward path, and remove the diagnostic before final validation. Do not leave debug instrumentation in `model_new.py`.
- A benchmark-only sweep after changing `model_new.py`—including a hardcoded `config`, dispatch branch, or wrapper argument—is a tuning probe. Before treating that selection as final, run verification on the exact final file, then benchmark. A passing verification of an earlier/default config does not validate the new selection.
- Do not treat compile success as correctness validation. If a new candidate is compiled, run verification before using its benchmark as final evidence. If verification fails, reject the faster candidate or revert to the last verified version.

## 23. Clock-State-Aware Benchmark Comparison

- Absolute candidate latencies in this environment can change by roughly 2× between runs as GPU clock state changes. Do not rank tuning variants solely by absolute microseconds measured in separate runs.
- Compare candidate and `torch.compile` baseline in the same benchmark invocation, or repeat/interleave measurements and report the candidate/baseline ratio. A result is more trustworthy when the ratio is stable across clock states.
- Do not use a fast but unverified benchmark under one clock state to justify leaving a different final configuration. Final evidence must come from the exact final files after the required compile → verify → benchmark order.

## 24. Workspace Write Payloads (Tool/Workspace Only)

- For workspace writes, pass `operation: "write"`, `path`, and `content` as separate fields. Do not omit `operation` or put file content/Markdown in the `operation` field.
- A `ValueError: operation must be one of ...` or a missing-`operation` error is a tool payload problem, not a CUDA kernel or binding bug. Retry the write/read correctly; do not change kernel math or allocation logic in response to that error alone.
- After a failed or ambiguous write, read the target file before compiling or making further edits. A failed tool call does not prove the file is correct or incorrect.
- Environment warnings such as “Failed to find real location of python.exe” alongside an explicit compile/verify success are not CUDA errors; use the explicit status and actionable compiler/runtime output.

**Scope:** Sections 25–28 apply to CUDA extension episodes using this workspace. They do not relax Sections 1–24 or the existing forbidden/allowed lists. Section 28 applies only to tool/workspace operations and explicitly does not classify tool transport, encoding, payload, or path failures as CUDA coding mistakes.

## 25. Output Tensor Lifetime and Buffer-Reuse Safety

- Treat “previously returned outputs must remain unchanged after later calls” as a hard lifetime constraint. Never write into, recycle, or reuse storage that may still be referenced by any earlier returned tensor. The verifier/harness may retain outputs across repeated calls.
- For AXPBY-like elementwise ops that return a new tensor, default to the allowed per-call `torch::empty_like(...)` and rely on PyTorch’s allocator for caching. Do not add a custom cross-call output pool merely to reduce allocation overhead unless you can prove with public APIs that no live tensor references the buffer.
- Do not use `TensorImpl::use_count`, `Tensor::use_count`, `storage().use_count`, intrusive-pointer counts, or other refcount internals to decide whether an output buffer is recyclable. Such APIs are unstable/version-dependent, and an observed build failed with `"use_count": is not a member of "c10::TensorImpl"`. A single refcount is also not a contract-safe proof against live Python/harness references.
- If an attempted refcount API does not compile, remove that pooling attempt or use per-call allocation. Do not add guessed headers, private ATen/TensorImpl APIs, or unrelated kernel changes to force it to compile.
- A growing pool is a failure signal. If diagnostic output shows the pool size increasing monotonically, or benchmark latency degrades over repeated calls, the buffers are not safely recyclable. Remove the pool and return to per-call allocation. Do not report a faster benchmark from a pool that was not proven to preserve all previous outputs.
- If you experiment with a pre-reserved chunk or suballocation, keep size/offset metadata on the host. Do not write metadata to device memory through a host pointer; that is a host/device memory bug, not an allocation-tuning detail. Ensure each live output has distinct usable storage or an ownership scheme that cannot overwrite a live result.
- Never alias or overwrite input tensors `a`/`b` to save allocation. Compute into a distinct output. When using typed pointers, validate dtype, device, contiguity, and shape assumptions; if buffer reuse is attempted, match all relevant tensor properties, not just shape.

## 26. No Probe Code in Final Production Paths

- A verification PASS does not mean the binding is clean. Probe output can run during verification and benchmark while tests still pass. Before finalizing, inspect `kernels/*.cu`, `kernels/*_binding.cpp`, and `model_new.py` for diagnostic `printf`/`std::printf`, counters, timing probes, pool-inspection logs, and comments such as `PROBE ONLY`.
- Remove diagnostic code embedded in otherwise final files, not only separate `probe_*.cpp` files. Do not leave global probe state, pool-size printing, or debug-only branches reachable from `model_new.py`.
- After removing probe code, recompile if any C++/CUDA/binding file changed, then run verification and benchmark on the exact cleaned files. Verification or benchmark output from the probe version is not evidence for the cleaned version.
- Temporary probe files must be deleted before final validation. If a probe-specific symbol remains declared or referenced, recompile so the extension cannot retain a stale or unintended path.

## 27. Finish on Validated Files and Submit the Report

- Do not end an episode after a verify-only step when the performance gate has not been measured on the exact final files. The final sequence is: finish edits/cleanup → compile if needed → verification → benchmark on the same files → submit the final report.
- Reserve enough budget for that final sequence. If the remaining budget is small, stop optional pooling/refcount experiments and preserve or restore the last candidate that passed compile, verification, and benchmark. Do not start a speculative optimization that may leave the workspace unvalidated.
- A benchmark from an earlier candidate is not final if later writes changed `model_new.py`, binding files, or kernels. Report only measurements actually obtained from the final validated files; if the final files were not benchmarked, say so and treat the performance goal as unconfirmed.
- If a late buffer-pool or binding experiment fails compile, fails verification, or shows degradation, revert to the known-good per-call allocation version and rerun the required gates rather than leaving the experimental binding in place.

## 28. Tool/Workspace Failure Separation (Scoped)

- A missing `operation` field or `ValueError: operation must be one of ...` is a tooling payload error. Retry the write/read with `operation`, `path`, and `content` as separate fields; do not change kernel math, output pooling, or binding allocation in response to that error alone.
- After any failed or ambiguous write, read the target file before compiling. A failed tool call does not prove the file is unchanged or correct.
- Environment warnings such as failure to locate `python.exe` alongside an explicit compile/verify success are not CUDA errors; use the explicit status and actionable compiler/runtime output. A log-path, encoding, transport, or payload failure is not evidence that the CUDA implementation is wrong.


## Repeated appendix, copy 3

**Scope of appended rules:** Sections 9–12 apply to CUDA extension episodes using this workspace. Section 12 applies only to tool/workspace operations and explicitly does not classify tool transport/encoding failures as CUDA coding errors. None of these rules relax the existing forbidden/allowed lists.

## 9. Compile-Safe Binding and Allocation

- Use only the public allocation exception already allowed by this document. For bindings, prefer the documented `torch::empty_like(...)` allocation path.
- Do not use `at::detail::empty_cuda`, other `at::detail::*` symbols, or version-private ATen internals to bypass the dispatcher. These are not stable extension APIs.
- If compilation reports an error such as `'empty_cuda': is not a member of 'at::detail'`, replace the internal allocation call with the documented allocation path. Do not keep the internal call or add guessed headers to force it to compile.
- Fix binding compile errors in the binding files, not by changing kernel math. A missing binding symbol or undeclared allocation API is not evidence that the CUDA kernel algorithm is wrong.
- Do not replace a compiling binding with a speculative “lean hot path” using internal APIs. Allocation micro-optimizations are lower priority than compile success and correctness; if attempted, isolate the change and keep a known-good fallback.
- When a binding specializes to float32/contiguous CUDA tensors, enforce that specialization with `TORCH_CHECK`/equivalent validation or ensure `model_new.py` guarantees it. Do not reinterpret arbitrary tensor storage as `float*`.

## 10. Final Validation Gate

- The final submitted files must be the exact files that passed compile, verification, and benchmark. Any write after the last successful verification/benchmark invalidates those results.
- Final sequence: finish cleanup and final edits, then compile, then run verification, then run profiling/benchmark. Do not stop immediately after a write.
- Do not perform a post-benchmark speculative rewrite of `kernels/*` or `model_new.py` and declare completion. If optimizing after a passing result, preserve or restore the known-good version; re-run compile, verification, and benchmark; if the new version fails, revert and leave the known-good version as final.
- A successful earlier compile does not excuse a later compile failure. Do not submit an uncompiled final candidate.
- Final cleanup counts as an edit. Run compile, verification, and benchmark after cleanup, or perform cleanup before the final validation sequence.
- On compile failure, use the first actionable compiler error. If the reported compile log path cannot be read, use the compile tool output or rerun compile; do not make unrelated kernel changes based on a log-path failure.

## 11. ModelNew and Wrapper Semantics

- Keep `ModelNew.forward` as the normal method and let the harness use `nn.Module.__call__`.
- Do not add `__call__ = forward` or otherwise bypass `nn.Module` hook machinery as a speculative shortcut. It can change harness/hook behavior. Use such a bypass only if explicitly required and revalidated; otherwise remove it before final validation.
- Preserve the constructor and `forward` signatures expected by `model.py`.
- Keep `model_new.py` limited to allowed tensor creation and custom extension ops. Do not add unsupported Python-side torch operations to compensate for binding or kernel issues.

## 12. Tool/Workspace Hygiene (Not CUDA Coding Errors)

- For `workspace` writes, pass `operation: "write"`, `path`, and `content` as separate fields. Do not put file content or Markdown into the `operation` field.
- Treat `ValueError: unknown operation`, missing `operation` errors, malformed tool payloads, and log encoding/mojibake as tooling issues, not CUDA coding mistakes.
- After any failed or ambiguous write, read the target file before compiling. A failed tool call does not prove the file is unchanged or correct.
- If a write/read fails due to tool transport or encoding, retry the tool call cleanly. Do not change CUDA kernel or binding logic solely in response to the tool failure.
- A missing utility file or unreadable path is a workspace/file-availability issue, not evidence that the kernel is incorrect. Use the documented utilities or list the directory to discover the actual files.

**Scope:** Sections 13–16 apply to CUDA extension episodes using this workspace. They do not relax Sections 1–12 or the existing forbidden/allowed lists. Section 16 applies only to tool/workspace operations and does not classify tool transport/encoding failures as CUDA coding errors.

## 13. Post-Tuning Validation State

- Treat `model_new.py` as part of the correctness surface. Changing the selected kernel `config`, wrapper arguments, or forward path can change numerical behavior; a benchmark-only run after such a change does not validate it.
- After any candidate change to `kernels/*`, bindings, or `model_new.py`, run verification before using or reporting benchmark results. A fast benchmark from an unverified candidate is only a performance probe.
- The final validation sequence must run on the exact final files: finish all edits and cleanup, compile if any C++/CUDA/binding file changed, run verification, then run benchmark. Any write after verification makes that verification stale.
- If a compile and a later write occur in the same step, do not count that compile as final validation for the later file. Ensure the last compile precedes the verified final state, or explicitly confirm no compiled file changed and keep the binary consistent with the verified files.
- Do not end an episode immediately after a benchmark-only edit. If the step budget is nearly exhausted, preserve or restore the last fully validated version rather than making an unverified last-moment change.
- If verification fails after a tuning change, reject the faster benchmark. Either fix the kernel/config and rerun compile, verification, and benchmark, or revert to the last verified version.

## 14. Tuned Config and Shape-Specialized Path Safety

- Do not hardcode a tuned `config` in `model_new.py` solely because it benchmarked fastest. Verify that exact config on all verification cases and seeds before leaving it as the final selection.
- For shape-specialized dispatch, ensure every non-default config either has a correct fallback or is unreachable from the final `model_new.py`. A fast path that is wrong for one harness shape must not remain selected.
- When a config-specific case fails, first inspect dispatch guards, supported `F`/`N` assumptions, and fallback coverage. Do not change unrelated kernel math or move the failure into Python.
- Auto/default passing does not validate a hardcoded config. After switching `model_new.py` from auto/default to a fixed config, rerun verification on that exact final config.
- Diagnostic, no-op, copy-only, or debug configs must not be reachable in the final model path.

## 15. Final Cleanup Specifics

- Before final validation, remove diagnostic no-op/copy kernels, unused tuning branches, and intermediate kernel versions from `kernels/` unless they are still required by the final default path.
- If multiple tuning configs remain, ensure they are all semantically correct for supported shapes, or remove/disable the unused ones. Do not leave a known-wrong config available to be selected accidentally.
- Do not submit a kernel containing diagnostic configs that produce no output or copy only, even if they were useful during tuning.
- Cleanup is an edit. After cleanup, run the full final sequence on the exact cleaned files; a successful earlier compile or benchmark does not validate cleaned files.

## 16. Tool Output Interpretation

- Environment warnings such as “Failed to find real location of python.exe” accompanying a successful compile are not compiler errors or CUDA kernel bugs. Use the compile success/failure status and actionable compiler errors.
- If a write/read fails because of a malformed workspace payload or missing `operation`, retry the tool call cleanly. A tool failure by itself is not evidence that the CUDA kernel or binding logic is wrong.
- After any failed or ambiguous write, read the target file before compiling. Do not change CUDA kernel or binding logic solely in response to a tool transport/encoding/payload failure.
- If a reported compile-log path cannot be read, use the compile tool output or rerun compile; do not make unrelated kernel changes based on a log-path failure.

**Scope:** Sections 17–20 apply to CUDA extension episodes using this workspace and do not relax the existing forbidden/allowed lists or validation rules. Section 20 applies only to tool/workspace operations and does not classify tool transport/encoding/path failures as CUDA coding errors.

## 17. Validation Order Is Non-Negotiable

- Compile success, verification success, and benchmark success are separate gates. A successful compile does not validate kernel math or wrapper semantics, and a benchmark without preceding verification on the same files is not evidence of correctness.
- After any edit to `kernels/*`, binding files, or `model_new.py`, use this order: finish edits/cleanup → compile if any C++/CUDA/binding file changed → verification → profiling/benchmark. The final benchmark must run on the exact files that passed verification.
- Do not make a late `model_new.py` config or forward-path edit and stop after a benchmark-only run. If the edit selects a different config, dispatch branch, or wrapper path, rerun verification on that exact file, then rerun the benchmark.
- A benchmark-only run may be used only as a tuning probe. Preserve the last verified selection and either verify the final selected file or revert to the verified one.
- Do not end an episode on a write, a compile-only step, or a benchmark-only step after a change. If step budget is nearly exhausted, do not start another speculative optimization; leave or restore the last fully validated files.
- A write after the last verification invalidates that verification. A compile after the last verification still leaves verification stale; recompile, verify, and benchmark before completion.
- If a benchmark-only run is followed by another edit, the earlier benchmark is not final evidence for the new file.

## 18. Reject Unverified Speedups

- If verification fails after a kernel, binding, or wrapper change, reject that candidate even if its benchmark is faster. Fix the failing path or revert to the last verified implementation, then recompile, verify, and benchmark.
- Do not continue tuning `model_new.py` configs while the underlying kernel is failing verification. Correctness must be restored first.
- When a rewrite breaks correctness, use the last passing kernel as the baseline and prefer a minimal fix over another broad rewrite. Keep a known-good copy until the replacement passes verification.
- A config that is fast for one shape but unverified or incorrect for another harness case must not remain selected in final `model_new.py`.
- Do not leave a faster but unverified or known-wrong config, dispatch branch, or wrapper path reachable from the final model path.

## 19. Probe and Debug Artifact Hygiene

- Temporary probe/diagnostic kernels, alternate bindings, and patch/debug files must not be part of the final module path. If they are added under `kernels/` or referenced by a binding, remove them and rerun compile, verification, and benchmark on the cleaned files.
- Do not use a probe benchmark as evidence that the final implementation is correct.
- Do not leave temporary patch files, probe launchers, or notes in `kernels/`; cleanup is an edit and must precede the final validation sequence.
- After deleting a probe file or reverting a binding, recompile so the extension cannot retain stale symbols or an unintended old path.

## 20. Tool Log Path Failures Are Not CUDA Signals

Scope: tool/workspace operations only; these rules do not classify tool transport/encoding/path failures as CUDA coding mistakes.

- A `FileNotFoundError` or unreadable path for a compile, verification, or profiling log is a tool/path issue. Use the tool’s returned output/status or rerun the tool; do not change kernel math, binding allocation, or `model_new.py` based on the failed read alone.
- If a log path cannot be read, do not infer pass/fail from the missing file. Rerun verification/benchmark or use the tool result already returned.
- After a failed or ambiguous workspace read/write, read the target file before compiling or editing further. A failed tool call is not evidence that the file is correct or incorrect.
- Environment warnings such as failure to locate `python.exe` alongside an explicit compile/verify/benchmark success are not CUDA kernel errors; use the explicit status and actionable compiler/runtime errors.

**Scope:** Sections 21–24 apply to CUDA extension episodes using this workspace. They do not relax the existing forbidden/allowed lists or validation requirements. Section 24 applies only to tool/workspace operations and does not classify tool payload or environment failures as CUDA coding mistakes.

## 21. Vectorized Reduction Semantics and Debug Isolation

- When a vector load (`float2`, `float4`) spans the non-reduced axis, each component belongs to a different output column and must have its own reduction accumulator and its own reciprocal normalization value. Do not sum the components into one scalar, and do not broadcast one scalar result across the vector lanes.
- Store and combine vector partials as vector/per-lane values in shared memory (for example, `float4` accumulators), not as one scalar per thread. Apply the same per-lane requirement when scaling and storing the output.
- A vectorized reduction failure with most elements mismatched and the same wrong values across different block shapes is evidence to inspect lane packing/reduction semantics first, not automatically a race or launch-config problem. Isolate it by testing the scalar path, a vectorized copy/indexing path without the reduction, and then the vectorized reduction.
- Keep diagnostic configs that skip reduction or copy data out of the final path. After fixing the vector path, verify both vector-eligible shapes and non-vector fallback shapes.

## 22. Forward-Path Diagnostics and Post-Tuning Validation

- Do not add PyTorch reference calculations (`pow`, `mean`, `sqrt`, etc.) or other debug tensor ops to `model_new.py`/`ModelNew.forward` to diagnose a kernel. The verifier rejects PyTorch computation during candidate forward; that failure is an instrumentation violation, not evidence that the CUDA math is wrong.
- Perform any reference comparison outside the submitted forward path, and remove the diagnostic before final validation. Do not leave debug instrumentation in `model_new.py`.
- A benchmark-only sweep after changing `model_new.py`—including a hardcoded `config`, dispatch branch, or wrapper argument—is a tuning probe. Before treating that selection as final, run verification on the exact final file, then benchmark. A passing verification of an earlier/default config does not validate the new selection.
- Do not treat compile success as correctness validation. If a new candidate is compiled, run verification before using its benchmark as final evidence. If verification fails, reject the faster candidate or revert to the last verified version.

## 23. Clock-State-Aware Benchmark Comparison

- Absolute candidate latencies in this environment can change by roughly 2× between runs as GPU clock state changes. Do not rank tuning variants solely by absolute microseconds measured in separate runs.
- Compare candidate and `torch.compile` baseline in the same benchmark invocation, or repeat/interleave measurements and report the candidate/baseline ratio. A result is more trustworthy when the ratio is stable across clock states.
- Do not use a fast but unverified benchmark under one clock state to justify leaving a different final configuration. Final evidence must come from the exact final files after the required compile → verify → benchmark order.

## 24. Workspace Write Payloads (Tool/Workspace Only)

- For workspace writes, pass `operation: "write"`, `path`, and `content` as separate fields. Do not omit `operation` or put file content/Markdown in the `operation` field.
- A `ValueError: operation must be one of ...` or a missing-`operation` error is a tool payload problem, not a CUDA kernel or binding bug. Retry the write/read correctly; do not change kernel math or allocation logic in response to that error alone.
- After a failed or ambiguous write, read the target file before compiling or making further edits. A failed tool call does not prove the file is correct or incorrect.
- Environment warnings such as “Failed to find real location of python.exe” alongside an explicit compile/verify success are not CUDA errors; use the explicit status and actionable compiler/runtime output.

**Scope:** Sections 25–28 apply to CUDA extension episodes using this workspace. They do not relax Sections 1–24 or the existing forbidden/allowed lists. Section 28 applies only to tool/workspace operations and explicitly does not classify tool transport, encoding, payload, or path failures as CUDA coding mistakes.

## 25. Output Tensor Lifetime and Buffer-Reuse Safety

- Treat “previously returned outputs must remain unchanged after later calls” as a hard lifetime constraint. Never write into, recycle, or reuse storage that may still be referenced by any earlier returned tensor. The verifier/harness may retain outputs across repeated calls.
- For AXPBY-like elementwise ops that return a new tensor, default to the allowed per-call `torch::empty_like(...)` and rely on PyTorch’s allocator for caching. Do not add a custom cross-call output pool merely to reduce allocation overhead unless you can prove with public APIs that no live tensor references the buffer.
- Do not use `TensorImpl::use_count`, `Tensor::use_count`, `storage().use_count`, intrusive-pointer counts, or other refcount internals to decide whether an output buffer is recyclable. Such APIs are unstable/version-dependent, and an observed build failed with `"use_count": is not a member of "c10::TensorImpl"`. A single refcount is also not a contract-safe proof against live Python/harness references.
- If an attempted refcount API does not compile, remove that pooling attempt or use per-call allocation. Do not add guessed headers, private ATen/TensorImpl APIs, or unrelated kernel changes to force it to compile.
- A growing pool is a failure signal. If diagnostic output shows the pool size increasing monotonically, or benchmark latency degrades over repeated calls, the buffers are not safely recyclable. Remove the pool and return to per-call allocation. Do not report a faster benchmark from a pool that was not proven to preserve all previous outputs.
- If you experiment with a pre-reserved chunk or suballocation, keep size/offset metadata on the host. Do not write metadata to device memory through a host pointer; that is a host/device memory bug, not an allocation-tuning detail. Ensure each live output has distinct usable storage or an ownership scheme that cannot overwrite a live result.
- Never alias or overwrite input tensors `a`/`b` to save allocation. Compute into a distinct output. When using typed pointers, validate dtype, device, contiguity, and shape assumptions; if buffer reuse is attempted, match all relevant tensor properties, not just shape.

## 26. No Probe Code in Final Production Paths

- A verification PASS does not mean the binding is clean. Probe output can run during verification and benchmark while tests still pass. Before finalizing, inspect `kernels/*.cu`, `kernels/*_binding.cpp`, and `model_new.py` for diagnostic `printf`/`std::printf`, counters, timing probes, pool-inspection logs, and comments such as `PROBE ONLY`.
- Remove diagnostic code embedded in otherwise final files, not only separate `probe_*.cpp` files. Do not leave global probe state, pool-size printing, or debug-only branches reachable from `model_new.py`.
- After removing probe code, recompile if any C++/CUDA/binding file changed, then run verification and benchmark on the exact cleaned files. Verification or benchmark output from the probe version is not evidence for the cleaned version.
- Temporary probe files must be deleted before final validation. If a probe-specific symbol remains declared or referenced, recompile so the extension cannot retain a stale or unintended path.

## 27. Finish on Validated Files and Submit the Report

- Do not end an episode after a verify-only step when the performance gate has not been measured on the exact final files. The final sequence is: finish edits/cleanup → compile if needed → verification → benchmark on the same files → submit the final report.
- Reserve enough budget for that final sequence. If the remaining budget is small, stop optional pooling/refcount experiments and preserve or restore the last candidate that passed compile, verification, and benchmark. Do not start a speculative optimization that may leave the workspace unvalidated.
- A benchmark from an earlier candidate is not final if later writes changed `model_new.py`, binding files, or kernels. Report only measurements actually obtained from the final validated files; if the final files were not benchmarked, say so and treat the performance goal as unconfirmed.
- If a late buffer-pool or binding experiment fails compile, fails verification, or shows degradation, revert to the known-good per-call allocation version and rerun the required gates rather than leaving the experimental binding in place.

## 28. Tool/Workspace Failure Separation (Scoped)

- A missing `operation` field or `ValueError: operation must be one of ...` is a tooling payload error. Retry the write/read with `operation`, `path`, and `content` as separate fields; do not change kernel math, output pooling, or binding allocation in response to that error alone.
- After any failed or ambiguous write, read the target file before compiling. A failed tool call does not prove the file is unchanged or correct.
- Environment warnings such as failure to locate `python.exe` alongside an explicit compile/verify success are not CUDA errors; use the explicit status and actionable compiler/runtime output. A log-path, encoding, transport, or payload failure is not evidence that the CUDA implementation is wrong.


## Repeated appendix, copy 4

**Scope of appended rules:** Sections 9–12 apply to CUDA extension episodes using this workspace. Section 12 applies only to tool/workspace operations and explicitly does not classify tool transport/encoding failures as CUDA coding errors. None of these rules relax the existing forbidden/allowed lists.

## 9. Compile-Safe Binding and Allocation

- Use only the public allocation exception already allowed by this document. For bindings, prefer the documented `torch::empty_like(...)` allocation path.
- Do not use `at::detail::empty_cuda`, other `at::detail::*` symbols, or version-private ATen internals to bypass the dispatcher. These are not stable extension APIs.
- If compilation reports an error such as `'empty_cuda': is not a member of 'at::detail'`, replace the internal allocation call with the documented allocation path. Do not keep the internal call or add guessed headers to force it to compile.
- Fix binding compile errors in the binding files, not by changing kernel math. A missing binding symbol or undeclared allocation API is not evidence that the CUDA kernel algorithm is wrong.
- Do not replace a compiling binding with a speculative “lean hot path” using internal APIs. Allocation micro-optimizations are lower priority than compile success and correctness; if attempted, isolate the change and keep a known-good fallback.
- When a binding specializes to float32/contiguous CUDA tensors, enforce that specialization with `TORCH_CHECK`/equivalent validation or ensure `model_new.py` guarantees it. Do not reinterpret arbitrary tensor storage as `float*`.

## 10. Final Validation Gate

- The final submitted files must be the exact files that passed compile, verification, and benchmark. Any write after the last successful verification/benchmark invalidates those results.
- Final sequence: finish cleanup and final edits, then compile, then run verification, then run profiling/benchmark. Do not stop immediately after a write.
- Do not perform a post-benchmark speculative rewrite of `kernels/*` or `model_new.py` and declare completion. If optimizing after a passing result, preserve or restore the known-good version; re-run compile, verification, and benchmark; if the new version fails, revert and leave the known-good version as final.
- A successful earlier compile does not excuse a later compile failure. Do not submit an uncompiled final candidate.
- Final cleanup counts as an edit. Run compile, verification, and benchmark after cleanup, or perform cleanup before the final validation sequence.
- On compile failure, use the first actionable compiler error. If the reported compile log path cannot be read, use the compile tool output or rerun compile; do not make unrelated kernel changes based on a log-path failure.

## 11. ModelNew and Wrapper Semantics

- Keep `ModelNew.forward` as the normal method and let the harness use `nn.Module.__call__`.
- Do not add `__call__ = forward` or otherwise bypass `nn.Module` hook machinery as a speculative shortcut. It can change harness/hook behavior. Use such a bypass only if explicitly required and revalidated; otherwise remove it before final validation.
- Preserve the constructor and `forward` signatures expected by `model.py`.
- Keep `model_new.py` limited to allowed tensor creation and custom extension ops. Do not add unsupported Python-side torch operations to compensate for binding or kernel issues.

## 12. Tool/Workspace Hygiene (Not CUDA Coding Errors)

- For `workspace` writes, pass `operation: "write"`, `path`, and `content` as separate fields. Do not put file content or Markdown into the `operation` field.
- Treat `ValueError: unknown operation`, missing `operation` errors, malformed tool payloads, and log encoding/mojibake as tooling issues, not CUDA coding mistakes.
- After any failed or ambiguous write, read the target file before compiling. A failed tool call does not prove the file is unchanged or correct.
- If a write/read fails due to tool transport or encoding, retry the tool call cleanly. Do not change CUDA kernel or binding logic solely in response to the tool failure.
- A missing utility file or unreadable path is a workspace/file-availability issue, not evidence that the kernel is incorrect. Use the documented utilities or list the directory to discover the actual files.

**Scope:** Sections 13–16 apply to CUDA extension episodes using this workspace. They do not relax Sections 1–12 or the existing forbidden/allowed lists. Section 16 applies only to tool/workspace operations and does not classify tool transport/encoding failures as CUDA coding errors.

## 13. Post-Tuning Validation State

- Treat `model_new.py` as part of the correctness surface. Changing the selected kernel `config`, wrapper arguments, or forward path can change numerical behavior; a benchmark-only run after such a change does not validate it.
- After any candidate change to `kernels/*`, bindings, or `model_new.py`, run verification before using or reporting benchmark results. A fast benchmark from an unverified candidate is only a performance probe.
- The final validation sequence must run on the exact final files: finish all edits and cleanup, compile if any C++/CUDA/binding file changed, run verification, then run benchmark. Any write after verification makes that verification stale.
- If a compile and a later write occur in the same step, do not count that compile as final validation for the later file. Ensure the last compile precedes the verified final state, or explicitly confirm no compiled file changed and keep the binary consistent with the verified files.
- Do not end an episode immediately after a benchmark-only edit. If the step budget is nearly exhausted, preserve or restore the last fully validated version rather than making an unverified last-moment change.
- If verification fails after a tuning change, reject the faster benchmark. Either fix the kernel/config and rerun compile, verification, and benchmark, or revert to the last verified version.

## 14. Tuned Config and Shape-Specialized Path Safety

- Do not hardcode a tuned `config` in `model_new.py` solely because it benchmarked fastest. Verify that exact config on all verification cases and seeds before leaving it as the final selection.
- For shape-specialized dispatch, ensure every non-default config either has a correct fallback or is unreachable from the final `model_new.py`. A fast path that is wrong for one harness shape must not remain selected.
- When a config-specific case fails, first inspect dispatch guards, supported `F`/`N` assumptions, and fallback coverage. Do not change unrelated kernel math or move the failure into Python.
- Auto/default passing does not validate a hardcoded config. After switching `model_new.py` from auto/default to a fixed config, rerun verification on that exact final config.
- Diagnostic, no-op, copy-only, or debug configs must not be reachable in the final model path.

## 15. Final Cleanup Specifics

- Before final validation, remove diagnostic no-op/copy kernels, unused tuning branches, and intermediate kernel versions from `kernels/` unless they are still required by the final default path.
- If multiple tuning configs remain, ensure they are all semantically correct for supported shapes, or remove/disable the unused ones. Do not leave a known-wrong config available to be selected accidentally.
- Do not submit a kernel containing diagnostic configs that produce no output or copy only, even if they were useful during tuning.
- Cleanup is an edit. After cleanup, run the full final sequence on the exact cleaned files; a successful earlier compile or benchmark does not validate cleaned files.

## 16. Tool Output Interpretation

- Environment warnings such as “Failed to find real location of python.exe” accompanying a successful compile are not compiler errors or CUDA kernel bugs. Use the compile success/failure status and actionable compiler errors.
- If a write/read fails because of a malformed workspace payload or missing `operation`, retry the tool call cleanly. A tool failure by itself is not evidence that the CUDA kernel or binding logic is wrong.
- After any failed or ambiguous write, read the target file before compiling. Do not change CUDA kernel or binding logic solely in response to a tool transport/encoding/payload failure.
- If a reported compile-log path cannot be read, use the compile tool output or rerun compile; do not make unrelated kernel changes based on a log-path failure.

**Scope:** Sections 17–20 apply to CUDA extension episodes using this workspace and do not relax the existing forbidden/allowed lists or validation rules. Section 20 applies only to tool/workspace operations and does not classify tool transport/encoding/path failures as CUDA coding errors.

## 17. Validation Order Is Non-Negotiable

- Compile success, verification success, and benchmark success are separate gates. A successful compile does not validate kernel math or wrapper semantics, and a benchmark without preceding verification on the same files is not evidence of correctness.
- After any edit to `kernels/*`, binding files, or `model_new.py`, use this order: finish edits/cleanup → compile if any C++/CUDA/binding file changed → verification → profiling/benchmark. The final benchmark must run on the exact files that passed verification.
- Do not make a late `model_new.py` config or forward-path edit and stop after a benchmark-only run. If the edit selects a different config, dispatch branch, or wrapper path, rerun verification on that exact file, then rerun the benchmark.
- A benchmark-only run may be used only as a tuning probe. Preserve the last verified selection and either verify the final selected file or revert to the verified one.
- Do not end an episode on a write, a compile-only step, or a benchmark-only step after a change. If step budget is nearly exhausted, do not start another speculative optimization; leave or restore the last fully validated files.
- A write after the last verification invalidates that verification. A compile after the last verification still leaves verification stale; recompile, verify, and benchmark before completion.
- If a benchmark-only run is followed by another edit, the earlier benchmark is not final evidence for the new file.

## 18. Reject Unverified Speedups

- If verification fails after a kernel, binding, or wrapper change, reject that candidate even if its benchmark is faster. Fix the failing path or revert to the last verified implementation, then recompile, verify, and benchmark.
- Do not continue tuning `model_new.py` configs while the underlying kernel is failing verification. Correctness must be restored first.
- When a rewrite breaks correctness, use the last passing kernel as the baseline and prefer a minimal fix over another broad rewrite. Keep a known-good copy until the replacement passes verification.
- A config that is fast for one shape but unverified or incorrect for another harness case must not remain selected in final `model_new.py`.
- Do not leave a faster but unverified or known-wrong config, dispatch branch, or wrapper path reachable from the final model path.

## 19. Probe and Debug Artifact Hygiene

- Temporary probe/diagnostic kernels, alternate bindings, and patch/debug files must not be part of the final module path. If they are added under `kernels/` or referenced by a binding, remove them and rerun compile, verification, and benchmark on the cleaned files.
- Do not use a probe benchmark as evidence that the final implementation is correct.
- Do not leave temporary patch files, probe launchers, or notes in `kernels/`; cleanup is an edit and must precede the final validation sequence.
- After deleting a probe file or reverting a binding, recompile so the extension cannot retain stale symbols or an unintended old path.

## 20. Tool Log Path Failures Are Not CUDA Signals

Scope: tool/workspace operations only; these rules do not classify tool transport/encoding/path failures as CUDA coding mistakes.

- A `FileNotFoundError` or unreadable path for a compile, verification, or profiling log is a tool/path issue. Use the tool’s returned output/status or rerun the tool; do not change kernel math, binding allocation, or `model_new.py` based on the failed read alone.
- If a log path cannot be read, do not infer pass/fail from the missing file. Rerun verification/benchmark or use the tool result already returned.
- After a failed or ambiguous workspace read/write, read the target file before compiling or editing further. A failed tool call is not evidence that the file is correct or incorrect.
- Environment warnings such as failure to locate `python.exe` alongside an explicit compile/verify/benchmark success are not CUDA kernel errors; use the explicit status and actionable compiler/runtime errors.

**Scope:** Sections 21–24 apply to CUDA extension episodes using this workspace. They do not relax the existing forbidden/allowed lists or validation requirements. Section 24 applies only to tool/workspace operations and does not classify tool payload or environment failures as CUDA coding mistakes.

## 21. Vectorized Reduction Semantics and Debug Isolation

- When a vector load (`float2`, `float4`) spans the non-reduced axis, each component belongs to a different output column and must have its own reduction accumulator and its own reciprocal normalization value. Do not sum the components into one scalar, and do not broadcast one scalar result across the vector lanes.
- Store and combine vector partials as vector/per-lane values in shared memory (for example, `float4` accumulators), not as one scalar per thread. Apply the same per-lane requirement when scaling and storing the output.
- A vectorized reduction failure with most elements mismatched and the same wrong values across different block shapes is evidence to inspect lane packing/reduction semantics first, not automatically a race or launch-config problem. Isolate it by testing the scalar path, a vectorized copy/indexing path without the reduction, and then the vectorized reduction.
- Keep diagnostic configs that skip reduction or copy data out of the final path. After fixing the vector path, verify both vector-eligible shapes and non-vector fallback shapes.

## 22. Forward-Path Diagnostics and Post-Tuning Validation

- Do not add PyTorch reference calculations (`pow`, `mean`, `sqrt`, etc.) or other debug tensor ops to `model_new.py`/`ModelNew.forward` to diagnose a kernel. The verifier rejects PyTorch computation during candidate forward; that failure is an instrumentation violation, not evidence that the CUDA math is wrong.
- Perform any reference comparison outside the submitted forward path, and remove the diagnostic before final validation. Do not leave debug instrumentation in `model_new.py`.
- A benchmark-only sweep after changing `model_new.py`—including a hardcoded `config`, dispatch branch, or wrapper argument—is a tuning probe. Before treating that selection as final, run verification on the exact final file, then benchmark. A passing verification of an earlier/default config does not validate the new selection.
- Do not treat compile success as correctness validation. If a new candidate is compiled, run verification before using its benchmark as final evidence. If verification fails, reject the faster candidate or revert to the last verified version.

## 23. Clock-State-Aware Benchmark Comparison

- Absolute candidate latencies in this environment can change by roughly 2× between runs as GPU clock state changes. Do not rank tuning variants solely by absolute microseconds measured in separate runs.
- Compare candidate and `torch.compile` baseline in the same benchmark invocation, or repeat/interleave measurements and report the candidate/baseline ratio. A result is more trustworthy when the ratio is stable across clock states.
- Do not use a fast but unverified benchmark under one clock state to justify leaving a different final configuration. Final evidence must come from the exact final files after the required compile → verify → benchmark order.

## 24. Workspace Write Payloads (Tool/Workspace Only)

- For workspace writes, pass `operation: "write"`, `path`, and `content` as separate fields. Do not omit `operation` or put file content/Markdown in the `operation` field.
- A `ValueError: operation must be one of ...` or a missing-`operation` error is a tool payload problem, not a CUDA kernel or binding bug. Retry the write/read correctly; do not change kernel math or allocation logic in response to that error alone.
- After a failed or ambiguous write, read the target file before compiling or making further edits. A failed tool call does not prove the file is correct or incorrect.
- Environment warnings such as “Failed to find real location of python.exe” alongside an explicit compile/verify success are not CUDA errors; use the explicit status and actionable compiler/runtime output.

**Scope:** Sections 25–28 apply to CUDA extension episodes using this workspace. They do not relax Sections 1–24 or the existing forbidden/allowed lists. Section 28 applies only to tool/workspace operations and explicitly does not classify tool transport, encoding, payload, or path failures as CUDA coding mistakes.

## 25. Output Tensor Lifetime and Buffer-Reuse Safety

- Treat “previously returned outputs must remain unchanged after later calls” as a hard lifetime constraint. Never write into, recycle, or reuse storage that may still be referenced by any earlier returned tensor. The verifier/harness may retain outputs across repeated calls.
- For AXPBY-like elementwise ops that return a new tensor, default to the allowed per-call `torch::empty_like(...)` and rely on PyTorch’s allocator for caching. Do not add a custom cross-call output pool merely to reduce allocation overhead unless you can prove with public APIs that no live tensor references the buffer.
- Do not use `TensorImpl::use_count`, `Tensor::use_count`, `storage().use_count`, intrusive-pointer counts, or other refcount internals to decide whether an output buffer is recyclable. Such APIs are unstable/version-dependent, and an observed build failed with `"use_count": is not a member of "c10::TensorImpl"`. A single refcount is also not a contract-safe proof against live Python/harness references.
- If an attempted refcount API does not compile, remove that pooling attempt or use per-call allocation. Do not add guessed headers, private ATen/TensorImpl APIs, or unrelated kernel changes to force it to compile.
- A growing pool is a failure signal. If diagnostic output shows the pool size increasing monotonically, or benchmark latency degrades over repeated calls, the buffers are not safely recyclable. Remove the pool and return to per-call allocation. Do not report a faster benchmark from a pool that was not proven to preserve all previous outputs.
- If you experiment with a pre-reserved chunk or suballocation, keep size/offset metadata on the host. Do not write metadata to device memory through a host pointer; that is a host/device memory bug, not an allocation-tuning detail. Ensure each live output has distinct usable storage or an ownership scheme that cannot overwrite a live result.
- Never alias or overwrite input tensors `a`/`b` to save allocation. Compute into a distinct output. When using typed pointers, validate dtype, device, contiguity, and shape assumptions; if buffer reuse is attempted, match all relevant tensor properties, not just shape.

## 26. No Probe Code in Final Production Paths

- A verification PASS does not mean the binding is clean. Probe output can run during verification and benchmark while tests still pass. Before finalizing, inspect `kernels/*.cu`, `kernels/*_binding.cpp`, and `model_new.py` for diagnostic `printf`/`std::printf`, counters, timing probes, pool-inspection logs, and comments such as `PROBE ONLY`.
- Remove diagnostic code embedded in otherwise final files, not only separate `probe_*.cpp` files. Do not leave global probe state, pool-size printing, or debug-only branches reachable from `model_new.py`.
- After removing probe code, recompile if any C++/CUDA/binding file changed, then run verification and benchmark on the exact cleaned files. Verification or benchmark output from the probe version is not evidence for the cleaned version.
- Temporary probe files must be deleted before final validation. If a probe-specific symbol remains declared or referenced, recompile so the extension cannot retain a stale or unintended path.

## 27. Finish on Validated Files and Submit the Report

- Do not end an episode after a verify-only step when the performance gate has not been measured on the exact final files. The final sequence is: finish edits/cleanup → compile if needed → verification → benchmark on the same files → submit the final report.
- Reserve enough budget for that final sequence. If the remaining budget is small, stop optional pooling/refcount experiments and preserve or restore the last candidate that passed compile, verification, and benchmark. Do not start a speculative optimization that may leave the workspace unvalidated.
- A benchmark from an earlier candidate is not final if later writes changed `model_new.py`, binding files, or kernels. Report only measurements actually obtained from the final validated files; if the final files were not benchmarked, say so and treat the performance goal as unconfirmed.
- If a late buffer-pool or binding experiment fails compile, fails verification, or shows degradation, revert to the known-good per-call allocation version and rerun the required gates rather than leaving the experimental binding in place.

## 28. Tool/Workspace Failure Separation (Scoped)

- A missing `operation` field or `ValueError: operation must be one of ...` is a tooling payload error. Retry the write/read with `operation`, `path`, and `content` as separate fields; do not change kernel math, output pooling, or binding allocation in response to that error alone.
- After any failed or ambiguous write, read the target file before compiling. A failed tool call does not prove the file is unchanged or correct.
- Environment warnings such as failure to locate `python.exe` alongside an explicit compile/verify success are not CUDA errors; use the explicit status and actionable compiler/runtime output. A log-path, encoding, transport, or payload failure is not evidence that the CUDA implementation is wrong.


## Repeated appendix, copy 5

**Scope of appended rules:** Sections 9–12 apply to CUDA extension episodes using this workspace. Section 12 applies only to tool/workspace operations and explicitly does not classify tool transport/encoding failures as CUDA coding errors. None of these rules relax the existing forbidden/allowed lists.

## 9. Compile-Safe Binding and Allocation

- Use only the public allocation exception already allowed by this document. For bindings, prefer the documented `torch::empty_like(...)` allocation path.
- Do not use `at::detail::empty_cuda`, other `at::detail::*` symbols, or version-private ATen internals to bypass the dispatcher. These are not stable extension APIs.
- If compilation reports an error such as `'empty_cuda': is not a member of 'at::detail'`, replace the internal allocation call with the documented allocation path. Do not keep the internal call or add guessed headers to force it to compile.
- Fix binding compile errors in the binding files, not by changing kernel math. A missing binding symbol or undeclared allocation API is not evidence that the CUDA kernel algorithm is wrong.
- Do not replace a compiling binding with a speculative “lean hot path” using internal APIs. Allocation micro-optimizations are lower priority than compile success and correctness; if attempted, isolate the change and keep a known-good fallback.
- When a binding specializes to float32/contiguous CUDA tensors, enforce that specialization with `TORCH_CHECK`/equivalent validation or ensure `model_new.py` guarantees it. Do not reinterpret arbitrary tensor storage as `float*`.

## 10. Final Validation Gate

- The final submitted files must be the exact files that passed compile, verification, and benchmark. Any write after the last successful verification/benchmark invalidates those results.
- Final sequence: finish cleanup and final edits, then compile, then run verification, then run profiling/benchmark. Do not stop immediately after a write.
- Do not perform a post-benchmark speculative rewrite of `kernels/*` or `model_new.py` and declare completion. If optimizing after a passing result, preserve or restore the known-good version; re-run compile, verification, and benchmark; if the new version fails, revert and leave the known-good version as final.
- A successful earlier compile does not excuse a later compile failure. Do not submit an uncompiled final candidate.
- Final cleanup counts as an edit. Run compile, verification, and benchmark after cleanup, or perform cleanup before the final validation sequence.
- On compile failure, use the first actionable compiler error. If the reported compile log path cannot be read, use the compile tool output or rerun compile; do not make unrelated kernel changes based on a log-path failure.

## 11. ModelNew and Wrapper Semantics

- Keep `ModelNew.forward` as the normal method and let the harness use `nn.Module.__call__`.
- Do not add `__call__ = forward` or otherwise bypass `nn.Module` hook machinery as a speculative shortcut. It can change harness/hook behavior. Use such a bypass only if explicitly required and revalidated; otherwise remove it before final validation.
- Preserve the constructor and `forward` signatures expected by `model.py`.
- Keep `model_new.py` limited to allowed tensor creation and custom extension ops. Do not add unsupported Python-side torch operations to compensate for binding or kernel issues.

## 12. Tool/Workspace Hygiene (Not CUDA Coding Errors)

- For `workspace` writes, pass `operation: "write"`, `path`, and `content` as separate fields. Do not put file content or Markdown into the `operation` field.
- Treat `ValueError: unknown operation`, missing `operation` errors, malformed tool payloads, and log encoding/mojibake as tooling issues, not CUDA coding mistakes.
- After any failed or ambiguous write, read the target file before compiling. A failed tool call does not prove the file is unchanged or correct.
- If a write/read fails due to tool transport or encoding, retry the tool call cleanly. Do not change CUDA kernel or binding logic solely in response to the tool failure.
- A missing utility file or unreadable path is a workspace/file-availability issue, not evidence that the kernel is incorrect. Use the documented utilities or list the directory to discover the actual files.

**Scope:** Sections 13–16 apply to CUDA extension episodes using this workspace. They do not relax Sections 1–12 or the existing forbidden/allowed lists. Section 16 applies only to tool/workspace operations and does not classify tool transport/encoding failures as CUDA coding errors.

## 13. Post-Tuning Validation State

- Treat `model_new.py` as part of the correctness surface. Changing the selected kernel `config`, wrapper arguments, or forward path can change numerical behavior; a benchmark-only run after such a change does not validate it.
- After any candidate change to `kernels/*`, bindings, or `model_new.py`, run verification before using or reporting benchmark results. A fast benchmark from an unverified candidate is only a performance probe.
- The final validation sequence must run on the exact final files: finish all edits and cleanup, compile if any C++/CUDA/binding file changed, run verification, then run benchmark. Any write after verification makes that verification stale.
- If a compile and a later write occur in the same step, do not count that compile as final validation for the later file. Ensure the last compile precedes the verified final state, or explicitly confirm no compiled file changed and keep the binary consistent with the verified files.
- Do not end an episode immediately after a benchmark-only edit. If the step budget is nearly exhausted, preserve or restore the last fully validated version rather than making an unverified last-moment change.
- If verification fails after a tuning change, reject the faster benchmark. Either fix the kernel/config and rerun compile, verification, and benchmark, or revert to the last verified version.

## 14. Tuned Config and Shape-Specialized Path Safety

- Do not hardcode a tuned `config` in `model_new.py` solely because it benchmarked fastest. Verify that exact config on all verification cases and seeds before leaving it as the final selection.
- For shape-specialized dispatch, ensure every non-default config either has a correct fallback or is unreachable from the final `model_new.py`. A fast path that is wrong for one harness shape must not remain selected.
- When a config-specific case fails, first inspect dispatch guards, supported `F`/`N` assumptions, and fallback coverage. Do not change unrelated kernel math or move the failure into Python.
- Auto/default passing does not validate a hardcoded config. After switching `model_new.py` from auto/default to a fixed config, rerun verification on that exact final config.
- Diagnostic, no-op, copy-only, or debug configs must not be reachable in the final model path.

## 15. Final Cleanup Specifics

- Before final validation, remove diagnostic no-op/copy kernels, unused tuning branches, and intermediate kernel versions from `kernels/` unless they are still required by the final default path.
- If multiple tuning configs remain, ensure they are all semantically correct for supported shapes, or remove/disable the unused ones. Do not leave a known-wrong config available to be selected accidentally.
- Do not submit a kernel containing diagnostic configs that produce no output or copy only, even if they were useful during tuning.
- Cleanup is an edit. After cleanup, run the full final sequence on the exact cleaned files; a successful earlier compile or benchmark does not validate cleaned files.

## 16. Tool Output Interpretation

- Environment warnings such as “Failed to find real location of python.exe” accompanying a successful compile are not compiler errors or CUDA kernel bugs. Use the compile success/failure status and actionable compiler errors.
- If a write/read fails because of a malformed workspace payload or missing `operation`, retry the tool call cleanly. A tool failure by itself is not evidence that the CUDA kernel or binding logic is wrong.
- After any failed or ambiguous write, read the target file before compiling. Do not change CUDA kernel or binding logic solely in response to a tool transport/encoding/payload failure.
- If a reported compile-log path cannot be read, use the compile tool output or rerun compile; do not make unrelated kernel changes based on a log-path failure.

**Scope:** Sections 17–20 apply to CUDA extension episodes using this workspace and do not relax the existing forbidden/allowed lists or validation rules. Section 20 applies only to tool/workspace operations and does not classify tool transport/encoding/path failures as CUDA coding errors.

## 17. Validation Order Is Non-Negotiable

- Compile success, verification success, and benchmark success are separate gates. A successful compile does not validate kernel math or wrapper semantics, and a benchmark without preceding verification on the same files is not evidence of correctness.
- After any edit to `kernels/*`, binding files, or `model_new.py`, use this order: finish edits/cleanup → compile if any C++/CUDA/binding file changed → verification → profiling/benchmark. The final benchmark must run on the exact files that passed verification.
- Do not make a late `model_new.py` config or forward-path edit and stop after a benchmark-only run. If the edit selects a different config, dispatch branch, or wrapper path, rerun verification on that exact file, then rerun the benchmark.
- A benchmark-only run may be used only as a tuning probe. Preserve the last verified selection and either verify the final selected file or revert to the verified one.
- Do not end an episode on a write, a compile-only step, or a benchmark-only step after a change. If step budget is nearly exhausted, do not start another speculative optimization; leave or restore the last fully validated files.
- A write after the last verification invalidates that verification. A compile after the last verification still leaves verification stale; recompile, verify, and benchmark before completion.
- If a benchmark-only run is followed by another edit, the earlier benchmark is not final evidence for the new file.

## 18. Reject Unverified Speedups

- If verification fails after a kernel, binding, or wrapper change, reject that candidate even if its benchmark is faster. Fix the failing path or revert to the last verified implementation, then recompile, verify, and benchmark.
- Do not continue tuning `model_new.py` configs while the underlying kernel is failing verification. Correctness must be restored first.
- When a rewrite breaks correctness, use the last passing kernel as the baseline and prefer a minimal fix over another broad rewrite. Keep a known-good copy until the replacement passes verification.
- A config that is fast for one shape but unverified or incorrect for another harness case must not remain selected in final `model_new.py`.
- Do not leave a faster but unverified or known-wrong config, dispatch branch, or wrapper path reachable from the final model path.

## 19. Probe and Debug Artifact Hygiene

- Temporary probe/diagnostic kernels, alternate bindings, and patch/debug files must not be part of the final module path. If they are added under `kernels/` or referenced by a binding, remove them and rerun compile, verification, and benchmark on the cleaned files.
- Do not use a probe benchmark as evidence that the final implementation is correct.
- Do not leave temporary patch files, probe launchers, or notes in `kernels/`; cleanup is an edit and must precede the final validation sequence.
- After deleting a probe file or reverting a binding, recompile so the extension cannot retain stale symbols or an unintended old path.

## 20. Tool Log Path Failures Are Not CUDA Signals

Scope: tool/workspace operations only; these rules do not classify tool transport/encoding/path failures as CUDA coding mistakes.

- A `FileNotFoundError` or unreadable path for a compile, verification, or profiling log is a tool/path issue. Use the tool’s returned output/status or rerun the tool; do not change kernel math, binding allocation, or `model_new.py` based on the failed read alone.
- If a log path cannot be read, do not infer pass/fail from the missing file. Rerun verification/benchmark or use the tool result already returned.
- After a failed or ambiguous workspace read/write, read the target file before compiling or editing further. A failed tool call is not evidence that the file is correct or incorrect.
- Environment warnings such as failure to locate `python.exe` alongside an explicit compile/verify/benchmark success are not CUDA kernel errors; use the explicit status and actionable compiler/runtime errors.

**Scope:** Sections 21–24 apply to CUDA extension episodes using this workspace. They do not relax the existing forbidden/allowed lists or validation requirements. Section 24 applies only to tool/workspace operations and does not classify tool payload or environment failures as CUDA coding mistakes.

## 21. Vectorized Reduction Semantics and Debug Isolation

- When a vector load (`float2`, `float4`) spans the non-reduced axis, each component belongs to a different output column and must have its own reduction accumulator and its own reciprocal normalization value. Do not sum the components into one scalar, and do not broadcast one scalar result across the vector lanes.
- Store and combine vector partials as vector/per-lane values in shared memory (for example, `float4` accumulators), not as one scalar per thread. Apply the same per-lane requirement when scaling and storing the output.
- A vectorized reduction failure with most elements mismatched and the same wrong values across different block shapes is evidence to inspect lane packing/reduction semantics first, not automatically a race or launch-config problem. Isolate it by testing the scalar path, a vectorized copy/indexing path without the reduction, and then the vectorized reduction.
- Keep diagnostic configs that skip reduction or copy data out of the final path. After fixing the vector path, verify both vector-eligible shapes and non-vector fallback shapes.

## 22. Forward-Path Diagnostics and Post-Tuning Validation

- Do not add PyTorch reference calculations (`pow`, `mean`, `sqrt`, etc.) or other debug tensor ops to `model_new.py`/`ModelNew.forward` to diagnose a kernel. The verifier rejects PyTorch computation during candidate forward; that failure is an instrumentation violation, not evidence that the CUDA math is wrong.
- Perform any reference comparison outside the submitted forward path, and remove the diagnostic before final validation. Do not leave debug instrumentation in `model_new.py`.
- A benchmark-only sweep after changing `model_new.py`—including a hardcoded `config`, dispatch branch, or wrapper argument—is a tuning probe. Before treating that selection as final, run verification on the exact final file, then benchmark. A passing verification of an earlier/default config does not validate the new selection.
- Do not treat compile success as correctness validation. If a new candidate is compiled, run verification before using its benchmark as final evidence. If verification fails, reject the faster candidate or revert to the last verified version.

## 23. Clock-State-Aware Benchmark Comparison

- Absolute candidate latencies in this environment can change by roughly 2× between runs as GPU clock state changes. Do not rank tuning variants solely by absolute microseconds measured in separate runs.
- Compare candidate and `torch.compile` baseline in the same benchmark invocation, or repeat/interleave measurements and report the candidate/baseline ratio. A result is more trustworthy when the ratio is stable across clock states.
- Do not use a fast but unverified benchmark under one clock state to justify leaving a different final configuration. Final evidence must come from the exact final files after the required compile → verify → benchmark order.

## 24. Workspace Write Payloads (Tool/Workspace Only)

- For workspace writes, pass `operation: "write"`, `path`, and `content` as separate fields. Do not omit `operation` or put file content/Markdown in the `operation` field.
- A `ValueError: operation must be one of ...` or a missing-`operation` error is a tool payload problem, not a CUDA kernel or binding bug. Retry the write/read correctly; do not change kernel math or allocation logic in response to that error alone.
- After a failed or ambiguous write, read the target file before compiling or making further edits. A failed tool call does not prove the file is correct or incorrect.
- Environment warnings such as “Failed to find real location of python.exe” alongside an explicit compile/verify success are not CUDA errors; use the explicit status and actionable compiler/runtime output.

**Scope:** Sections 25–28 apply to CUDA extension episodes using this workspace. They do not relax Sections 1–24 or the existing forbidden/allowed lists. Section 28 applies only to tool/workspace operations and explicitly does not classify tool transport, encoding, payload, or path failures as CUDA coding mistakes.

## 25. Output Tensor Lifetime and Buffer-Reuse Safety

- Treat “previously returned outputs must remain unchanged after later calls” as a hard lifetime constraint. Never write into, recycle, or reuse storage that may still be referenced by any earlier returned tensor. The verifier/harness may retain outputs across repeated calls.
- For AXPBY-like elementwise ops that return a new tensor, default to the allowed per-call `torch::empty_like(...)` and rely on PyTorch’s allocator for caching. Do not add a custom cross-call output pool merely to reduce allocation overhead unless you can prove with public APIs that no live tensor references the buffer.
- Do not use `TensorImpl::use_count`, `Tensor::use_count`, `storage().use_count`, intrusive-pointer counts, or other refcount internals to decide whether an output buffer is recyclable. Such APIs are unstable/version-dependent, and an observed build failed with `"use_count": is not a member of "c10::TensorImpl"`. A single refcount is also not a contract-safe proof against live Python/harness references.
- If an attempted refcount API does not compile, remove that pooling attempt or use per-call allocation. Do not add guessed headers, private ATen/TensorImpl APIs, or unrelated kernel changes to force it to compile.
- A growing pool is a failure signal. If diagnostic output shows the pool size increasing monotonically, or benchmark latency degrades over repeated calls, the buffers are not safely recyclable. Remove the pool and return to per-call allocation. Do not report a faster benchmark from a pool that was not proven to preserve all previous outputs.
- If you experiment with a pre-reserved chunk or suballocation, keep size/offset metadata on the host. Do not write metadata to device memory through a host pointer; that is a host/device memory bug, not an allocation-tuning detail. Ensure each live output has distinct usable storage or an ownership scheme that cannot overwrite a live result.
- Never alias or overwrite input tensors `a`/`b` to save allocation. Compute into a distinct output. When using typed pointers, validate dtype, device, contiguity, and shape assumptions; if buffer reuse is attempted, match all relevant tensor properties, not just shape.

## 26. No Probe Code in Final Production Paths

- A verification PASS does not mean the binding is clean. Probe output can run during verification and benchmark while tests still pass. Before finalizing, inspect `kernels/*.cu`, `kernels/*_binding.cpp`, and `model_new.py` for diagnostic `printf`/`std::printf`, counters, timing probes, pool-inspection logs, and comments such as `PROBE ONLY`.
- Remove diagnostic code embedded in otherwise final files, not only separate `probe_*.cpp` files. Do not leave global probe state, pool-size printing, or debug-only branches reachable from `model_new.py`.
- After removing probe code, recompile if any C++/CUDA/binding file changed, then run verification and benchmark on the exact cleaned files. Verification or benchmark output from the probe version is not evidence for the cleaned version.
- Temporary probe files must be deleted before final validation. If a probe-specific symbol remains declared or referenced, recompile so the extension cannot retain a stale or unintended path.

## 27. Finish on Validated Files and Submit the Report

- Do not end an episode after a verify-only step when the performance gate has not been measured on the exact final files. The final sequence is: finish edits/cleanup → compile if needed → verification → benchmark on the same files → submit the final report.
- Reserve enough budget for that final sequence. If the remaining budget is small, stop optional pooling/refcount experiments and preserve or restore the last candidate that passed compile, verification, and benchmark. Do not start a speculative optimization that may leave the workspace unvalidated.
- A benchmark from an earlier candidate is not final if later writes changed `model_new.py`, binding files, or kernels. Report only measurements actually obtained from the final validated files; if the final files were not benchmarked, say so and treat the performance goal as unconfirmed.
- If a late buffer-pool or binding experiment fails compile, fails verification, or shows degradation, revert to the known-good per-call allocation version and rerun the required gates rather than leaving the experimental binding in place.

## 28. Tool/Workspace Failure Separation (Scoped)

- A missing `operation` field or `ValueError: operation must be one of ...` is a tooling payload error. Retry the write/read with `operation`, `path`, and `content` as separate fields; do not change kernel math, output pooling, or binding allocation in response to that error alone.
- After any failed or ambiguous write, read the target file before compiling. A failed tool call does not prove the file is unchanged or correct.
- Environment warnings such as failure to locate `python.exe` alongside an explicit compile/verify success are not CUDA errors; use the explicit status and actionable compiler/runtime output. A log-path, encoding, transport, or payload failure is not evidence that the CUDA implementation is wrong.


## Repeated appendix, copy 6

**Scope of appended rules:** Sections 9–12 apply to CUDA extension episodes using this workspace. Section 12 applies only to tool/workspace operations and explicitly does not classify tool transport/encoding failures as CUDA coding errors. None of these rules relax the existing forbidden/allowed lists.

## 9. Compile-Safe Binding and Allocation

- Use only the public allocation exception already allowed by this document. For bindings, prefer the documented `torch::empty_like(...)` allocation path.
- Do not use `at::detail::empty_cuda`, other `at::detail::*` symbols, or version-private ATen internals to bypass the dispatcher. These are not stable extension APIs.
- If compilation reports an error such as `'empty_cuda': is not a member of 'at::detail'`, replace the internal allocation call with the documented allocation path. Do not keep the internal call or add guessed headers to force it to compile.
- Fix binding compile errors in the binding files, not by changing kernel math. A missing binding symbol or undeclared allocation API is not evidence that the CUDA kernel algorithm is wrong.
- Do not replace a compiling binding with a speculative “lean hot path” using internal APIs. Allocation micro-optimizations are lower priority than compile success and correctness; if attempted, isolate the change and keep a known-good fallback.
- When a binding specializes to float32/contiguous CUDA tensors, enforce that specialization with `TORCH_CHECK`/equivalent validation or ensure `model_new.py` guarantees it. Do not reinterpret arbitrary tensor storage as `float*`.

## 10. Final Validation Gate

- The final submitted files must be the exact files that passed compile, verification, and benchmark. Any write after the last successful verification/benchmark invalidates those results.
- Final sequence: finish cleanup and final edits, then compile, then run verification, then run profiling/benchmark. Do not stop immediately after a write.
- Do not perform a post-benchmark speculative rewrite of `kernels/*` or `model_new.py` and declare completion. If optimizing after a passing result, preserve or restore the known-good version; re-run compile, verification, and benchmark; if the new version fails, revert and leave the known-good version as final.
- A successful earlier compile does not excuse a later compile failure. Do not submit an uncompiled final candidate.
- Final cleanup counts as an edit. Run compile, verification, and benchmark after cleanup, or perform cleanup before the final validation sequence.
- On compile failure, use the first actionable compiler error. If the reported compile log path cannot be read, use the compile tool output or rerun compile; do not make unrelated kernel changes based on a log-path failure.

## 11. ModelNew and Wrapper Semantics

- Keep `ModelNew.forward` as the normal method and let the harness use `nn.Module.__call__`.
- Do not add `__call__ = forward` or otherwise bypass `nn.Module` hook machinery as a speculative shortcut. It can change harness/hook behavior. Use such a bypass only if explicitly required and revalidated; otherwise remove it before final validation.
- Preserve the constructor and `forward` signatures expected by `model.py`.
- Keep `model_new.py` limited to allowed tensor creation and custom extension ops. Do not add unsupported Python-side torch operations to compensate for binding or kernel issues.

## 12. Tool/Workspace Hygiene (Not CUDA Coding Errors)

- For `workspace` writes, pass `operation: "write"`, `path`, and `content` as separate fields. Do not put file content or Markdown into the `operation` field.
- Treat `ValueError: unknown operation`, missing `operation` errors, malformed tool payloads, and log encoding/mojibake as tooling issues, not CUDA coding mistakes.
- After any failed or ambiguous write, read the target file before compiling. A failed tool call does not prove the file is unchanged or correct.
- If a write/read fails due to tool transport or encoding, retry the tool call cleanly. Do not change CUDA kernel or binding logic solely in response to the tool failure.
- A missing utility file or unreadable path is a workspace/file-availability issue, not evidence that the kernel is incorrect. Use the documented utilities or list the directory to discover the actual files.

**Scope:** Sections 13–16 apply to CUDA extension episodes using this workspace. They do not relax Sections 1–12 or the existing forbidden/allowed lists. Section 16 applies only to tool/workspace operations and does not classify tool transport/encoding failures as CUDA coding errors.

## 13. Post-Tuning Validation State

- Treat `model_new.py` as part of the correctness surface. Changing the selected kernel `config`, wrapper arguments, or forward path can change numerical behavior; a benchmark-only run after such a change does not validate it.
- After any candidate change to `kernels/*`, bindings, or `model_new.py`, run verification before using or reporting benchmark results. A fast benchmark from an unverified candidate is only a performance probe.
- The final validation sequence must run on the exact final files: finish all edits and cleanup, compile if any C++/CUDA/binding file changed, run verification, then run benchmark. Any write after verification makes that verification stale.
- If a compile and a later write occur in the same step, do not count that compile as final validation for the later file. Ensure the last compile precedes the verified final state, or explicitly confirm no compiled file changed and keep the binary consistent with the verified files.
- Do not end an episode immediately after a benchmark-only edit. If the step budget is nearly exhausted, preserve or restore the last fully validated version rather than making an unverified last-moment change.
- If verification fails after a tuning change, reject the faster benchmark. Either fix the kernel/config and rerun compile, verification, and benchmark, or revert to the last verified version.

## 14. Tuned Config and Shape-Specialized Path Safety

- Do not hardcode a tuned `config` in `model_new.py` solely because it benchmarked fastest. Verify that exact config on all verification cases and seeds before leaving it as the final selection.
- For shape-specialized dispatch, ensure every non-default config either has a correct fallback or is unreachable from the final `model_new.py`. A fast path that is wrong for one harness shape must not remain selected.
- When a config-specific case fails, first inspect dispatch guards, supported `F`/`N` assumptions, and fallback coverage. Do not change unrelated kernel math or move the failure into Python.
- Auto/default passing does not validate a hardcoded config. After switching `model_new.py` from auto/default to a fixed config, rerun verification on that exact final config.
- Diagnostic, no-op, copy-only, or debug configs must not be reachable in the final model path.

## 15. Final Cleanup Specifics

- Before final validation, remove diagnostic no-op/copy kernels, unused tuning branches, and intermediate kernel versions from `kernels/` unless they are still required by the final default path.
- If multiple tuning configs remain, ensure they are all semantically correct for supported shapes, or remove/disable the unused ones. Do not leave a known-wrong config available to be selected accidentally.
- Do not submit a kernel containing diagnostic configs that produce no output or copy only, even if they were useful during tuning.
- Cleanup is an edit. After cleanup, run the full final sequence on the exact cleaned files; a successful earlier compile or benchmark does not validate cleaned files.

## 16. Tool Output Interpretation

- Environment warnings such as “Failed to find real location of python.exe” accompanying a successful compile are not compiler errors or CUDA kernel bugs. Use the compile success/failure status and actionable compiler errors.
- If a write/read fails because of a malformed workspace payload or missing `operation`, retry the tool call cleanly. A tool failure by itself is not evidence that the CUDA kernel or binding logic is wrong.
- After any failed or ambiguous write, read the target file before compiling. Do not change CUDA kernel or binding logic solely in response to a tool transport/encoding/payload failure.
- If a reported compile-log path cannot be read, use the compile tool output or rerun compile; do not make unrelated kernel changes based on a log-path failure.

**Scope:** Sections 17–20 apply to CUDA extension episodes using this workspace and do not relax the existing forbidden/allowed lists or validation rules. Section 20 applies only to tool/workspace operations and does not classify tool transport/encoding/path failures as CUDA coding errors.

## 17. Validation Order Is Non-Negotiable

- Compile success, verification success, and benchmark success are separate gates. A successful compile does not validate kernel math or wrapper semantics, and a benchmark without preceding verification on the same files is not evidence of correctness.
- After any edit to `kernels/*`, binding files, or `model_new.py`, use this order: finish edits/cleanup → compile if any C++/CUDA/binding file changed → verification → profiling/benchmark. The final benchmark must run on the exact files that passed verification.
- Do not make a late `model_new.py` config or forward-path edit and stop after a benchmark-only run. If the edit selects a different config, dispatch branch, or wrapper path, rerun verification on that exact file, then rerun the benchmark.
- A benchmark-only run may be used only as a tuning probe. Preserve the last verified selection and either verify the final selected file or revert to the verified one.
- Do not end an episode on a write, a compile-only step, or a benchmark-only step after a change. If step budget is nearly exhausted, do not start another speculative optimization; leave or restore the last fully validated files.
- A write after the last verification invalidates that verification. A compile after the last verification still leaves verification stale; recompile, verify, and benchmark before completion.
- If a benchmark-only run is followed by another edit, the earlier benchmark is not final evidence for the new file.

## 18. Reject Unverified Speedups

- If verification fails after a kernel, binding, or wrapper change, reject that candidate even if its benchmark is faster. Fix the failing path or revert to the last verified implementation, then recompile, verify, and benchmark.
- Do not continue tuning `model_new.py` configs while the underlying kernel is failing verification. Correctness must be restored first.
- When a rewrite breaks correctness, use the last passing kernel as the baseline and prefer a minimal fix over another broad rewrite. Keep a known-good copy until the replacement passes verification.
- A config that is fast for one shape but unverified or incorrect for another harness case must not remain selected in final `model_new.py`.
- Do not leave a faster but unverified or known-wrong config, dispatch branch, or wrapper path reachable from the final model path.

## 19. Probe and Debug Artifact Hygiene

- Temporary probe/diagnostic kernels, alternate bindings, and patch/debug files must not be part of the final module path. If they are added under `kernels/` or referenced by a binding, remove them and rerun compile, verification, and benchmark on the cleaned files.
- Do not use a probe benchmark as evidence that the final implementation is correct.
- Do not leave temporary patch files, probe launchers, or notes in `kernels/`; cleanup is an edit and must precede the final validation sequence.
- After deleting a probe file or reverting a binding, recompile so the extension cannot retain stale symbols or an unintended old path.

## 20. Tool Log Path Failures Are Not CUDA Signals

Scope: tool/workspace operations only; these rules do not classify tool transport/encoding/path failures as CUDA coding mistakes.

- A `FileNotFoundError` or unreadable path for a compile, verification, or profiling log is a tool/path issue. Use the tool’s returned output/status or rerun the tool; do not change kernel math, binding allocation, or `model_new.py` based on the failed read alone.
- If a log path cannot be read, do not infer pass/fail from the missing file. Rerun verification/benchmark or use the tool result already returned.
- After a failed or ambiguous workspace read/write, read the target file before compiling or editing further. A failed tool call is not evidence that the file is correct or incorrect.
- Environment warnings such as failure to locate `python.exe` alongside an explicit compile/verify/benchmark success are not CUDA kernel errors; use the explicit status and actionable compiler/runtime errors.

**Scope:** Sections 21–24 apply to CUDA extension episodes using this workspace. They do not relax the existing forbidden/allowed lists or validation requirements. Section 24 applies only to tool/workspace operations and does not classify tool payload or environment failures as CUDA coding mistakes.

## 21. Vectorized Reduction Semantics and Debug Isolation

- When a vector load (`float2`, `float4`) spans the non-reduced axis, each component belongs to a different output column and must have its own reduction accumulator and its own reciprocal normalization value. Do not sum the components into one scalar, and do not broadcast one scalar result across the vector lanes.
- Store and combine vector partials as vector/per-lane values in shared memory (for example, `float4` accumulators), not as one scalar per thread. Apply the same per-lane requirement when scaling and storing the output.
- A vectorized reduction failure with most elements mismatched and the same wrong values across different block shapes is evidence to inspect lane packing/reduction semantics first, not automatically a race or launch-config problem. Isolate it by testing the scalar path, a vectorized copy/indexing path without the reduction, and then the vectorized reduction.
- Keep diagnostic configs that skip reduction or copy data out of the final path. After fixing the vector path, verify both vector-eligible shapes and non-vector fallback shapes.

## 22. Forward-Path Diagnostics and Post-Tuning Validation

- Do not add PyTorch reference calculations (`pow`, `mean`, `sqrt`, etc.) or other debug tensor ops to `model_new.py`/`ModelNew.forward` to diagnose a kernel. The verifier rejects PyTorch computation during candidate forward; that failure is an instrumentation violation, not evidence that the CUDA math is wrong.
- Perform any reference comparison outside the submitted forward path, and remove the diagnostic before final validation. Do not leave debug instrumentation in `model_new.py`.
- A benchmark-only sweep after changing `model_new.py`—including a hardcoded `config`, dispatch branch, or wrapper argument—is a tuning probe. Before treating that selection as final, run verification on the exact final file, then benchmark. A passing verification of an earlier/default config does not validate the new selection.
- Do not treat compile success as correctness validation. If a new candidate is compiled, run verification before using its benchmark as final evidence. If verification fails, reject the faster candidate or revert to the last verified version.

## 23. Clock-State-Aware Benchmark Comparison

- Absolute candidate latencies in this environment can change by roughly 2× between runs as GPU clock state changes. Do not rank tuning variants solely by absolute microseconds measured in separate runs.
- Compare candidate and `torch.compile` baseline in the same benchmark invocation, or repeat/interleave measurements and report the candidate/baseline ratio. A result is more trustworthy when the ratio is stable across clock states.
- Do not use a fast but unverified benchmark under one clock state to justify leaving a different final configuration. Final evidence must come from the exact final files after the required compile → verify → benchmark order.

## 24. Workspace Write Payloads (Tool/Workspace Only)

- For workspace writes, pass `operation: "write"`, `path`, and `content` as separate fields. Do not omit `operation` or put file content/Markdown in the `operation` field.
- A `ValueError: operation must be one of ...` or a missing-`operation` error is a tool payload problem, not a CUDA kernel or binding bug. Retry the write/read correctly; do not change kernel math or allocation logic in response to that error alone.
- After a failed or ambiguous write, read the target file before compiling or making further edits. A failed tool call does not prove the file is correct or incorrect.
- Environment warnings such as “Failed to find real location of python.exe” alongside an explicit compile/verify success are not CUDA errors; use the explicit status and actionable compiler/runtime output.

**Scope:** Sections 25–28 apply to CUDA extension episodes using this workspace. They do not relax Sections 1–24 or the existing forbidden/allowed lists. Section 28 applies only to tool/workspace operations and explicitly does not classify tool transport, encoding, payload, or path failures as CUDA coding mistakes.

## 25. Output Tensor Lifetime and Buffer-Reuse Safety

- Treat “previously returned outputs must remain unchanged after later calls” as a hard lifetime constraint. Never write into, recycle, or reuse storage that may still be referenced by any earlier returned tensor. The verifier/harness may retain outputs across repeated calls.
- For AXPBY-like elementwise ops that return a new tensor, default to the allowed per-call `torch::empty_like(...)` and rely on PyTorch’s allocator for caching. Do not add a custom cross-call output pool merely to reduce allocation overhead unless you can prove with public APIs that no live tensor references the buffer.
- Do not use `TensorImpl::use_count`, `Tensor::use_count`, `storage().use_count`, intrusive-pointer counts, or other refcount internals to decide whether an output buffer is recyclable. Such APIs are unstable/version-dependent, and an observed build failed with `"use_count": is not a member of "c10::TensorImpl"`. A single refcount is also not a contract-safe proof against live Python/harness references.
- If an attempted refcount API does not compile, remove that pooling attempt or use per-call allocation. Do not add guessed headers, private ATen/TensorImpl APIs, or unrelated kernel changes to force it to compile.
- A growing pool is a failure signal. If diagnostic output shows the pool size increasing monotonically, or benchmark latency degrades over repeated calls, the buffers are not safely recyclable. Remove the pool and return to per-call allocation. Do not report a faster benchmark from a pool that was not proven to preserve all previous outputs.
- If you experiment with a pre-reserved chunk or suballocation, keep size/offset metadata on the host. Do not write metadata to device memory through a host pointer; that is a host/device memory bug, not an allocation-tuning detail. Ensure each live output has distinct usable storage or an ownership scheme that cannot overwrite a live result.
- Never alias or overwrite input tensors `a`/`b` to save allocation. Compute into a distinct output. When using typed pointers, validate dtype, device, contiguity, and shape assumptions; if buffer reuse is attempted, match all relevant tensor properties, not just shape.

## 26. No Probe Code in Final Production Paths

- A verification PASS does not mean the binding is clean. Probe output can run during verification and benchmark while tests still pass. Before finalizing, inspect `kernels/*.cu`, `kernels/*_binding.cpp`, and `model_new.py` for diagnostic `printf`/`std::printf`, counters, timing probes, pool-inspection logs, and comments such as `PROBE ONLY`.
- Remove diagnostic code embedded in otherwise final files, not only separate `probe_*.cpp` files. Do not leave global probe state, pool-size printing, or debug-only branches reachable from `model_new.py`.
- After removing probe code, recompile if any C++/CUDA/binding file changed, then run verification and benchmark on the exact cleaned files. Verification or benchmark output from the probe version is not evidence for the cleaned version.
- Temporary probe files must be deleted before final validation. If a probe-specific symbol remains declared or referenced, recompile so the extension cannot retain a stale or unintended path.

## 27. Finish on Validated Files and Submit the Report

- Do not end an episode after a verify-only step when the performance gate has not been measured on the exact final files. The final sequence is: finish edits/cleanup → compile if needed → verification → benchmark on the same files → submit the final report.
- Reserve enough budget for that final sequence. If the remaining budget is small, stop optional pooling/refcount experiments and preserve or restore the last candidate that passed compile, verification, and benchmark. Do not start a speculative optimization that may leave the workspace unvalidated.
- A benchmark from an earlier candidate is not final if later writes changed `model_new.py`, binding files, or kernels. Report only measurements actually obtained from the final validated files; if the final files were not benchmarked, say so and treat the performance goal as unconfirmed.
- If a late buffer-pool or binding experiment fails compile, fails verification, or shows degradation, revert to the known-good per-call allocation version and rerun the required gates rather than leaving the experimental binding in place.

## 28. Tool/Workspace Failure Separation (Scoped)

- A missing `operation` field or `ValueError: operation must be one of ...` is a tooling payload error. Retry the write/read with `operation`, `path`, and `content` as separate fields; do not change kernel math, output pooling, or binding allocation in response to that error alone.
- After any failed or ambiguous write, read the target file before compiling. A failed tool call does not prove the file is unchanged or correct.
- Environment warnings such as failure to locate `python.exe` alongside an explicit compile/verify success are not CUDA errors; use the explicit status and actionable compiler/runtime output. A log-path, encoding, transport, or payload failure is not evidence that the CUDA implementation is wrong.


## Repeated appendix, copy 7

**Scope of appended rules:** Sections 9–12 apply to CUDA extension episodes using this workspace. Section 12 applies only to tool/workspace operations and explicitly does not classify tool transport/encoding failures as CUDA coding errors. None of these rules relax the existing forbidden/allowed lists.

## 9. Compile-Safe Binding and Allocation

- Use only the public allocation exception already allowed by this document. For bindings, prefer the documented `torch::empty_like(...)` allocation path.
- Do not use `at::detail::empty_cuda`, other `at::detail::*` symbols, or version-private ATen internals to bypass the dispatcher. These are not stable extension APIs.
- If compilation reports an error such as `'empty_cuda': is not a member of 'at::detail'`, replace the internal allocation call with the documented allocation path. Do not keep the internal call or add guessed headers to force it to compile.
- Fix binding compile errors in the binding files, not by changing kernel math. A missing binding symbol or undeclared allocation API is not evidence that the CUDA kernel algorithm is wrong.
- Do not replace a compiling binding with a speculative “lean hot path” using internal APIs. Allocation micro-optimizations are lower priority than compile success and correctness; if attempted, isolate the change and keep a known-good fallback.
- When a binding specializes to float32/contiguous CUDA tensors, enforce that specialization with `TORCH_CHECK`/equivalent validation or ensure `model_new.py` guarantees it. Do not reinterpret arbitrary tensor storage as `float*`.

## 10. Final Validation Gate

- The final submitted files must be the exact files that passed compile, verification, and benchmark. Any write after the last successful verification/benchmark invalidates those results.
- Final sequence: finish cleanup and final edits, then compile, then run verification, then run profiling/benchmark. Do not stop immediately after a write.
- Do not perform a post-benchmark speculative rewrite of `kernels/*` or `model_new.py` and declare completion. If optimizing after a passing result, preserve or restore the known-good version; re-run compile, verification, and benchmark; if the new version fails, revert and leave the known-good version as final.
- A successful earlier compile does not excuse a later compile failure. Do not submit an uncompiled final candidate.
- Final cleanup counts as an edit. Run compile, verification, and benchmark after cleanup, or perform cleanup before the final validation sequence.
- On compile failure, use the first actionable compiler error. If the reported compile log path cannot be read, use the compile tool output or rerun compile; do not make unrelated kernel changes based on a log-path failure.

## 11. ModelNew and Wrapper Semantics

- Keep `ModelNew.forward` as the normal method and let the harness use `nn.Module.__call__`.
- Do not add `__call__ = forward` or otherwise bypass `nn.Module` hook machinery as a speculative shortcut. It can change harness/hook behavior. Use such a bypass only if explicitly required and revalidated; otherwise remove it before final validation.
- Preserve the constructor and `forward` signatures expected by `model.py`.
- Keep `model_new.py` limited to allowed tensor creation and custom extension ops. Do not add unsupported Python-side torch operations to compensate for binding or kernel issues.

## 12. Tool/Workspace Hygiene (Not CUDA Coding Errors)

- For `workspace` writes, pass `operation: "write"`, `path`, and `content` as separate fields. Do not put file content or Markdown into the `operation` field.
- Treat `ValueError: unknown operation`, missing `operation` errors, malformed tool payloads, and log encoding/mojibake as tooling issues, not CUDA coding mistakes.
- After any failed or ambiguous write, read the target file before compiling. A failed tool call does not prove the file is unchanged or correct.
- If a write/read fails due to tool transport or encoding, retry the tool call cleanly. Do not change CUDA kernel or binding logic solely in response to the tool failure.
- A missing utility file or unreadable path is a workspace/file-availability issue, not evidence that the kernel is incorrect. Use the documented utilities or list the directory to discover the actual files.

**Scope:** Sections 13–16 apply to CUDA extension episodes using this workspace. They do not relax Sections 1–12 or the existing forbidden/allowed lists. Section 16 applies only to tool/workspace operations and does not classify tool transport/encoding failures as CUDA coding errors.

## 13. Post-Tuning Validation State

- Treat `model_new.py` as part of the correctness surface. Changing the selected kernel `config`, wrapper arguments, or forward path can change numerical behavior; a benchmark-only run after such a change does not validate it.
- After any candidate change to `kernels/*`, bindings, or `model_new.py`, run verification before using or reporting benchmark results. A fast benchmark from an unverified candidate is only a performance probe.
- The final validation sequence must run on the exact final files: finish all edits and cleanup, compile if any C++/CUDA/binding file changed, run verification, then run benchmark. Any write after verification makes that verification stale.
- If a compile and a later write occur in the same step, do not count that compile as final validation for the later file. Ensure the last compile precedes the verified final state, or explicitly confirm no compiled file changed and keep the binary consistent with the verified files.
- Do not end an episode immediately after a benchmark-only edit. If the step budget is nearly exhausted, preserve or restore the last fully validated version rather than making an unverified last-moment change.
- If verification fails after a tuning change, reject the faster benchmark. Either fix the kernel/config and rerun compile, verification, and benchmark, or revert to the last verified version.

## 14. Tuned Config and Shape-Specialized Path Safety

- Do not hardcode a tuned `config` in `model_new.py` solely because it benchmarked fastest. Verify that exact config on all verification cases and seeds before leaving it as the final selection.
- For shape-specialized dispatch, ensure every non-default config either has a correct fallback or is unreachable from the final `model_new.py`. A fast path that is wrong for one harness shape must not remain selected.
- When a config-specific case fails, first inspect dispatch guards, supported `F`/`N` assumptions, and fallback coverage. Do not change unrelated kernel math or move the failure into Python.
- Auto/default passing does not validate a hardcoded config. After switching `model_new.py` from auto/default to a fixed config, rerun verification on that exact final config.
- Diagnostic, no-op, copy-only, or debug configs must not be reachable in the final model path.

## 15. Final Cleanup Specifics

- Before final validation, remove diagnostic no-op/copy kernels, unused tuning branches, and intermediate kernel versions from `kernels/` unless they are still required by the final default path.
- If multiple tuning configs remain, ensure they are all semantically correct for supported shapes, or remove/disable the unused ones. Do not leave a known-wrong config available to be selected accidentally.
- Do not submit a kernel containing diagnostic configs that produce no output or copy only, even if they were useful during tuning.
- Cleanup is an edit. After cleanup, run the full final sequence on the exact cleaned files; a successful earlier compile or benchmark does not validate cleaned files.

## 16. Tool Output Interpretation

- Environment warnings such as “Failed to find real location of python.exe” accompanying a successful compile are not compiler errors or CUDA kernel bugs. Use the compile success/failure status and actionable compiler errors.
- If a write/read fails because of a malformed workspace payload or missing `operation`, retry the tool call cleanly. A tool failure by itself is not evidence that the CUDA kernel or binding logic is wrong.
- After any failed or ambiguous write, read the target file before compiling. Do not change CUDA kernel or binding logic solely in response to a tool transport/encoding/payload failure.
- If a reported compile-log path cannot be read, use the compile tool output or rerun compile; do not make unrelated kernel changes based on a log-path failure.

**Scope:** Sections 17–20 apply to CUDA extension episodes using this workspace and do not relax the existing forbidden/allowed lists or validation rules. Section 20 applies only to tool/workspace operations and does not classify tool transport/encoding/path failures as CUDA coding errors.

## 17. Validation Order Is Non-Negotiable

- Compile success, verification success, and benchmark success are separate gates. A successful compile does not validate kernel math or wrapper semantics, and a benchmark without preceding verification on the same files is not evidence of correctness.
- After any edit to `kernels/*`, binding files, or `model_new.py`, use this order: finish edits/cleanup → compile if any C++/CUDA/binding file changed → verification → profiling/benchmark. The final benchmark must run on the exact files that passed verification.
- Do not make a late `model_new.py` config or forward-path edit and stop after a benchmark-only run. If the edit selects a different config, dispatch branch, or wrapper path, rerun verification on that exact file, then rerun the benchmark.
- A benchmark-only run may be used only as a tuning probe. Preserve the last verified selection and either verify the final selected file or revert to the verified one.
- Do not end an episode on a write, a compile-only step, or a benchmark-only step after a change. If step budget is nearly exhausted, do not start another speculative optimization; leave or restore the last fully validated files.
- A write after the last verification invalidates that verification. A compile after the last verification still leaves verification stale; recompile, verify, and benchmark before completion.
- If a benchmark-only run is followed by another edit, the earlier benchmark is not final evidence for the new file.

## 18. Reject Unverified Speedups

- If verification fails after a kernel, binding, or wrapper change, reject that candidate even if its benchmark is faster. Fix the failing path or revert to the last verified implementation, then recompile, verify, and benchmark.
- Do not continue tuning `model_new.py` configs while the underlying kernel is failing verification. Correctness must be restored first.
- When a rewrite breaks correctness, use the last passing kernel as the baseline and prefer a minimal fix over another broad rewrite. Keep a known-good copy until the replacement passes verification.
- A config that is fast for one shape but unverified or incorrect for another harness case must not remain selected in final `model_new.py`.
- Do not leave a faster but unverified or known-wrong config, dispatch branch, or wrapper path reachable from the final model path.

## 19. Probe and Debug Artifact Hygiene

- Temporary probe/diagnostic kernels, alternate bindings, and patch/debug files must not be part of the final module path. If they are added under `kernels/` or referenced by a binding, remove them and rerun compile, verification, and benchmark on the cleaned files.
- Do not use a probe benchmark as evidence that the final implementation is correct.
- Do not leave temporary patch files, probe launchers, or notes in `kernels/`; cleanup is an edit and must precede the final validation sequence.
- After deleting a probe file or reverting a binding, recompile so the extension cannot retain stale symbols or an unintended old path.

## 20. Tool Log Path Failures Are Not CUDA Signals

Scope: tool/workspace operations only; these rules do not classify tool transport/encoding/path failures as CUDA coding mistakes.

- A `FileNotFoundError` or unreadable path for a compile, verification, or profiling log is a tool/path issue. Use the tool’s returned output/status or rerun the tool; do not change kernel math, binding allocation, or `model_new.py` based on the failed read alone.
- If a log path cannot be read, do not infer pass/fail from the missing file. Rerun verification/benchmark or use the tool result already returned.
- After a failed or ambiguous workspace read/write, read the target file before compiling or editing further. A failed tool call is not evidence that the file is correct or incorrect.
- Environment warnings such as failure to locate `python.exe` alongside an explicit compile/verify/benchmark success are not CUDA kernel errors; use the explicit status and actionable compiler/runtime errors.

**Scope:** Sections 21–24 apply to CUDA extension episodes using this workspace. They do not relax the existing forbidden/allowed lists or validation requirements. Section 24 applies only to tool/workspace operations and does not classify tool payload or environment failures as CUDA coding mistakes.

## 21. Vectorized Reduction Semantics and Debug Isolation

- When a vector load (`float2`, `float4`) spans the non-reduced axis, each component belongs to a different output column and must have its own reduction accumulator and its own reciprocal normalization value. Do not sum the components into one scalar, and do not broadcast one scalar result across the vector lanes.
- Store and combine vector partials as vector/per-lane values in shared memory (for example, `float4` accumulators), not as one scalar per thread. Apply the same per-lane requirement when scaling and storing the output.
- A vectorized reduction failure with most elements mismatched and the same wrong values across different block shapes is evidence to inspect lane packing/reduction semantics first, not automatically a race or launch-config problem. Isolate it by testing the scalar path, a vectorized copy/indexing path without the reduction, and then the vectorized reduction.
- Keep diagnostic configs that skip reduction or copy data out of the final path. After fixing the vector path, verify both vector-eligible shapes and non-vector fallback shapes.

## 22. Forward-Path Diagnostics and Post-Tuning Validation

- Do not add PyTorch reference calculations (`pow`, `mean`, `sqrt`, etc.) or other debug tensor ops to `model_new.py`/`ModelNew.forward` to diagnose a kernel. The verifier rejects PyTorch computation during candidate forward; that failure is an instrumentation violation, not evidence that the CUDA math is wrong.
- Perform any reference comparison outside the submitted forward path, and remove the diagnostic before final validation. Do not leave debug instrumentation in `model_new.py`.
- A benchmark-only sweep after changing `model_new.py`—including a hardcoded `config`, dispatch branch, or wrapper argument—is a tuning probe. Before treating that selection as final, run verification on the exact final file, then benchmark. A passing verification of an earlier/default config does not validate the new selection.
- Do not treat compile success as correctness validation. If a new candidate is compiled, run verification before using its benchmark as final evidence. If verification fails, reject the faster candidate or revert to the last verified version.

## 23. Clock-State-Aware Benchmark Comparison

- Absolute candidate latencies in this environment can change by roughly 2× between runs as GPU clock state changes. Do not rank tuning variants solely by absolute microseconds measured in separate runs.
- Compare candidate and `torch.compile` baseline in the same benchmark invocation, or repeat/interleave measurements and report the candidate/baseline ratio. A result is more trustworthy when the ratio is stable across clock states.
- Do not use a fast but unverified benchmark under one clock state to justify leaving a different final configuration. Final evidence must come from the exact final files after the required compile → verify → benchmark order.

## 24. Workspace Write Payloads (Tool/Workspace Only)

- For workspace writes, pass `operation: "write"`, `path`, and `content` as separate fields. Do not omit `operation` or put file content/Markdown in the `operation` field.
- A `ValueError: operation must be one of ...` or a missing-`operation` error is a tool payload problem, not a CUDA kernel or binding bug. Retry the write/read correctly; do not change kernel math or allocation logic in response to that error alone.
- After a failed or ambiguous write, read the target file before compiling or making further edits. A failed tool call does not prove the file is correct or incorrect.
- Environment warnings such as “Failed to find real location of python.exe” alongside an explicit compile/verify success are not CUDA errors; use the explicit status and actionable compiler/runtime output.

**Scope:** Sections 25–28 apply to CUDA extension episodes using this workspace. They do not relax Sections 1–24 or the existing forbidden/allowed lists. Section 28 applies only to tool/workspace operations and explicitly does not classify tool transport, encoding, payload, or path failures as CUDA coding mistakes.

## 25. Output Tensor Lifetime and Buffer-Reuse Safety

- Treat “previously returned outputs must remain unchanged after later calls” as a hard lifetime constraint. Never write into, recycle, or reuse storage that may still be referenced by any earlier returned tensor. The verifier/harness may retain outputs across repeated calls.
- For AXPBY-like elementwise ops that return a new tensor, default to the allowed per-call `torch::empty_like(...)` and rely on PyTorch’s allocator for caching. Do not add a custom cross-call output pool merely to reduce allocation overhead unless you can prove with public APIs that no live tensor references the buffer.
- Do not use `TensorImpl::use_count`, `Tensor::use_count`, `storage().use_count`, intrusive-pointer counts, or other refcount internals to decide whether an output buffer is recyclable. Such APIs are unstable/version-dependent, and an observed build failed with `"use_count": is not a member of "c10::TensorImpl"`. A single refcount is also not a contract-safe proof against live Python/harness references.
- If an attempted refcount API does not compile, remove that pooling attempt or use per-call allocation. Do not add guessed headers, private ATen/TensorImpl APIs, or unrelated kernel changes to force it to compile.
- A growing pool is a failure signal. If diagnostic output shows the pool size increasing monotonically, or benchmark latency degrades over repeated calls, the buffers are not safely recyclable. Remove the pool and return to per-call allocation. Do not report a faster benchmark from a pool that was not proven to preserve all previous outputs.
- If you experiment with a pre-reserved chunk or suballocation, keep size/offset metadata on the host. Do not write metadata to device memory through a host pointer; that is a host/device memory bug, not an allocation-tuning detail. Ensure each live output has distinct usable storage or an ownership scheme that cannot overwrite a live result.
- Never alias or overwrite input tensors `a`/`b` to save allocation. Compute into a distinct output. When using typed pointers, validate dtype, device, contiguity, and shape assumptions; if buffer reuse is attempted, match all relevant tensor properties, not just shape.

## 26. No Probe Code in Final Production Paths

- A verification PASS does not mean the binding is clean. Probe output can run during verification and benchmark while tests still pass. Before finalizing, inspect `kernels/*.cu`, `kernels/*_binding.cpp`, and `model_new.py` for diagnostic `printf`/`std::printf`, counters, timing probes, pool-inspection logs, and comments such as `PROBE ONLY`.
- Remove diagnostic code embedded in otherwise final files, not only separate `probe_*.cpp` files. Do not leave global probe state, pool-size printing, or debug-only branches reachable from `model_new.py`.
- After removing probe code, recompile if any C++/CUDA/binding file changed, then run verification and benchmark on the exact cleaned files. Verification or benchmark output from the probe version is not evidence for the cleaned version.
- Temporary probe files must be deleted before final validation. If a probe-specific symbol remains declared or referenced, recompile so the extension cannot retain a stale or unintended path.

## 27. Finish on Validated Files and Submit the Report

- Do not end an episode after a verify-only step when the performance gate has not been measured on the exact final files. The final sequence is: finish edits/cleanup → compile if needed → verification → benchmark on the same files → submit the final report.
- Reserve enough budget for that final sequence. If the remaining budget is small, stop optional pooling/refcount experiments and preserve or restore the last candidate that passed compile, verification, and benchmark. Do not start a speculative optimization that may leave the workspace unvalidated.
- A benchmark from an earlier candidate is not final if later writes changed `model_new.py`, binding files, or kernels. Report only measurements actually obtained from the final validated files; if the final files were not benchmarked, say so and treat the performance goal as unconfirmed.
- If a late buffer-pool or binding experiment fails compile, fails verification, or shows degradation, revert to the known-good per-call allocation version and rerun the required gates rather than leaving the experimental binding in place.

## 28. Tool/Workspace Failure Separation (Scoped)

- A missing `operation` field or `ValueError: operation must be one of ...` is a tooling payload error. Retry the write/read with `operation`, `path`, and `content` as separate fields; do not change kernel math, output pooling, or binding allocation in response to that error alone.
- After any failed or ambiguous write, read the target file before compiling. A failed tool call does not prove the file is unchanged or correct.
- Environment warnings such as failure to locate `python.exe` alongside an explicit compile/verify success are not CUDA errors; use the explicit status and actionable compiler/runtime output. A log-path, encoding, transport, or payload failure is not evidence that the CUDA implementation is wrong.


## Repeated appendix, copy 8

**Scope of appended rules:** Sections 9–12 apply to CUDA extension episodes using this workspace. Section 12 applies only to tool/workspace operations and explicitly does not classify tool transport/encoding failures as CUDA coding errors. None of these rules relax the existing forbidden/allowed lists.

## 9. Compile-Safe Binding and Allocation

- Use only the public allocation exception already allowed by this document. For bindings, prefer the documented `torch::empty_like(...)` allocation path.
- Do not use `at::detail::empty_cuda`, other `at::detail::*` symbols, or version-private ATen internals to bypass the dispatcher. These are not stable extension APIs.
- If compilation reports an error such as `'empty_cuda': is not a member of 'at::detail'`, replace the internal allocation call with the documented allocation path. Do not keep the internal call or add guessed headers to force it to compile.
- Fix binding compile errors in the binding files, not by changing kernel math. A missing binding symbol or undeclared allocation API is not evidence that the CUDA kernel algorithm is wrong.
- Do not replace a compiling binding with a speculative “lean hot path” using internal APIs. Allocation micro-optimizations are lower priority than compile success and correctness; if attempted, isolate the change and keep a known-good fallback.
- When a binding specializes to float32/contiguous CUDA tensors, enforce that specialization with `TORCH_CHECK`/equivalent validation or ensure `model_new.py` guarantees it. Do not reinterpret arbitrary tensor storage as `float*`.

## 10. Final Validation Gate

- The final submitted files must be the exact files that passed compile, verification, and benchmark. Any write after the last successful verification/benchmark invalidates those results.
- Final sequence: finish cleanup and final edits, then compile, then run verification, then run profiling/benchmark. Do not stop immediately after a write.
- Do not perform a post-benchmark speculative rewrite of `kernels/*` or `model_new.py` and declare completion. If optimizing after a passing result, preserve or restore the known-good version; re-run compile, verification, and benchmark; if the new version fails, revert and leave the known-good version as final.
- A successful earlier compile does not excuse a later compile failure. Do not submit an uncompiled final candidate.
- Final cleanup counts as an edit. Run compile, verification, and benchmark after cleanup, or perform cleanup before the final validation sequence.
- On compile failure, use the first actionable compiler error. If the reported compile log path cannot be read, use the compile tool output or rerun compile; do not make unrelated kernel changes based on a log-path failure.

## 11. ModelNew and Wrapper Semantics

- Keep `ModelNew.forward` as the normal method and let the harness use `nn.Module.__call__`.
- Do not add `__call__ = forward` or otherwise bypass `nn.Module` hook machinery as a speculative shortcut. It can change harness/hook behavior. Use such a bypass only if explicitly required and revalidated; otherwise remove it before final validation.
- Preserve the constructor and `forward` signatures expected by `model.py`.
- Keep `model_new.py` limited to allowed tensor creation and custom extension ops. Do not add unsupported Python-side torch operations to compensate for binding or kernel issues.

## 12. Tool/Workspace Hygiene (Not CUDA Coding Errors)

- For `workspace` writes, pass `operation: "write"`, `path`, and `content` as separate fields. Do not put file content or Markdown into the `operation` field.
- Treat `ValueError: unknown operation`, missing `operation` errors, malformed tool payloads, and log encoding/mojibake as tooling issues, not CUDA coding mistakes.
- After any failed or ambiguous write, read the target file before compiling. A failed tool call does not prove the file is unchanged or correct.
- If a write/read fails due to tool transport or encoding, retry the tool call cleanly. Do not change CUDA kernel or binding logic solely in response to the tool failure.
- A missing utility file or unreadable path is a workspace/file-availability issue, not evidence that the kernel is incorrect. Use the documented utilities or list the directory to discover the actual files.

**Scope:** Sections 13–16 apply to CUDA extension episodes using this workspace. They do not relax Sections 1–12 or the existing forbidden/allowed lists. Section 16 applies only to tool/workspace operations and does not classify tool transport/encoding failures as CUDA coding errors.

## 13. Post-Tuning Validation State

- Treat `model_new.py` as part of the correctness surface. Changing the selected kernel `config`, wrapper arguments, or forward path can change numerical behavior; a benchmark-only run after such a change does not validate it.
- After any candidate change to `kernels/*`, bindings, or `model_new.py`, run verification before using or reporting benchmark results. A fast benchmark from an unverified candidate is only a performance probe.
- The final validation sequence must run on the exact final files: finish all edits and cleanup, compile if any C++/CUDA/binding file changed, run verification, then run benchmark. Any write after verification makes that verification stale.
- If a compile and a later write occur in the same step, do not count that compile as final validation for the later file. Ensure the last compile precedes the verified final state, or explicitly confirm no compiled file changed and keep the binary consistent with the verified files.
- Do not end an episode immediately after a benchmark-only edit. If the step budget is nearly exhausted, preserve or restore the last fully validated version rather than making an unverified last-moment change.
- If verification fails after a tuning change, reject the faster benchmark. Either fix the kernel/config and rerun compile, verification, and benchmark, or revert to the last verified version.

## 14. Tuned Config and Shape-Specialized Path Safety

- Do not hardcode a tuned `config` in `model_new.py` solely because it benchmarked fastest. Verify that exact config on all verification cases and seeds before leaving it as the final selection.
- For shape-specialized dispatch, ensure every non-default config either has a correct fallback or is unreachable from the final `model_new.py`. A fast path that is wrong for one harness shape must not remain selected.
- When a config-specific case fails, first inspect dispatch guards, supported `F`/`N` assumptions, and fallback coverage. Do not change unrelated kernel math or move the failure into Python.
- Auto/default passing does not validate a hardcoded config. After switching `model_new.py` from auto/default to a fixed config, rerun verification on that exact final config.
- Diagnostic, no-op, copy-only, or debug configs must not be reachable in the final model path.

## 15. Final Cleanup Specifics

- Before final validation, remove diagnostic no-op/copy kernels, unused tuning branches, and intermediate kernel versions from `kernels/` unless they are still required by the final default path.
- If multiple tuning configs remain, ensure they are all semantically correct for supported shapes, or remove/disable the unused ones. Do not leave a known-wrong config available to be selected accidentally.
- Do not submit a kernel containing diagnostic configs that produce no output or copy only, even if they were useful during tuning.
- Cleanup is an edit. After cleanup, run the full final sequence on the exact cleaned files; a successful earlier compile or benchmark does not validate cleaned files.

## 16. Tool Output Interpretation

- Environment warnings such as “Failed to find real location of python.exe” accompanying a successful compile are not compiler errors or CUDA kernel bugs. Use the compile success/failure status and actionable compiler errors.
- If a write/read fails because of a malformed workspace payload or missing `operation`, retry the tool call cleanly. A tool failure by itself is not evidence that the CUDA kernel or binding logic is wrong.
- After any failed or ambiguous write, read the target file before compiling. Do not change CUDA kernel or binding logic solely in response to a tool transport/encoding/payload failure.
- If a reported compile-log path cannot be read, use the compile tool output or rerun compile; do not make unrelated kernel changes based on a log-path failure.

**Scope:** Sections 17–20 apply to CUDA extension episodes using this workspace and do not relax the existing forbidden/allowed lists or validation rules. Section 20 applies only to tool/workspace operations and does not classify tool transport/encoding/path failures as CUDA coding errors.

## 17. Validation Order Is Non-Negotiable

- Compile success, verification success, and benchmark success are separate gates. A successful compile does not validate kernel math or wrapper semantics, and a benchmark without preceding verification on the same files is not evidence of correctness.
- After any edit to `kernels/*`, binding files, or `model_new.py`, use this order: finish edits/cleanup → compile if any C++/CUDA/binding file changed → verification → profiling/benchmark. The final benchmark must run on the exact files that passed verification.
- Do not make a late `model_new.py` config or forward-path edit and stop after a benchmark-only run. If the edit selects a different config, dispatch branch, or wrapper path, rerun verification on that exact file, then rerun the benchmark.
- A benchmark-only run may be used only as a tuning probe. Preserve the last verified selection and either verify the final selected file or revert to the verified one.
- Do not end an episode on a write, a compile-only step, or a benchmark-only step after a change. If step budget is nearly exhausted, do not start another speculative optimization; leave or restore the last fully validated files.
- A write after the last verification invalidates that verification. A compile after the last verification still leaves verification stale; recompile, verify, and benchmark before completion.
- If a benchmark-only run is followed by another edit, the earlier benchmark is not final evidence for the new file.

## 18. Reject Unverified Speedups

- If verification fails after a kernel, binding, or wrapper change, reject that candidate even if its benchmark is faster. Fix the failing path or revert to the last verified implementation, then recompile, verify, and benchmark.
- Do not continue tuning `model_new.py` configs while the underlying kernel is failing verification. Correctness must be restored first.
- When a rewrite breaks correctness, use the last passing kernel as the baseline and prefer a minimal fix over another broad rewrite. Keep a known-good copy until the replacement passes verification.
- A config that is fast for one shape but unverified or incorrect for another harness case must not remain selected in final `model_new.py`.
- Do not leave a faster but unverified or known-wrong config, dispatch branch, or wrapper path reachable from the final model path.

## 19. Probe and Debug Artifact Hygiene

- Temporary probe/diagnostic kernels, alternate bindings, and patch/debug files must not be part of the final module path. If they are added under `kernels/` or referenced by a binding, remove them and rerun compile, verification, and benchmark on the cleaned files.
- Do not use a probe benchmark as evidence that the final implementation is correct.
- Do not leave temporary patch files, probe launchers, or notes in `kernels/`; cleanup is an edit and must precede the final validation sequence.
- After deleting a probe file or reverting a binding, recompile so the extension cannot retain stale symbols or an unintended old path.

## 20. Tool Log Path Failures Are Not CUDA Signals

Scope: tool/workspace operations only; these rules do not classify tool transport/encoding/path failures as CUDA coding mistakes.

- A `FileNotFoundError` or unreadable path for a compile, verification, or profiling log is a tool/path issue. Use the tool’s returned output/status or rerun the tool; do not change kernel math, binding allocation, or `model_new.py` based on the failed read alone.
- If a log path cannot be read, do not infer pass/fail from the missing file. Rerun verification/benchmark or use the tool result already returned.
- After a failed or ambiguous workspace read/write, read the target file before compiling or editing further. A failed tool call is not evidence that the file is correct or incorrect.
- Environment warnings such as failure to locate `python.exe` alongside an explicit compile/verify/benchmark success are not CUDA kernel errors; use the explicit status and actionable compiler/runtime errors.

**Scope:** Sections 21–24 apply to CUDA extension episodes using this workspace. They do not relax the existing forbidden/allowed lists or validation requirements. Section 24 applies only to tool/workspace operations and does not classify tool payload or environment failures as CUDA coding mistakes.

## 21. Vectorized Reduction Semantics and Debug Isolation

- When a vector load (`float2`, `float4`) spans the non-reduced axis, each component belongs to a different output column and must have its own reduction accumulator and its own reciprocal normalization value. Do not sum the components into one scalar, and do not broadcast one scalar result across the vector lanes.
- Store and combine vector partials as vector/per-lane values in shared memory (for example, `float4` accumulators), not as one scalar per thread. Apply the same per-lane requirement when scaling and storing the output.
- A vectorized reduction failure with most elements mismatched and the same wrong values across different block shapes is evidence to inspect lane packing/reduction semantics first, not automatically a race or launch-config problem. Isolate it by testing the scalar path, a vectorized copy/indexing path without the reduction, and then the vectorized reduction.
- Keep diagnostic configs that skip reduction or copy data out of the final path. After fixing the vector path, verify both vector-eligible shapes and non-vector fallback shapes.

## 22. Forward-Path Diagnostics and Post-Tuning Validation

- Do not add PyTorch reference calculations (`pow`, `mean`, `sqrt`, etc.) or other debug tensor ops to `model_new.py`/`ModelNew.forward` to diagnose a kernel. The verifier rejects PyTorch computation during candidate forward; that failure is an instrumentation violation, not evidence that the CUDA math is wrong.
- Perform any reference comparison outside the submitted forward path, and remove the diagnostic before final validation. Do not leave debug instrumentation in `model_new.py`.
- A benchmark-only sweep after changing `model_new.py`—including a hardcoded `config`, dispatch branch, or wrapper argument—is a tuning probe. Before treating that selection as final, run verification on the exact final file, then benchmark. A passing verification of an earlier/default config does not validate the new selection.
- Do not treat compile success as correctness validation. If a new candidate is compiled, run verification before using its benchmark as final evidence. If verification fails, reject the faster candidate or revert to the last verified version.

## 23. Clock-State-Aware Benchmark Comparison

- Absolute candidate latencies in this environment can change by roughly 2× between runs as GPU clock state changes. Do not rank tuning variants solely by absolute microseconds measured in separate runs.
- Compare candidate and `torch.compile` baseline in the same benchmark invocation, or repeat/interleave measurements and report the candidate/baseline ratio. A result is more trustworthy when the ratio is stable across clock states.
- Do not use a fast but unverified benchmark under one clock state to justify leaving a different final configuration. Final evidence must come from the exact final files after the required compile → verify → benchmark order.

## 24. Workspace Write Payloads (Tool/Workspace Only)

- For workspace writes, pass `operation: "write"`, `path`, and `content` as separate fields. Do not omit `operation` or put file content/Markdown in the `operation` field.
- A `ValueError: operation must be one of ...` or a missing-`operation` error is a tool payload problem, not a CUDA kernel or binding bug. Retry the write/read correctly; do not change kernel math or allocation logic in response to that error alone.
- After a failed or ambiguous write, read the target file before compiling or making further edits. A failed tool call does not prove the file is correct or incorrect.
- Environment warnings such as “Failed to find real location of python.exe” alongside an explicit compile/verify success are not CUDA errors; use the explicit status and actionable compiler/runtime output.

**Scope:** Sections 25–28 apply to CUDA extension episodes using this workspace. They do not relax Sections 1–24 or the existing forbidden/allowed lists. Section 28 applies only to tool/workspace operations and explicitly does not classify tool transport, encoding, payload, or path failures as CUDA coding mistakes.

## 25. Output Tensor Lifetime and Buffer-Reuse Safety

- Treat “previously returned outputs must remain unchanged after later calls” as a hard lifetime constraint. Never write into, recycle, or reuse storage that may still be referenced by any earlier returned tensor. The verifier/harness may retain outputs across repeated calls.
- For AXPBY-like elementwise ops that return a new tensor, default to the allowed per-call `torch::empty_like(...)` and rely on PyTorch’s allocator for caching. Do not add a custom cross-call output pool merely to reduce allocation overhead unless you can prove with public APIs that no live tensor references the buffer.
- Do not use `TensorImpl::use_count`, `Tensor::use_count`, `storage().use_count`, intrusive-pointer counts, or other refcount internals to decide whether an output buffer is recyclable. Such APIs are unstable/version-dependent, and an observed build failed with `"use_count": is not a member of "c10::TensorImpl"`. A single refcount is also not a contract-safe proof against live Python/harness references.
- If an attempted refcount API does not compile, remove that pooling attempt or use per-call allocation. Do not add guessed headers, private ATen/TensorImpl APIs, or unrelated kernel changes to force it to compile.
- A growing pool is a failure signal. If diagnostic output shows the pool size increasing monotonically, or benchmark latency degrades over repeated calls, the buffers are not safely recyclable. Remove the pool and return to per-call allocation. Do not report a faster benchmark from a pool that was not proven to preserve all previous outputs.
- If you experiment with a pre-reserved chunk or suballocation, keep size/offset metadata on the host. Do not write metadata to device memory through a host pointer; that is a host/device memory bug, not an allocation-tuning detail. Ensure each live output has distinct usable storage or an ownership scheme that cannot overwrite a live result.
- Never alias or overwrite input tensors `a`/`b` to save allocation. Compute into a distinct output. When using typed pointers, validate dtype, device, contiguity, and shape assumptions; if buffer reuse is attempted, match all relevant tensor properties, not just shape.

## 26. No Probe Code in Final Production Paths

- A verification PASS does not mean the binding is clean. Probe output can run during verification and benchmark while tests still pass. Before finalizing, inspect `kernels/*.cu`, `kernels/*_binding.cpp`, and `model_new.py` for diagnostic `printf`/`std::printf`, counters, timing probes, pool-inspection logs, and comments such as `PROBE ONLY`.
- Remove diagnostic code embedded in otherwise final files, not only separate `probe_*.cpp` files. Do not leave global probe state, pool-size printing, or debug-only branches reachable from `model_new.py`.
- After removing probe code, recompile if any C++/CUDA/binding file changed, then run verification and benchmark on the exact cleaned files. Verification or benchmark output from the probe version is not evidence for the cleaned version.
- Temporary probe files must be deleted before final validation. If a probe-specific symbol remains declared or referenced, recompile so the extension cannot retain a stale or unintended path.

## 27. Finish on Validated Files and Submit the Report

- Do not end an episode after a verify-only step when the performance gate has not been measured on the exact final files. The final sequence is: finish edits/cleanup → compile if needed → verification → benchmark on the same files → submit the final report.
- Reserve enough budget for that final sequence. If the remaining budget is small, stop optional pooling/refcount experiments and preserve or restore the last candidate that passed compile, verification, and benchmark. Do not start a speculative optimization that may leave the workspace unvalidated.
- A benchmark from an earlier candidate is not final if later writes changed `model_new.py`, binding files, or kernels. Report only measurements actually obtained from the final validated files; if the final files were not benchmarked, say so and treat the performance goal as unconfirmed.
- If a late buffer-pool or binding experiment fails compile, fails verification, or shows degradation, revert to the known-good per-call allocation version and rerun the required gates rather than leaving the experimental binding in place.

## 28. Tool/Workspace Failure Separation (Scoped)

- A missing `operation` field or `ValueError: operation must be one of ...` is a tooling payload error. Retry the write/read with `operation`, `path`, and `content` as separate fields; do not change kernel math, output pooling, or binding allocation in response to that error alone.
- After any failed or ambiguous write, read the target file before compiling. A failed tool call does not prove the file is unchanged or correct.
- Environment warnings such as failure to locate `python.exe` alongside an explicit compile/verify success are not CUDA errors; use the explicit status and actionable compiler/runtime output. A log-path, encoding, transport, or payload failure is not evidence that the CUDA implementation is wrong.


## Repeated appendix, copy 9

**Scope of appended rules:** Sections 9–12 apply to CUDA extension episodes using this workspace. Section 12 applies only to tool/workspace operations and explicitly does not classify tool transport/encoding failures as CUDA coding errors. None of these rules relax the existing forbidden/allowed lists.

## 9. Compile-Safe Binding and Allocation

- Use only the public allocation exception already allowed by this document. For bindings, prefer the documented `torch::empty_like(...)` allocation path.
- Do not use `at::detail::empty_cuda`, other `at::detail::*` symbols, or version-private ATen internals to bypass the dispatcher. These are not stable extension APIs.
- If compilation reports an error such as `'empty_cuda': is not a member of 'at::detail'`, replace the internal allocation call with the documented allocation path. Do not keep the internal call or add guessed headers to force it to compile.
- Fix binding compile errors in the binding files, not by changing kernel math. A missing binding symbol or undeclared allocation API is not evidence that the CUDA kernel algorithm is wrong.
- Do not replace a compiling binding with a speculative “lean hot path” using internal APIs. Allocation micro-optimizations are lower priority than compile success and correctness; if attempted, isolate the change and keep a known-good fallback.
- When a binding specializes to float32/contiguous CUDA tensors, enforce that specialization with `TORCH_CHECK`/equivalent validation or ensure `model_new.py` guarantees it. Do not reinterpret arbitrary tensor storage as `float*`.

## 10. Final Validation Gate

- The final submitted files must be the exact files that passed compile, verification, and benchmark. Any write after the last successful verification/benchmark invalidates those results.
- Final sequence: finish cleanup and final edits, then compile, then run verification, then run profiling/benchmark. Do not stop immediately after a write.
- Do not perform a post-benchmark speculative rewrite of `kernels/*` or `model_new.py` and declare completion. If optimizing after a passing result, preserve or restore the known-good version; re-run compile, verification, and benchmark; if the new version fails, revert and leave the known-good version as final.
- A successful earlier compile does not excuse a later compile failure. Do not submit an uncompiled final candidate.
- Final cleanup counts as an edit. Run compile, verification, and benchmark after cleanup, or perform cleanup before the final validation sequence.
- On compile failure, use the first actionable compiler error. If the reported compile log path cannot be read, use the compile tool output or rerun compile; do not make unrelated kernel changes based on a log-path failure.

## 11. ModelNew and Wrapper Semantics

- Keep `ModelNew.forward` as the normal method and let the harness use `nn.Module.__call__`.
- Do not add `__call__ = forward` or otherwise bypass `nn.Module` hook machinery as a speculative shortcut. It can change harness/hook behavior. Use such a bypass only if explicitly required and revalidated; otherwise remove it before final validation.
- Preserve the constructor and `forward` signatures expected by `model.py`.
- Keep `model_new.py` limited to allowed tensor creation and custom extension ops. Do not add unsupported Python-side torch operations to compensate for binding or kernel issues.

## 12. Tool/Workspace Hygiene (Not CUDA Coding Errors)

- For `workspace` writes, pass `operation: "write"`, `path`, and `content` as separate fields. Do not put file content or Markdown into the `operation` field.
- Treat `ValueError: unknown operation`, missing `operation` errors, malformed tool payloads, and log encoding/mojibake as tooling issues, not CUDA coding mistakes.
- After any failed or ambiguous write, read the target file before compiling. A failed tool call does not prove the file is unchanged or correct.
- If a write/read fails due to tool transport or encoding, retry the tool call cleanly. Do not change CUDA kernel or binding logic solely in response to the tool failure.
- A missing utility file or unreadable path is a workspace/file-availability issue, not evidence that the kernel is incorrect. Use the documented utilities or list the directory to discover the actual files.

**Scope:** Sections 13–16 apply to CUDA extension episodes using this workspace. They do not relax Sections 1–12 or the existing forbidden/allowed lists. Section 16 applies only to tool/workspace operations and does not classify tool transport/encoding failures as CUDA coding errors.

## 13. Post-Tuning Validation State

- Treat `model_new.py` as part of the correctness surface. Changing the selected kernel `config`, wrapper arguments, or forward path can change numerical behavior; a benchmark-only run after such a change does not validate it.
- After any candidate change to `kernels/*`, bindings, or `model_new.py`, run verification before using or reporting benchmark results. A fast benchmark from an unverified candidate is only a performance probe.
- The final validation sequence must run on the exact final files: finish all edits and cleanup, compile if any C++/CUDA/binding file changed, run verification, then run benchmark. Any write after verification makes that verification stale.
- If a compile and a later write occur in the same step, do not count that compile as final validation for the later file. Ensure the last compile precedes the verified final state, or explicitly confirm no compiled file changed and keep the binary consistent with the verified files.
- Do not end an episode immediately after a benchmark-only edit. If the step budget is nearly exhausted, preserve or restore the last fully validated version rather than making an unverified last-moment change.
- If verification fails after a tuning change, reject the faster benchmark. Either fix the kernel/config and rerun compile, verification, and benchmark, or revert to the last verified version.

## 14. Tuned Config and Shape-Specialized Path Safety

- Do not hardcode a tuned `config` in `model_new.py` solely because it benchmarked fastest. Verify that exact config on all verification cases and seeds before leaving it as the final selection.
- For shape-specialized dispatch, ensure every non-default config either has a correct fallback or is unreachable from the final `model_new.py`. A fast path that is wrong for one harness shape must not remain selected.
- When a config-specific case fails, first inspect dispatch guards, supported `F`/`N` assumptions, and fallback coverage. Do not change unrelated kernel math or move the failure into Python.
- Auto/default passing does not validate a hardcoded config. After switching `model_new.py` from auto/default to a fixed config, rerun verification on that exact final config.
- Diagnostic, no-op, copy-only, or debug configs must not be reachable in the final model path.

## 15. Final Cleanup Specifics

- Before final validation, remove diagnostic no-op/copy kernels, unused tuning branches, and intermediate kernel versions from `kernels/` unless they are still required by the final default path.
- If multiple tuning configs remain, ensure they are all semantically correct for supported shapes, or remove/disable the unused ones. Do not leave a known-wrong config available to be selected accidentally.
- Do not submit a kernel containing diagnostic configs that produce no output or copy only, even if they were useful during tuning.
- Cleanup is an edit. After cleanup, run the full final sequence on the exact cleaned files; a successful earlier compile or benchmark does not validate cleaned files.

## 16. Tool Output Interpretation

- Environment warnings such as “Failed to find real location of python.exe” accompanying a successful compile are not compiler errors or CUDA kernel bugs. Use the compile success/failure status and actionable compiler errors.
- If a write/read fails because of a malformed workspace payload or missing `operation`, retry the tool call cleanly. A tool failure by itself is not evidence that the CUDA kernel or binding logic is wrong.
- After any failed or ambiguous write, read the target file before compiling. Do not change CUDA kernel or binding logic solely in response to a tool transport/encoding/payload failure.
- If a reported compile-log path cannot be read, use the compile tool output or rerun compile; do not make unrelated kernel changes based on a log-path failure.

**Scope:** Sections 17–20 apply to CUDA extension episodes using this workspace and do not relax the existing forbidden/allowed lists or validation rules. Section 20 applies only to tool/workspace operations and does not classify tool transport/encoding/path failures as CUDA coding errors.

## 17. Validation Order Is Non-Negotiable

- Compile success, verification success, and benchmark success are separate gates. A successful compile does not validate kernel math or wrapper semantics, and a benchmark without preceding verification on the same files is not evidence of correctness.
- After any edit to `kernels/*`, binding files, or `model_new.py`, use this order: finish edits/cleanup → compile if any C++/CUDA/binding file changed → verification → profiling/benchmark. The final benchmark must run on the exact files that passed verification.
- Do not make a late `model_new.py` config or forward-path edit and stop after a benchmark-only run. If the edit selects a different config, dispatch branch, or wrapper path, rerun verification on that exact file, then rerun the benchmark.
- A benchmark-only run may be used only as a tuning probe. Preserve the last verified selection and either verify the final selected file or revert to the verified one.
- Do not end an episode on a write, a compile-only step, or a benchmark-only step after a change. If step budget is nearly exhausted, do not start another speculative optimization; leave or restore the last fully validated files.
- A write after the last verification invalidates that verification. A compile after the last verification still leaves verification stale; recompile, verify, and benchmark before completion.
- If a benchmark-only run is followed by another edit, the earlier benchmark is not final evidence for the new file.

## 18. Reject Unverified Speedups

- If verification fails after a kernel, binding, or wrapper change, reject that candidate even if its benchmark is faster. Fix the failing path or revert to the last verified implementation, then recompile, verify, and benchmark.
- Do not continue tuning `model_new.py` configs while the underlying kernel is failing verification. Correctness must be restored first.
- When a rewrite breaks correctness, use the last passing kernel as the baseline and prefer a minimal fix over another broad rewrite. Keep a known-good copy until the replacement passes verification.
- A config that is fast for one shape but unverified or incorrect for another harness case must not remain selected in final `model_new.py`.
- Do not leave a faster but unverified or known-wrong config, dispatch branch, or wrapper path reachable from the final model path.

## 19. Probe and Debug Artifact Hygiene

- Temporary probe/diagnostic kernels, alternate bindings, and patch/debug files must not be part of the final module path. If they are added under `kernels/` or referenced by a binding, remove them and rerun compile, verification, and benchmark on the cleaned files.
- Do not use a probe benchmark as evidence that the final implementation is correct.
- Do not leave temporary patch files, probe launchers, or notes in `kernels/`; cleanup is an edit and must precede the final validation sequence.
- After deleting a probe file or reverting a binding, recompile so the extension cannot retain stale symbols or an unintended old path.

## 20. Tool Log Path Failures Are Not CUDA Signals

Scope: tool/workspace operations only; these rules do not classify tool transport/encoding/path failures as CUDA coding mistakes.

- A `FileNotFoundError` or unreadable path for a compile, verification, or profiling log is a tool/path issue. Use the tool’s returned output/status or rerun the tool; do not change kernel math, binding allocation, or `model_new.py` based on the failed read alone.
- If a log path cannot be read, do not infer pass/fail from the missing file. Rerun verification/benchmark or use the tool result already returned.
- After a failed or ambiguous workspace read/write, read the target file before compiling or editing further. A failed tool call is not evidence that the file is correct or incorrect.
- Environment warnings such as failure to locate `python.exe` alongside an explicit compile/verify/benchmark success are not CUDA kernel errors; use the explicit status and actionable compiler/runtime errors.

**Scope:** Sections 21–24 apply to CUDA extension episodes using this workspace. They do not relax the existing forbidden/allowed lists or validation requirements. Section 24 applies only to tool/workspace operations and does not classify tool payload or environment failures as CUDA coding mistakes.

## 21. Vectorized Reduction Semantics and Debug Isolation

- When a vector load (`float2`, `float4`) spans the non-reduced axis, each component belongs to a different output column and must have its own reduction accumulator and its own reciprocal normalization value. Do not sum the components into one scalar, and do not broadcast one scalar result across the vector lanes.
- Store and combine vector partials as vector/per-lane values in shared memory (for example, `float4` accumulators), not as one scalar per thread. Apply the same per-lane requirement when scaling and storing the output.
- A vectorized reduction failure with most elements mismatched and the same wrong values across different block shapes is evidence to inspect lane packing/reduction semantics first, not automatically a race or launch-config problem. Isolate it by testing the scalar path, a vectorized copy/indexing path without the reduction, and then the vectorized reduction.
- Keep diagnostic configs that skip reduction or copy data out of the final path. After fixing the vector path, verify both vector-eligible shapes and non-vector fallback shapes.

## 22. Forward-Path Diagnostics and Post-Tuning Validation

- Do not add PyTorch reference calculations (`pow`, `mean`, `sqrt`, etc.) or other debug tensor ops to `model_new.py`/`ModelNew.forward` to diagnose a kernel. The verifier rejects PyTorch computation during candidate forward; that failure is an instrumentation violation, not evidence that the CUDA math is wrong.
- Perform any reference comparison outside the submitted forward path, and remove the diagnostic before final validation. Do not leave debug instrumentation in `model_new.py`.
- A benchmark-only sweep after changing `model_new.py`—including a hardcoded `config`, dispatch branch, or wrapper argument—is a tuning probe. Before treating that selection as final, run verification on the exact final file, then benchmark. A passing verification of an earlier/default config does not validate the new selection.
- Do not treat compile success as correctness validation. If a new candidate is compiled, run verification before using its benchmark as final evidence. If verification fails, reject the faster candidate or revert to the last verified version.

## 23. Clock-State-Aware Benchmark Comparison

- Absolute candidate latencies in this environment can change by roughly 2× between runs as GPU clock state changes. Do not rank tuning variants solely by absolute microseconds measured in separate runs.
- Compare candidate and `torch.compile` baseline in the same benchmark invocation, or repeat/interleave measurements and report the candidate/baseline ratio. A result is more trustworthy when the ratio is stable across clock states.
- Do not use a fast but unverified benchmark under one clock state to justify leaving a different final configuration. Final evidence must come from the exact final files after the required compile → verify → benchmark order.

## 24. Workspace Write Payloads (Tool/Workspace Only)

- For workspace writes, pass `operation: "write"`, `path`, and `content` as separate fields. Do not omit `operation` or put file content/Markdown in the `operation` field.
- A `ValueError: operation must be one of ...` or a missing-`operation` error is a tool payload problem, not a CUDA kernel or binding bug. Retry the write/read correctly; do not change kernel math or allocation logic in response to that error alone.
- After a failed or ambiguous write, read the target file before compiling or making further edits. A failed tool call does not prove the file is correct or incorrect.
- Environment warnings such as “Failed to find real location of python.exe” alongside an explicit compile/verify success are not CUDA errors; use the explicit status and actionable compiler/runtime output.

**Scope:** Sections 25–28 apply to CUDA extension episodes using this workspace. They do not relax Sections 1–24 or the existing forbidden/allowed lists. Section 28 applies only to tool/workspace operations and explicitly does not classify tool transport, encoding, payload, or path failures as CUDA coding mistakes.

## 25. Output Tensor Lifetime and Buffer-Reuse Safety

- Treat “previously returned outputs must remain unchanged after later calls” as a hard lifetime constraint. Never write into, recycle, or reuse storage that may still be referenced by any earlier returned tensor. The verifier/harness may retain outputs across repeated calls.
- For AXPBY-like elementwise ops that return a new tensor, default to the allowed per-call `torch::empty_like(...)` and rely on PyTorch’s allocator for caching. Do not add a custom cross-call output pool merely to reduce allocation overhead unless you can prove with public APIs that no live tensor references the buffer.
- Do not use `TensorImpl::use_count`, `Tensor::use_count`, `storage().use_count`, intrusive-pointer counts, or other refcount internals to decide whether an output buffer is recyclable. Such APIs are unstable/version-dependent, and an observed build failed with `"use_count": is not a member of "c10::TensorImpl"`. A single refcount is also not a contract-safe proof against live Python/harness references.
- If an attempted refcount API does not compile, remove that pooling attempt or use per-call allocation. Do not add guessed headers, private ATen/TensorImpl APIs, or unrelated kernel changes to force it to compile.
- A growing pool is a failure signal. If diagnostic output shows the pool size increasing monotonically, or benchmark latency degrades over repeated calls, the buffers are not safely recyclable. Remove the pool and return to per-call allocation. Do not report a faster benchmark from a pool that was not proven to preserve all previous outputs.
- If you experiment with a pre-reserved chunk or suballocation, keep size/offset metadata on the host. Do not write metadata to device memory through a host pointer; that is a host/device memory bug, not an allocation-tuning detail. Ensure each live output has distinct usable storage or an ownership scheme that cannot overwrite a live result.
- Never alias or overwrite input tensors `a`/`b` to save allocation. Compute into a distinct output. When using typed pointers, validate dtype, device, contiguity, and shape assumptions; if buffer reuse is attempted, match all relevant tensor properties, not just shape.

## 26. No Probe Code in Final Production Paths

- A verification PASS does not mean the binding is clean. Probe output can run during verification and benchmark while tests still pass. Before finalizing, inspect `kernels/*.cu`, `kernels/*_binding.cpp`, and `model_new.py` for diagnostic `printf`/`std::printf`, counters, timing probes, pool-inspection logs, and comments such as `PROBE ONLY`.
- Remove diagnostic code embedded in otherwise final files, not only separate `probe_*.cpp` files. Do not leave global probe state, pool-size printing, or debug-only branches reachable from `model_new.py`.
- After removing probe code, recompile if any C++/CUDA/binding file changed, then run verification and benchmark on the exact cleaned files. Verification or benchmark output from the probe version is not evidence for the cleaned version.
- Temporary probe files must be deleted before final validation. If a probe-specific symbol remains declared or referenced, recompile so the extension cannot retain a stale or unintended path.

## 27. Finish on Validated Files and Submit the Report

- Do not end an episode after a verify-only step when the performance gate has not been measured on the exact final files. The final sequence is: finish edits/cleanup → compile if needed → verification → benchmark on the same files → submit the final report.
- Reserve enough budget for that final sequence. If the remaining budget is small, stop optional pooling/refcount experiments and preserve or restore the last candidate that passed compile, verification, and benchmark. Do not start a speculative optimization that may leave the workspace unvalidated.
- A benchmark from an earlier candidate is not final if later writes changed `model_new.py`, binding files, or kernels. Report only measurements actually obtained from the final validated files; if the final files were not benchmarked, say so and treat the performance goal as unconfirmed.
- If a late buffer-pool or binding experiment fails compile, fails verification, or shows degradation, revert to the known-good per-call allocation version and rerun the required gates rather than leaving the experimental binding in place.

## 28. Tool/Workspace Failure Separation (Scoped)

- A missing `operation` field or `ValueError: operation must be one of ...` is a tooling payload error. Retry the write/read with `operation`, `path`, and `content` as separate fields; do not change kernel math, output pooling, or binding allocation in response to that error alone.
- After any failed or ambiguous write, read the target file before compiling. A failed tool call does not prove the file is unchanged or correct.
- Environment warnings such as failure to locate `python.exe` alongside an explicit compile/verify success are not CUDA errors; use the explicit status and actionable compiler/runtime output. A log-path, encoding, transport, or payload failure is not evidence that the CUDA implementation is wrong.


## Repeated appendix, copy 10

**Scope of appended rules:** Sections 9–12 apply to CUDA extension episodes using this workspace. Section 12 applies only to tool/workspace operations and explicitly does not classify tool transport/encoding failures as CUDA coding errors. None of these rules relax the existing forbidden/allowed lists.

## 9. Compile-Safe Binding and Allocation

- Use only the public allocation exception already allowed by this document. For bindings, prefer the documented `torch::empty_like(...)` allocation path.
- Do not use `at::detail::empty_cuda`, other `at::detail::*` symbols, or version-private ATen internals to bypass the dispatcher. These are not stable extension APIs.
- If compilation reports an error such as `'empty_cuda': is not a member of 'at::detail'`, replace the internal allocation call with the documented allocation path. Do not keep the internal call or add guessed headers to force it to compile.
- Fix binding compile errors in the binding files, not by changing kernel math. A missing binding symbol or undeclared allocation API is not evidence that the CUDA kernel algorithm is wrong.
- Do not replace a compiling binding with a speculative “lean hot path” using internal APIs. Allocation micro-optimizations are lower priority than compile success and correctness; if attempted, isolate the change and keep a known-good fallback.
- When a binding specializes to float32/contiguous CUDA tensors, enforce that specialization with `TORCH_CHECK`/equivalent validation or ensure `model_new.py` guarantees it. Do not reinterpret arbitrary tensor storage as `float*`.

## 10. Final Validation Gate

- The final submitted files must be the exact files that passed compile, verification, and benchmark. Any write after the last successful verification/benchmark invalidates those results.
- Final sequence: finish cleanup and final edits, then compile, then run verification, then run profiling/benchmark. Do not stop immediately after a write.
- Do not perform a post-benchmark speculative rewrite of `kernels/*` or `model_new.py` and declare completion. If optimizing after a passing result, preserve or restore the known-good version; re-run compile, verification, and benchmark; if the new version fails, revert and leave the known-good version as final.
- A successful earlier compile does not excuse a later compile failure. Do not submit an uncompiled final candidate.
- Final cleanup counts as an edit. Run compile, verification, and benchmark after cleanup, or perform cleanup before the final validation sequence.
- On compile failure, use the first actionable compiler error. If the reported compile log path cannot be read, use the compile tool output or rerun compile; do not make unrelated kernel changes based on a log-path failure.

## 11. ModelNew and Wrapper Semantics

- Keep `ModelNew.forward` as the normal method and let the harness use `nn.Module.__call__`.
- Do not add `__call__ = forward` or otherwise bypass `nn.Module` hook machinery as a speculative shortcut. It can change harness/hook behavior. Use such a bypass only if explicitly required and revalidated; otherwise remove it before final validation.
- Preserve the constructor and `forward` signatures expected by `model.py`.
- Keep `model_new.py` limited to allowed tensor creation and custom extension ops. Do not add unsupported Python-side torch operations to compensate for binding or kernel issues.

## 12. Tool/Workspace Hygiene (Not CUDA Coding Errors)

- For `workspace` writes, pass `operation: "write"`, `path`, and `content` as separate fields. Do not put file content or Markdown into the `operation` field.
- Treat `ValueError: unknown operation`, missing `operation` errors, malformed tool payloads, and log encoding/mojibake as tooling issues, not CUDA coding mistakes.
- After any failed or ambiguous write, read the target file before compiling. A failed tool call does not prove the file is unchanged or correct.
- If a write/read fails due to tool transport or encoding, retry the tool call cleanly. Do not change CUDA kernel or binding logic solely in response to the tool failure.
- A missing utility file or unreadable path is a workspace/file-availability issue, not evidence that the kernel is incorrect. Use the documented utilities or list the directory to discover the actual files.

**Scope:** Sections 13–16 apply to CUDA extension episodes using this workspace. They do not relax Sections 1–12 or the existing forbidden/allowed lists. Section 16 applies only to tool/workspace operations and does not classify tool transport/encoding failures as CUDA coding errors.

## 13. Post-Tuning Validation State

- Treat `model_new.py` as part of the correctness surface. Changing the selected kernel `config`, wrapper arguments, or forward path can change numerical behavior; a benchmark-only run after such a change does not validate it.
- After any candidate change to `kernels/*`, bindings, or `model_new.py`, run verification before using or reporting benchmark results. A fast benchmark from an unverified candidate is only a performance probe.
- The final validation sequence must run on the exact final files: finish all edits and cleanup, compile if any C++/CUDA/binding file changed, run verification, then run benchmark. Any write after verification makes that verification stale.
- If a compile and a later write occur in the same step, do not count that compile as final validation for the later file. Ensure the last compile precedes the verified final state, or explicitly confirm no compiled file changed and keep the binary consistent with the verified files.
- Do not end an episode immediately after a benchmark-only edit. If the step budget is nearly exhausted, preserve or restore the last fully validated version rather than making an unverified last-moment change.
- If verification fails after a tuning change, reject the faster benchmark. Either fix the kernel/config and rerun compile, verification, and benchmark, or revert to the last verified version.

## 14. Tuned Config and Shape-Specialized Path Safety

- Do not hardcode a tuned `config` in `model_new.py` solely because it benchmarked fastest. Verify that exact config on all verification cases and seeds before leaving it as the final selection.
- For shape-specialized dispatch, ensure every non-default config either has a correct fallback or is unreachable from the final `model_new.py`. A fast path that is wrong for one harness shape must not remain selected.
- When a config-specific case fails, first inspect dispatch guards, supported `F`/`N` assumptions, and fallback coverage. Do not change unrelated kernel math or move the failure into Python.
- Auto/default passing does not validate a hardcoded config. After switching `model_new.py` from auto/default to a fixed config, rerun verification on that exact final config.
- Diagnostic, no-op, copy-only, or debug configs must not be reachable in the final model path.

## 15. Final Cleanup Specifics

- Before final validation, remove diagnostic no-op/copy kernels, unused tuning branches, and intermediate kernel versions from `kernels/` unless they are still required by the final default path.
- If multiple tuning configs remain, ensure they are all semantically correct for supported shapes, or remove/disable the unused ones. Do not leave a known-wrong config available to be selected accidentally.
- Do not submit a kernel containing diagnostic configs that produce no output or copy only, even if they were useful during tuning.
- Cleanup is an edit. After cleanup, run the full final sequence on the exact cleaned files; a successful earlier compile or benchmark does not validate cleaned files.

## 16. Tool Output Interpretation

- Environment warnings such as “Failed to find real location of python.exe” accompanying a successful compile are not compiler errors or CUDA kernel bugs. Use the compile success/failure status and actionable compiler errors.
- If a write/read fails because of a malformed workspace payload or missing `operation`, retry the tool call cleanly. A tool failure by itself is not evidence that the CUDA kernel or binding logic is wrong.
- After any failed or ambiguous write, read the target file before compiling. Do not change CUDA kernel or binding logic solely in response to a tool transport/encoding/payload failure.
- If a reported compile-log path cannot be read, use the compile tool output or rerun compile; do not make unrelated kernel changes based on a log-path failure.

**Scope:** Sections 17–20 apply to CUDA extension episodes using this workspace and do not relax the existing forbidden/allowed lists or validation rules. Section 20 applies only to tool/workspace operations and does not classify tool transport/encoding/path failures as CUDA coding errors.

## 17. Validation Order Is Non-Negotiable

- Compile success, verification success, and benchmark success are separate gates. A successful compile does not validate kernel math or wrapper semantics, and a benchmark without preceding verification on the same files is not evidence of correctness.
- After any edit to `kernels/*`, binding files, or `model_new.py`, use this order: finish edits/cleanup → compile if any C++/CUDA/binding file changed → verification → profiling/benchmark. The final benchmark must run on the exact files that passed verification.
- Do not make a late `model_new.py` config or forward-path edit and stop after a benchmark-only run. If the edit selects a different config, dispatch branch, or wrapper path, rerun verification on that exact file, then rerun the benchmark.
- A benchmark-only run may be used only as a tuning probe. Preserve the last verified selection and either verify the final selected file or revert to the verified one.
- Do not end an episode on a write, a compile-only step, or a benchmark-only step after a change. If step budget is nearly exhausted, do not start another speculative optimization; leave or restore the last fully validated files.
- A write after the last verification invalidates that verification. A compile after the last verification still leaves verification stale; recompile, verify, and benchmark before completion.
- If a benchmark-only run is followed by another edit, the earlier benchmark is not final evidence for the new file.

## 18. Reject Unverified Speedups

- If verification fails after a kernel, binding, or wrapper change, reject that candidate even if its benchmark is faster. Fix the failing path or revert to the last verified implementation, then recompile, verify, and benchmark.
- Do not continue tuning `model_new.py` configs while the underlying kernel is failing verification. Correctness must be restored first.
- When a rewrite breaks correctness, use the last passing kernel as the baseline and prefer a minimal fix over another broad rewrite. Keep a known-good copy until the replacement passes verification.
- A config that is fast for one shape but unverified or incorrect for another harness case must not remain selected in final `model_new.py`.
- Do not leave a faster but unverified or known-wrong config, dispatch branch, or wrapper path reachable from the final model path.

## 19. Probe and Debug Artifact Hygiene

- Temporary probe/diagnostic kernels, alternate bindings, and patch/debug files must not be part of the final module path. If they are added under `kernels/` or referenced by a binding, remove them and rerun compile, verification, and benchmark on the cleaned files.
- Do not use a probe benchmark as evidence that the final implementation is correct.
- Do not leave temporary patch files, probe launchers, or notes in `kernels/`; cleanup is an edit and must precede the final validation sequence.
- After deleting a probe file or reverting a binding, recompile so the extension cannot retain stale symbols or an unintended old path.

## 20. Tool Log Path Failures Are Not CUDA Signals

Scope: tool/workspace operations only; these rules do not classify tool transport/encoding/path failures as CUDA coding mistakes.

- A `FileNotFoundError` or unreadable path for a compile, verification, or profiling log is a tool/path issue. Use the tool’s returned output/status or rerun the tool; do not change kernel math, binding allocation, or `model_new.py` based on the failed read alone.
- If a log path cannot be read, do not infer pass/fail from the missing file. Rerun verification/benchmark or use the tool result already returned.
- After a failed or ambiguous workspace read/write, read the target file before compiling or editing further. A failed tool call is not evidence that the file is correct or incorrect.
- Environment warnings such as failure to locate `python.exe` alongside an explicit compile/verify/benchmark success are not CUDA kernel errors; use the explicit status and actionable compiler/runtime errors.

**Scope:** Sections 21–24 apply to CUDA extension episodes using this workspace. They do not relax the existing forbidden/allowed lists or validation requirements. Section 24 applies only to tool/workspace operations and does not classify tool payload or environment failures as CUDA coding mistakes.

## 21. Vectorized Reduction Semantics and Debug Isolation

- When a vector load (`float2`, `float4`) spans the non-reduced axis, each component belongs to a different output column and must have its own reduction accumulator and its own reciprocal normalization value. Do not sum the components into one scalar, and do not broadcast one scalar result across the vector lanes.
- Store and combine vector partials as vector/per-lane values in shared memory (for example, `float4` accumulators), not as one scalar per thread. Apply the same per-lane requirement when scaling and storing the output.
- A vectorized reduction failure with most elements mismatched and the same wrong values across different block shapes is evidence to inspect lane packing/reduction semantics first, not automatically a race or launch-config problem. Isolate it by testing the scalar path, a vectorized copy/indexing path without the reduction, and then the vectorized reduction.
- Keep diagnostic configs that skip reduction or copy data out of the final path. After fixing the vector path, verify both vector-eligible shapes and non-vector fallback shapes.

## 22. Forward-Path Diagnostics and Post-Tuning Validation

- Do not add PyTorch reference calculations (`pow`, `mean`, `sqrt`, etc.) or other debug tensor ops to `model_new.py`/`ModelNew.forward` to diagnose a kernel. The verifier rejects PyTorch computation during candidate forward; that failure is an instrumentation violation, not evidence that the CUDA math is wrong.
- Perform any reference comparison outside the submitted forward path, and remove the diagnostic before final validation. Do not leave debug instrumentation in `model_new.py`.
- A benchmark-only sweep after changing `model_new.py`—including a hardcoded `config`, dispatch branch, or wrapper argument—is a tuning probe. Before treating that selection as final, run verification on the exact final file, then benchmark. A passing verification of an earlier/default config does not validate the new selection.
- Do not treat compile success as correctness validation. If a new candidate is compiled, run verification before using its benchmark as final evidence. If verification fails, reject the faster candidate or revert to the last verified version.

## 23. Clock-State-Aware Benchmark Comparison

- Absolute candidate latencies in this environment can change by roughly 2× between runs as GPU clock state changes. Do not rank tuning variants solely by absolute microseconds measured in separate runs.
- Compare candidate and `torch.compile` baseline in the same benchmark invocation, or repeat/interleave measurements and report the candidate/baseline ratio. A result is more trustworthy when the ratio is stable across clock states.
- Do not use a fast but unverified benchmark under one clock state to justify leaving a different final configuration. Final evidence must come from the exact final files after the required compile → verify → benchmark order.

## 24. Workspace Write Payloads (Tool/Workspace Only)

- For workspace writes, pass `operation: "write"`, `path`, and `content` as separate fields. Do not omit `operation` or put file content/Markdown in the `operation` field.
- A `ValueError: operation must be one of ...` or a missing-`operation` error is a tool payload problem, not a CUDA kernel or binding bug. Retry the write/read correctly; do not change kernel math or allocation logic in response to that error alone.
- After a failed or ambiguous write, read the target file before compiling or making further edits. A failed tool call does not prove the file is correct or incorrect.
- Environment warnings such as “Failed to find real location of python.exe” alongside an explicit compile/verify success are not CUDA errors; use the explicit status and actionable compiler/runtime output.

**Scope:** Sections 25–28 apply to CUDA extension episodes using this workspace. They do not relax Sections 1–24 or the existing forbidden/allowed lists. Section 28 applies only to tool/workspace operations and explicitly does not classify tool transport, encoding, payload, or path failures as CUDA coding mistakes.

## 25. Output Tensor Lifetime and Buffer-Reuse Safety

- Treat “previously returned outputs must remain unchanged after later calls” as a hard lifetime constraint. Never write into, recycle, or reuse storage that may still be referenced by any earlier returned tensor. The verifier/harness may retain outputs across repeated calls.
- For AXPBY-like elementwise ops that return a new tensor, default to the allowed per-call `torch::empty_like(...)` and rely on PyTorch’s allocator for caching. Do not add a custom cross-call output pool merely to reduce allocation overhead unless you can prove with public APIs that no live tensor references the buffer.
- Do not use `TensorImpl::use_count`, `Tensor::use_count`, `storage().use_count`, intrusive-pointer counts, or other refcount internals to decide whether an output buffer is recyclable. Such APIs are unstable/version-dependent, and an observed build failed with `"use_count": is not a member of "c10::TensorImpl"`. A single refcount is also not a contract-safe proof against live Python/harness references.
- If an attempted refcount API does not compile, remove that pooling attempt or use per-call allocation. Do not add guessed headers, private ATen/TensorImpl APIs, or unrelated kernel changes to force it to compile.
- A growing pool is a failure signal. If diagnostic output shows the pool size increasing monotonically, or benchmark latency degrades over repeated calls, the buffers are not safely recyclable. Remove the pool and return to per-call allocation. Do not report a faster benchmark from a pool that was not proven to preserve all previous outputs.
- If you experiment with a pre-reserved chunk or suballocation, keep size/offset metadata on the host. Do not write metadata to device memory through a host pointer; that is a host/device memory bug, not an allocation-tuning detail. Ensure each live output has distinct usable storage or an ownership scheme that cannot overwrite a live result.
- Never alias or overwrite input tensors `a`/`b` to save allocation. Compute into a distinct output. When using typed pointers, validate dtype, device, contiguity, and shape assumptions; if buffer reuse is attempted, match all relevant tensor properties, not just shape.

## 26. No Probe Code in Final Production Paths

- A verification PASS does not mean the binding is clean. Probe output can run during verification and benchmark while tests still pass. Before finalizing, inspect `kernels/*.cu`, `kernels/*_binding.cpp`, and `model_new.py` for diagnostic `printf`/`std::printf`, counters, timing probes, pool-inspection logs, and comments such as `PROBE ONLY`.
- Remove diagnostic code embedded in otherwise final files, not only separate `probe_*.cpp` files. Do not leave global probe state, pool-size printing, or debug-only branches reachable from `model_new.py`.
- After removing probe code, recompile if any C++/CUDA/binding file changed, then run verification and benchmark on the exact cleaned files. Verification or benchmark output from the probe version is not evidence for the cleaned version.
- Temporary probe files must be deleted before final validation. If a probe-specific symbol remains declared or referenced, recompile so the extension cannot retain a stale or unintended path.

## 27. Finish on Validated Files and Submit the Report

- Do not end an episode after a verify-only step when the performance gate has not been measured on the exact final files. The final sequence is: finish edits/cleanup → compile if needed → verification → benchmark on the same files → submit the final report.
- Reserve enough budget for that final sequence. If the remaining budget is small, stop optional pooling/refcount experiments and preserve or restore the last candidate that passed compile, verification, and benchmark. Do not start a speculative optimization that may leave the workspace unvalidated.
- A benchmark from an earlier candidate is not final if later writes changed `model_new.py`, binding files, or kernels. Report only measurements actually obtained from the final validated files; if the final files were not benchmarked, say so and treat the performance goal as unconfirmed.
- If a late buffer-pool or binding experiment fails compile, fails verification, or shows degradation, revert to the known-good per-call allocation version and rerun the required gates rather than leaving the experimental binding in place.

## 28. Tool/Workspace Failure Separation (Scoped)

- A missing `operation` field or `ValueError: operation must be one of ...` is a tooling payload error. Retry the write/read with `operation`, `path`, and `content` as separate fields; do not change kernel math, output pooling, or binding allocation in response to that error alone.
- After any failed or ambiguous write, read the target file before compiling. A failed tool call does not prove the file is unchanged or correct.
- Environment warnings such as failure to locate `python.exe` alongside an explicit compile/verify success are not CUDA errors; use the explicit status and actionable compiler/runtime output. A log-path, encoding, transport, or payload failure is not evidence that the CUDA implementation is wrong.


## Repeated appendix, copy 11

**Scope of appended rules:** Sections 9–12 apply to CUDA extension episodes using this workspace. Section 12 applies only to tool/workspace operations and explicitly does not classify tool transport/encoding failures as CUDA coding errors. None of these rules relax the existing forbidden/allowed lists.

## 9. Compile-Safe Binding and Allocation

- Use only the public allocation exception already allowed by this document. For bindings, prefer the documented `torch::empty_like(...)` allocation path.
- Do not use `at::detail::empty_cuda`, other `at::detail::*` symbols, or version-private ATen internals to bypass the dispatcher. These are not stable extension APIs.
- If compilation reports an error such as `'empty_cuda': is not a member of 'at::detail'`, replace the internal allocation call with the documented allocation path. Do not keep the internal call or add guessed headers to force it to compile.
- Fix binding compile errors in the binding files, not by changing kernel math. A missing binding symbol or undeclared allocation API is not evidence that the CUDA kernel algorithm is wrong.
- Do not replace a compiling binding with a speculative “lean hot path” using internal APIs. Allocation micro-optimizations are lower priority than compile success and correctness; if attempted, isolate the change and keep a known-good fallback.
- When a binding specializes to float32/contiguous CUDA tensors, enforce that specialization with `TORCH_CHECK`/equivalent validation or ensure `model_new.py` guarantees it. Do not reinterpret arbitrary tensor storage as `float*`.

## 10. Final Validation Gate

- The final submitted files must be the exact files that passed compile, verification, and benchmark. Any write after the last successful verification/benchmark invalidates those results.
- Final sequence: finish cleanup and final edits, then compile, then run verification, then run profiling/benchmark. Do not stop immediately after a write.
- Do not perform a post-benchmark speculative rewrite of `kernels/*` or `model_new.py` and declare completion. If optimizing after a passing result, preserve or restore the known-good version; re-run compile, verification, and benchmark; if the new version fails, revert and leave the known-good version as final.
- A successful earlier compile does not excuse a later compile failure. Do not submit an uncompiled final candidate.
- Final cleanup counts as an edit. Run compile, verification, and benchmark after cleanup, or perform cleanup before the final validation sequence.
- On compile failure, use the first actionable compiler error. If the reported compile log path cannot be read, use the compile tool output or rerun compile; do not make unrelated kernel changes based on a log-path failure.

## 11. ModelNew and Wrapper Semantics

- Keep `ModelNew.forward` as the normal method and let the harness use `nn.Module.__call__`.
- Do not add `__call__ = forward` or otherwise bypass `nn.Module` hook machinery as a speculative shortcut. It can change harness/hook behavior. Use such a bypass only if explicitly required and revalidated; otherwise remove it before final validation.
- Preserve the constructor and `forward` signatures expected by `model.py`.
- Keep `model_new.py` limited to allowed tensor creation and custom extension ops. Do not add unsupported Python-side torch operations to compensate for binding or kernel issues.

## 12. Tool/Workspace Hygiene (Not CUDA Coding Errors)

- For `workspace` writes, pass `operation: "write"`, `path`, and `content` as separate fields. Do not put file content or Markdown into the `operation` field.
- Treat `ValueError: unknown operation`, missing `operation` errors, malformed tool payloads, and log encoding/mojibake as tooling issues, not CUDA coding mistakes.
- After any failed or ambiguous write, read the target file before compiling. A failed tool call does not prove the file is unchanged or correct.
- If a write/read fails due to tool transport or encoding, retry the tool call cleanly. Do not change CUDA kernel or binding logic solely in response to the tool failure.
- A missing utility file or unreadable path is a workspace/file-availability issue, not evidence that the kernel is incorrect. Use the documented utilities or list the directory to discover the actual files.

**Scope:** Sections 13–16 apply to CUDA extension episodes using this workspace. They do not relax Sections 1–12 or the existing forbidden/allowed lists. Section 16 applies only to tool/workspace operations and does not classify tool transport/encoding failures as CUDA coding errors.

## 13. Post-Tuning Validation State

- Treat `model_new.py` as part of the correctness surface. Changing the selected kernel `config`, wrapper arguments, or forward path can change numerical behavior; a benchmark-only run after such a change does not validate it.
- After any candidate change to `kernels/*`, bindings, or `model_new.py`, run verification before using or reporting benchmark results. A fast benchmark from an unverified candidate is only a performance probe.
- The final validation sequence must run on the exact final files: finish all edits and cleanup, compile if any C++/CUDA/binding file changed, run verification, then run benchmark. Any write after verification makes that verification stale.
- If a compile and a later write occur in the same step, do not count that compile as final validation for the later file. Ensure the last compile precedes the verified final state, or explicitly confirm no compiled file changed and keep the binary consistent with the verified files.
- Do not end an episode immediately after a benchmark-only edit. If the step budget is nearly exhausted, preserve or restore the last fully validated version rather than making an unverified last-moment change.
- If verification fails after a tuning change, reject the faster benchmark. Either fix the kernel/config and rerun compile, verification, and benchmark, or revert to the last verified version.

## 14. Tuned Config and Shape-Specialized Path Safety

- Do not hardcode a tuned `config` in `model_new.py` solely because it benchmarked fastest. Verify that exact config on all verification cases and seeds before leaving it as the final selection.
- For shape-specialized dispatch, ensure every non-default config either has a correct fallback or is unreachable from the final `model_new.py`. A fast path that is wrong for one harness shape must not remain selected.
- When a config-specific case fails, first inspect dispatch guards, supported `F`/`N` assumptions, and fallback coverage. Do not change unrelated kernel math or move the failure into Python.
- Auto/default passing does not validate a hardcoded config. After switching `model_new.py` from auto/default to a fixed config, rerun verification on that exact final config.
- Diagnostic, no-op, copy-only, or debug configs must not be reachable in the final model path.

## 15. Final Cleanup Specifics

- Before final validation, remove diagnostic no-op/copy kernels, unused tuning branches, and intermediate kernel versions from `kernels/` unless they are still required by the final default path.
- If multiple tuning configs remain, ensure they are all semantically correct for supported shapes, or remove/disable the unused ones. Do not leave a known-wrong config available to be selected accidentally.
- Do not submit a kernel containing diagnostic configs that produce no output or copy only, even if they were useful during tuning.
- Cleanup is an edit. After cleanup, run the full final sequence on the exact cleaned files; a successful earlier compile or benchmark does not validate cleaned files.

## 16. Tool Output Interpretation

- Environment warnings such as “Failed to find real location of python.exe” accompanying a successful compile are not compiler errors or CUDA kernel bugs. Use the compile success/failure status and actionable compiler errors.
- If a write/read fails because of a malformed workspace payload or missing `operation`, retry the tool call cleanly. A tool failure by itself is not evidence that the CUDA kernel or binding logic is wrong.
- After any failed or ambiguous write, read the target file before compiling. Do not change CUDA kernel or binding logic solely in response to a tool transport/encoding/payload failure.
- If a reported compile-log path cannot be read, use the compile tool output or rerun compile; do not make unrelated kernel changes based on a log-path failure.

**Scope:** Sections 17–20 apply to CUDA extension episodes using this workspace and do not relax the existing forbidden/allowed lists or validation rules. Section 20 applies only to tool/workspace operations and does not classify tool transport/encoding/path failures as CUDA coding errors.

## 17. Validation Order Is Non-Negotiable

- Compile success, verification success, and benchmark success are separate gates. A successful compile does not validate kernel math or wrapper semantics, and a benchmark without preceding verification on the same files is not evidence of correctness.
- After any edit to `kernels/*`, binding files, or `model_new.py`, use this order: finish edits/cleanup → compile if any C++/CUDA/binding file changed → verification → profiling/benchmark. The final benchmark must run on the exact files that passed verification.
- Do not make a late `model_new.py` config or forward-path edit and stop after a benchmark-only run. If the edit selects a different config, dispatch branch, or wrapper path, rerun verification on that exact file, then rerun the benchmark.
- A benchmark-only run may be used only as a tuning probe. Preserve the last verified selection and either verify the final selected file or revert to the verified one.
- Do not end an episode on a write, a compile-only step, or a benchmark-only step after a change. If step budget is nearly exhausted, do not start another speculative optimization; leave or restore the last fully validated files.
- A write after the last verification invalidates that verification. A compile after the last verification still leaves verification stale; recompile, verify, and benchmark before completion.
- If a benchmark-only run is followed by another edit, the earlier benchmark is not final evidence for the new file.

## 18. Reject Unverified Speedups

- If verification fails after a kernel, binding, or wrapper change, reject that candidate even if its benchmark is faster. Fix the failing path or revert to the last verified implementation, then recompile, verify, and benchmark.
- Do not continue tuning `model_new.py` configs while the underlying kernel is failing verification. Correctness must be restored first.
- When a rewrite breaks correctness, use the last passing kernel as the baseline and prefer a minimal fix over another broad rewrite. Keep a known-good copy until the replacement passes verification.
- A config that is fast for one shape but unverified or incorrect for another harness case must not remain selected in final `model_new.py`.
- Do not leave a faster but unverified or known-wrong config, dispatch branch, or wrapper path reachable from the final model path.

## 19. Probe and Debug Artifact Hygiene

- Temporary probe/diagnostic kernels, alternate bindings, and patch/debug files must not be part of the final module path. If they are added under `kernels/` or referenced by a binding, remove them and rerun compile, verification, and benchmark on the cleaned files.
- Do not use a probe benchmark as evidence that the final implementation is correct.
- Do not leave temporary patch files, probe launchers, or notes in `kernels/`; cleanup is an edit and must precede the final validation sequence.
- After deleting a probe file or reverting a binding, recompile so the extension cannot retain stale symbols or an unintended old path.

## 20. Tool Log Path Failures Are Not CUDA Signals

Scope: tool/workspace operations only; these rules do not classify tool transport/encoding/path failures as CUDA coding mistakes.

- A `FileNotFoundError` or unreadable path for a compile, verification, or profiling log is a tool/path issue. Use the tool’s returned output/status or rerun the tool; do not change kernel math, binding allocation, or `model_new.py` based on the failed read alone.
- If a log path cannot be read, do not infer pass/fail from the missing file. Rerun verification/benchmark or use the tool result already returned.
- After a failed or ambiguous workspace read/write, read the target file before compiling or editing further. A failed tool call is not evidence that the file is correct or incorrect.
- Environment warnings such as failure to locate `python.exe` alongside an explicit compile/verify/benchmark success are not CUDA kernel errors; use the explicit status and actionable compiler/runtime errors.

**Scope:** Sections 21–24 apply to CUDA extension episodes using this workspace. They do not relax the existing forbidden/allowed lists or validation requirements. Section 24 applies only to tool/workspace operations and does not classify tool payload or environment failures as CUDA coding mistakes.

## 21. Vectorized Reduction Semantics and Debug Isolation

- When a vector load (`float2`, `float4`) spans the non-reduced axis, each component belongs to a different output column and must have its own reduction accumulator and its own reciprocal normalization value. Do not sum the components into one scalar, and do not broadcast one scalar result across the vector lanes.
- Store and combine vector partials as vector/per-lane values in shared memory (for example, `float4` accumulators), not as one scalar per thread. Apply the same per-lane requirement when scaling and storing the output.
- A vectorized reduction failure with most elements mismatched and the same wrong values across different block shapes is evidence to inspect lane packing/reduction semantics first, not automatically a race or launch-config problem. Isolate it by testing the scalar path, a vectorized copy/indexing path without the reduction, and then the vectorized reduction.
- Keep diagnostic configs that skip reduction or copy data out of the final path. After fixing the vector path, verify both vector-eligible shapes and non-vector fallback shapes.

## 22. Forward-Path Diagnostics and Post-Tuning Validation

- Do not add PyTorch reference calculations (`pow`, `mean`, `sqrt`, etc.) or other debug tensor ops to `model_new.py`/`ModelNew.forward` to diagnose a kernel. The verifier rejects PyTorch computation during candidate forward; that failure is an instrumentation violation, not evidence that the CUDA math is wrong.
- Perform any reference comparison outside the submitted forward path, and remove the diagnostic before final validation. Do not leave debug instrumentation in `model_new.py`.
- A benchmark-only sweep after changing `model_new.py`—including a hardcoded `config`, dispatch branch, or wrapper argument—is a tuning probe. Before treating that selection as final, run verification on the exact final file, then benchmark. A passing verification of an earlier/default config does not validate the new selection.
- Do not treat compile success as correctness validation. If a new candidate is compiled, run verification before using its benchmark as final evidence. If verification fails, reject the faster candidate or revert to the last verified version.

## 23. Clock-State-Aware Benchmark Comparison

- Absolute candidate latencies in this environment can change by roughly 2× between runs as GPU clock state changes. Do not rank tuning variants solely by absolute microseconds measured in separate runs.
- Compare candidate and `torch.compile` baseline in the same benchmark invocation, or repeat/interleave measurements and report the candidate/baseline ratio. A result is more trustworthy when the ratio is stable across clock states.
- Do not use a fast but unverified benchmark under one clock state to justify leaving a different final configuration. Final evidence must come from the exact final files after the required compile → verify → benchmark order.

## 24. Workspace Write Payloads (Tool/Workspace Only)

- For workspace writes, pass `operation: "write"`, `path`, and `content` as separate fields. Do not omit `operation` or put file content/Markdown in the `operation` field.
- A `ValueError: operation must be one of ...` or a missing-`operation` error is a tool payload problem, not a CUDA kernel or binding bug. Retry the write/read correctly; do not change kernel math or allocation logic in response to that error alone.
- After a failed or ambiguous write, read the target file before compiling or making further edits. A failed tool call does not prove the file is correct or incorrect.
- Environment warnings such as “Failed to find real location of python.exe” alongside an explicit compile/verify success are not CUDA errors; use the explicit status and actionable compiler/runtime output.

**Scope:** Sections 25–28 apply to CUDA extension episodes using this workspace. They do not relax Sections 1–24 or the existing forbidden/allowed lists. Section 28 applies only to tool/workspace operations and explicitly does not classify tool transport, encoding, payload, or path failures as CUDA coding mistakes.

## 25. Output Tensor Lifetime and Buffer-Reuse Safety

- Treat “previously returned outputs must remain unchanged after later calls” as a hard lifetime constraint. Never write into, recycle, or reuse storage that may still be referenced by any earlier returned tensor. The verifier/harness may retain outputs across repeated calls.
- For AXPBY-like elementwise ops that return a new tensor, default to the allowed per-call `torch::empty_like(...)` and rely on PyTorch’s allocator for caching. Do not add a custom cross-call output pool merely to reduce allocation overhead unless you can prove with public APIs that no live tensor references the buffer.
- Do not use `TensorImpl::use_count`, `Tensor::use_count`, `storage().use_count`, intrusive-pointer counts, or other refcount internals to decide whether an output buffer is recyclable. Such APIs are unstable/version-dependent, and an observed build failed with `"use_count": is not a member of "c10::TensorImpl"`. A single refcount is also not a contract-safe proof against live Python/harness references.
- If an attempted refcount API does not compile, remove that pooling attempt or use per-call allocation. Do not add guessed headers, private ATen/TensorImpl APIs, or unrelated kernel changes to force it to compile.
- A growing pool is a failure signal. If diagnostic output shows the pool size increasing monotonically, or benchmark latency degrades over repeated calls, the buffers are not safely recyclable. Remove the pool and return to per-call allocation. Do not report a faster benchmark from a pool that was not proven to preserve all previous outputs.
- If you experiment with a pre-reserved chunk or suballocation, keep size/offset metadata on the host. Do not write metadata to device memory through a host pointer; that is a host/device memory bug, not an allocation-tuning detail. Ensure each live output has distinct usable storage or an ownership scheme that cannot overwrite a live result.
- Never alias or overwrite input tensors `a`/`b` to save allocation. Compute into a distinct output. When using typed pointers, validate dtype, device, contiguity, and shape assumptions; if buffer reuse is attempted, match all relevant tensor properties, not just shape.

## 26. No Probe Code in Final Production Paths

- A verification PASS does not mean the binding is clean. Probe output can run during verification and benchmark while tests still pass. Before finalizing, inspect `kernels/*.cu`, `kernels/*_binding.cpp`, and `model_new.py` for diagnostic `printf`/`std::printf`, counters, timing probes, pool-inspection logs, and comments such as `PROBE ONLY`.
- Remove diagnostic code embedded in otherwise final files, not only separate `probe_*.cpp` files. Do not leave global probe state, pool-size printing, or debug-only branches reachable from `model_new.py`.
- After removing probe code, recompile if any C++/CUDA/binding file changed, then run verification and benchmark on the exact cleaned files. Verification or benchmark output from the probe version is not evidence for the cleaned version.
- Temporary probe files must be deleted before final validation. If a probe-specific symbol remains declared or referenced, recompile so the extension cannot retain a stale or unintended path.

## 27. Finish on Validated Files and Submit the Report

- Do not end an episode after a verify-only step when the performance gate has not been measured on the exact final files. The final sequence is: finish edits/cleanup → compile if needed → verification → benchmark on the same files → submit the final report.
- Reserve enough budget for that final sequence. If the remaining budget is small, stop optional pooling/refcount experiments and preserve or restore the last candidate that passed compile, verification, and benchmark. Do not start a speculative optimization that may leave the workspace unvalidated.
- A benchmark from an earlier candidate is not final if later writes changed `model_new.py`, binding files, or kernels. Report only measurements actually obtained from the final validated files; if the final files were not benchmarked, say so and treat the performance goal as unconfirmed.
- If a late buffer-pool or binding experiment fails compile, fails verification, or shows degradation, revert to the known-good per-call allocation version and rerun the required gates rather than leaving the experimental binding in place.

## 28. Tool/Workspace Failure Separation (Scoped)

- A missing `operation` field or `ValueError: operation must be one of ...` is a tooling payload error. Retry the write/read with `operation`, `path`, and `content` as separate fields; do not change kernel math, output pooling, or binding allocation in response to that error alone.
- After any failed or ambiguous write, read the target file before compiling. A failed tool call does not prove the file is unchanged or correct.
- Environment warnings such as failure to locate `python.exe` alongside an explicit compile/verify success are not CUDA errors; use the explicit status and actionable compiler/runtime output. A log-path, encoding, transport, or payload failure is not evidence that the CUDA implementation is wrong.


## Repeated appendix, copy 12

**Scope of appended rules:** Sections 9–12 apply to CUDA extension episodes using this workspace. Section 12 applies only to tool/workspace operations and explicitly does not classify tool transport/encoding failures as CUDA coding errors. None of these rules relax the existing forbidden/allowed lists.

## 9. Compile-Safe Binding and Allocation

- Use only the public allocation exception already allowed by this document. For bindings, prefer the documented `torch::empty_like(...)` allocation path.
- Do not use `at::detail::empty_cuda`, other `at::detail::*` symbols, or version-private ATen internals to bypass the dispatcher. These are not stable extension APIs.
- If compilation reports an error such as `'empty_cuda': is not a member of 'at::detail'`, replace the internal allocation call with the documented allocation path. Do not keep the internal call or add guessed headers to force it to compile.
- Fix binding compile errors in the binding files, not by changing kernel math. A missing binding symbol or undeclared allocation API is not evidence that the CUDA kernel algorithm is wrong.
- Do not replace a compiling binding with a speculative “lean hot path” using internal APIs. Allocation micro-optimizations are lower priority than compile success and correctness; if attempted, isolate the change and keep a known-good fallback.
- When a binding specializes to float32/contiguous CUDA tensors, enforce that specialization with `TORCH_CHECK`/equivalent validation or ensure `model_new.py` guarantees it. Do not reinterpret arbitrary tensor storage as `float*`.

## 10. Final Validation Gate

- The final submitted files must be the exact files that passed compile, verification, and benchmark. Any write after the last successful verification/benchmark invalidates those results.
- Final sequence: finish cleanup and final edits, then compile, then run verification, then run profiling/benchmark. Do not stop immediately after a write.
- Do not perform a post-benchmark speculative rewrite of `kernels/*` or `model_new.py` and declare completion. If optimizing after a passing result, preserve or restore the known-good version; re-run compile, verification, and benchmark; if the new version fails, revert and leave the known-good version as final.
- A successful earlier compile does not excuse a later compile failure. Do not submit an uncompiled final candidate.
- Final cleanup counts as an edit. Run compile, verification, and benchmark after cleanup, or perform cleanup before the final validation sequence.
- On compile failure, use the first actionable compiler error. If the reported compile log path cannot be read, use the compile tool output or rerun compile; do not make unrelated kernel changes based on a log-path failure.

## 11. ModelNew and Wrapper Semantics

- Keep `ModelNew.forward` as the normal method and let the harness use `nn.Module.__call__`.
- Do not add `__call__ = forward` or otherwise bypass `nn.Module` hook machinery as a speculative shortcut. It can change harness/hook behavior. Use such a bypass only if explicitly required and revalidated; otherwise remove it before final validation.
- Preserve the constructor and `forward` signatures expected by `model.py`.
- Keep `model_new.py` limited to allowed tensor creation and custom extension ops. Do not add unsupported Python-side torch operations to compensate for binding or kernel issues.

## 12. Tool/Workspace Hygiene (Not CUDA Coding Errors)

- For `workspace` writes, pass `operation: "write"`, `path`, and `content` as separate fields. Do not put file content or Markdown into the `operation` field.
- Treat `ValueError: unknown operation`, missing `operation` errors, malformed tool payloads, and log encoding/mojibake as tooling issues, not CUDA coding mistakes.
- After any failed or ambiguous write, read the target file before compiling. A failed tool call does not prove the file is unchanged or correct.
- If a write/read fails due to tool transport or encoding, retry the tool call cleanly. Do not change CUDA kernel or binding logic solely in response to the tool failure.
- A missing utility file or unreadable path is a workspace/file-availability issue, not evidence that the kernel is incorrect. Use the documented utilities or list the directory to discover the actual files.

**Scope:** Sections 13–16 apply to CUDA extension episodes using this workspace. They do not relax Sections 1–12 or the existing forbidden/allowed lists. Section 16 applies only to tool/workspace operations and does not classify tool transport/encoding failures as CUDA coding errors.

## 13. Post-Tuning Validation State

- Treat `model_new.py` as part of the correctness surface. Changing the selected kernel `config`, wrapper arguments, or forward path can change numerical behavior; a benchmark-only run after such a change does not validate it.
- After any candidate change to `kernels/*`, bindings, or `model_new.py`, run verification before using or reporting benchmark results. A fast benchmark from an unverified candidate is only a performance probe.
- The final validation sequence must run on the exact final files: finish all edits and cleanup, compile if any C++/CUDA/binding file changed, run verification, then run benchmark. Any write after verification makes that verification stale.
- If a compile and a later write occur in the same step, do not count that compile as final validation for the later file. Ensure the last compile precedes the verified final state, or explicitly confirm no compiled file changed and keep the binary consistent with the verified files.
- Do not end an episode immediately after a benchmark-only edit. If the step budget is nearly exhausted, preserve or restore the last fully validated version rather than making an unverified last-moment change.
- If verification fails after a tuning change, reject the faster benchmark. Either fix the kernel/config and rerun compile, verification, and benchmark, or revert to the last verified version.

## 14. Tuned Config and Shape-Specialized Path Safety

- Do not hardcode a tuned `config` in `model_new.py` solely because it benchmarked fastest. Verify that exact config on all verification cases and seeds before leaving it as the final selection.
- For shape-specialized dispatch, ensure every non-default config either has a correct fallback or is unreachable from the final `model_new.py`. A fast path that is wrong for one harness shape must not remain selected.
- When a config-specific case fails, first inspect dispatch guards, supported `F`/`N` assumptions, and fallback coverage. Do not change unrelated kernel math or move the failure into Python.
- Auto/default passing does not validate a hardcoded config. After switching `model_new.py` from auto/default to a fixed config, rerun verification on that exact final config.
- Diagnostic, no-op, copy-only, or debug configs must not be reachable in the final model path.

## 15. Final Cleanup Specifics

- Before final validation, remove diagnostic no-op/copy kernels, unused tuning branches, and intermediate kernel versions from `kernels/` unless they are still required by the final default path.
- If multiple tuning configs remain, ensure they are all semantically correct for supported shapes, or remove/disable the unused ones. Do not leave a known-wrong config available to be selected accidentally.
- Do not submit a kernel containing diagnostic configs that produce no output or copy only, even if they were useful during tuning.
- Cleanup is an edit. After cleanup, run the full final sequence on the exact cleaned files; a successful earlier compile or benchmark does not validate cleaned files.

## 16. Tool Output Interpretation

- Environment warnings such as “Failed to find real location of python.exe” accompanying a successful compile are not compiler errors or CUDA kernel bugs. Use the compile success/failure status and actionable compiler errors.
- If a write/read fails because of a malformed workspace payload or missing `operation`, retry the tool call cleanly. A tool failure by itself is not evidence that the CUDA kernel or binding logic is wrong.
- After any failed or ambiguous write, read the target file before compiling. Do not change CUDA kernel or binding logic solely in response to a tool transport/encoding/payload failure.
- If a reported compile-log path cannot be read, use the compile tool output or rerun compile; do not make unrelated kernel changes based on a log-path failure.

**Scope:** Sections 17–20 apply to CUDA extension episodes using this workspace and do not relax the existing forbidden/allowed lists or validation rules. Section 20 applies only to tool/workspace operations and does not classify tool transport/encoding/path failures as CUDA coding errors.

## 17. Validation Order Is Non-Negotiable

- Compile success, verification success, and benchmark success are separate gates. A successful compile does not validate kernel math or wrapper semantics, and a benchmark without preceding verification on the same files is not evidence of correctness.
- After any edit to `kernels/*`, binding files, or `model_new.py`, use this order: finish edits/cleanup → compile if any C++/CUDA/binding file changed → verification → profiling/benchmark. The final benchmark must run on the exact files that passed verification.
- Do not make a late `model_new.py` config or forward-path edit and stop after a benchmark-only run. If the edit selects a different config, dispatch branch, or wrapper path, rerun verification on that exact file, then rerun the benchmark.
- A benchmark-only run may be used only as a tuning probe. Preserve the last verified selection and either verify the final selected file or revert to the verified one.
- Do not end an episode on a write, a compile-only step, or a benchmark-only step after a change. If step budget is nearly exhausted, do not start another speculative optimization; leave or restore the last fully validated files.
- A write after the last verification invalidates that verification. A compile after the last verification still leaves verification stale; recompile, verify, and benchmark before completion.
- If a benchmark-only run is followed by another edit, the earlier benchmark is not final evidence for the new file.

## 18. Reject Unverified Speedups

- If verification fails after a kernel, binding, or wrapper change, reject that candidate even if its benchmark is faster. Fix the failing path or revert to the last verified implementation, then recompile, verify, and benchmark.
- Do not continue tuning `model_new.py` configs while the underlying kernel is failing verification. Correctness must be restored first.
- When a rewrite breaks correctness, use the last passing kernel as the baseline and prefer a minimal fix over another broad rewrite. Keep a known-good copy until the replacement passes verification.
- A config that is fast for one shape but unverified or incorrect for another harness case must not remain selected in final `model_new.py`.
- Do not leave a faster but unverified or known-wrong config, dispatch branch, or wrapper path reachable from the final model path.

## 19. Probe and Debug Artifact Hygiene

- Temporary probe/diagnostic kernels, alternate bindings, and patch/debug files must not be part of the final module path. If they are added under `kernels/` or referenced by a binding, remove them and rerun compile, verification, and benchmark on the cleaned files.
- Do not use a probe benchmark as evidence that the final implementation is correct.
- Do not leave temporary patch files, probe launchers, or notes in `kernels/`; cleanup is an edit and must precede the final validation sequence.
- After deleting a probe file or reverting a binding, recompile so the extension cannot retain stale symbols or an unintended old path.

## 20. Tool Log Path Failures Are Not CUDA Signals

Scope: tool/workspace operations only; these rules do not classify tool transport/encoding/path failures as CUDA coding mistakes.

- A `FileNotFoundError` or unreadable path for a compile, verification, or profiling log is a tool/path issue. Use the tool’s returned output/status or rerun the tool; do not change kernel math, binding allocation, or `model_new.py` based on the failed read alone.
- If a log path cannot be read, do not infer pass/fail from the missing file. Rerun verification/benchmark or use the tool result already returned.
- After a failed or ambiguous workspace read/write, read the target file before compiling or editing further. A failed tool call is not evidence that the file is correct or incorrect.
- Environment warnings such as failure to locate `python.exe` alongside an explicit compile/verify/benchmark success are not CUDA kernel errors; use the explicit status and actionable compiler/runtime errors.

**Scope:** Sections 21–24 apply to CUDA extension episodes using this workspace. They do not relax the existing forbidden/allowed lists or validation requirements. Section 24 applies only to tool/workspace operations and does not classify tool payload or environment failures as CUDA coding mistakes.

## 21. Vectorized Reduction Semantics and Debug Isolation

- When a vector load (`float2`, `float4`) spans the non-reduced axis, each component belongs to a different output column and must have its own reduction accumulator and its own reciprocal normalization value. Do not sum the components into one scalar, and do not broadcast one scalar result across the vector lanes.
- Store and combine vector partials as vector/per-lane values in shared memory (for example, `float4` accumulators), not as one scalar per thread. Apply the same per-lane requirement when scaling and storing the output.
- A vectorized reduction failure with most elements mismatched and the same wrong values across different block shapes is evidence to inspect lane packing/reduction semantics first, not automatically a race or launch-config problem. Isolate it by testing the scalar path, a vectorized copy/indexing path without the reduction, and then the vectorized reduction.
- Keep diagnostic configs that skip reduction or copy data out of the final path. After fixing the vector path, verify both vector-eligible shapes and non-vector fallback shapes.

## 22. Forward-Path Diagnostics and Post-Tuning Validation

- Do not add PyTorch reference calculations (`pow`, `mean`, `sqrt`, etc.) or other debug tensor ops to `model_new.py`/`ModelNew.forward` to diagnose a kernel. The verifier rejects PyTorch computation during candidate forward; that failure is an instrumentation violation, not evidence that the CUDA math is wrong.
- Perform any reference comparison outside the submitted forward path, and remove the diagnostic before final validation. Do not leave debug instrumentation in `model_new.py`.
- A benchmark-only sweep after changing `model_new.py`—including a hardcoded `config`, dispatch branch, or wrapper argument—is a tuning probe. Before treating that selection as final, run verification on the exact final file, then benchmark. A passing verification of an earlier/default config does not validate the new selection.
- Do not treat compile success as correctness validation. If a new candidate is compiled, run verification before using its benchmark as final evidence. If verification fails, reject the faster candidate or revert to the last verified version.

## 23. Clock-State-Aware Benchmark Comparison

- Absolute candidate latencies in this environment can change by roughly 2× between runs as GPU clock state changes. Do not rank tuning variants solely by absolute microseconds measured in separate runs.
- Compare candidate and `torch.compile` baseline in the same benchmark invocation, or repeat/interleave measurements and report the candidate/baseline ratio. A result is more trustworthy when the ratio is stable across clock states.
- Do not use a fast but unverified benchmark under one clock state to justify leaving a different final configuration. Final evidence must come from the exact final files after the required compile → verify → benchmark order.

## 24. Workspace Write Payloads (Tool/Workspace Only)

- For workspace writes, pass `operation: "write"`, `path`, and `content` as separate fields. Do not omit `operation` or put file content/Markdown in the `operation` field.
- A `ValueError: operation must be one of ...` or a missing-`operation` error is a tool payload problem, not a CUDA kernel or binding bug. Retry the write/read correctly; do not change kernel math or allocation logic in response to that error alone.
- After a failed or ambiguous write, read the target file before compiling or making further edits. A failed tool call does not prove the file is correct or incorrect.
- Environment warnings such as “Failed to find real location of python.exe” alongside an explicit compile/verify success are not CUDA errors; use the explicit status and actionable compiler/runtime output.

**Scope:** Sections 25–28 apply to CUDA extension episodes using this workspace. They do not relax Sections 1–24 or the existing forbidden/allowed lists. Section 28 applies only to tool/workspace operations and explicitly does not classify tool transport, encoding, payload, or path failures as CUDA coding mistakes.

## 25. Output Tensor Lifetime and Buffer-Reuse Safety

- Treat “previously returned outputs must remain unchanged after later calls” as a hard lifetime constraint. Never write into, recycle, or reuse storage that may still be referenced by any earlier returned tensor. The verifier/harness may retain outputs across repeated calls.
- For AXPBY-like elementwise ops that return a new tensor, default to the allowed per-call `torch::empty_like(...)` and rely on PyTorch’s allocator for caching. Do not add a custom cross-call output pool merely to reduce allocation overhead unless you can prove with public APIs that no live tensor references the buffer.
- Do not use `TensorImpl::use_count`, `Tensor::use_count`, `storage().use_count`, intrusive-pointer counts, or other refcount internals to decide whether an output buffer is recyclable. Such APIs are unstable/version-dependent, and an observed build failed with `"use_count": is not a member of "c10::TensorImpl"`. A single refcount is also not a contract-safe proof against live Python/harness references.
- If an attempted refcount API does not compile, remove that pooling attempt or use per-call allocation. Do not add guessed headers, private ATen/TensorImpl APIs, or unrelated kernel changes to force it to compile.
- A growing pool is a failure signal. If diagnostic output shows the pool size increasing monotonically, or benchmark latency degrades over repeated calls, the buffers are not safely recyclable. Remove the pool and return to per-call allocation. Do not report a faster benchmark from a pool that was not proven to preserve all previous outputs.
- If you experiment with a pre-reserved chunk or suballocation, keep size/offset metadata on the host. Do not write metadata to device memory through a host pointer; that is a host/device memory bug, not an allocation-tuning detail. Ensure each live output has distinct usable storage or an ownership scheme that cannot overwrite a live result.
- Never alias or overwrite input tensors `a`/`b` to save allocation. Compute into a distinct output. When using typed pointers, validate dtype, device, contiguity, and shape assumptions; if buffer reuse is attempted, match all relevant tensor properties, not just shape.

## 26. No Probe Code in Final Production Paths

- A verification PASS does not mean the binding is clean. Probe output can run during verification and benchmark while tests still pass. Before finalizing, inspect `kernels/*.cu`, `kernels/*_binding.cpp`, and `model_new.py` for diagnostic `printf`/`std::printf`, counters, timing probes, pool-inspection logs, and comments such as `PROBE ONLY`.
- Remove diagnostic code embedded in otherwise final files, not only separate `probe_*.cpp` files. Do not leave global probe state, pool-size printing, or debug-only branches reachable from `model_new.py`.
- After removing probe code, recompile if any C++/CUDA/binding file changed, then run verification and benchmark on the exact cleaned files. Verification or benchmark output from the probe version is not evidence for the cleaned version.
- Temporary probe files must be deleted before final validation. If a probe-specific symbol remains declared or referenced, recompile so the extension cannot retain a stale or unintended path.

## 27. Finish on Validated Files and Submit the Report

- Do not end an episode after a verify-only step when the performance gate has not been measured on the exact final files. The final sequence is: finish edits/cleanup → compile if needed → verification → benchmark on the same files → submit the final report.
- Reserve enough budget for that final sequence. If the remaining budget is small, stop optional pooling/refcount experiments and preserve or restore the last candidate that passed compile, verification, and benchmark. Do not start a speculative optimization that may leave the workspace unvalidated.
- A benchmark from an earlier candidate is not final if later writes changed `model_new.py`, binding files, or kernels. Report only measurements actually obtained from the final validated files; if the final files were not benchmarked, say so and treat the performance goal as unconfirmed.
- If a late buffer-pool or binding experiment fails compile, fails verification, or shows degradation, revert to the known-good per-call allocation version and rerun the required gates rather than leaving the experimental binding in place.

## 28. Tool/Workspace Failure Separation (Scoped)

- A missing `operation` field or `ValueError: operation must be one of ...` is a tooling payload error. Retry the write/read with `operation`, `path`, and `content` as separate fields; do not change kernel math, output pooling, or binding allocation in response to that error alone.
- After any failed or ambiguous write, read the target file before compiling. A failed tool call does not prove the file is unchanged or correct.
- Environment warnings such as failure to locate `python.exe` alongside an explicit compile/verify success are not CUDA errors; use the explicit status and actionable compiler/runtime output. A log-path, encoding, transport, or payload failure is not evidence that the CUDA implementation is wrong.


## Repeated appendix, copy 13

**Scope of appended rules:** Sections 9–12 apply to CUDA extension episodes using this workspace. Section 12 applies only to tool/workspace operations and explicitly does not classify tool transport/encoding failures as CUDA coding errors. None of these rules relax the existing forbidden/allowed lists.

## 9. Compile-Safe Binding and Allocation

- Use only the public allocation exception already allowed by this document. For bindings, prefer the documented `torch::empty_like(...)` allocation path.
- Do not use `at::detail::empty_cuda`, other `at::detail::*` symbols, or version-private ATen internals to bypass the dispatcher. These are not stable extension APIs.
- If compilation reports an error such as `'empty_cuda': is not a member of 'at::detail'`, replace the internal allocation call with the documented allocation path. Do not keep the internal call or add guessed headers to force it to compile.
- Fix binding compile errors in the binding files, not by changing kernel math. A missing binding symbol or undeclared allocation API is not evidence that the CUDA kernel algorithm is wrong.
- Do not replace a compiling binding with a speculative “lean hot path” using internal APIs. Allocation micro-optimizations are lower priority than compile success and correctness; if attempted, isolate the change and keep a known-good fallback.
- When a binding specializes to float32/contiguous CUDA tensors, enforce that specialization with `TORCH_CHECK`/equivalent validation or ensure `model_new.py` guarantees it. Do not reinterpret arbitrary tensor storage as `float*`.

## 10. Final Validation Gate

- The final submitted files must be the exact files that passed compile, verification, and benchmark. Any write after the last successful verification/benchmark invalidates those results.
- Final sequence: finish cleanup and final edits, then compile, then run verification, then run profiling/benchmark. Do not stop immediately after a write.
- Do not perform a post-benchmark speculative rewrite of `kernels/*` or `model_new.py` and declare completion. If optimizing after a passing result, preserve or restore the known-good version; re-run compile, verification, and benchmark; if the new version fails, revert and leave the known-good version as final.
- A successful earlier compile does not excuse a later compile failure. Do not submit an uncompiled final candidate.
- Final cleanup counts as an edit. Run compile, verification, and benchmark after cleanup, or perform cleanup before the final validation sequence.
- On compile failure, use the first actionable compiler error. If the reported compile log path cannot be read, use the compile tool output or rerun compile; do not make unrelated kernel changes based on a log-path failure.

## 11. ModelNew and Wrapper Semantics

- Keep `ModelNew.forward` as the normal method and let the harness use `nn.Module.__call__`.
- Do not add `__call__ = forward` or otherwise bypass `nn.Module` hook machinery as a speculative shortcut. It can change harness/hook behavior. Use such a bypass only if explicitly required and revalidated; otherwise remove it before final validation.
- Preserve the constructor and `forward` signatures expected by `model.py`.
- Keep `model_new.py` limited to allowed tensor creation and custom extension ops. Do not add unsupported Python-side torch operations to compensate for binding or kernel issues.

## 12. Tool/Workspace Hygiene (Not CUDA Coding Errors)

- For `workspace` writes, pass `operation: "write"`, `path`, and `content` as separate fields. Do not put file content or Markdown into the `operation` field.
- Treat `ValueError: unknown operation`, missing `operation` errors, malformed tool payloads, and log encoding/mojibake as tooling issues, not CUDA coding mistakes.
- After any failed or ambiguous write, read the target file before compiling. A failed tool call does not prove the file is unchanged or correct.
- If a write/read fails due to tool transport or encoding, retry the tool call cleanly. Do not change CUDA kernel or binding logic solely in response to the tool failure.
- A missing utility file or unreadable path is a workspace/file-availability issue, not evidence that the kernel is incorrect. Use the documented utilities or list the directory to discover the actual files.

**Scope:** Sections 13–16 apply to CUDA extension episodes using this workspace. They do not relax Sections 1–12 or the existing forbidden/allowed lists. Section 16 applies only to tool/workspace operations and does not classify tool transport/encoding failures as CUDA coding errors.

## 13. Post-Tuning Validation State

- Treat `model_new.py` as part of the correctness surface. Changing the selected kernel `config`, wrapper arguments, or forward path can change numerical behavior; a benchmark-only run after such a change does not validate it.
- After any candidate change to `kernels/*`, bindings, or `model_new.py`, run verification before using or reporting benchmark results. A fast benchmark from an unverified candidate is only a performance probe.
- The final validation sequence must run on the exact final files: finish all edits and cleanup, compile if any C++/CUDA/binding file changed, run verification, then run benchmark. Any write after verification makes that verification stale.
- If a compile and a later write occur in the same step, do not count that compile as final validation for the later file. Ensure the last compile precedes the verified final state, or explicitly confirm no compiled file changed and keep the binary consistent with the verified files.
- Do not end an episode immediately after a benchmark-only edit. If the step budget is nearly exhausted, preserve or restore the last fully validated version rather than making an unverified last-moment change.
- If verification fails after a tuning change, reject the faster benchmark. Either fix the kernel/config and rerun compile, verification, and benchmark, or revert to the last verified version.

## 14. Tuned Config and Shape-Specialized Path Safety

- Do not hardcode a tuned `config` in `model_new.py` solely because it benchmarked fastest. Verify that exact config on all verification cases and seeds before leaving it as the final selection.
- For shape-specialized dispatch, ensure every non-default config either has a correct fallback or is unreachable from the final `model_new.py`. A fast path that is wrong for one harness shape must not remain selected.
- When a config-specific case fails, first inspect dispatch guards, supported `F`/`N` assumptions, and fallback coverage. Do not change unrelated kernel math or move the failure into Python.
- Auto/default passing does not validate a hardcoded config. After switching `model_new.py` from auto/default to a fixed config, rerun verification on that exact final config.
- Diagnostic, no-op, copy-only, or debug configs must not be reachable in the final model path.

## 15. Final Cleanup Specifics

- Before final validation, remove diagnostic no-op/copy kernels, unused tuning branches, and intermediate kernel versions from `kernels/` unless they are still required by the final default path.
- If multiple tuning configs remain, ensure they are all semantically correct for supported shapes, or remove/disable the unused ones. Do not leave a known-wrong config available to be selected accidentally.
- Do not submit a kernel containing diagnostic configs that produce no output or copy only, even if they were useful during tuning.
- Cleanup is an edit. After cleanup, run the full final sequence on the exact cleaned files; a successful earlier compile or benchmark does not validate cleaned files.

## 16. Tool Output Interpretation

- Environment warnings such as “Failed to find real location of python.exe” accompanying a successful compile are not compiler errors or CUDA kernel bugs. Use the compile success/failure status and actionable compiler errors.
- If a write/read fails because of a malformed workspace payload or missing `operation`, retry the tool call cleanly. A tool failure by itself is not evidence that the CUDA kernel or binding logic is wrong.
- After any failed or ambiguous write, read the target file before compiling. Do not change CUDA kernel or binding logic solely in response to a tool transport/encoding/payload failure.
- If a reported compile-log path cannot be read, use the compile tool output or rerun compile; do not make unrelated kernel changes based on a log-path failure.

**Scope:** Sections 17–20 apply to CUDA extension episodes using this workspace and do not relax the existing forbidden/allowed lists or validation rules. Section 20 applies only to tool/workspace operations and does not classify tool transport/encoding/path failures as CUDA coding errors.

## 17. Validation Order Is Non-Negotiable

- Compile success, verification success, and benchmark success are separate gates. A successful compile does not validate kernel math or wrapper semantics, and a benchmark without preceding verification on the same files is not evidence of correctness.
- After any edit to `kernels/*`, binding files, or `model_new.py`, use this order: finish edits/cleanup → compile if any C++/CUDA/binding file changed → verification → profiling/benchmark. The final benchmark must run on the exact files that passed verification.
- Do not make a late `model_new.py` config or forward-path edit and stop after a benchmark-only run. If the edit selects a different config, dispatch branch, or wrapper path, rerun verification on that exact file, then rerun the benchmark.
- A benchmark-only run may be used only as a tuning probe. Preserve the last verified selection and either verify the final selected file or revert to the verified one.
- Do not end an episode on a write, a compile-only step, or a benchmark-only step after a change. If step budget is nearly exhausted, do not start another speculative optimization; leave or restore the last fully validated files.
- A write after the last verification invalidates that verification. A compile after the last verification still leaves verification stale; recompile, verify, and benchmark before completion.
- If a benchmark-only run is followed by another edit, the earlier benchmark is not final evidence for the new file.

## 18. Reject Unverified Speedups

- If verification fails after a kernel, binding, or wrapper change, reject that candidate even if its benchmark is faster. Fix the failing path or revert to the last verified implementation, then recompile, verify, and benchmark.
- Do not continue tuning `model_new.py` configs while the underlying kernel is failing verification. Correctness must be restored first.
- When a rewrite breaks correctness, use the last passing kernel as the baseline and prefer a minimal fix over another broad rewrite. Keep a known-good copy until the replacement passes verification.
- A config that is fast for one shape but unverified or incorrect for another harness case must not remain selected in final `model_new.py`.
- Do not leave a faster but unverified or known-wrong config, dispatch branch, or wrapper path reachable from the final model path.

## 19. Probe and Debug Artifact Hygiene

- Temporary probe/diagnostic kernels, alternate bindings, and patch/debug files must not be part of the final module path. If they are added under `kernels/` or referenced by a binding, remove them and rerun compile, verification, and benchmark on the cleaned files.
- Do not use a probe benchmark as evidence that the final implementation is correct.
- Do not leave temporary patch files, probe launchers, or notes in `kernels/`; cleanup is an edit and must precede the final validation sequence.
- After deleting a probe file or reverting a binding, recompile so the extension cannot retain stale symbols or an unintended old path.

## 20. Tool Log Path Failures Are Not CUDA Signals

Scope: tool/workspace operations only; these rules do not classify tool transport/encoding/path failures as CUDA coding mistakes.

- A `FileNotFoundError` or unreadable path for a compile, verification, or profiling log is a tool/path issue. Use the tool’s returned output/status or rerun the tool; do not change kernel math, binding allocation, or `model_new.py` based on the failed read alone.
- If a log path cannot be read, do not infer pass/fail from the missing file. Rerun verification/benchmark or use the tool result already returned.
- After a failed or ambiguous workspace read/write, read the target file before compiling or editing further. A failed tool call is not evidence that the file is correct or incorrect.
- Environment warnings such as failure to locate `python.exe` alongside an explicit compile/verify/benchmark success are not CUDA kernel errors; use the explicit status and actionable compiler/runtime errors.

**Scope:** Sections 21–24 apply to CUDA extension episodes using this workspace. They do not relax the existing forbidden/allowed lists or validation requirements. Section 24 applies only to tool/workspace operations and does not classify tool payload or environment failures as CUDA coding mistakes.

## 21. Vectorized Reduction Semantics and Debug Isolation

- When a vector load (`float2`, `float4`) spans the non-reduced axis, each component belongs to a different output column and must have its own reduction accumulator and its own reciprocal normalization value. Do not sum the components into one scalar, and do not broadcast one scalar result across the vector lanes.
- Store and combine vector partials as vector/per-lane values in shared memory (for example, `float4` accumulators), not as one scalar per thread. Apply the same per-lane requirement when scaling and storing the output.
- A vectorized reduction failure with most elements mismatched and the same wrong values across different block shapes is evidence to inspect lane packing/reduction semantics first, not automatically a race or launch-config problem. Isolate it by testing the scalar path, a vectorized copy/indexing path without the reduction, and then the vectorized reduction.
- Keep diagnostic configs that skip reduction or copy data out of the final path. After fixing the vector path, verify both vector-eligible shapes and non-vector fallback shapes.

## 22. Forward-Path Diagnostics and Post-Tuning Validation

- Do not add PyTorch reference calculations (`pow`, `mean`, `sqrt`, etc.) or other debug tensor ops to `model_new.py`/`ModelNew.forward` to diagnose a kernel. The verifier rejects PyTorch computation during candidate forward; that failure is an instrumentation violation, not evidence that the CUDA math is wrong.
- Perform any reference comparison outside the submitted forward path, and remove the diagnostic before final validation. Do not leave debug instrumentation in `model_new.py`.
- A benchmark-only sweep after changing `model_new.py`—including a hardcoded `config`, dispatch branch, or wrapper argument—is a tuning probe. Before treating that selection as final, run verification on the exact final file, then benchmark. A passing verification of an earlier/default config does not validate the new selection.
- Do not treat compile success as correctness validation. If a new candidate is compiled, run verification before using its benchmark as final evidence. If verification fails, reject the faster candidate or revert to the last verified version.

## 23. Clock-State-Aware Benchmark Comparison

- Absolute candidate latencies in this environment can change by roughly 2× between runs as GPU clock state changes. Do not rank tuning variants solely by absolute microseconds measured in separate runs.
- Compare candidate and `torch.compile` baseline in the same benchmark invocation, or repeat/interleave measurements and report the candidate/baseline ratio. A result is more trustworthy when the ratio is stable across clock states.
- Do not use a fast but unverified benchmark under one clock state to justify leaving a different final configuration. Final evidence must come from the exact final files after the required compile → verify → benchmark order.

## 24. Workspace Write Payloads (Tool/Workspace Only)

- For workspace writes, pass `operation: "write"`, `path`, and `content` as separate fields. Do not omit `operation` or put file content/Markdown in the `operation` field.
- A `ValueError: operation must be one of ...` or a missing-`operation` error is a tool payload problem, not a CUDA kernel or binding bug. Retry the write/read correctly; do not change kernel math or allocation logic in response to that error alone.
- After a failed or ambiguous write, read the target file before compiling or making further edits. A failed tool call does not prove the file is correct or incorrect.
- Environment warnings such as “Failed to find real location of python.exe” alongside an explicit compile/verify success are not CUDA errors; use the explicit status and actionable compiler/runtime output.

**Scope:** Sections 25–28 apply to CUDA extension episodes using this workspace. They do not relax Sections 1–24 or the existing forbidden/allowed lists. Section 28 applies only to tool/workspace operations and explicitly does not classify tool transport, encoding, payload, or path failures as CUDA coding mistakes.

## 25. Output Tensor Lifetime and Buffer-Reuse Safety

- Treat “previously returned outputs must remain unchanged after later calls” as a hard lifetime constraint. Never write into, recycle, or reuse storage that may still be referenced by any earlier returned tensor. The verifier/harness may retain outputs across repeated calls.
- For AXPBY-like elementwise ops that return a new tensor, default to the allowed per-call `torch::empty_like(...)` and rely on PyTorch’s allocator for caching. Do not add a custom cross-call output pool merely to reduce allocation overhead unless you can prove with public APIs that no live tensor references the buffer.
- Do not use `TensorImpl::use_count`, `Tensor::use_count`, `storage().use_count`, intrusive-pointer counts, or other refcount internals to decide whether an output buffer is recyclable. Such APIs are unstable/version-dependent, and an observed build failed with `"use_count": is not a member of "c10::TensorImpl"`. A single refcount is also not a contract-safe proof against live Python/harness references.
- If an attempted refcount API does not compile, remove that pooling attempt or use per-call allocation. Do not add guessed headers, private ATen/TensorImpl APIs, or unrelated kernel changes to force it to compile.
- A growing pool is a failure signal. If diagnostic output shows the pool size increasing monotonically, or benchmark latency degrades over repeated calls, the buffers are not safely recyclable. Remove the pool and return to per-call allocation. Do not report a faster benchmark from a pool that was not proven to preserve all previous outputs.
- If you experiment with a pre-reserved chunk or suballocation, keep size/offset metadata on the host. Do not write metadata to device memory through a host pointer; that is a host/device memory bug, not an allocation-tuning detail. Ensure each live output has distinct usable storage or an ownership scheme that cannot overwrite a live result.
- Never alias or overwrite input tensors `a`/`b` to save allocation. Compute into a distinct output. When using typed pointers, validate dtype, device, contiguity, and shape assumptions; if buffer reuse is attempted, match all relevant tensor properties, not just shape.

## 26. No Probe Code in Final Production Paths

- A verification PASS does not mean the binding is clean. Probe output can run during verification and benchmark while tests still pass. Before finalizing, inspect `kernels/*.cu`, `kernels/*_binding.cpp`, and `model_new.py` for diagnostic `printf`/`std::printf`, counters, timing probes, pool-inspection logs, and comments such as `PROBE ONLY`.
- Remove diagnostic code embedded in otherwise final files, not only separate `probe_*.cpp` files. Do not leave global probe state, pool-size printing, or debug-only branches reachable from `model_new.py`.
- After removing probe code, recompile if any C++/CUDA/binding file changed, then run verification and benchmark on the exact cleaned files. Verification or benchmark output from the probe version is not evidence for the cleaned version.
- Temporary probe files must be deleted before final validation. If a probe-specific symbol remains declared or referenced, recompile so the extension cannot retain a stale or unintended path.

## 27. Finish on Validated Files and Submit the Report

- Do not end an episode after a verify-only step when the performance gate has not been measured on the exact final files. The final sequence is: finish edits/cleanup → compile if needed → verification → benchmark on the same files → submit the final report.
- Reserve enough budget for that final sequence. If the remaining budget is small, stop optional pooling/refcount experiments and preserve or restore the last candidate that passed compile, verification, and benchmark. Do not start a speculative optimization that may leave the workspace unvalidated.
- A benchmark from an earlier candidate is not final if later writes changed `model_new.py`, binding files, or kernels. Report only measurements actually obtained from the final validated files; if the final files were not benchmarked, say so and treat the performance goal as unconfirmed.
- If a late buffer-pool or binding experiment fails compile, fails verification, or shows degradation, revert to the known-good per-call allocation version and rerun the required gates rather than leaving the experimental binding in place.

## 28. Tool/Workspace Failure Separation (Scoped)

- A missing `operation` field or `ValueError: operation must be one of ...` is a tooling payload error. Retry the write/read with `operation`, `path`, and `content` as separate fields; do not change kernel math, output pooling, or binding allocation in response to that error alone.
- After any failed or ambiguous write, read the target file before compiling. A failed tool call does not prove the file is unchanged or correct.
- Environment warnings such as failure to locate `python.exe` alongside an explicit compile/verify success are not CUDA errors; use the explicit status and actionable compiler/runtime output. A log-path, encoding, transport, or payload failure is not evidence that the CUDA implementation is wrong.


## Repeated appendix, copy 14

**Scope of appended rules:** Sections 9–12 apply to CUDA extension episodes using this workspace. Section 12 applies only to tool/workspace operations and explicitly does not classify tool transport/encoding failures as CUDA coding errors. None of these rules relax the existing forbidden/allowed lists.

## 9. Compile-Safe Binding and Allocation

- Use only the public allocation exception already allowed by this document. For bindings, prefer the documented `torch::empty_like(...)` allocation path.
- Do not use `at::detail::empty_cuda`, other `at::detail::*` symbols, or version-private ATen internals to bypass the dispatcher. These are not stable extension APIs.
- If compilation reports an error such as `'empty_cuda': is not a member of 'at::detail'`, replace the internal allocation call with the documented allocation path. Do not keep the internal call or add guessed headers to force it to compile.
- Fix binding compile errors in the binding files, not by changing kernel math. A missing binding symbol or undeclared allocation API is not evidence that the CUDA kernel algorithm is wrong.
- Do not replace a compiling binding with a speculative “lean hot path” using internal APIs. Allocation micro-optimizations are lower priority than compile success and correctness; if attempted, isolate the change and keep a known-good fallback.
- When a binding specializes to float32/contiguous CUDA tensors, enforce that specialization with `TORCH_CHECK`/equivalent validation or ensure `model_new.py` guarantees it. Do not reinterpret arbitrary tensor storage as `float*`.

## 10. Final Validation Gate

- The final submitted files must be the exact files that passed compile, verification, and benchmark. Any write after the last successful verification/benchmark invalidates those results.
- Final sequence: finish cleanup and final edits, then compile, then run verification, then run profiling/benchmark. Do not stop immediately after a write.
- Do not perform a post-benchmark speculative rewrite of `kernels/*` or `model_new.py` and declare completion. If optimizing after a passing result, preserve or restore the known-good version; re-run compile, verification, and benchmark; if the new version fails, revert and leave the known-good version as final.
- A successful earlier compile does not excuse a later compile failure. Do not submit an uncompiled final candidate.
- Final cleanup counts as an edit. Run compile, verification, and benchmark after cleanup, or perform cleanup before the final validation sequence.
- On compile failure, use the first actionable compiler error. If the reported compile log path cannot be read, use the compile tool output or rerun compile; do not make unrelated kernel changes based on a log-path failure.

## 11. ModelNew and Wrapper Semantics

- Keep `ModelNew.forward` as the normal method and let the harness use `nn.Module.__call__`.
- Do not add `__call__ = forward` or otherwise bypass `nn.Module` hook machinery as a speculative shortcut. It can change harness/hook behavior. Use such a bypass only if explicitly required and revalidated; otherwise remove it before final validation.
- Preserve the constructor and `forward` signatures expected by `model.py`.
- Keep `model_new.py` limited to allowed tensor creation and custom extension ops. Do not add unsupported Python-side torch operations to compensate for binding or kernel issues.

## 12. Tool/Workspace Hygiene (Not CUDA Coding Errors)

- For `workspace` writes, pass `operation: "write"`, `path`, and `content` as separate fields. Do not put file content or Markdown into the `operation` field.
- Treat `ValueError: unknown operation`, missing `operation` errors, malformed tool payloads, and log encoding/mojibake as tooling issues, not CUDA coding mistakes.
- After any failed or ambiguous write, read the target file before compiling. A failed tool call does not prove the file is unchanged or correct.
- If a write/read fails due to tool transport or encoding, retry the tool call cleanly. Do not change CUDA kernel or binding logic solely in response to the tool failure.
- A missing utility file or unreadable path is a workspace/file-availability issue, not evidence that the kernel is incorrect. Use the documented utilities or list the directory to discover the actual files.

**Scope:** Sections 13–16 apply to CUDA extension episodes using this workspace. They do not relax Sections 1–12 or the existing forbidden/allowed lists. Section 16 applies only to tool/workspace operations and does not classify tool transport/encoding failures as CUDA coding errors.

## 13. Post-Tuning Validation State

- Treat `model_new.py` as part of the correctness surface. Changing the selected kernel `config`, wrapper arguments, or forward path can change numerical behavior; a benchmark-only run after such a change does not validate it.
- After any candidate change to `kernels/*`, bindings, or `model_new.py`, run verification before using or reporting benchmark results. A fast benchmark from an unverified candidate is only a performance probe.
- The final validation sequence must run on the exact final files: finish all edits and cleanup, compile if any C++/CUDA/binding file changed, run verification, then run benchmark. Any write after verification makes that verification stale.
- If a compile and a later write occur in the same step, do not count that compile as final validation for the later file. Ensure the last compile precedes the verified final state, or explicitly confirm no compiled file changed and keep the binary consistent with the verified files.
- Do not end an episode immediately after a benchmark-only edit. If the step budget is nearly exhausted, preserve or restore the last fully validated version rather than making an unverified last-moment change.
- If verification fails after a tuning change, reject the faster benchmark. Either fix the kernel/config and rerun compile, verification, and benchmark, or revert to the last verified version.

## 14. Tuned Config and Shape-Specialized Path Safety

- Do not hardcode a tuned `config` in `model_new.py` solely because it benchmarked fastest. Verify that exact config on all verification cases and seeds before leaving it as the final selection.
- For shape-specialized dispatch, ensure every non-default config either has a correct fallback or is unreachable from the final `model_new.py`. A fast path that is wrong for one harness shape must not remain selected.
- When a config-specific case fails, first inspect dispatch guards, supported `F`/`N` assumptions, and fallback coverage. Do not change unrelated kernel math or move the failure into Python.
- Auto/default passing does not validate a hardcoded config. After switching `model_new.py` from auto/default to a fixed config, rerun verification on that exact final config.
- Diagnostic, no-op, copy-only, or debug configs must not be reachable in the final model path.

## 15. Final Cleanup Specifics

- Before final validation, remove diagnostic no-op/copy kernels, unused tuning branches, and intermediate kernel versions from `kernels/` unless they are still required by the final default path.
- If multiple tuning configs remain, ensure they are all semantically correct for supported shapes, or remove/disable the unused ones. Do not leave a known-wrong config available to be selected accidentally.
- Do not submit a kernel containing diagnostic configs that produce no output or copy only, even if they were useful during tuning.
- Cleanup is an edit. After cleanup, run the full final sequence on the exact cleaned files; a successful earlier compile or benchmark does not validate cleaned files.

## 16. Tool Output Interpretation

- Environment warnings such as “Failed to find real location of python.exe” accompanying a successful compile are not compiler errors or CUDA kernel bugs. Use the compile success/failure status and actionable compiler errors.
- If a write/read fails because of a malformed workspace payload or missing `operation`, retry the tool call cleanly. A tool failure by itself is not evidence that the CUDA kernel or binding logic is wrong.
- After any failed or ambiguous write, read the target file before compiling. Do not change CUDA kernel or binding logic solely in response to a tool transport/encoding/payload failure.
- If a reported compile-log path cannot be read, use the compile tool output or rerun compile; do not make unrelated kernel changes based on a log-path failure.

**Scope:** Sections 17–20 apply to CUDA extension episodes using this workspace and do not relax the existing forbidden/allowed lists or validation rules. Section 20 applies only to tool/workspace operations and does not classify tool transport/encoding/path failures as CUDA coding errors.

## 17. Validation Order Is Non-Negotiable

- Compile success, verification success, and benchmark success are separate gates. A successful compile does not validate kernel math or wrapper semantics, and a benchmark without preceding verification on the same files is not evidence of correctness.
- After any edit to `kernels/*`, binding files, or `model_new.py`, use this order: finish edits/cleanup → compile if any C++/CUDA/binding file changed → verification → profiling/benchmark. The final benchmark must run on the exact files that passed verification.
- Do not make a late `model_new.py` config or forward-path edit and stop after a benchmark-only run. If the edit selects a different config, dispatch branch, or wrapper path, rerun verification on that exact file, then rerun the benchmark.
- A benchmark-only run may be used only as a tuning probe. Preserve the last verified selection and either verify the final selected file or revert to the verified one.
- Do not end an episode on a write, a compile-only step, or a benchmark-only step after a change. If step budget is nearly exhausted, do not start another speculative optimization; leave or restore the last fully validated files.
- A write after the last verification invalidates that verification. A compile after the last verification still leaves verification stale; recompile, verify, and benchmark before completion.
- If a benchmark-only run is followed by another edit, the earlier benchmark is not final evidence for the new file.

## 18. Reject Unverified Speedups

- If verification fails after a kernel, binding, or wrapper change, reject that candidate even if its benchmark is faster. Fix the failing path or revert to the last verified implementation, then recompile, verify, and benchmark.
- Do not continue tuning `model_new.py` configs while the underlying kernel is failing verification. Correctness must be restored first.
- When a rewrite breaks correctness, use the last passing kernel as the baseline and prefer a minimal fix over another broad rewrite. Keep a known-good copy until the replacement passes verification.
- A config that is fast for one shape but unverified or incorrect for another harness case must not remain selected in final `model_new.py`.
- Do not leave a faster but unverified or known-wrong config, dispatch branch, or wrapper path reachable from the final model path.

## 19. Probe and Debug Artifact Hygiene

- Temporary probe/diagnostic kernels, alternate bindings, and patch/debug files must not be part of the final module path. If they are added under `kernels/` or referenced by a binding, remove them and rerun compile, verification, and benchmark on the cleaned files.
- Do not use a probe benchmark as evidence that the final implementation is correct.
- Do not leave temporary patch files, probe launchers, or notes in `kernels/`; cleanup is an edit and must precede the final validation sequence.
- After deleting a probe file or reverting a binding, recompile so the extension cannot retain stale symbols or an unintended old path.

## 20. Tool Log Path Failures Are Not CUDA Signals

Scope: tool/workspace operations only; these rules do not classify tool transport/encoding/path failures as CUDA coding mistakes.

- A `FileNotFoundError` or unreadable path for a compile, verification, or profiling log is a tool/path issue. Use the tool’s returned output/status or rerun the tool; do not change kernel math, binding allocation, or `model_new.py` based on the failed read alone.
- If a log path cannot be read, do not infer pass/fail from the missing file. Rerun verification/benchmark or use the tool result already returned.
- After a failed or ambiguous workspace read/write, read the target file before compiling or editing further. A failed tool call is not evidence that the file is correct or incorrect.
- Environment warnings such as failure to locate `python.exe` alongside an explicit compile/verify/benchmark success are not CUDA kernel errors; use the explicit status and actionable compiler/runtime errors.

**Scope:** Sections 21–24 apply to CUDA extension episodes using this workspace. They do not relax the existing forbidden/allowed lists or validation requirements. Section 24 applies only to tool/workspace operations and does not classify tool payload or environment failures as CUDA coding mistakes.

## 21. Vectorized Reduction Semantics and Debug Isolation

- When a vector load (`float2`, `float4`) spans the non-reduced axis, each component belongs to a different output column and must have its own reduction accumulator and its own reciprocal normalization value. Do not sum the components into one scalar, and do not broadcast one scalar result across the vector lanes.
- Store and combine vector partials as vector/per-lane values in shared memory (for example, `float4` accumulators), not as one scalar per thread. Apply the same per-lane requirement when scaling and storing the output.
- A vectorized reduction failure with most elements mismatched and the same wrong values across different block shapes is evidence to inspect lane packing/reduction semantics first, not automatically a race or launch-config problem. Isolate it by testing the scalar path, a vectorized copy/indexing path without the reduction, and then the vectorized reduction.
- Keep diagnostic configs that skip reduction or copy data out of the final path. After fixing the vector path, verify both vector-eligible shapes and non-vector fallback shapes.

## 22. Forward-Path Diagnostics and Post-Tuning Validation

- Do not add PyTorch reference calculations (`pow`, `mean`, `sqrt`, etc.) or other debug tensor ops to `model_new.py`/`ModelNew.forward` to diagnose a kernel. The verifier rejects PyTorch computation during candidate forward; that failure is an instrumentation violation, not evidence that the CUDA math is wrong.
- Perform any reference comparison outside the submitted forward path, and remove the diagnostic before final validation. Do not leave debug instrumentation in `model_new.py`.
- A benchmark-only sweep after changing `model_new.py`—including a hardcoded `config`, dispatch branch, or wrapper argument—is a tuning probe. Before treating that selection as final, run verification on the exact final file, then benchmark. A passing verification of an earlier/default config does not validate the new selection.
- Do not treat compile success as correctness validation. If a new candidate is compiled, run verification before using its benchmark as final evidence. If verification fails, reject the faster candidate or revert to the last verified version.

## 23. Clock-State-Aware Benchmark Comparison

- Absolute candidate latencies in this environment can change by roughly 2× between runs as GPU clock state changes. Do not rank tuning variants solely by absolute microseconds measured in separate runs.
- Compare candidate and `torch.compile` baseline in the same benchmark invocation, or repeat/interleave measurements and report the candidate/baseline ratio. A result is more trustworthy when the ratio is stable across clock states.
- Do not use a fast but unverified benchmark under one clock state to justify leaving a different final configuration. Final evidence must come from the exact final files after the required compile → verify → benchmark order.

## 24. Workspace Write Payloads (Tool/Workspace Only)

- For workspace writes, pass `operation: "write"`, `path`, and `content` as separate fields. Do not omit `operation` or put file content/Markdown in the `operation` field.
- A `ValueError: operation must be one of ...` or a missing-`operation` error is a tool payload problem, not a CUDA kernel or binding bug. Retry the write/read correctly; do not change kernel math or allocation logic in response to that error alone.
- After a failed or ambiguous write, read the target file before compiling or making further edits. A failed tool call does not prove the file is correct or incorrect.
- Environment warnings such as “Failed to find real location of python.exe” alongside an explicit compile/verify success are not CUDA errors; use the explicit status and actionable compiler/runtime output.

**Scope:** Sections 25–28 apply to CUDA extension episodes using this workspace. They do not relax Sections 1–24 or the existing forbidden/allowed lists. Section 28 applies only to tool/workspace operations and explicitly does not classify tool transport, encoding, payload, or path failures as CUDA coding mistakes.

## 25. Output Tensor Lifetime and Buffer-Reuse Safety

- Treat “previously returned outputs must remain unchanged after later calls” as a hard lifetime constraint. Never write into, recycle, or reuse storage that may still be referenced by any earlier returned tensor. The verifier/harness may retain outputs across repeated calls.
- For AXPBY-like elementwise ops that return a new tensor, default to the allowed per-call `torch::empty_like(...)` and rely on PyTorch’s allocator for caching. Do not add a custom cross-call output pool merely to reduce allocation overhead unless you can prove with public APIs that no live tensor references the buffer.
- Do not use `TensorImpl::use_count`, `Tensor::use_count`, `storage().use_count`, intrusive-pointer counts, or other refcount internals to decide whether an output buffer is recyclable. Such APIs are unstable/version-dependent, and an observed build failed with `"use_count": is not a member of "c10::TensorImpl"`. A single refcount is also not a contract-safe proof against live Python/harness references.
- If an attempted refcount API does not compile, remove that pooling attempt or use per-call allocation. Do not add guessed headers, private ATen/TensorImpl APIs, or unrelated kernel changes to force it to compile.
- A growing pool is a failure signal. If diagnostic output shows the pool size increasing monotonically, or benchmark latency degrades over repeated calls, the buffers are not safely recyclable. Remove the pool and return to per-call allocation. Do not report a faster benchmark from a pool that was not proven to preserve all previous outputs.
- If you experiment with a pre-reserved chunk or suballocation, keep size/offset metadata on the host. Do not write metadata to device memory through a host pointer; that is a host/device memory bug, not an allocation-tuning detail. Ensure each live output has distinct usable storage or an ownership scheme that cannot overwrite a live result.
- Never alias or overwrite input tensors `a`/`b` to save allocation. Compute into a distinct output. When using typed pointers, validate dtype, device, contiguity, and shape assumptions; if buffer reuse is attempted, match all relevant tensor properties, not just shape.

## 26. No Probe Code in Final Production Paths

- A verification PASS does not mean the binding is clean. Probe output can run during verification and benchmark while tests still pass. Before finalizing, inspect `kernels/*.cu`, `kernels/*_binding.cpp`, and `model_new.py` for diagnostic `printf`/`std::printf`, counters, timing probes, pool-inspection logs, and comments such as `PROBE ONLY`.
- Remove diagnostic code embedded in otherwise final files, not only separate `probe_*.cpp` files. Do not leave global probe state, pool-size printing, or debug-only branches reachable from `model_new.py`.
- After removing probe code, recompile if any C++/CUDA/binding file changed, then run verification and benchmark on the exact cleaned files. Verification or benchmark output from the probe version is not evidence for the cleaned version.
- Temporary probe files must be deleted before final validation. If a probe-specific symbol remains declared or referenced, recompile so the extension cannot retain a stale or unintended path.

## 27. Finish on Validated Files and Submit the Report

- Do not end an episode after a verify-only step when the performance gate has not been measured on the exact final files. The final sequence is: finish edits/cleanup → compile if needed → verification → benchmark on the same files → submit the final report.
- Reserve enough budget for that final sequence. If the remaining budget is small, stop optional pooling/refcount experiments and preserve or restore the last candidate that passed compile, verification, and benchmark. Do not start a speculative optimization that may leave the workspace unvalidated.
- A benchmark from an earlier candidate is not final if later writes changed `model_new.py`, binding files, or kernels. Report only measurements actually obtained from the final validated files; if the final files were not benchmarked, say so and treat the performance goal as unconfirmed.
- If a late buffer-pool or binding experiment fails compile, fails verification, or shows degradation, revert to the known-good per-call allocation version and rerun the required gates rather than leaving the experimental binding in place.

## 28. Tool/Workspace Failure Separation (Scoped)

- A missing `operation` field or `ValueError: operation must be one of ...` is a tooling payload error. Retry the write/read with `operation`, `path`, and `content` as separate fields; do not change kernel math, output pooling, or binding allocation in response to that error alone.
- After any failed or ambiguous write, read the target file before compiling. A failed tool call does not prove the file is unchanged or correct.
- Environment warnings such as failure to locate `python.exe` alongside an explicit compile/verify success are not CUDA errors; use the explicit status and actionable compiler/runtime output. A log-path, encoding, transport, or payload failure is not evidence that the CUDA implementation is wrong.


## Repeated appendix, copy 15

**Scope of appended rules:** Sections 9–12 apply to CUDA extension episodes using this workspace. Section 12 applies only to tool/workspace operations and explicitly does not classify tool transport/encoding failures as CUDA coding errors. None of these rules relax the existing forbidden/allowed lists.

## 9. Compile-Safe Binding and Allocation

- Use only the public allocation exception already allowed by this document. For bindings, prefer the documented `torch::empty_like(...)` allocation path.
- Do not use `at::detail::empty_cuda`, other `at::detail::*` symbols, or version-private ATen internals to bypass the dispatcher. These are not stable extension APIs.
- If compilation reports an error such as `'empty_cuda': is not a member of 'at::detail'`, replace the internal allocation call with the documented allocation path. Do not keep the internal call or add guessed headers to force it to compile.
- Fix binding compile errors in the binding files, not by changing kernel math. A missing binding symbol or undeclared allocation API is not evidence that the CUDA kernel algorithm is wrong.
- Do not replace a compiling binding with a speculative “lean hot path” using internal APIs. Allocation micro-optimizations are lower priority than compile success and correctness; if attempted, isolate the change and keep a known-good fallback.
- When a binding specializes to float32/contiguous CUDA tensors, enforce that specialization with `TORCH_CHECK`/equivalent validation or ensure `model_new.py` guarantees it. Do not reinterpret arbitrary tensor storage as `float*`.

## 10. Final Validation Gate

- The final submitted files must be the exact files that passed compile, verification, and benchmark. Any write after the last successful verification/benchmark invalidates those results.
- Final sequence: finish cleanup and final edits, then compile, then run verification, then run profiling/benchmark. Do not stop immediately after a write.
- Do not perform a post-benchmark speculative rewrite of `kernels/*` or `model_new.py` and declare completion. If optimizing after a passing result, preserve or restore the known-good version; re-run compile, verification, and benchmark; if the new version fails, revert and leave the known-good version as final.
- A successful earlier compile does not excuse a later compile failure. Do not submit an uncompiled final candidate.
- Final cleanup counts as an edit. Run compile, verification, and benchmark after cleanup, or perform cleanup before the final validation sequence.
- On compile failure, use the first actionable compiler error. If the reported compile log path cannot be read, use the compile tool output or rerun compile; do not make unrelated kernel changes based on a log-path failure.

## 11. ModelNew and Wrapper Semantics

- Keep `ModelNew.forward` as the normal method and let the harness use `nn.Module.__call__`.
- Do not add `__call__ = forward` or otherwise bypass `nn.Module` hook machinery as a speculative shortcut. It can change harness/hook behavior. Use such a bypass only if explicitly required and revalidated; otherwise remove it before final validation.
- Preserve the constructor and `forward` signatures expected by `model.py`.
- Keep `model_new.py` limited to allowed tensor creation and custom extension ops. Do not add unsupported Python-side torch operations to compensate for binding or kernel issues.

## 12. Tool/Workspace Hygiene (Not CUDA Coding Errors)

- For `workspace` writes, pass `operation: "write"`, `path`, and `content` as separate fields. Do not put file content or Markdown into the `operation` field.
- Treat `ValueError: unknown operation`, missing `operation` errors, malformed tool payloads, and log encoding/mojibake as tooling issues, not CUDA coding mistakes.
- After any failed or ambiguous write, read the target file before compiling. A failed tool call does not prove the file is unchanged or correct.
- If a write/read fails due to tool transport or encoding, retry the tool call cleanly. Do not change CUDA kernel or binding logic solely in response to the tool failure.
- A missing utility file or unreadable path is a workspace/file-availability issue, not evidence that the kernel is incorrect. Use the documented utilities or list the directory to discover the actual files.

**Scope:** Sections 13–16 apply to CUDA extension episodes using this workspace. They do not relax Sections 1–12 or the existing forbidden/allowed lists. Section 16 applies only to tool/workspace operations and does not classify tool transport/encoding failures as CUDA coding errors.

## 13. Post-Tuning Validation State

- Treat `model_new.py` as part of the correctness surface. Changing the selected kernel `config`, wrapper arguments, or forward path can change numerical behavior; a benchmark-only run after such a change does not validate it.
- After any candidate change to `kernels/*`, bindings, or `model_new.py`, run verification before using or reporting benchmark results. A fast benchmark from an unverified candidate is only a performance probe.
- The final validation sequence must run on the exact final files: finish all edits and cleanup, compile if any C++/CUDA/binding file changed, run verification, then run benchmark. Any write after verification makes that verification stale.
- If a compile and a later write occur in the same step, do not count that compile as final validation for the later file. Ensure the last compile precedes the verified final state, or explicitly confirm no compiled file changed and keep the binary consistent with the verified files.
- Do not end an episode immediately after a benchmark-only edit. If the step budget is nearly exhausted, preserve or restore the last fully validated version rather than making an unverified last-moment change.
- If verification fails after a tuning change, reject the faster benchmark. Either fix the kernel/config and rerun compile, verification, and benchmark, or revert to the last verified version.

## 14. Tuned Config and Shape-Specialized Path Safety

- Do not hardcode a tuned `config` in `model_new.py` solely because it benchmarked fastest. Verify that exact config on all verification cases and seeds before leaving it as the final selection.
- For shape-specialized dispatch, ensure every non-default config either has a correct fallback or is unreachable from the final `model_new.py`. A fast path that is wrong for one harness shape must not remain selected.
- When a config-specific case fails, first inspect dispatch guards, supported `F`/`N` assumptions, and fallback coverage. Do not change unrelated kernel math or move the failure into Python.
- Auto/default passing does not validate a hardcoded config. After switching `model_new.py` from auto/default to a fixed config, rerun verification on that exact final config.
- Diagnostic, no-op, copy-only, or debug configs must not be reachable in the final model path.

## 15. Final Cleanup Specifics

- Before final validation, remove diagnostic no-op/copy kernels, unused tuning branches, and intermediate kernel versions from `kernels/` unless they are still required by the final default path.
- If multiple tuning configs remain, ensure they are all semantically correct for supported shapes, or remove/disable the unused ones. Do not leave a known-wrong config available to be selected accidentally.
- Do not submit a kernel containing diagnostic configs that produce no output or copy only, even if they were useful during tuning.
- Cleanup is an edit. After cleanup, run the full final sequence on the exact cleaned files; a successful earlier compile or benchmark does not validate cleaned files.

## 16. Tool Output Interpretation

- Environment warnings such as “Failed to find real location of python.exe” accompanying a successful compile are not compiler errors or CUDA kernel bugs. Use the compile success/failure status and actionable compiler errors.
- If a write/read fails because of a malformed workspace payload or missing `operation`, retry the tool call cleanly. A tool failure by itself is not evidence that the CUDA kernel or binding logic is wrong.
- After any failed or ambiguous write, read the target file before compiling. Do not change CUDA kernel or binding logic solely in response to a tool transport/encoding/payload failure.
- If a reported compile-log path cannot be read, use the compile tool output or rerun compile; do not make unrelated kernel changes based on a log-path failure.

**Scope:** Sections 17–20 apply to CUDA extension episodes using this workspace and do not relax the existing forbidden/allowed lists or validation rules. Section 20 applies only to tool/workspace operations and does not classify tool transport/encoding/path failures as CUDA coding errors.

## 17. Validation Order Is Non-Negotiable

- Compile success, verification success, and benchmark success are separate gates. A successful compile does not validate kernel math or wrapper semantics, and a benchmark without preceding verification on the same files is not evidence of correctness.
- After any edit to `kernels/*`, binding files, or `model_new.py`, use this order: finish edits/cleanup → compile if any C++/CUDA/binding file changed → verification → profiling/benchmark. The final benchmark must run on the exact files that passed verification.
- Do not make a late `model_new.py` config or forward-path edit and stop after a benchmark-only run. If the edit selects a different config, dispatch branch, or wrapper path, rerun verification on that exact file, then rerun the benchmark.
- A benchmark-only run may be used only as a tuning probe. Preserve the last verified selection and either verify the final selected file or revert to the verified one.
- Do not end an episode on a write, a compile-only step, or a benchmark-only step after a change. If step budget is nearly exhausted, do not start another speculative optimization; leave or restore the last fully validated files.
- A write after the last verification invalidates that verification. A compile after the last verification still leaves verification stale; recompile, verify, and benchmark before completion.
- If a benchmark-only run is followed by another edit, the earlier benchmark is not final evidence for the new file.

## 18. Reject Unverified Speedups

- If verification fails after a kernel, binding, or wrapper change, reject that candidate even if its benchmark is faster. Fix the failing path or revert to the last verified implementation, then recompile, verify, and benchmark.
- Do not continue tuning `model_new.py` configs while the underlying kernel is failing verification. Correctness must be restored first.
- When a rewrite breaks correctness, use the last passing kernel as the baseline and prefer a minimal fix over another broad rewrite. Keep a known-good copy until the replacement passes verification.
- A config that is fast for one shape but unverified or incorrect for another harness case must not remain selected in final `model_new.py`.
- Do not leave a faster but unverified or known-wrong config, dispatch branch, or wrapper path reachable from the final model path.

## 19. Probe and Debug Artifact Hygiene

- Temporary probe/diagnostic kernels, alternate bindings, and patch/debug files must not be part of the final module path. If they are added under `kernels/` or referenced by a binding, remove them and rerun compile, verification, and benchmark on the cleaned files.
- Do not use a probe benchmark as evidence that the final implementation is correct.
- Do not leave temporary patch files, probe launchers, or notes in `kernels/`; cleanup is an edit and must precede the final validation sequence.
- After deleting a probe file or reverting a binding, recompile so the extension cannot retain stale symbols or an unintended old path.

## 20. Tool Log Path Failures Are Not CUDA Signals

Scope: tool/workspace operations only; these rules do not classify tool transport/encoding/path failures as CUDA coding mistakes.

- A `FileNotFoundError` or unreadable path for a compile, verification, or profiling log is a tool/path issue. Use the tool’s returned output/status or rerun the tool; do not change kernel math, binding allocation, or `model_new.py` based on the failed read alone.
- If a log path cannot be read, do not infer pass/fail from the missing file. Rerun verification/benchmark or use the tool result already returned.
- After a failed or ambiguous workspace read/write, read the target file before compiling or editing further. A failed tool call is not evidence that the file is correct or incorrect.
- Environment warnings such as failure to locate `python.exe` alongside an explicit compile/verify/benchmark success are not CUDA kernel errors; use the explicit status and actionable compiler/runtime errors.

**Scope:** Sections 21–24 apply to CUDA extension episodes using this workspace. They do not relax the existing forbidden/allowed lists or validation requirements. Section 24 applies only to tool/workspace operations and does not classify tool payload or environment failures as CUDA coding mistakes.

## 21. Vectorized Reduction Semantics and Debug Isolation

- When a vector load (`float2`, `float4`) spans the non-reduced axis, each component belongs to a different output column and must have its own reduction accumulator and its own reciprocal normalization value. Do not sum the components into one scalar, and do not broadcast one scalar result across the vector lanes.
- Store and combine vector partials as vector/per-lane values in shared memory (for example, `float4` accumulators), not as one scalar per thread. Apply the same per-lane requirement when scaling and storing the output.
- A vectorized reduction failure with most elements mismatched and the same wrong values across different block shapes is evidence to inspect lane packing/reduction semantics first, not automatically a race or launch-config problem. Isolate it by testing the scalar path, a vectorized copy/indexing path without the reduction, and then the vectorized reduction.
- Keep diagnostic configs that skip reduction or copy data out of the final path. After fixing the vector path, verify both vector-eligible shapes and non-vector fallback shapes.

## 22. Forward-Path Diagnostics and Post-Tuning Validation

- Do not add PyTorch reference calculations (`pow`, `mean`, `sqrt`, etc.) or other debug tensor ops to `model_new.py`/`ModelNew.forward` to diagnose a kernel. The verifier rejects PyTorch computation during candidate forward; that failure is an instrumentation violation, not evidence that the CUDA math is wrong.
- Perform any reference comparison outside the submitted forward path, and remove the diagnostic before final validation. Do not leave debug instrumentation in `model_new.py`.
- A benchmark-only sweep after changing `model_new.py`—including a hardcoded `config`, dispatch branch, or wrapper argument—is a tuning probe. Before treating that selection as final, run verification on the exact final file, then benchmark. A passing verification of an earlier/default config does not validate the new selection.
- Do not treat compile success as correctness validation. If a new candidate is compiled, run verification before using its benchmark as final evidence. If verification fails, reject the faster candidate or revert to the last verified version.

## 23. Clock-State-Aware Benchmark Comparison

- Absolute candidate latencies in this environment can change by roughly 2× between runs as GPU clock state changes. Do not rank tuning variants solely by absolute microseconds measured in separate runs.
- Compare candidate and `torch.compile` baseline in the same benchmark invocation, or repeat/interleave measurements and report the candidate/baseline ratio. A result is more trustworthy when the ratio is stable across clock states.
- Do not use a fast but unverified benchmark under one clock state to justify leaving a different final configuration. Final evidence must come from the exact final files after the required compile → verify → benchmark order.

## 24. Workspace Write Payloads (Tool/Workspace Only)

- For workspace writes, pass `operation: "write"`, `path`, and `content` as separate fields. Do not omit `operation` or put file content/Markdown in the `operation` field.
- A `ValueError: operation must be one of ...` or a missing-`operation` error is a tool payload problem, not a CUDA kernel or binding bug. Retry the write/read correctly; do not change kernel math or allocation logic in response to that error alone.
- After a failed or ambiguous write, read the target file before compiling or making further edits. A failed tool call does not prove the file is correct or incorrect.
- Environment warnings such as “Failed to find real location of python.exe” alongside an explicit compile/verify success are not CUDA errors; use the explicit status and actionable compiler/runtime output.

**Scope:** Sections 25–28 apply to CUDA extension episodes using this workspace. They do not relax Sections 1–24 or the existing forbidden/allowed lists. Section 28 applies only to tool/workspace operations and explicitly does not classify tool transport, encoding, payload, or path failures as CUDA coding mistakes.

## 25. Output Tensor Lifetime and Buffer-Reuse Safety

- Treat “previously returned outputs must remain unchanged after later calls” as a hard lifetime constraint. Never write into, recycle, or reuse storage that may still be referenced by any earlier returned tensor. The verifier/harness may retain outputs across repeated calls.
- For AXPBY-like elementwise ops that return a new tensor, default to the allowed per-call `torch::empty_like(...)` and rely on PyTorch’s allocator for caching. Do not add a custom cross-call output pool merely to reduce allocation overhead unless you can prove with public APIs that no live tensor references the buffer.
- Do not use `TensorImpl::use_count`, `Tensor::use_count`, `storage().use_count`, intrusive-pointer counts, or other refcount internals to decide whether an output buffer is recyclable. Such APIs are unstable/version-dependent, and an observed build failed with `"use_count": is not a member of "c10::TensorImpl"`. A single refcount is also not a contract-safe proof against live Python/harness references.
- If an attempted refcount API does not compile, remove that pooling attempt or use per-call allocation. Do not add guessed headers, private ATen/TensorImpl APIs, or unrelated kernel changes to force it to compile.
- A growing pool is a failure signal. If diagnostic output shows the pool size increasing monotonically, or benchmark latency degrades over repeated calls, the buffers are not safely recyclable. Remove the pool and return to per-call allocation. Do not report a faster benchmark from a pool that was not proven to preserve all previous outputs.
- If you experiment with a pre-reserved chunk or suballocation, keep size/offset metadata on the host. Do not write metadata to device memory through a host pointer; that is a host/device memory bug, not an allocation-tuning detail. Ensure each live output has distinct usable storage or an ownership scheme that cannot overwrite a live result.
- Never alias or overwrite input tensors `a`/`b` to save allocation. Compute into a distinct output. When using typed pointers, validate dtype, device, contiguity, and shape assumptions; if buffer reuse is attempted, match all relevant tensor properties, not just shape.

## 26. No Probe Code in Final Production Paths

- A verification PASS does not mean the binding is clean. Probe output can run during verification and benchmark while tests still pass. Before finalizing, inspect `kernels/*.cu`, `kernels/*_binding.cpp`, and `model_new.py` for diagnostic `printf`/`std::printf`, counters, timing probes, pool-inspection logs, and comments such as `PROBE ONLY`.
- Remove diagnostic code embedded in otherwise final files, not only separate `probe_*.cpp` files. Do not leave global probe state, pool-size printing, or debug-only branches reachable from `model_new.py`.
- After removing probe code, recompile if any C++/CUDA/binding file changed, then run verification and benchmark on the exact cleaned files. Verification or benchmark output from the probe version is not evidence for the cleaned version.
- Temporary probe files must be deleted before final validation. If a probe-specific symbol remains declared or referenced, recompile so the extension cannot retain a stale or unintended path.

## 27. Finish on Validated Files and Submit the Report

- Do not end an episode after a verify-only step when the performance gate has not been measured on the exact final files. The final sequence is: finish edits/cleanup → compile if needed → verification → benchmark on the same files → submit the final report.
- Reserve enough budget for that final sequence. If the remaining budget is small, stop optional pooling/refcount experiments and preserve or restore the last candidate that passed compile, verification, and benchmark. Do not start a speculative optimization that may leave the workspace unvalidated.
- A benchmark from an earlier candidate is not final if later writes changed `model_new.py`, binding files, or kernels. Report only measurements actually obtained from the final validated files; if the final files were not benchmarked, say so and treat the performance goal as unconfirmed.
- If a late buffer-pool or binding experiment fails compile, fails verification, or shows degradation, revert to the known-good per-call allocation version and rerun the required gates rather than leaving the experimental binding in place.

## 28. Tool/Workspace Failure Separation (Scoped)

- A missing `operation` field or `ValueError: operation must be one of ...` is a tooling payload error. Retry the write/read with `operation`, `path`, and `content` as separate fields; do not change kernel math, output pooling, or binding allocation in response to that error alone.
- After any failed or ambiguous write, read the target file before compiling. A failed tool call does not prove the file is unchanged or correct.
- Environment warnings such as failure to locate `python.exe` alongside an explicit compile/verify success are not CUDA errors; use the explicit status and actionable compiler/runtime output. A log-path, encoding, transport, or payload failure is not evidence that the CUDA implementation is wrong.


## Repeated appendix, copy 16

**Scope of appended rules:** Sections 9–12 apply to CUDA extension episodes using this workspace. Section 12 applies only to tool/workspace operations and explicitly does not classify tool transport/encoding failures as CUDA coding errors. None of these rules relax the existing forbidden/allowed lists.

## 9. Compile-Safe Binding and Allocation

- Use only the public allocation exception already allowed by this document. For bindings, prefer the documented `torch::empty_like(...)` allocation path.
- Do not use `at::detail::empty_cuda`, other `at::detail::*` symbols, or version-private ATen internals to bypass the dispatcher. These are not stable extension APIs.
- If compilation reports an error such as `'empty_cuda': is not a member of 'at::detail'`, replace the internal allocation call with the documented allocation path. Do not keep the internal call or add guessed headers to force it to compile.
- Fix binding compile errors in the binding files, not by changing kernel math. A missing binding symbol or undeclared allocation API is not evidence that the CUDA kernel algorithm is wrong.
- Do not replace a compiling binding with a speculative “lean hot path” using internal APIs. Allocation micro-optimizations are lower priority than compile success and correctness; if attempted, isolate the change and keep a known-good fallback.
- When a binding specializes to float32/contiguous CUDA tensors, enforce that specialization with `TORCH_CHECK`/equivalent validation or ensure `model_new.py` guarantees it. Do not reinterpret arbitrary tensor storage as `float*`.

## 10. Final Validation Gate

- The final submitted files must be the exact files that passed compile, verification, and benchmark. Any write after the last successful verification/benchmark invalidates those results.
- Final sequence: finish cleanup and final edits, then compile, then run verification, then run profiling/benchmark. Do not stop immediately after a write.
- Do not perform a post-benchmark speculative rewrite of `kernels/*` or `model_new.py` and declare completion. If optimizing after a passing result, preserve or restore the known-good version; re-run compile, verification, and benchmark; if the new version fails, revert and leave the known-good version as final.
- A successful earlier compile does not excuse a later compile failure. Do not submit an uncompiled final candidate.
- Final cleanup counts as an edit. Run compile, verification, and benchmark after cleanup, or perform cleanup before the final validation sequence.
- On compile failure, use the first actionable compiler error. If the reported compile log path cannot be read, use the compile tool output or rerun compile; do not make unrelated kernel changes based on a log-path failure.

## 11. ModelNew and Wrapper Semantics

- Keep `ModelNew.forward` as the normal method and let the harness use `nn.Module.__call__`.
- Do not add `__call__ = forward` or otherwise bypass `nn.Module` hook machinery as a speculative shortcut. It can change harness/hook behavior. Use such a bypass only if explicitly required and revalidated; otherwise remove it before final validation.
- Preserve the constructor and `forward` signatures expected by `model.py`.
- Keep `model_new.py` limited to allowed tensor creation and custom extension ops. Do not add unsupported Python-side torch operations to compensate for binding or kernel issues.

## 12. Tool/Workspace Hygiene (Not CUDA Coding Errors)

- For `workspace` writes, pass `operation: "write"`, `path`, and `content` as separate fields. Do not put file content or Markdown into the `operation` field.
- Treat `ValueError: unknown operation`, missing `operation` errors, malformed tool payloads, and log encoding/mojibake as tooling issues, not CUDA coding mistakes.
- After any failed or ambiguous write, read the target file before compiling. A failed tool call does not prove the file is unchanged or correct.
- If a write/read fails due to tool transport or encoding, retry the tool call cleanly. Do not change CUDA kernel or binding logic solely in response to the tool failure.
- A missing utility file or unreadable path is a workspace/file-availability issue, not evidence that the kernel is incorrect. Use the documented utilities or list the directory to discover the actual files.

**Scope:** Sections 13–16 apply to CUDA extension episodes using this workspace. They do not relax Sections 1–12 or the existing forbidden/allowed lists. Section 16 applies only to tool/workspace operations and does not classify tool transport/encoding failures as CUDA coding errors.

## 13. Post-Tuning Validation State

- Treat `model_new.py` as part of the correctness surface. Changing the selected kernel `config`, wrapper arguments, or forward path can change numerical behavior; a benchmark-only run after such a change does not validate it.
- After any candidate change to `kernels/*`, bindings, or `model_new.py`, run verification before using or reporting benchmark results. A fast benchmark from an unverified candidate is only a performance probe.
- The final validation sequence must run on the exact final files: finish all edits and cleanup, compile if any C++/CUDA/binding file changed, run verification, then run benchmark. Any write after verification makes that verification stale.
- If a compile and a later write occur in the same step, do not count that compile as final validation for the later file. Ensure the last compile precedes the verified final state, or explicitly confirm no compiled file changed and keep the binary consistent with the verified files.
- Do not end an episode immediately after a benchmark-only edit. If the step budget is nearly exhausted, preserve or restore the last fully validated version rather than making an unverified last-moment change.
- If verification fails after a tuning change, reject the faster benchmark. Either fix the kernel/config and rerun compile, verification, and benchmark, or revert to the last verified version.

## 14. Tuned Config and Shape-Specialized Path Safety

- Do not hardcode a tuned `config` in `model_new.py` solely because it benchmarked fastest. Verify that exact config on all verification cases and seeds before leaving it as the final selection.
- For shape-specialized dispatch, ensure every non-default config either has a correct fallback or is unreachable from the final `model_new.py`. A fast path that is wrong for one harness shape must not remain selected.
- When a config-specific case fails, first inspect dispatch guards, supported `F`/`N` assumptions, and fallback coverage. Do not change unrelated kernel math or move the failure into Python.
- Auto/default passing does not validate a hardcoded config. After switching `model_new.py` from auto/default to a fixed config, rerun verification on that exact final config.
- Diagnostic, no-op, copy-only, or debug configs must not be reachable in the final model path.

## 15. Final Cleanup Specifics

- Before final validation, remove diagnostic no-op/copy kernels, unused tuning branches, and intermediate kernel versions from `kernels/` unless they are still required by the final default path.
- If multiple tuning configs remain, ensure they are all semantically correct for supported shapes, or remove/disable the unused ones. Do not leave a known-wrong config available to be selected accidentally.
- Do not submit a kernel containing diagnostic configs that produce no output or copy only, even if they were useful during tuning.
- Cleanup is an edit. After cleanup, run the full final sequence on the exact cleaned files; a successful earlier compile or benchmark does not validate cleaned files.

## 16. Tool Output Interpretation

- Environment warnings such as “Failed to find real location of python.exe” accompanying a successful compile are not compiler errors or CUDA kernel bugs. Use the compile success/failure status and actionable compiler errors.
- If a write/read fails because of a malformed workspace payload or missing `operation`, retry the tool call cleanly. A tool failure by itself is not evidence that the CUDA kernel or binding logic is wrong.
- After any failed or ambiguous write, read the target file before compiling. Do not change CUDA kernel or binding logic solely in response to a tool transport/encoding/payload failure.
- If a reported compile-log path cannot be read, use the compile tool output or rerun compile; do not make unrelated kernel changes based on a log-path failure.

**Scope:** Sections 17–20 apply to CUDA extension episodes using this workspace and do not relax the existing forbidden/allowed lists or validation rules. Section 20 applies only to tool/workspace operations and does not classify tool transport/encoding/path failures as CUDA coding errors.

## 17. Validation Order Is Non-Negotiable

- Compile success, verification success, and benchmark success are separate gates. A successful compile does not validate kernel math or wrapper semantics, and a benchmark without preceding verification on the same files is not evidence of correctness.
- After any edit to `kernels/*`, binding files, or `model_new.py`, use this order: finish edits/cleanup → compile if any C++/CUDA/binding file changed → verification → profiling/benchmark. The final benchmark must run on the exact files that passed verification.
- Do not make a late `model_new.py` config or forward-path edit and stop after a benchmark-only run. If the edit selects a different config, dispatch branch, or wrapper path, rerun verification on that exact file, then rerun the benchmark.
- A benchmark-only run may be used only as a tuning probe. Preserve the last verified selection and either verify the final selected file or revert to the verified one.
- Do not end an episode on a write, a compile-only step, or a benchmark-only step after a change. If step budget is nearly exhausted, do not start another speculative optimization; leave or restore the last fully validated files.
- A write after the last verification invalidates that verification. A compile after the last verification still leaves verification stale; recompile, verify, and benchmark before completion.
- If a benchmark-only run is followed by another edit, the earlier benchmark is not final evidence for the new file.

## 18. Reject Unverified Speedups

- If verification fails after a kernel, binding, or wrapper change, reject that candidate even if its benchmark is faster. Fix the failing path or revert to the last verified implementation, then recompile, verify, and benchmark.
- Do not continue tuning `model_new.py` configs while the underlying kernel is failing verification. Correctness must be restored first.
- When a rewrite breaks correctness, use the last passing kernel as the baseline and prefer a minimal fix over another broad rewrite. Keep a known-good copy until the replacement passes verification.
- A config that is fast for one shape but unverified or incorrect for another harness case must not remain selected in final `model_new.py`.
- Do not leave a faster but unverified or known-wrong config, dispatch branch, or wrapper path reachable from the final model path.

## 19. Probe and Debug Artifact Hygiene

- Temporary probe/diagnostic kernels, alternate bindings, and patch/debug files must not be part of the final module path. If they are added under `kernels/` or referenced by a binding, remove them and rerun compile, verification, and benchmark on the cleaned files.
- Do not use a probe benchmark as evidence that the final implementation is correct.
- Do not leave temporary patch files, probe launchers, or notes in `kernels/`; cleanup is an edit and must precede the final validation sequence.
- After deleting a probe file or reverting a binding, recompile so the extension cannot retain stale symbols or an unintended old path.

## 20. Tool Log Path Failures Are Not CUDA Signals

Scope: tool/workspace operations only; these rules do not classify tool transport/encoding/path failures as CUDA coding mistakes.

- A `FileNotFoundError` or unreadable path for a compile, verification, or profiling log is a tool/path issue. Use the tool’s returned output/status or rerun the tool; do not change kernel math, binding allocation, or `model_new.py` based on the failed read alone.
- If a log path cannot be read, do not infer pass/fail from the missing file. Rerun verification/benchmark or use the tool result already returned.
- After a failed or ambiguous workspace read/write, read the target file before compiling or editing further. A failed tool call is not evidence that the file is correct or incorrect.
- Environment warnings such as failure to locate `python.exe` alongside an explicit compile/verify/benchmark success are not CUDA kernel errors; use the explicit status and actionable compiler/runtime errors.

**Scope:** Sections 21–24 apply to CUDA extension episodes using this workspace. They do not relax the existing forbidden/allowed lists or validation requirements. Section 24 applies only to tool/workspace operations and does not classify tool payload or environment failures as CUDA coding mistakes.

## 21. Vectorized Reduction Semantics and Debug Isolation

- When a vector load (`float2`, `float4`) spans the non-reduced axis, each component belongs to a different output column and must have its own reduction accumulator and its own reciprocal normalization value. Do not sum the components into one scalar, and do not broadcast one scalar result across the vector lanes.
- Store and combine vector partials as vector/per-lane values in shared memory (for example, `float4` accumulators), not as one scalar per thread. Apply the same per-lane requirement when scaling and storing the output.
- A vectorized reduction failure with most elements mismatched and the same wrong values across different block shapes is evidence to inspect lane packing/reduction semantics first, not automatically a race or launch-config problem. Isolate it by testing the scalar path, a vectorized copy/indexing path without the reduction, and then the vectorized reduction.
- Keep diagnostic configs that skip reduction or copy data out of the final path. After fixing the vector path, verify both vector-eligible shapes and non-vector fallback shapes.

## 22. Forward-Path Diagnostics and Post-Tuning Validation

- Do not add PyTorch reference calculations (`pow`, `mean`, `sqrt`, etc.) or other debug tensor ops to `model_new.py`/`ModelNew.forward` to diagnose a kernel. The verifier rejects PyTorch computation during candidate forward; that failure is an instrumentation violation, not evidence that the CUDA math is wrong.
- Perform any reference comparison outside the submitted forward path, and remove the diagnostic before final validation. Do not leave debug instrumentation in `model_new.py`.
- A benchmark-only sweep after changing `model_new.py`—including a hardcoded `config`, dispatch branch, or wrapper argument—is a tuning probe. Before treating that selection as final, run verification on the exact final file, then benchmark. A passing verification of an earlier/default config does not validate the new selection.
- Do not treat compile success as correctness validation. If a new candidate is compiled, run verification before using its benchmark as final evidence. If verification fails, reject the faster candidate or revert to the last verified version.

## 23. Clock-State-Aware Benchmark Comparison

- Absolute candidate latencies in this environment can change by roughly 2× between runs as GPU clock state changes. Do not rank tuning variants solely by absolute microseconds measured in separate runs.
- Compare candidate and `torch.compile` baseline in the same benchmark invocation, or repeat/interleave measurements and report the candidate/baseline ratio. A result is more trustworthy when the ratio is stable across clock states.
- Do not use a fast but unverified benchmark under one clock state to justify leaving a different final configuration. Final evidence must come from the exact final files after the required compile → verify → benchmark order.

## 24. Workspace Write Payloads (Tool/Workspace Only)

- For workspace writes, pass `operation: "write"`, `path`, and `content` as separate fields. Do not omit `operation` or put file content/Markdown in the `operation` field.
- A `ValueError: operation must be one of ...` or a missing-`operation` error is a tool payload problem, not a CUDA kernel or binding bug. Retry the write/read correctly; do not change kernel math or allocation logic in response to that error alone.
- After a failed or ambiguous write, read the target file before compiling or making further edits. A failed tool call does not prove the file is correct or incorrect.
- Environment warnings such as “Failed to find real location of python.exe” alongside an explicit compile/verify success are not CUDA errors; use the explicit status and actionable compiler/runtime output.

**Scope:** Sections 25–28 apply to CUDA extension episodes using this workspace. They do not relax Sections 1–24 or the existing forbidden/allowed lists. Section 28 applies only to tool/workspace operations and explicitly does not classify tool transport, encoding, payload, or path failures as CUDA coding mistakes.

## 25. Output Tensor Lifetime and Buffer-Reuse Safety

- Treat “previously returned outputs must remain unchanged after later calls” as a hard lifetime constraint. Never write into, recycle, or reuse storage that may still be referenced by any earlier returned tensor. The verifier/harness may retain outputs across repeated calls.
- For AXPBY-like elementwise ops that return a new tensor, default to the allowed per-call `torch::empty_like(...)` and rely on PyTorch’s allocator for caching. Do not add a custom cross-call output pool merely to reduce allocation overhead unless you can prove with public APIs that no live tensor references the buffer.
- Do not use `TensorImpl::use_count`, `Tensor::use_count`, `storage().use_count`, intrusive-pointer counts, or other refcount internals to decide whether an output buffer is recyclable. Such APIs are unstable/version-dependent, and an observed build failed with `"use_count": is not a member of "c10::TensorImpl"`. A single refcount is also not a contract-safe proof against live Python/harness references.
- If an attempted refcount API does not compile, remove that pooling attempt or use per-call allocation. Do not add guessed headers, private ATen/TensorImpl APIs, or unrelated kernel changes to force it to compile.
- A growing pool is a failure signal. If diagnostic output shows the pool size increasing monotonically, or benchmark latency degrades over repeated calls, the buffers are not safely recyclable. Remove the pool and return to per-call allocation. Do not report a faster benchmark from a pool that was not proven to preserve all previous outputs.
- If you experiment with a pre-reserved chunk or suballocation, keep size/offset metadata on the host. Do not write metadata to device memory through a host pointer; that is a host/device memory bug, not an allocation-tuning detail. Ensure each live output has distinct usable storage or an ownership scheme that cannot overwrite a live result.
- Never alias or overwrite input tensors `a`/`b` to save allocation. Compute into a distinct output. When using typed pointers, validate dtype, device, contiguity, and shape assumptions; if buffer reuse is attempted, match all relevant tensor properties, not just shape.

## 26. No Probe Code in Final Production Paths

- A verification PASS does not mean the binding is clean. Probe output can run during verification and benchmark while tests still pass. Before finalizing, inspect `kernels/*.cu`, `kernels/*_binding.cpp`, and `model_new.py` for diagnostic `printf`/`std::printf`, counters, timing probes, pool-inspection logs, and comments such as `PROBE ONLY`.
- Remove diagnostic code embedded in otherwise final files, not only separate `probe_*.cpp` files. Do not leave global probe state, pool-size printing, or debug-only branches reachable from `model_new.py`.
- After removing probe code, recompile if any C++/CUDA/binding file changed, then run verification and benchmark on the exact cleaned files. Verification or benchmark output from the probe version is not evidence for the cleaned version.
- Temporary probe files must be deleted before final validation. If a probe-specific symbol remains declared or referenced, recompile so the extension cannot retain a stale or unintended path.

## 27. Finish on Validated Files and Submit the Report

- Do not end an episode after a verify-only step when the performance gate has not been measured on the exact final files. The final sequence is: finish edits/cleanup → compile if needed → verification → benchmark on the same files → submit the final report.
- Reserve enough budget for that final sequence. If the remaining budget is small, stop optional pooling/refcount experiments and preserve or restore the last candidate that passed compile, verification, and benchmark. Do not start a speculative optimization that may leave the workspace unvalidated.
- A benchmark from an earlier candidate is not final if later writes changed `model_new.py`, binding files, or kernels. Report only measurements actually obtained from the final validated files; if the final files were not benchmarked, say so and treat the performance goal as unconfirmed.
- If a late buffer-pool or binding experiment fails compile, fails verification, or shows degradation, revert to the known-good per-call allocation version and rerun the required gates rather than leaving the experimental binding in place.

## 28. Tool/Workspace Failure Separation (Scoped)

- A missing `operation` field or `ValueError: operation must be one of ...` is a tooling payload error. Retry the write/read with `operation`, `path`, and `content` as separate fields; do not change kernel math, output pooling, or binding allocation in response to that error alone.
- After any failed or ambiguous write, read the target file before compiling. A failed tool call does not prove the file is unchanged or correct.
- Environment warnings such as failure to locate `python.exe` alongside an explicit compile/verify success are not CUDA errors; use the explicit status and actionable compiler/runtime output. A log-path, encoding, transport, or payload failure is not evidence that the CUDA implementation is wrong.
