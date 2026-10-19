## Concrete discrepancies

**1. C01 — strengthened objective/criterion wording**
Original: “minimum requirement of 5% faster than torch.compile baseline”; “🎯 MINIMUM REQUIREMENT: At least 5% faster than torch.compile (≤ 0.95× baseline time)”.
Optimized: “[C01] Requirement: Produce correct CUDA C++ extension output and achieve candidate latency <= 0.95 times torch.compile, then seek the best attainable performance.”
Discrepancy: The original states the minimum as a relative speed claim (“5% faster”), then clarifies with a token example. The optimized rewrites the pass criterion only as a latency-ratio inequality. That is a different formulation and can be read as making the ratio itself the required condition irrespective of how the baseline time was measured. Also the original’s explicit “minimum requirement” framing is compressed into a conjunction with correctness.
Minimal correction: “Produce correct output; the minimum is at least 5% faster than torch.compile (for example ≤ 0.95× baseline time), then seek the best attainable performance.”

**2. C16 — altered compile requirement**
Original: “Compile with architecture-specific optimizations” followed by the concrete command `TORCH_CUDA_ARCH_LIST=9.0 bash utils/compile.sh`, and “Use sudo to run sandbox utilities”.
Optimized: “[C16] Requirement: Compile for the actual GPU architecture, then execute verification and profiling using the supplied platform-equivalent commands.”
Discrepancy: “actual GPU architecture” is not in the original; the original instructs architecture-specific compilation and gives `9.0` as the command value. “Platform-equivalent commands” is an addition that changes the required command semantics, and the `sudo` qualifier for the sandbox utilities is dropped.
Minimal correction: “Compile with architecture-specific optimizations (e.g. `TORCH_CUDA_ARCH_LIST=9.0 bash utils/compile.sh`), then run `sudo python3 -m utils.verification` and `sudo python3 -m utils.profiling`.”

**3. C21 — recommendation strength strengthened to requirement**
Original: “Pass config parameters through bindings - Enable runtime tuning without recompilation” (Key Reminders; example binding shows a config parameter).
Optimized: “[C21] Requirement: Expose configuration parameters through bindings for runtime tuning without recompilation.”
Discrepancy: The original presents this as a key reminder/example practice, not a MUST-level requirement equivalent to the forbidden-modification rules. The optimized elevates it to “Requirement,” strengthening applicability.
Minimal correction: “Example practice: Pass config parameters through bindings to enable runtime tuning without recompilation.”

**4. C23 — strengthened timing/scope of documentation obligation**
Original: “For each iteration: 1. **Document expectation**: ‘Fusion will eliminate 3 kernels, expect ~20% speedup’” and reminder “Document performance expectations - Before implementing, state expected gains”.
Optimized: “[C23] Requirement: Before implementing each optimization, document the expected effect and why it should help.”
Discrepancy: The original requires documenting an expected effect (with a quantified example) per iteration; it does not require a separate “why it should help” rationale. The optimized adds a new obligation (“why it should help”) and shifts from per-iteration to per-optimization.
Minimal correction: “Requirement: For each iteration, before implementing, document the expected effect (for example expected speedup).”

**5. C25 — strengthened from analysis step to separate requirement**
Original: “4. **Measure and analyze**: Understand why performance changed”.
Optimized: “[C25] Requirement: Measure the optimization and explain why performance changed.”
Discrepancy: “Understand” is an internal analysis instruction; “explain” introduces an output/documentation obligation not present in the original.
Minimal correction: “Requirement: Measure and analyze the optimization to understand why performance changed.”

**6. C28 — narrowed applicability**
Original: “Boundary conditions (tid < size)”; checklist “Boundary Checks: Validate all array accesses (tid < size)”; “Thread Bounds: Check tid < N before array access”.
Optimized: “[C28] Conditional advice: Check thread bounds and handle non-divisible input sizes.”
Discrepancy: The original requires validating all array accesses with a bounds check; the optimized recasts this as advice and adds “non-divisible input sizes,” which is not the same statement and narrows the general bounds requirement.
Minimal correction: “Advice: Validate all array accesses (e.g. tid < size); handle thread bounds.”

**7. C32 — merged/mutated NaN guidance**
Original correctness checklist: “**Numerical Stability**: Handle NaN/Inf, use stable algorithms”; issue table: “NaN/Inf results | 1. Check division by zero 2. Verify numerical stability 3. Add bounds checking”.
Optimized: “[C32] Conditional advice: Use numerically stable algorithms and investigate NaN/Inf, division by zero and missing bounds checks.”
Discrepancy: The original says to *handle* NaN/Inf and *add bounds checking* as remedies; the optimized reduces this to “investigate … missing bounds checks,” weakening the corrective action.
Minimal correction: “Advice: Use stable algorithms; handle NaN/Inf, check division by zero, and add bounds checking.”

**8. C43 — weakened prescription**
Original table: “Slower than baseline | No fusion | Combine kernels”; “Low SM efficiency | Poor occupancy | Tune block size”; “Low memory throughput | Uncoalesced access | Restructure memory pattern”; “High kernel count | Missing fusion | Implement compound operations”.
Optimized: “[C43] Conditional advice: For poor performance, investigate missing fusion, occupancy, coalescing and excessive kernel count as relevant to the observed symptom.”
Discrepancy: Original pairs each symptom with a specific corrective action; optimized turns them into a generic “investigate” list and drops the required remedies (combine kernels, tune block size, restructure memory pattern, implement compound operations).
Minimal correction: “Advice: Slower than baseline → combine kernels (fusion); low SM efficiency → tune block size/occupancy; low memory throughput → restructure for coalescing; high kernel count → implement compound/fused operations.”

**9. C11/C12 — example status and content drift**
Original binding example includes validation only for cuda/contiguous/float32 (`TORCH_CHECK(input.is_cuda()...`, `is_contiguous()`, `dtype()==kFloat32`) and obtains the stream via `c10::cuda::getCurrentCUDAStream().stream()`.
Optimized C11: “Validate CUDA placement, contiguity and supported dtype in the binding.” C12: “Launch on the current PyTorch CUDA stream.”
Discrepancy: Minor—original shows a concrete float32 check and concrete stream API; optimized generalizes “float32” to “supported dtype” and omits the concrete API. This is mostly paraphrase, but “supported dtype” could be read as permitting types the original example did not validate. Flagged only as qualification drift.
Minimal correction: Keep example wording: validate CUDA, contiguous, `torch::kFloat32`; use `c10::cuda::getCurrentCUDAStream().stream()`.

**10. C04 — Conv/ConvTranspose strengthening/qualification**
Original: “cuDNN (MANDATORY for Conv/ConvTranspose)” and “NO third-party libraries: Except cuBLAS (GEMM only) and cuDNN (Conv only)”.
Optimized: “Use no third-party compute libraries except cuBLAS for GEMM and cuDNN for convolution; use cuDNN for Conv and ConvTranspose.”
Discrepancy: Slight scope shift: original says cuDNN “Conv only” in the exception list but also “MANDATORY for Conv/ConvTranspose”; the optimized merges these. No semantic loss, but the original’s internal tension (“Conv only” vs ConvTranspose) is left unresolved. Ambiguity exists in the original; do not invent a resolution.
Minimal correction: none required beyond preserving both phrases; note original ambiguity.

**11. C26 — iteration guidance made conditional/quantified differently**
Original: “First 1-2 iterations: Achieve the minimum 5% improvement”; “Next 3-5 iterations: Push for maximum possible speedup”; “Continue until no further improvements possible or diminishing returns”.
Optimized: “[C26] … until no further improvement or diminishing returns; the source suggests 1-2 initial and 3-5 further iterations.”
Discrepancy: “3-5 further iterations” vs original “Next 3-5 iterations” (i.e., iterations 2-7, not additional beyond the first phase). Slight numeric shift.
Minimal correction: “the source suggests 1-2 initial iterations to meet the minimum, then 3-5 iterations pushing for maximum speedup.”

## Coverage statement

All C01–C45 map to content present in the original; no obligation class is wholly absent. The substantive issues are: C01 reformulates the pass criterion; C16 changes the compile command/architecture wording and drops `sudo`; C21, C23, C25 strengthen reminders/analysis into requirement-level output obligations (C23 also adds “why it should help”); C28 weakens a universal bounds-check instruction into conditional advice and adds non-divisibility handling; C32 weakens corrective NaN/bounds actions to investigation; C43 weakens symptom→remedy prescriptions into a generic investigate list; C26 slightly shifts iteration counts. C04 contains an ambiguity already present in the original (“cuDNN (Conv only)” vs “MANDATORY for Conv/ConvTranspose”). C10–C13, C14–C15, C17–C20, C22, C24, C27, C29–C31, C33–C42, C44–C45 are preserved in substance (some as paraphrase). External experiment budget references are identical for all conditions and are not counted as discrepancies.