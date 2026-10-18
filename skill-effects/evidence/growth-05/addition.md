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