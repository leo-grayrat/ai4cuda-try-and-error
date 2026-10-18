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