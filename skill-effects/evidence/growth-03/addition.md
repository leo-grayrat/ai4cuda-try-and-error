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