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