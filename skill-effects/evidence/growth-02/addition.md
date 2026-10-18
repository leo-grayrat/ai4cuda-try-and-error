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