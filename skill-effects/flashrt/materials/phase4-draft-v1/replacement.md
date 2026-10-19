### Phase 4: Validation and iteration

#### Exploration and queue

Complete at least 10 distinct deployment variants, exercising different optimization ideas. A bug fix to the same idea does not count as a distinct variant. Ten is a floor: continue until every queued hypothesis has a measured result or satisfies one of the narrow terminal exceptions below, and the definition-of-done checklist is satisfied. Cover the axes surfaced by IR analysis, including pipelining/streaming, other model parallelism, and co-location/device placement. Include both isolated levers and combinations. Prefer measuring isolated levers against the user's reference before combining them; name a combined variant for the levers it stacks, not `combined_best`. Do not start by combining multiple untested optimizations.

Before implementing any variant, save an explicit ordered initial queue in the experiment log. Each row contains its name, one-sentence hypothesis, specific motivating IR analysis result, predicted impact on active targets with a rough magnitude, and status: `planned`, `in_progress`, `measured`, `blocked`, or `discarded`. Include every opportunity surfaced by IR analysis, below-IR-granularity levers identified in Phase 2, and combinations. Rank by IR-analysis-based expected impact, irrespective of implementation cost.

Update the queue between every measured variant: record its result, add hypotheses suggested by wins, regressions, or invalidated earlier assumptions, and reconsider remaining ordering. Apply the ten-variant floor to the final queue. An early large win calls for more exploration; it does not justify skipping ambitious or smaller-gain variants.

Every follow-up mentioned in a log entry must become a queue row in the same edit that finalizes the entry, or receive the written removal reason and evidence specified below. A prediction of another variant's numerical results likewise requires that variant to be queued as `planned` before finalizing the current entry. Free-text follow-ups and predictions never replace implementation and measurement.

When two measured variants receive `promote` on different active optimization axes and neither strictly dominates the other across the ranked axes, enqueue `combined(A, B)` as `planned` before either isolated variant is closed out. This includes primary/secondary axes under an app override. Measure that combination or satisfy the single legal-removal exception below.

Every queued variant needs its own independent implementation, correctness check, and benchmark. A related, scaled, simplified, or combined variant cannot stand in for it: their compute, memory, communication, contention, synchronization, and implementation profiles differ. Domination is a claim between measured variants. Expected regression, small impact, communication cost, lack of critical-path relevance, or an apparently superior measured sibling cannot justify skipping an unmeasured variant. Record such predictions as hypotheses and test them.

#### Work scope and persistence

Write code only under `flashrt/apps/<app>/backend/`. Keep each benchmarked variant in its own subpackage/config rather than mutating a previously good variant in place.

**Read-only areas are not a blocker — vendor the file into your workspace.** When behavior in a read-only file must change, vendor the necessary file or subset into the variant directory, modify and import the copy, and add a one-line source-path/divergence comment. Validate numerical parity of unchanged paths against the original, which remains the oracle. This applies to shared adapters, models, runtime, `flashrt_kernels/`, and other read-only code.

Audit reusable runtime support before dismissing a direction: grep `flashrt/*` for the relevant abstraction and read at least the most promising file. If manual implementation is needed, proceed. Do not defer a direction because of implementation complexity or missing runtime coverage. The experiment log must never say “audited `flashrt/<path>` and it does not cover this case because …,” “no relevant support found in the shared runtime after grepping for …,” or “too complex.”

**Implementation and debugging cost are never valid rationale for skipping, deferring, downgrading, substituting, or rotating off of a queued variant.** There is no session deadline or self-imposed wall-clock/per-variant limit. Implementation cost, debugging difficulty, integration, new infrastructure, distributed/async composition, missing read-only edit permission, existing wins, anticipated user permission to stop, or reaching ten variants never justify skipping, downgrading, proxying, substituting, deferring, or rotating away from a queued variant. Work on the current variant until it reaches a terminal status. If no progress is possible after a serious attempt, apply the strict environmental-blocker criteria below.

**Forbidden session-handoff language.** The log, README, and final status describe this session's actual state. They must not contain these phrases or technical paraphrases: “next session,” “future session,” “follow-on session,” “pick this up later,” “I'll come back to this,” “come back to it,” “the next session's first target,” “beyond this session's scope,” “deferred to a later session,” “a future agent can,” “out of scope for this session,” or “left for follow-up.” Every queued variant remains this session's responsibility until terminal.

#### Terminal statuses and removal evidence

**Terminal statuses.** `planned` and `in_progress` are non-terminal. The terminal statuses are:

- `measured`: launched end-to-end, with an executed correctness verdict (PASS or FAIL), its persisted harness log, and benchmark JSON from a real end-to-end run.
- `discarded`: the IR analysis incorrectly declared a transformation legal; supply the dependency it missed, a minimal on-disk reproduction showing that even a correctly implemented transformation corrupts output, and the erroneous IR file/operator declaration. Correct the IR and re-run validation in the same edit. Include a failing transformation-legality test under `flashrt/apps/<app>/backend/tests/` and the self-framing paragraph below.
- `blocked`: all three criteria below hold, with the required artifacts and self-framing paragraph. This is an extremely rare exception, not an implementation-cost escape.

The IR-legality error is the single acceptable reason for dropping a planned variant without measuring it. Do not silently drop rows. Suspected domination, prediction, implementation cost, presumed user wishes, or another variant's measurement are not removal grounds. If the evidence does not establish the chosen exception, restore `in_progress` and continue.

**What `blocked` means (strict).** All three `blocked` criteria must hold:

1. Preserve actual variant code, including vendored files, under the backend directory. Launch it end-to-end, or as close as the failure allows, and preserve a specific reproducible failure. Deleted drafts and mental arguments do not count.
2. The failure is environmental and cannot be resolved by any code you could write under the backend directory. Exactly three categories qualify: (a) required hardware failure, such as unavailable GPU/NVLink or OOM persisting after the most aggressive supported sharding, evidenced by `nvidia-smi`, `dmesg`, or CUDA errors; (b) missing unavailable weights or an external service with no substitute, evidenced by filesystem/HTTP errors; (c) a third-party library bug in the preconfigured environment, evidenced by an isolated pytest-style reproduction checked into backend `tests/` that exercises the library alone and establishes correct API use.
3. Save and cite the concrete artifact under the backend directory, including its path in the queue row (for example `blocker_artifact: tests/fixtures/pp4_cuda_oom.log`). Prose descriptions of difficulty do not count.

Agent-written bugs, including wrappers, workers, harness helpers, vendored code, and distributed protocols, require fixing. An apparent shared-runtime failure requires an isolated reproduction without the wrapper; if that works, the wrapper remains the agent's problem. An unexplained wait is an unknown to investigate, not a blocker. No complexity, integration effort, session time, tokens, or patience can satisfy the environmental criterion.

**Adversarial self-framing (required).** Every `blocked`/`discarded` row, and every log block for an unmeasured removed variant, requires a dedicated experiment-log paragraph in this form:

> A skeptic reading this could argue it is implementation-cost skipping re-dressed. The concrete evidence it is not is <specific on-disk pointer>.

For `discarded`, point to both the IR correction made in the same edit and the failing transformation-legality test. For `blocked`, point to the category-specific hardware, resource, or isolated-library evidence above. General difficulty arguments, eloquent justifications, or related measurements do not qualify. Correct the status if a convincing category-specific artifact pointer cannot be supplied.

#### Investigating silent or slow workers

A worker silent for five minutes is in an unknown state. Autotune, JIT compilation, cold weights, NCCL initialization, and process contention for disk caches can take minutes without logs. Investigate rather than waiting indefinitely or declaring a hang:

1. Sample actual work and wait state: `nvidia-smi`; `ps -p <pid> -o stat,wchan,cmd`; `cat /proc/<pid>/wchan`; and Python stacks via `py-spy dump --pid <pid>` or `gdb -p <pid>` plus `py-bt`. A futex wait alone proves neither deadlock nor progress. Compare snapshots; live `torch._inductor.select_algorithm` frames can indicate autotune, while all workers parked without progress need further diagnosis.
2. Sample cache growth, e.g. `watch -n 5 'du -sh /tmp/torchinductor_*'`, using the active cache location. Growth demonstrates forward progress; static size with CPU activity could reflect another phase or a stall and needs diagnosis.
3. Test multi-process cache-lock contention against the same code path on one process and the same hardware. If the single-process case is fast, prewarm its cache or reduce initialization concurrency, e.g. `TORCHINDUCTOR_COMPILE_THREADS=1` or one rank at a time.
4. Document and wait for genuine slow progress. Record cold-launch wall time in the experiment log; a twenty-minute cold compilation is a one-time cost, not a blocker.
5. If snapshots demonstrate deadlock, localize it. Fix agent-owned code. A shared/third-party deadlock, even with the isolated reproduction above, does not itself permit `blocked`: attempt vendoring/patching, alternative shared primitives, process-launch/sharding/concurrency workarounds, and designs that avoid the broken path. Only after every such workaround has been attempted and itself produces the required isolated-library failure evidence can `blocked` qualify. Its self-framing paragraph must cite each workaround's specific failure log, not merely the initial hang.

Use saved utilization snapshots, cache deltas, stacks, and isolated workaround reproductions as evidence. Time spent watching an unchanged log is not evidence. Until diagnosis establishes the applicable criteria, retain `in_progress` and investigate.

#### Each iteration, in order

1. **Decide:** log the transformation, specific IR analysis establishing legality (state independence, streaming overlap, or pipeline stage), and predicted effect on every active target. Defaults are first-output latency and sustainable-output-rate/smoothness; honor app overrides. After the first iteration, state what the previous measurement taught and how it motivated this hypothesis.
2. **Implement:** obey the backend-only scope and separate variant subpackage/config rule above.
3. **Verify correctness:** run a harness launching reference and candidate backends as subprocesses, driving both via the adapter with simulated inputs, collecting and comparing outputs, and cleanly terminating both. Read its output and confirm PASS. Never promote a failing variant; record its failure. Correctness checking is mandatory.
4. **Benchmark:** run the candidate as a subprocess through the adapter, measure every active target, cleanly terminate it, read actual numbers, and persist results under backend `benchmarks/results/`. Use timestamps for first-output latency, a steady-rate sweep for sustainable-rate/smoothness, and measurements of app-specific targets. Benchmarking is mandatory.
5. **Iterate:** record results, decide whether to promote the variant, update/add/reorder queue rows, and form the next hypothesis. Record regressions as evidence. Before writing the next hypothesis, check that the current variant has both a persisted correctness verdict/log (PASS or FAIL) and a benchmark JSON. Missing either means the iteration is unfinished.

The first deployment-iteration log entry must benchmark the user's `flashrt/apps/<app>/reference/` backend through the same harness on the same hardware. That remains the baseline for every variant's delta; promoted agent variants never replace the anchor.

Stop only when all queue rows are terminal and the definition-of-done checklist is satisfied. A measured row may have verdict `needs-followup` when it created a new queue entry; an unmeasured row cannot use that verdict.
