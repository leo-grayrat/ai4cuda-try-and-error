# AGENTS.md

This file is the entry point for any coding agent operating on this
repo. It follows the [agents.md](https://agents.md) convention: the
nearest `AGENTS.md` to a file you are editing wins, so always check
for and read nested `AGENTS.md` files under `flashrt/apps/`,
`flashrt/ir/`, and `assets/` before you touch code in those trees.

All paths in this file are **relative to the repo root**, which is the
directory containing this `AGENTS.md`. Run every command from here.

## What this repo is

This repo is a harness for a specific experiment:

1. A **user** writes a simple, strictly-sequential, single-GPU reference
   *backend* implementation of some interactive real-time application
   pipeline, plus the *frontend* the application will ultimately be
   driven by, plus the *adapter* that defines the backend-facing IPC
   contract between the two. The reference backend lives under
   `flashrt/apps/<app>/reference/`. The frontend and adapter are user-written
   too, but they are shared artifacts (they work against *any*
   backend for the app) and live beside it in the app directory —
   `flashrt/apps/<app>/frontend/` and `flashrt/apps/<app>/adapter.py`. Each
   `<app>` is one such pipeline; the repo is designed so the user can
   add any number of apps side-by-side. The reference backend is
   intentionally slow, but clean and easy to read — it is the
   *specification* of the pipeline's behavior.
2. An **IR framework** under `flashrt/ir/` provides data structures
   and tools for representing computation graphs as explicit
   Intermediate Representations. The IR makes data dependencies and
   state dependencies visible as graph structure, so that parallelism
   and scheduling strategies can be derived mechanically rather than
   inferred from imperative code.
3. A **coding agent** (that is you) reads the user's reference
   implementation for a given `<app>`, converts it to an IR graph
   using the framework, validates the IR for correctness, then
   derives an efficient, possibly multi-GPU, streaming deployment
   under `flashrt/apps/<app>/backend/<platform>/`. The deployment must be
   **correct** relative to the reference and must be **better** on the
   optimization target(s) defined below.
4. The frontend and adapter are not modified. Your new backend must
   honor the same adapter contract the user's reference backend
   honors, so that once you finish optimizing, the user can launch
   *your* backend and drive it with the same unchanged frontend under
   `flashrt/apps/<app>/frontend/`. You, however, are not expected (or
   allowed) to use that frontend yourself as a test harness — see
   "Testing without a human in the loop" below.

Variants are **platform-partitioned**. Emit yours under
`flashrt/apps/<app>/backend/<platform>/` and their run scripts under
`examples/<app>/<platform>/`, where `<platform>` is `cuda` or `rocm`
matching the platform you target; register them in
`backend/<platform>/launch.py`. The shared `backend/launch.py` dispatcher
selects the current platform's set at runtime — do not modify it. The
shared runtime resolves the platform automatically
(`flashrt.platforms.current_platform`), so attention backends and kernels
are already selected for you; set `attention_backend: auto` in configs
unless you need a specific one. Resolve repo-root or app-directory paths
with `flashrt.paths.repo_root()` / `app_dir(<app>)`; never count `..` from
`__file__` (a variant's nesting depth is not fixed).

`flashrt/` also ships a shared runtime library
(`flashrt/channels/`, `flashrt/distributed/`, `flashrt/runner/`,
`flashrt/pipeline/`, `flashrt/models/`, `flashrt/layers/`,
`flashrt/memory/`, `flashrt/loader/`, `flashrt/utils/`,
`flashrt/platforms/`, `flashrt/rt_config.py`, etc.); the **IR
framework** under `flashrt/ir/` (see "The IR workflow" below); a
per-app **adapter** at `flashrt/apps/<app>/adapter.py` that defines
the backend-facing interface the frontend speaks (shared-memory
buffers, control opcodes, input / output shapes, session lifecycle);
the user's **frontend** for each app under
`flashrt/apps/<app>/frontend/`; and a shared **assets** directory
`assets/` at the repo root holding the static data (reference
images, clips, etc.) an app's backend / frontend / tests consume.
Both the reference and your own backend implementation are expected
to be compatible with these. Treat them as stable third-party code:
read freely, import freely (for the adapter and IR in particular), do
not edit.

## Your mission

For the `<app>` the user points you at, produce an implementation under
`flashrt/apps/<app>/backend/<platform>/` that **maximizes the system's interactivity**
along the optimization target(s) that apply to the app, while
preserving correctness relative to the reference implementation under
`flashrt/apps/<app>/reference/`.

### Default optimization targets

Unless the user specifies otherwise (see "Per-app overrides" below),
the agent co-optimizes two complementary targets:

1. **Latency to first output (per request).** For a given user input
   *I* that enters the backend at wall-clock time `t_in`, define
   `t_first_out` as the wall-clock time the *first* chunk of *I*'s
   corresponding output becomes available on the backend's output-side
   contract. The target is to minimize `t_first_out - t_in` — how
   quickly the user perceives any response. This dominates one-shot
   pipelines and the perceived initial latency of streaming
   pipelines. Report as a distribution (median and p95) over multiple
   inputs.

2. **Maximum sustainable output rate (with smoothness).** For a
   streaming output, the system has to keep emitting chunks
   continuously without the consumer ever stalling. The target is to
   maximize the steady-state output frequency the system can hold
   *without falling behind*. Concretely:

   a. **Sustainable rate.** Drive the backend with a steady offered
      input rate and measure the highest input rate at which the
      output rate matches it indefinitely (the consumer never has to
      wait, the backend's input queue does not grow without bound).
      Report this rate (chunks/s, frames/s, tokens/s — whatever the
      app's natural output unit is). For one-shot pipelines that
      produce exactly one output chunk per input, this collapses to
      "1 / latency-to-first-output" and stops being a separate
      number.
   b. **Smoothness, as a side condition.** A high median rate can
      still produce jittery / stuttering output if the *gaps between
      successive output chunks* are uneven, or if some chunks slip
      past the consumer's per-chunk deadline. A variant that improves
      the median rate while widening the inter-chunk-gap distribution
      has not improved the sustainable rate. Record the inter-chunk-
      gap distribution (median, p95, and the fraction of gaps
      exceeding the app's natural chunk-period budget) alongside the
      sustainable-rate number whenever the app produces more than one
      output chunk per input.

For pipelines where one input produces exactly one output chunk,
target 2 collapses into target 1 and you optimize a single number.
For streaming output pipelines the two diverge, and a transformation
that improves one may leave the other unchanged (or even regress
it). Treat both as co-equal targets: every variant in your
experiment log must report distributions for both (plus the
smoothness side condition where applicable), and you do not get to
optimize one in isolation.

### Per-app overrides

Different applications optimize for different things, and the user
may specify a different target when they assign you the app. Common
alternatives the agent should accept and follow:

- **System throughput** (aggregate requests served per unit time
  across many concurrent inputs). The relevant numbers are aggregate
  request rate at fixed latency budget, or latency at fixed offered
  load — not per-request first-output latency in isolation. The
  variant queue's predicted-impact column and per-variant Δ
  measurements should report against the throughput target instead.
- **Cost per request / cost per unit output** (when the user is
  optimizing $/token, $/frame, GPU-seconds, etc.). Variants should
  be measured against the cost-per-unit metric the user names, and
  perf wins that come from oversubscribing hardware in ways that
  improve $/unit but regress per-request latency are valid here.
- **A specific tail-latency percentile** (e.g., minimize p99 even at
  the expense of median).
- **Memory / KV-cache footprint at fixed latency** (when the agent's
  job is to fit a longer context window or larger batch into
  available VRAM rather than to make each step faster).
- **Anything else the user explicitly names.** Take the user's
  framing literally; do not silently fall back to the defaults.

When the user specifies an alternative target, treat the defaults
above as background constraints (the variant should not regress the
defaults arbitrarily — keep their distributions in the experiment
log as secondary evidence) and treat the user's named target as the
primary axis the variant queue is ranked against.

When the user does *not* specify, the two defaults above are what
applies. The agent does not invent a third target to chase.

You are free to reorganize compute, rearrange scheduling, redistribute
work across devices, and restructure data flow, *provided that* every
transformation you make preserves the data dependencies of the
reference implementation. The general rule: if a transformation changes
the set of values each tensor operation consumes (beyond floating-point
noise from non-associative reductions), it is a bug; if it does not, it
is fair game. Discovering which transformations are legal and which
ones actually help is explicitly part of the task — derive them from
the IR's data and state dependencies, do not copy a recipe.

### Two distinct optimization layers

Read the reference backend carefully and recognize that there are
(at least) two independent layers where the active optimization
target(s) can be improved:

1. **Application pipeline scheduling.** The reference backend may run
   its high-level stages strictly one-after-the-other. Analyze data
   dependencies between these stages to determine which can overlap
   or stream chunks to each other.
2. **Individual model inference.** Each model stage in the pipeline
   is itself a compute-heavy operation. For stages whose model
   implementation is visible to you in the codebase (i.e., you can
   read the forward pass, the data flow, the memory layout), analyze
   the data and resource dependencies *within* that model to figure
   out whether its inference can be distributed across multiple GPUs.
   Some stages may be served by opaque external engines whose
   internals you cannot inspect or modify — for those, treat them as
   black-box latency contributors and focus your optimization on how
   they are scheduled relative to other stages and on which device(s)
   they are placed, rather than on their internal execution.

Within layer 2, parallelism strategies divide along an axis worth
making explicit, because they attack the two default targets
differently (and, under per-app overrides, attack the user's named
target through different mechanisms):

- **Within one chunk's processing**: distribute the work that an
  exposed model does for *one* chunk across multiple devices, to
  shorten the critical path of that single chunk. This primarily
  attacks **target 1 (latency to first output)**. The IR's atomic
  operators do not decompose along this axis, so the IR analysis
  tools do not directly surface these opportunities — you must
  look inside individual model implementations to find them.
- **Across consecutive chunks**: when an exposed model produces
  multiple chunks via internal sub-operations whose state is
  disjoint, those sub-operations can be assigned to different
  devices and pipelined across chunks. This primarily attacks
  **target 2 (sustainable output rate and smoothness)**. The IR's
  `find_pipeline_stages` directly surfaces these.

These two are not mutually exclusive. The IR analysis tools surface
*some* model-parallelism opportunities directly; others exist below
the IR's granularity and require you to look beyond what the
analysis tools report. Do not anchor on what the IR surfaces — use
it as a starting point, not a ceiling.

**You must explore both layers.** An implementation that only
rearranges the pipeline schedule but leaves each model running
single-GPU, or one that only parallelizes a model stage but leaves
the pipeline fully sequential, is leaving wins on the active
target(s) on the table. Your experiment log should contain
deployment variants that address each layer independently as well
as variants that combine improvements from both.

### Device placement is yours to decide

The reference backend's device placement (which model runs on which
GPU) is part of its naive, unoptimized design — do not treat it as a
constraint. The reference may hard-code GPU indices for various
stages in its config, but your implementation should determine its
own placement strategy based on your own analysis of compute cost,
memory footprint, communication overhead, and opportunities for
overlap. Experiment with different placements as part of your
iteration loop; do not simply copy the reference's assignment.

## Environment

- **Python environment:** run *all* Python and CUDA code inside a
  single pre-configured environment that the user has prepared for
  this repo. Do not create new environments, do not `pip install`
  into the system Python, and do not assume the bare `python` on
  PATH will work without activation.

  Resolve which environment to use **once, at the very start of the
  session**, in this order:

  1. **Default: a `flashrt` conda environment, if it exists.** Probe
     for it before doing anything else — e.g., `conda env list | grep
     -E '(^|\s)flashrt(\s|$)'` or
     `conda run -n flashrt python -c 'import sys; print(sys.executable)'`.
     If the environment exists, activate it (`conda activate flashrt`)
     and use it for the rest of the session.
  2. **User-specified environment.** If the user told you which
     environment to use (a different conda env name, a venv path, an
     interpreter path, etc.) when assigning the task, use that one
     instead — the user's explicit choice always wins over the
     `flashrt` default.
  3. **Otherwise, ask the user once, before running anything.** If
     no `flashrt` conda env exists and the user has not specified an
     environment, ask which environment to use **at the very start
     of the session** so the answer is in hand for the full
     session's code. Do not guess, do not fall back to the system
     Python, and do not pick whatever `python` happens to resolve on
     PATH; the question must be resolved before any
     code-execution step.

  Once resolved, stick with that environment for the entire session.
  Activate it (`conda activate <name>` or
  `source <venv-path>/bin/activate`), or invoke its interpreter
  directly (`<env-path>/bin/python -u <script>`). When you spawn
  subprocesses from Python, pass `sys.executable` (the interpreter
  Python itself is running under) so the child inherits the same
  env, rather than relying on `python` resolving correctly from
  PATH.
- **Working directory:** the repo root (the directory containing this
  file). The package is installed into the environment
  (`pip install -e .`), so `flashrt` imports directly; no `PYTHONPATH`
  setup is needed.
- **Hardware:** the user will tell you in-session which GPUs you have
  access to and how (for example via `CUDA_VISIBLE_DEVICES`). If you
  have not been told, **ask** before running anything multi-GPU; do
  not guess.
- **Models and weights:** paths to model checkpoints are hard-coded in
  the user's YAML config for each app. Reuse those paths; do not
  re-download weights.

## What you may and may not modify

### Read-only (treat as a hard contract)

- **`flashrt/apps/<app>/reference/**`** — the user's reference
  *backend* implementation for every app, plus the app's runtime
  config (`flashrt/apps/<app>/config.yaml`) and backend launch
  scripts. This tree is the correctness oracle. Read it carefully; do
  not edit anything in it. If you think a file under a `reference/`
  tree needs to change, that is a signal the change belongs in
  `flashrt/apps/<app>/backend/` instead.
- **`assets/**`** — shared data assets (reference images, audio
  clips, sample inputs, etc.) in one flat directory at the repo root.
  These are not backend-specific: the reference backend, any agent backend,
  the frontend, and test / benchmark harnesses may all consume them
  through the app's runtime config or directly. Read freely, do not
  edit. If you need a new asset, add it under
  `flashrt/apps/<app>/backend/tests/fixtures/` instead.
- **`flashrt/apps/<app>/adapter.py`** — the shared adapter. For each
  app, this defines the frontend ↔ backend interface (shared-memory
  buffer names, control protocol, input / output shapes, session
  lifecycle). The frontend talks to *any* backend of an app through
  this adapter, and so must your backend. You are **explicitly allowed
  and encouraged to import** the adapter class when
  you write your own test / benchmark harnesses, since it is the
  canonical way to drive a running backend from a Python process. Do
  not edit the adapter itself.
- **`flashrt/apps/<app>/frontend/**`** — the shared frontend for each
  app: the user-written frontend code plus any frontend-side launch
  helpers and credential / token utilities. The
  frontend works against *any* backend of its app that honors the
  adapter contract. You must *not* run the frontend as part of your
  iteration loop, and you must not edit it, but your backend must
  remain compatible with it so the user can launch your backend and
  drive it with this unchanged frontend at the end of the session.
- **`flashrt/ir/**`** — the IR framework: graph data structures plus
  the subclassable `IROperator` base and `StateStore` protocol
  (`graph.py`), the `@ir_operator` decorator for function-style ops
  (`ops_base.py`), the sequential executor for validation
  (`executor.py`), and analysis tools for discovering parallelism and
  scheduling opportunities (`analysis.py`). Import and use freely, do
  not edit. See "The IR workflow" above and "Resources you can use"
  below for how to use this module.
- **`flashrt/{channels,configs,distributed,frontend,layers,loader,logging_utils,memory,models,pipeline,platforms,runner,third_party,utils,rt_config.py,envs.py,flashrt_args.py,hf_transformer_utils.py,logger.py}`** —
  the shared runtime library. Read freely, import freely, do not edit
  in place. If you believe the shared library is actually buggy or
  genuinely needs a new capability, the path forward is to work around
  it inside `flashrt/apps/<app>/backend/` — by vendoring the offending file
  per the "Read-only areas are not a blocker — vendor the file into
  your workspace" rule (Phase 4) and modifying the copy, or by
  re-implementing against shared primitives per the "Multi-GPU
  coordination" rule. The agent runs autonomously, so neither path
  requires escalating to the user; both are bounded by the on-disk
  documentation requirements those rules spell out.
- **`flashrt_kernels/`** — custom kernels. Read and call, do not edit.
- Top-level project files (`README.md`, `pyproject.toml`,
  `requirements/`, etc.) — do not edit.

### Writable (your workspace)

Everything you produce goes under **`flashrt/apps/<app>/backend/`** for the
app you have been asked to optimize. That directory does **not** exist
up-front: when you begin a new app, create it by copying the generic
skeleton at `flashrt/apps/_template/backend/`:

```bash
cp -r flashrt/apps/_template/backend flashrt/apps/<app>/backend
```

and then fill it in. See `flashrt/apps/AGENTS.md` for the layout
conventions, the harness patterns, the experiment-log format, and the
process hygiene rules that apply inside this directory. Do not create
files outside of `flashrt/apps/<app>/backend/` without explicit user approval,
and do not modify `flashrt/apps/_template/backend/` itself — treat it as a
read-only seed.

## Frontend compatibility contract

The frontend for an app (`flashrt/apps/<app>/frontend/`) talks to the
backend through the shared adapter (`flashrt/apps/<app>/adapter.py`). Your
backend must honor that exact contract so that, once you are finished,
the user can drop your backend in place of the reference backend and
drive it with the unchanged frontend. Read the adapter source and the
reference backend side-by-side to see exactly what the contract is —
what shared-memory buffers it creates, what control opcodes it
consumes and produces, what shapes / dtypes / cadences inputs and
outputs use, how a session is started, reset, and terminated.

Practically this means: whatever IPC primitives the adapter opens,
whatever control protocol it uses, whatever input cadence and output
cadence it expects — your backend must match. Do not change the
adapter; match it.

You do **not** use the user's frontend as your own test harness — see
the next section.

## The IR workflow

Your work proceeds in four phases. Phases 1 and 2 run once per app
and produce a validated IR that makes the reference's computation
structure explicit. Phases 3 and 4 form the iterative optimization
loop that uses the IR to generate and refine deployment variants.

### Phase 1: IR conversion

Read the user's reference backend under `flashrt/apps/<app>/reference/` and
convert the full exposed computation graph to IR using the framework
at `flashrt/ir/`. "Exposed" means all parts of the computation whose
implementation is visible to you — the worker's stage scheduling,
each model whose forward pass you can read, and their interactions.
Opaque engine-served stages (e.g., vLLM, vLLM-Omni) are represented
as black boxes you can schedule but not restructure.

You will produce these graphs (at minimum):

1. A **worker-level graph** representing the high-level pipeline
   stages (e.g., ASR → LLM → TTS → video generation). Each stage is
   an `IROperator` with:
   - `op_type`: `BLACK_BOX` for opaque engine-served models (LLM,
     TTS, ASR — you cannot see their internals) or `EXPOSED` for
     models whose forward pass is visible in the codebase, or
     `COMPOSITE` for stages that contain a sub-graph.
   - `stream_mode`: `STREAMING` if the stage produces output
     incrementally (enabling overlap with downstream stages), or
     `BATCH` if all output is materialized before the downstream
     can start.
   - `state_reads` / `state_writes`: names of `StateObject`s the
     operator accesses. Use scope `"chunk_persistent"` for state
     that accumulates across chunks (creates cross-chunk
     dependencies), `"session_init"` for write-once state,
     `"ephemeral"` for within-chunk temporaries.
   - Rate-conversion metadata on streaming edges where the source
     and destination have mismatched chunk sizes.

2. A **model-level graph** for each `EXPOSED` or `COMPOSITE` stage
   whose internals you can see. This graph represents the fine-grained
   operations *within* that stage (e.g., individual denoising steps,
   VAE decode, feature extraction). Each operation gets its own
   `IROperator` with explicit state dependencies.

   The key purpose of the model-level graph is to make **state
   independence** visible: if two operators write to *different* state
   objects, the analysis tools will identify them as candidates for
   pipeline parallelism across chunks (operator A for chunk N can
   overlap with operator B for chunk N+1).

Each operator's **implementation lives on the operator itself** — you
either subclass `IROperator` and override `execute(inputs, ctx, state)`
(the right pattern for parameterized ops like "denoising step k", where
one class is instantiated many times), or use the `@ir_operator(...)`
decorator from `flashrt.ir` over a function for one-off leaf ops. The
`SequentialExecutor` calls `op.execute(...)` directly; there is no
separate registry of callables to keep in sync with op names.

Persistent tensor state (KV caches, prefilled embeddings, cached
latents) is accessed through a `StateStore` handed to `execute`:
`state.get(name)` and `state.set(name, value)`. Both reject any name
not present in the operator's declared `state_reads` /
`state_writes`, which protects you from a common IR-conversion bug
— an operator whose declared dependencies don't match what its body
actually touches would silently poison the pipeline-stage analysis
without the enforcement. In-place mutation (e.g., a DiT runner
writing into a KV cache object) is expressed by calling only
`state.get(name)` on a name that's declared in `state_writes` and
mutating the returned object; `state.set` is for genuine rebinding.

Write your graph builders and operator definitions under
`flashrt/apps/<app>/backend/ir/`. See `flashrt/apps/AGENTS.md` for the
workspace layout.

### Phase 2: IR validation

Before you can trust the IR as a basis for optimization decisions,
you must verify that it faithfully represents the reference's
behavior. **This is non-negotiable.** Static analysis of the IR
(reading your graph builder, eyeballing operator dependencies,
visually checking that the graph "looks right") is not validation.
You must actually instantiate the `SequentialExecutor`, run it
against real input, and compare its output to the reference. An IR
that you have not executed and compared is unvalidated, and any
optimization decisions based on an unvalidated IR are speculative.
Treat unvalidated IR conversions the same as failed IR conversions:
the workflow stops at Phase 2.

The procedure:

1. Build the IR graph (operators carry their own `execute`) and pass
   it to `SequentialExecutor(graph)` (from `flashrt.ir`). If any
   persistent state needs to exist before the first operator runs —
   e.g., empty KV caches allocated at session start — pre-populate
   it with `executor.init_state({...})`; preamble operators can also
   populate state by calling `state.set(...)` during
   `run_preamble`.
2. Run the executor with test inputs and collect outputs.
3. Run the reference backend with the same inputs and collect outputs.
4. Compare outputs element-wise / chunk-wise with explicit tolerances.
   Document each tolerance in-comment with its justification.
5. If the comparison fails, fix the IR conversion and re-validate.
   Do not proceed to Phase 3 until validation passes.

Write the validation harness under `flashrt/apps/<app>/backend/tests/`.
The harness follows the same subprocess-driven adapter pattern
described in "Testing without a human in the loop" below: launch the
reference backend as a subprocess, drive it through the adapter with
simulated input from `tests/fixtures/`, cleanly terminate it, then
compare against the executor's output. The harness must print a
clear PASS / FAIL verdict to stdout, and you must read that verdict
before recording the validation result.

Once validation passes, run the analysis tools on your validated IR:

```python
from flashrt.ir import summarize_graph
print(summarize_graph(model_graph, op_latencies={...}))
```

Record the analysis output (pipeline stages, streaming overlaps,
critical path, independent operator pairs) in your experiment log.
This analysis is the basis for your Phase 3 decisions.

### Phase 3: Deployment

Use the IR analysis results to design an optimized deployment.
The analysis tells you:

- **Pipeline stages**: which operators can go on separate GPUs
  (from `find_pipeline_stages`).
- **Streaming overlaps**: which stages can overlap execution
  (from `find_streaming_overlaps`, including rate-conversion ratios).
- **Critical path**: the bottleneck chain through one chunk
  (from `compute_critical_path`).
- **Cross-chunk independent pairs**: which operators can run
  concurrently for different chunks (from
  `analyze_cross_chunk_dependencies`).

Translate these opportunities into a concrete deployment variant
under `flashrt/apps/<app>/backend/<variant_name>/`. A deployment variant
is a complete backend implementation (worker, model pipeline(s),
and any supporting code) that:

- Assigns operators to devices based on the pipeline stage analysis.
- Implements inter-device communication (e.g., `dist.send`/`recv`
  for latent transfer between pipeline stages).
- Implements streaming overlap where the IR analysis shows it is
  legal.
- Honors the same adapter contract as the reference backend.

### Phase 4: Validation and iteration

#### Exploration and queue

Complete at least 10 distinct deployment variants, exercising different optimization ideas. A bug fix to the same idea does not count as a distinct variant. Ten is a floor: continue until every queued hypothesis has a measured result or satisfies one of the narrow terminal exceptions below, and the definition-of-done checklist is satisfied. Cover the axes surfaced by IR analysis, including pipelining/streaming, other model parallelism, and co-location/device placement. Include both isolated levers and combinations. Prefer measuring isolated levers against the user's reference before combining them; name a combined variant for the levers it stacks, not `combined_best`. Do not start by combining multiple untested optimizations.

Before implementing any variant, save an explicit ordered initial queue in the experiment log. Each row contains its name, one-sentence hypothesis, specific motivating IR analysis result, predicted impact on active targets with a rough magnitude, and status: `planned`, `in_progress`, `measured`, `blocked`, or `discarded`. Include every opportunity surfaced by IR analysis, below-IR-granularity levers identified in Phase 2, and combinations. Rank by IR-analysis-based expected impact, irrespective of implementation cost.

Update the queue between every measured variant: record its result, add hypotheses suggested by wins, regressions, or invalidated earlier assumptions, and reconsider remaining ordering. Apply the ten-variant floor to the final queue. An early large win calls for more exploration; it does not justify skipping ambitious or smaller-gain variants.

Every follow-up mentioned in a log entry must become a queue row in the same edit that finalizes the entry, or receive the written removal reason and evidence specified below. A prediction of another variant's numerical results likewise requires that variant to be queued as `planned` before finalizing the current entry. Free-text follow-ups and predictions never replace implementation and measurement.

When two measured variants receive `promote` on different active optimization axes and neither strictly dominates the other across the ranked axes, enqueue `combined(A, B)` as `planned` before either isolated variant is closed out. This includes primary/secondary axes under an app override. Measure that combination or satisfy the single legal-removal exception below.

Every queued variant needs its own independent implementation, correctness check, and benchmark. A related, scaled, simplified, or combined variant cannot stand in for it: their compute, memory, communication, contention, synchronization, and implementation profiles differ. Domination is a claim between measured variants. Expected regression, small impact, communication cost, lack of critical-path relevance, or an apparently superior measured sibling cannot justify skipping an unmeasured variant. Record such predictions as hypotheses and test them.

#### Work scope and persistence

Write code only under `flashrt/apps/<app>/backend/`. Keep each benchmarked variant in its own subpackage/config rather than mutating a previously good variant in place. When behavior in a read-only file must change, vendor the necessary file or subset into the variant directory, modify and import the copy, and add a one-line source-path/divergence comment. Validate numerical parity of unchanged paths against the original, which remains the oracle. This applies to shared adapters, models, runtime, `flashrt_kernels/`, and other read-only code.

Audit reusable runtime support before dismissing a direction: grep `flashrt/*` for the relevant abstraction and read at least the most promising file. If manual implementation is needed, proceed. The experiment log may not use implementation complexity, missing runtime coverage, “audited ... and it does not cover this case,” or “no relevant support found ...” to defer a direction.

There is no session deadline or self-imposed wall-clock/per-variant limit. Implementation cost, debugging difficulty, integration, new infrastructure, distributed/async composition, missing read-only edit permission, existing wins, anticipated user permission to stop, or reaching ten variants never justify skipping, downgrading, proxying, substituting, deferring, or rotating away from a queued variant. Work on the current variant until it reaches a terminal status. If no progress is possible after a serious attempt, apply the strict environmental-blocker criteria below.

The log, README, and final status describe this session's actual state. They must not contain future-session handoff language or technical paraphrases, including “next/future/follow-on session,” “pick this up later,” “come back to it,” “out of scope for this session,” “left for follow-up,” “a future agent can,” or “deferred to a later session.” Every queued variant remains this session's responsibility until terminal.

#### Terminal statuses and removal evidence

`planned` and `in_progress` are non-terminal. The terminal statuses are:

- `measured`: launched end-to-end, with an executed correctness verdict (PASS or FAIL), its persisted harness log, and benchmark JSON from a real end-to-end run.
- `discarded`: the IR analysis incorrectly declared a transformation legal; supply the dependency it missed, a minimal on-disk reproduction showing that even a correctly implemented transformation corrupts output, and the erroneous IR file/operator declaration. Correct the IR and re-run validation in the same edit. Include a failing transformation-legality test under `flashrt/apps/<app>/backend/tests/` and the self-framing paragraph below.
- `blocked`: all three criteria below hold, with the required artifacts and self-framing paragraph. This is an extremely rare exception, not an implementation-cost escape.

The IR-legality error is the single acceptable reason for dropping a planned variant without measuring it. Do not silently drop rows. Suspected domination, prediction, implementation cost, presumed user wishes, or another variant's measurement are not removal grounds. If the evidence does not establish the chosen exception, restore `in_progress` and continue.

All three `blocked` criteria must hold:

1. Preserve actual variant code, including vendored files, under the backend directory. Launch it end-to-end, or as close as the failure allows, and preserve a specific reproducible failure. Deleted drafts and mental arguments do not count.
2. The failure is environmental and cannot be resolved by any code you could write under the backend directory. Exactly three categories qualify: (a) required hardware failure, such as unavailable GPU/NVLink or OOM persisting after the most aggressive supported sharding, evidenced by `nvidia-smi`, `dmesg`, or CUDA errors; (b) missing unavailable weights or an external service with no substitute, evidenced by filesystem/HTTP errors; (c) a third-party library bug in the preconfigured environment, evidenced by an isolated pytest-style reproduction checked into backend `tests/` that exercises the library alone and establishes correct API use.
3. Save and cite the concrete artifact under the backend directory, including its path in the queue row (for example `blocker_artifact: tests/fixtures/pp4_cuda_oom.log`). Prose descriptions of difficulty do not count.

Agent-written bugs, including wrappers, workers, harness helpers, vendored code, and distributed protocols, require fixing. An apparent shared-runtime failure requires an isolated reproduction without the wrapper; if that works, the wrapper remains the agent's problem. An unexplained wait is an unknown to investigate, not a blocker. No complexity, integration effort, session time, tokens, or patience can satisfy the environmental criterion.

Every `blocked`/`discarded` row, and every log block for an unmeasured removed variant, requires a dedicated experiment-log paragraph in this form:

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

### Verification and benchmarking are non-negotiable

This deserves its own section because it is the most common failure
mode. **At no point may you substitute static analysis (reading code,
inspecting graphs, reasoning about correctness) for actually running
a harness and comparing outputs.** This rule applies to every
verification step in the workflow:

- **Phase 2 IR validation.** You must instantiate the
  `SequentialExecutor`, run it on real input, and compare its output
  to the reference backend's output. Reading your graph builder and
  convincing yourself it "looks right" is not validation. An IR you
  have not executed and compared is unvalidated.
- **Phase 4 deployment correctness.** You must launch your candidate
  backend as a subprocess, drive it through the adapter, and compare
  its output to the reference. Reading your variant's pipeline /
  worker code and convincing yourself it preserves the IR's
  dependencies is not correctness. A variant you have not run
  end-to-end against the reference is unverified.
- **Phase 4 deployment benchmarking.** You must launch your
  candidate, drive it through the adapter, and read the harness's
  reported numbers for every active optimization target — measured
  timestamps for latency-to-first-output, a real steady-rate sweep
  for sustainable-output-rate (with the inter-chunk-gap distribution
  for the smoothness side condition where the app produces more
  than one output chunk per input), and any per-app override target.
  Estimated, extrapolated, or "expected" numbers are not benchmark
  results.

**The test and benchmark harnesses must actually run end-to-end.**
This means:

- The backend subprocess must start, initialize its models, create
  its shared-memory buffers, and enter its event loop.
- The harness must drive the backend through the adapter with real
  simulated input and read back real output.
- The backend subprocess must cleanly terminate (not just be
  abandoned in the background). See "Process cleanup" below.
- The harness must print a clear PASS/FAIL verdict (for correctness)
  or the actual measured numbers for every active target (for
  benchmarking) to stdout.
- **You must read this output** before recording anything in the
  experiment log. If the output says FAIL, or if no output appeared,
  or if the backend never started, the result is invalid.

**If you observe any of these symptoms, the run was not valid:**

- The harness exited before the backend finished processing.
- Backend processes are still running after the harness exited (check
  with `ps aux | grep python` or equivalent after every run).
- The harness printed no comparison results or no benchmark numbers.
- The backend printed errors or exceptions to stderr.

Do not record results from an invalid run. Fix the harness, re-run,
and only then record.

**What "PASS" means.** A validation result may be labeled **PASS** in
the experiment log only when it comes from comparing numerical
outputs of the reference backend against the numerical outputs of
the tested artifact (the IR executor in Phase 2, the candidate
backend in Phase 4) over a real simulated-input fixture, element-wise
/ chunk-wise, under a documented tolerance. Topology checks, mock or
synthetic-tensor round-trips, structural assertions, and other
static sanity checks are useful for debugging but **may not** be
recorded as PASS in the experiment log. If the only thing you ran
was a structural check, the log entry is at best a diagnostic — the
variant is still unvalidated until the numerical comparison runs.

### Process cleanup

**Every** test or benchmark script you write must guarantee that all
backend subprocesses it spawned are terminated before the script
exits, regardless of success or failure. This is a hard requirement.
Leaked processes corrupt subsequent runs, waste GPU memory, and —
most importantly — mean the harness did not actually complete, which
means the results you would record are meaningless.

Concretely:

- Launch backend subprocesses with `subprocess.Popen` or equivalent,
  and always wrap the interaction in a `try / finally` that calls
  `proc.terminate()` and `proc.wait()` (or `proc.kill()` after a
  timeout) in the `finally` block.
- After the `finally` block, verify with `os.waitpid` or a `ps`
  check that the process tree is actually gone. If the backend uses
  `torch.distributed` or spawns its own child workers, you may need
  to kill the entire process group (`os.killpg`).
- Clean up stale shared-memory handles (`/dev/shm/<prefix>*`) before
  *and* after each run.
- If a run crashes or times out, kill the backend, clean up IPC
  handles, and log the failure in the experiment log as an invalid
  run. Do not record timing data from a crashed run.
- **After every harness invocation, run a quick `ps` check** (or
  `pgrep -f` with your backend's script name) and confirm no orphan
  processes remain. If any do, kill them before proceeding.

### Testing without a human in the loop

Because there is no human interacting via the user's frontend in your
iteration loop, and because you are not allowed to use the frontend
itself as your harness, your tests drive the backend **directly
through the app's shared adapter (`flashrt/apps/<app>/adapter.py`)**.
The pattern:

1. Launch the backend as a subprocess (with explicit
   `CUDA_VISIBLE_DEVICES`).
2. Wait for the backend to be ready (e.g., poll the adapter's
   `start()` until it succeeds, or check that the shared-memory
   handles exist).
3. Drive the backend through the adapter with simulated input
   (pre-recorded fixtures, synthetic data, or a small scripted
   scenario) via whatever input-side methods the adapter exposes.
4. Read back the output via the adapter's output-side methods.
5. **Cleanly terminate the backend** — send the adapter's terminate
   signal, then `proc.terminate()` / `proc.wait()` in a `finally`
   block. Verify the process is gone.
6. Clean up shared-memory handles.

You are expected to write these harnesses yourself as part of the
loop — they are not provided. Keep them small, deterministic, and
checked into `flashrt/apps/<app>/backend/tests/` and `benchmarks/`.

Make sure correctness runs against the reference backend use the same
simulated input and the same adapter API as the candidate runs, so
that any difference in outputs is attributable to the candidate
implementation and not the harness. Pick explicit tolerances for any
floating-point comparison and document why those tolerances are
acceptable.

### Multi-GPU coordination: use the shared `torch.distributed` abstractions

**Default rule, in one line: use the shared-runtime abstractions
under `flashrt/distributed/`. Hand-rolling distributed
infrastructure is reserved for extreme circumstances and must be
justified on disk.**

The shared runtime under `flashrt/distributed/` provides the
process-group bootstrap, the model-parallel group construction
(world / SP / TP / DP), the group accessors and rank/world-size
helpers, the collective-communication ops, and the NCCL / CUDA
device-communicator plumbing. Read the module before designing any
multi-GPU variant; these are the primitives you reach for first,
every time.

When your deployment spans multiple GPUs for pipeline components
you are **directly orchestrating** — that is, exposed components
whose implementation you can read and whose execution you are
driving — these are what you call. This applies whether you are
distributing one exposed model's work across devices, or placing
different exposed components on different devices with inter-device
tensor transfers between them.

**Do not hand-roll a multi-GPU design that substitutes Python-level
coordination, custom rendezvous, custom process-group plumbing, or
direct CUDA copies for what the shared runtime already provides.**
Hand-rolled distributed infrastructure — your own
`init_process_group` wrapper, your own group-construction helpers,
your own broadcast / all-to-all / all-reduce loops, your own
peer-to-peer copy schemes via `.to(other_device)` substituting for
a collective, your own multi-CUDA-context managers in a single
process — is a reliable source of correctness and performance bugs.
Many parts of the shared runtime read implicit and explicit
globals (world size, SP/TP/DP group sizes and ranks, the active
process groups, the model-parallel-initialised flag, etc.) from
`flashrt.distributed`, and they condition behaviour on those
globals — a variant that bypasses the shared substrate does not
interoperate with those components even when its own internals look
correct in isolation, and the failure modes are subtle (silent
miscompare on one rank, deadlocks on a specific shape, caches and
buffers sized for the wrong world).

**When the shared abstraction genuinely does not cover a case you
need, you re-implement.** The agent runs autonomously; "the shared
library does not support this" is not by itself a reason to label
a variant `blocked` (the strict criteria above remain in force) and
is not a reason to escalate to the user. Re-implementations are
legitimate but subject to four explicit constraints, and skipping
any of them turns the re-implementation back into a hand-rolled
shortcut:

1. **Use sparingly.** First exhaust the existing API — try every
   shared primitive that could plausibly cover the case, including
   compositions and parameter changes (different parallel-group
   sizes, different rank counts, different role assignments,
   padding inputs to a shape the existing call accepts, etc.). The
   variant's experiment-log entry must record, in prose, which
   shared primitives were tried and why each one did not cover the
   case before any re-implementation appears in the workspace.
2. **Build on the shared substrate, do not replace it.** A
   legitimate re-implementation extends or composes the shared
   primitives — calls a lower-level shared helper directly,
   registers the result into the shared `parallel_state` so the
   rest of the runtime sees a normal world, etc. It does **not**
   stand up a parallel set of group accessors, rendezvous, or
   collective ops that the shared components cannot see.
3. **Preserve the implicit and explicit globals the rest of the
   codebase relies on.** Whatever distributed substrate the
   re-implementation produces, the shared globals (world size,
   SP / TP / DP group sizes and ranks, the active process groups,
   `model_parallel_is_initialized`, etc.) must be populated
   consistently with what shared consumers (model classes,
   attention backends, KV cache allocators, planning utilities,
   runners, etc.) expect to read. Audit the components your variant
   pulls in for which globals they read, and document the audit in
   the variant's experiment-log entry; an unpopulated or
   inconsistently-populated global is a silent-corruption bug, not
   a forward-compatible omission.
4. **Vendor the new code in the agent workspace.** Re-implementations
   live under `flashrt/apps/<app>/backend/`, never as edits to
   `flashrt/distributed/` (or any other read-only tree). If a
   future variant in the same app needs the same construct, it can
   import it from the agent workspace; cross-app reuse is out of
   scope.

A re-implementation that satisfies all four is acceptable. A
re-implementation that skips any of them is the hand-rolled
shortcut the rule above forbids, and the variant stays
`in_progress` until the missing constraint is met.

A variant that attempts multi-GPU coordination for an exposed
component via any mechanism other than the shared-runtime stack
described above — without the four constraints met — (e.g.,
in-process Python threading with manual `torch.cuda.device`
switches, per-device CUDA streams plus hand-rolled peer-to-peer
copies, plain `.to(other_device)` calls substituting for a
collective, custom rendezvous over TCP / files, ad-hoc
rank-bookkeeping in shared memory) is not a finished variant. The
shortcut implementation and any measurement you took with it are
**exploratory scaffolding**, not a completed queue entry: the queue
row for that variant stays `in_progress`, and the compliant
shared-runtime-based implementation is what must actually be built
and measured. The shortcut measurement is fine to log as
preliminary evidence — it informs ordering, memory estimates, and
hypotheses — but it does not satisfy the variant, cannot earn a
`promote` verdict, and is not a separate variant of its own. "I got
a nice number with a hand-rolled shortcut, so the
`torch.distributed` version is just an implementation detail" is
the proxy-substitution failure mode applied to the implementation
mechanism, and falls under the "No variant may stand in for
another" rule in Phase 4.

**Bugs in agent-written distributed code are not blockers.** When
the agent does write coordination code on top of the shared
abstractions — including any of the legitimate re-implementations
above — and that code hangs, errors, or produces wrong output, the
failure is implementation work the agent owns; re-read "Bugs in
code the agent wrote" in the strict-`blocked` criteria above. The
temptation in this exact situation is to attribute a hang in
agent-written multi-rank orchestration to a "library deadlock" and
label the variant blocked; that is the specific failure mode the
strict criteria are designed to catch. The discriminating evidence
is the criterion-(2c) isolated test: strip the wrapper, exercise
the same shared-runtime call alone, and see whether the failure
reproduces without the agent's code in the loop. If it does not,
the agent's code is at fault and the variant stays `in_progress`
until the agent fixes it.

This guidance is specifically about **exposed** pipeline components
that you are directly deploying. It does **not** apply to opaque
third-party engines that do their own distributed setup internally:
choosing that such an engine should run on its own GPU (or on its
own set of GPUs, if the engine natively supports multi-GPU) is a
co-location / device-placement decision, and the engine brings its
own distributed infrastructure. When you mix the two — your own
`torch.distributed` world plus one or more third-party engines that
do their own `torch.distributed` setup — see the env-var pitfall
immediately below.

### Known pitfall: `torch.distributed` env-var collisions

If your deployment code uses `torch.distributed` to stand up its own
process group, and it *also* launches or embeds a third-party engine
(many model-serving engines do their own `torch.distributed` setup
internally), the two `torch.distributed` initializers will fight over
a shared set of environment variables that the child process inherits
from your parent process by default. This manifests as mysterious
rank / world-size / NCCL init errors inside the third-party engine
that do not reproduce when the engine is launched standalone.

The fix: in your own deployment code, **before you launch a
third-party engine**, scrub the inherited `torch.distributed`
environment from the child's environment so the engine's own
initializer starts from a clean slate instead of accidentally
inheriting your world's settings. This is only necessary when *your*
process has already initialized a `torch.distributed` world and you
are now about to spawn / embed a child that will initialize its own;
if your deployment has no `torch.distributed` setup of its own, or
if you are not launching any third-party engines that do their own
distributed setup, you do not need to do this.

Concretely, before calling `subprocess.Popen` (or the engine's
constructor, if it initializes distributed internally on import /
instantiation), clear these variables from the environment that will
be passed to the child:

```
RANK
LOCAL_RANK
WORLD_SIZE
LOCAL_WORLD_SIZE
GROUP_RANK
ROLE_RANK
ROLE_NAME
OMP_NUM_THREADS
MASTER_ADDR
MASTER_PORT
TORCHELASTIC_USE_AGENT_STORE
TORCHELASTIC_MAX_RESTARTS
TORCHELASTIC_RUN_ID
TORCH_NCCL_ASYNC_ERROR_HANDLING
TORCHELASTIC_ERROR_FILE
```

For a separate subprocess, scrub the copy passed via `env=`:

```python
import os, subprocess

_DIST_VARS = (
    "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
    "GROUP_RANK", "ROLE_RANK", "ROLE_NAME", "OMP_NUM_THREADS",
    "MASTER_ADDR", "MASTER_PORT",
    "TORCHELASTIC_USE_AGENT_STORE", "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_RUN_ID", "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "TORCHELASTIC_ERROR_FILE",
)

child_env = os.environ.copy()
for _var in _DIST_VARS:
    child_env.pop(_var, None)
subprocess.Popen([...], env=child_env)
```

For an engine that initializes `torch.distributed` in-process (e.g.,
on import or on constructor call), scrub `os.environ` itself
immediately before the import / instantiation, and optionally restore
the values afterwards if your own setup still needs them.

## Resources you can use

The shared runtime library under `flashrt/` contains reusable
components you should read before deciding how to implement anything.
The list below is not prescriptive — it does not tell you *when* to
use these, just tells you that they *exist* — you are expected to
decide what fits your plan based on the reference's dependencies.

- **`flashrt/ir/`** — the IR framework. This is the primary tool you
  use in Phases 1–2.
  - **`flashrt/ir/graph.py`** — core data structures: `IRGraph`,
    `IROperator` (subclass and override `execute(inputs, ctx, state)`
    plus optionally `should_run(ctx)`), `IREdge`, `StateObject`,
    `StateStore` (the read/write handle `execute` receives),
    `TensorPort`, `OpType`, `StreamMode`. Build your IR graphs using
    these.
  - **`flashrt/ir/ops_base.py`** — `@ir_operator(...)` decorator that
    wraps a `(inputs, ctx, state) -> dict` function into an
    `IROperator` instance with the metadata baked in. Use it for
    one-off leaf ops; prefer subclassing for parameterized ops where
    one class is instantiated many times.
  - **`flashrt/ir/executor.py`** — `SequentialExecutor(graph)` that
    runs an `IRGraph` in topological order on a single device. Use
    this in Phase 2 to validate your IR conversion against the
    reference. It takes only the graph (operators carry their own
    `execute`), offers `init_state({...})` to pre-populate persistent
    state (e.g., empty KV caches allocated at session start), and
    provides `run_preamble()` and `run_chunk()` methods. The
    executor is context-agnostic — you define the context object
    your operators need, and it hands each op a `StateStore` scoped
    to the op's declared `state_reads` / `state_writes`.
  - **`flashrt/ir/analysis.py`** — analysis tools that operate on
    any `IRGraph`:
    - `analyze_cross_chunk_dependencies(graph)` — finds which
      operators share `chunk_persistent` state and therefore have
      cross-chunk dependencies. Returns independent pairs that can
      overlap across chunks.
    - `find_pipeline_stages(graph)` — partitions operators into
      pipeline stages by grouping shared-state operators. Each
      stage can run on a separate GPU.
    - `find_streaming_overlaps(graph)` — finds streaming edges
      with rate-conversion metadata.
    - `compute_critical_path(graph, op_latencies)` — finds the
      bottleneck path through one chunk.
    - `summarize_graph(graph, op_latencies=None)` — comprehensive
      human-readable summary of all the above. **Run this first**
      on any validated graph to understand the optimization space.
  - See `flashrt/ir/AGENTS.md` for detailed documentation.
- **`flashrt/pipeline/base.py`** — an abstract pipeline base class the
  reference pipelines inherit from. Your own pipelines can inherit
  from the same base.
- **`flashrt/runner/`** — wrappers around the major model stages
  (DiT, dual-DiT, VAE, text encoder). They handle model loading, LoRA
  application, attention backend selection, KV cache wiring, and CUDA
  graph capture.
- **`flashrt/memory/`** — KV cache implementations (including
  preallocated variants).
- **`flashrt/layers/attention_backends/`** — pluggable attention
  backends (`fa2`, `fa4`, `flashinfer`, etc.), selected via the
  runtime config.
- **`flashrt/distributed/parallel_state.py`** — a vLLM-derived
  `torch.distributed` bootstrap and process-group helper, including
  init / destroy, world / tensor-parallel / sequence-parallel groups,
  and process-group utilities. Use this instead of calling
  `torch.distributed` directly.
- **`flashrt/distributed/communication_op.py`** — collective
  communication ops layered on top of parallel_state.
- **`flashrt/distributed/vae_plan.py`** — a pre-existing
  sequence-parallel tile decomposition for one of the VAEs used in
  this repo. Read it if you think it is relevant; otherwise ignore it.
- **`flashrt/distributed/device_communicators/`** — NCCL / CUDA
  communicator plumbing.
- **`flashrt/channels/shm_channel/`** — shared-memory tensor and
  control buffers, which each app's adapter
  (`flashrt/apps/<app>/adapter.py`) uses for IPC between the backend
  and anything driving it (frontend or test harness). Your backend will need to create the buffers the
  adapter expects.
- **`flashrt/models/`** — the model implementations (DiT, VAE, text
  encoders) plus a loader / registry. Use these rather than
  reimplementing model code.
- **`flashrt/rt_config.RTConfig`** — the runtime config object the
  shared runtime takes everywhere. User configs dehydrate into this.
- **`flashrt/loader/`**, **`flashrt/platforms/`**, **`flashrt/utils/`**
  — supporting infrastructure (weight loading, CUDA platform
  detection, misc helpers).

You may also use third-party libraries already installed in the
pre-configured environment resolved at session start (see
"Environment" above for the resolution rule), e.g., whatever model
servers the user's reference worker already imports. Do not add new
heavy dependencies without asking.

## Communication conventions

- **Before launching any multi-GPU job or downloading anything,**
  confirm hardware availability with the user.
- **Do not modify anything outside `flashrt/apps/<app>/backend/`.** The agent
  runs autonomously; there is no "ask the user" escape valve here.
  When you would otherwise want to change a file under a read-only
  area, vendor it (per "Read-only areas are not a blocker — vendor
  the file into your workspace") or re-implement against the shared
  primitives in your workspace (per "Multi-GPU coordination"), both
  bounded by the on-disk documentation those rules require.
- **When a benchmark regresses,** record it in the experiment log
  anyway. Negative results are load-bearing.
- **After every change,** make sure the user's reference launch still
  imports cleanly — a quick `python -c "import
  flashrt.apps.<app>.reference"` is enough. If it doesn't, something you did
  leaked out of your workspace.
- **If the reference backend cannot launch end-to-end on your
  session's hardware** — config incompatibility, OOM, unavailable
  GPU index, missing weights, third-party-engine startup failure,
  etc. — stop and surface it to the user before starting Phase 3.
  Do **not** proceed to variant implementation or benchmarking
  without a working reference: every subsequent correctness
  comparison requires the reference as the oracle, and every
  subsequent benchmark Δ requires the reference as the anchor. A
  variant benchmarked against a reference that never actually ran
  is an invalid measurement, no matter how clean the candidate
  numbers look.

## Definition of done

A session is complete when, for the target `<app>`:

1. **IR conversion exists and is validated by execution.** The
   experiment log documents the IR conversion (Phase 1) and records
   a passing validation result (Phase 2) **from an actual run of the
   `SequentialExecutor` compared against the reference backend** —
   not from static inspection of the graph. The IR graph builders
   and operator implementations live under `flashrt/apps/<app>/backend/ir/`,
   and the validation harness under `flashrt/apps/<app>/backend/tests/`
   confirms the IR executor produces outputs matching the reference
   backend.
2. **IR analysis recorded.** The experiment log includes the output
   of `summarize_graph()` (or equivalent analysis) showing the
   discovered pipeline stages, streaming overlaps, cross-chunk
   dependencies, and critical path. This analysis is the documented
   basis for the deployment decisions in subsequent variants.
3. **Multiple deployment variants explored.** The experiment log
   contains at least 10 distinct deployment variants that address
   different optimization ideas across both optimization layers
   (application pipeline scheduling and model inference parallelism),
   covering both diverse strategies — pipelining / streaming, other
   model parallelisms, co-location / device placement — and
   combinations of those strategies (not all isolated-lever, not all
   combined). Each variant has a recorded correctness result and a
   recorded benchmark result from an actual, verified E2E run (not
   estimated or assumed). Failed variants count toward the minimum
   if they were actually run and the failure was recorded honestly.
   Each variant's hypothesis must cite a specific IR analysis result.
   The deployment-iteration section of the experiment log includes a
   visible variant queue with each entry's status (`planned` /
   `in_progress` / `measured` / `blocked` / `discarded`) updated
   between every measured variant, per the Phase 4 rules.

   **Every queue entry — not just the first 10 — must have a
   terminal status at session handoff.** Terminal means one of:
   `measured` (with correctness + benchmark artifacts on disk),
   `discarded` (with the single acceptable removal reason backed
   by an IR fix, plus the adversarial self-framing paragraph), or
   `blocked` (satisfying all three strict criteria with on-disk
   artifacts, plus the adversarial self-framing paragraph). A queue
   row that is `planned` or `in_progress` at the moment the session
   hands off means the session is **not** done, regardless of how
   many variants beyond 10 have been measured. In that state the
   agent keeps working — there is no session deadline, so the only
   way to finish is to finish.
4. **Best variant passes correctness.** The recommended variant under
   `flashrt/apps/<app>/backend/` passes your correctness harness against
   the reference, run end-to-end, with all processes cleanly
   terminated after the harness finishes.
5. **Benchmark with real numbers, on every active target.** That
   variant has documented distributions (median and p95) for every
   target the session is ranked against — under the defaults, that
   is **both** latency-to-first-output (`t_first_out - t_in`) and
   the maximum sustainable output rate (with the inter-chunk-gap
   distribution as the smoothness side condition, where the app
   produces more than one output chunk per input); under a per-app
   override, the user's named target plus the defaults kept as
   secondary evidence per the "Per-app overrides" rule. All numbers
   are measured on the same hardware as the reference baseline,
   produced by a benchmark script checked into
   `flashrt/apps/<app>/backend/benchmarks/`. The chosen variant must be
   meaningfully better than the reference on at least one active
   target and must not regress on any other active target. The
   benchmark numbers must come from an actual run where the backend
   actually processed real input — not from a run where the harness
   exited early, the backend crashed, or processes were left
   orphaned.
6. **No orphan processes.** After the final benchmark run, you have
   verified (with `ps` or equivalent) that no backend processes
   remain running.
7. **Documentation.** `flashrt/apps/<app>/backend/README.md` documents how
   to launch the chosen variant end-to-end against the user's
   unchanged frontend.
8. **Experiment history.** The experiment log documents the full
   arc: IR conversion, IR validation, IR analysis, and the chain
   of deployment variants derived from it. The log should make it
   clear that you explored multiple dimensions of the optimization
   space and that each deployment decision was grounded in the IR
   analysis, not guessed.
9. **Pre-handoff anti-deferral gate.** Before declaring the session
   complete, re-read the Phase-4 passages titled **"Implementation
   and debugging cost are never valid rationale ..."**, **"Read-
   only areas are not a blocker — vendor the file into your
   workspace"**, **"Terminal statuses"**, **"What `blocked` means
   (strict)"**, **"Forbidden session-handoff language"**, and
   **"Adversarial self-framing (required)"**. For each, in your
   final status message to the user, quote the section title and
   answer one of:

   - "This session satisfies this rule because <specific on-disk
     pointer or explicit reasoning>."
   - "This session violates this rule at <specific queue entry /
     log block> — returning to Phase 4 to fix."

   If any passage gets the second answer, the session is **not**
   done — go back to Phase 4, fix the violation, and re-run this
   gate. If you find yourself writing soft or general answers
   ("this session satisfies the rule because we worked hard"),
   that is itself a violation: the answers must be specific
   pointers.

   **The forbidden-language check in this gate is mechanical.**
   The agent must run a text search over the experiment log, the
   README, and the final status message for each forbidden phrase
   listed under "Forbidden session-handoff language" above. If any
   match is found, the session is not done; edit the offending
   text before handoff.
