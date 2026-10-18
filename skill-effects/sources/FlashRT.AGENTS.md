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

This is the tight iteration loop. **You must complete multiple
iterations** — a single deployment variant, no matter how clever,
is not sufficient. The user is testing whether you can
systematically explore the optimization space, not whether you can
write one plausible implementation on the first try. Your experiment
log should contain at least 10 distinct deployment variants that
exercise different optimization ideas (e.g., varying which models
are parallelized, varying the pipeline schedule, varying device
placement) before you may consider stopping. A variant that is "the
first idea but with a bug fix" does not count as a distinct variant.

**There is no session deadline.** Take the time the full queue
requires. This harness does not impose a wall-clock budget on you,
and you must not self-impose one. "Running out of time", "need to
wrap up", "I'll come back to this later", and "a follow-on session
can pick this up" are **not** valid reasons to skip, defer,
downgrade, proxy, or substitute any queued variant. The queue-
removal rule below enumerates the **single** valid reason for not
measuring a `planned` entry (the IR analysis was wrong about the
transformation's legality); if that reason does not apply, you
measure the variant — there is no alternative path to removing it,
no user approval shortcut, and no "defer to a future session"
escape hatch. The agent is expected to work autonomously; the
default is that every queued variant gets measured.

Within those 10+ variants, you must cover **both diverse strategies
and combinations of strategies**. Enough single-lever isolation
variants to cleanly attribute gains to individual ideas, and enough
multi-lever combined variants to show whether those ideas stack or
interact. A queue of 10 variants that all isolate different levers
but never combines them is insufficient; a queue of 10 combined
variants with no isolated measurements is also insufficient.
Diverse strategies means the variants collectively cover the axes
the IR analysis surfaced — pipelining / streaming, other model
parallelisms, and co-location / device placement — not just the
one or two that were easiest to implement first.

**10 is a floor, not a target.** Do not stop at 10 just because you
hit the quota. The rule is: run every hypothesis in your queue to a
measured result, regardless of whether you have already found a
large win, and regardless of whether your next hypothesis looks
smaller than a previous one. Stopping because "we already found
something good" is exactly the failure mode this rule exists to
prevent.

**Implementation and debugging cost are never valid rationale for
skipping, deferring, downgrading, substituting, or rotating off of a
queued variant.** There is no stuck-detection heuristic, no per-
variant time budget, no "rotate when hard", and no threshold after
which you are allowed to move on from an unmeasured variant to an
easier one. The following phrasings — and every technical-sounding
rephrasing of the same idea — are all cost-based rationales and are
forbidden as reasons to leave a queue entry unmeasured:

- "This would take too long" / "I am running low on session time"
- "Implementation is non-trivial" / "the integration is complex"
- "Debugging this is hard" / "the state machine is tangled"
- "X does not compose cleanly with Y"
- "This needs a redesigned scheduler / infrastructure / …"
- "I would need to modify a read-only file" (see below — vendor it)
- "The shortcut version is exploratory scaffolding and the full
  version is future work"

You work on the current variant until it reaches a terminal status
(see "Terminal statuses" below). You do not move on to a cheaper
variant because the current one is hard. If you genuinely cannot
make progress after a serious attempt, see "What `blocked` means
(strict)" below — that status has a specific, narrow definition
designed to rule out implementation-cost rationale entirely.

**Read-only areas are not a blocker — vendor the file into your
workspace.** If a variant requires changing behavior defined in a
file under a read-only area of the repo (`flashrt/*` outside
`flashrt/apps/<app>/backend/`, the shared adapter, the shared models, the
shared runtime, `flashrt_kernels/`, etc.), the correct move is to
**copy the file (or the subset you need) into
`flashrt/apps/<app>/backend/<variant>/` and modify the copy**, then import
from your copy instead of the original. This keeps the read-only
contract intact while giving you full freedom to iterate on the
logic. Document vendored copies with a one-line comment at the top
of the copied file pointing back to the original path and the
reason for divergence, and (since the original remains the oracle)
validate numerical parity of unchanged code paths through the
usual Phase-2 / Phase-4 comparisons. "I need to modify a read-only
file" is therefore **never** a legitimate reason to mark a variant
`blocked` or `discarded`; the path to unblock is always "vendor the
file and modify the copy." If you catch yourself about to write
"I'm not allowed to edit X", recognise that as this loophole and
vendor instead.

**Terminal statuses.** A queue entry reaches a terminal status only
when it is one of:

- `measured` — the variant has been launched end-to-end, has an
  executed correctness verdict (PASS / FAIL) with the harness log
  persisted on disk, and has a benchmark JSON on disk from a real
  E2E run.
- `discarded` — removed for the **single** acceptable reason
  described in "You may not silently drop a `planned` queue entry"
  below (the IR analysis was wrong about the transformation being
  legal), **plus** the adversarial self-framing paragraph described
  below. This status should be rare.
- `blocked` — reached only when *every* criterion in "What
  `blocked` means (strict)" below is demonstrably satisfied with
  on-disk artifacts the agent can point to, **plus** the
  adversarial self-framing paragraph. This status should be
  *extremely rare* — in a correctly-behaving session it is usually
  never used.

`planned` and `in_progress` are **not** terminal. A session that
hands off with any queue entry in a non-terminal status is not
complete (see "Definition of done"). The default expectation is
that every queued variant ends in `measured`; `discarded` and
`blocked` are narrow exceptions, and an agent that finds itself
using either frequently is almost certainly re-dressing
implementation cost.

**What `blocked` means (strict).** `blocked` is reserved for
situations where *no amount of code the agent could write under
`flashrt/apps/<app>/backend/` can resolve the failure*. It is self-
assigned by the agent — there is no human in the loop — but the
criteria below are deliberately strict enough that implementation-
cost situations cannot satisfy them. Before you may write
`status=blocked`, all three of the following must hold, and the
entry must link to a specific on-disk artifact demonstrating each:

1. **A real attempt happened, on disk.** You wrote code for this
   variant under `flashrt/apps/<app>/backend/` (including any vendored
   copies of read-only files per the section above), launched it
   end-to-end or as close to end-to-end as the failure allows, and
   observed a specific, reproducible failure. A partial draft you
   deleted does not count. A mental model of why it would be hard
   never counts. The current state of the variant's code must
   remain on disk (not deleted), so it can be re-run to reproduce
   the failure.

2. **The blocker is environmental, not implementation.** It must
   be something *no amount of code you could add to
   `flashrt/apps/<app>/backend/` can resolve*. Exactly three categories
   qualify:

   a. **Hardware failure.** A required GPU is not detected, NVLink
      is down, hardware OOM persists even after the variant has
      been sharded as aggressively as the hardware allows, etc.
      Must be backed by an `nvidia-smi` / `dmesg` / CUDA error log
      on disk.
   b. **Missing external resource.** A required model weights file
      is not on disk and cannot be downloaded, an external service
      is down with no substitute, etc. Must be backed by a
      filesystem or HTTP error on disk.
   c. **Third-party library bug in the pre-configured environment.**
      The
      bug must be reproduced in an isolated `pytest`-style test
      checked into `flashrt/apps/<app>/backend/tests/` that shows the
      failure is in the library itself, not in the agent's usage
      of it. "I could not figure out how to call it correctly" is
      implementation work, not a library bug.

   The following are explicitly **not** valid blockers:

   - Implementation complexity of any kind.
   - Debugging difficulty.
   - Integration work between existing components (coordinator,
     scheduler, dispatcher, RPC wrapper — you write it).
   - Needing to modify a read-only file — you vendor the copy; see
     above.
   - The fact that the variant spans multiple processes, GPUs, or
     frameworks.
   - Interaction between async / streaming / distributed patterns
     that you have not yet untangled. "Composing X with Y is
     non-trivial" means "this is implementation work", not
     "blocked".
   - Correctness failures in the variant's own code.
   - **Bugs in code the agent wrote.** If the variant hangs, errors,
     or produces wrong output because of a defect in any code the
     agent itself authored — variant worker, harness helper, vendored
     copy, custom protocol, anything under `flashrt/apps/<app>/backend/` —
     the answer is to **fix the bug**, not to declare the variant
     blocked. A bug in agent-written code is implementation work by
     definition; it cannot satisfy the "no amount of code under
     `flashrt/apps/<app>/backend/` can resolve" test in criterion (2) above
     because the resolving code is literally the bug fix the agent
     owes. This holds whether the buggy code is a thin wrapper around
     a shared-runtime call, a custom orchestration layer, or anything
     in between — the location of the bug does not change who owns
     the fix.

     Subtle case: when an agent-written wrapper around a shared-runtime
     call hangs or errors and the agent suspects the shared call
     itself, that suspicion is a hypothesis to test, not a conclusion.
     Before assigning `blocked` on a "third-party library bug" basis,
     the agent must produce the isolated `pytest`-style reproduction
     described under criterion (2c), running the **shared-runtime
     call alone** with no agent-written wrapper around it. If the
     isolated call works, the bug is in the wrapper and the variant
     stays `in_progress`.

   - **An unexplained long-running operation the agent has not
     investigated.** A worker that produces no new log output for
     several minutes is *not yet* a hang and *not yet* a blocker; it
     is an unknown that the agent is responsible for diagnosing
     before any terminal status can be assigned. See "Hangs vs.
     slow operations" below.

   - Running low on session time / tokens / patience.

3. **The log entry cites a specific on-disk artifact path for the
   failure.** A traceback, a harness stderr log, a `pytest` failure
   report, a system-level error message. Saved under
   `flashrt/apps/<app>/backend/` and linked from the queue row as e.g.
   `blocker_artifact: tests/fixtures/pp4_cuda_oom.log`. The artifact
   must clearly exhibit one of the three environmental categories
   above. Prose descriptions of difficulty are not artifacts.

If any step of your reasoning for `blocked` lives in the "not
valid blockers" list — including phrasings that have been
rewritten to sound technical — the status does not apply and the
variant is `in_progress`. You keep working on it. There is no
rotation-off, no time-box, and no escape hatch.

**Hangs vs. slow operations: investigate, do not assume.** A backend
that has stopped producing log output is *not* automatically a hang
and *not* automatically a deadlock. First-launch autotune (cuDNN
benchmark mode, triton autotune via inductor, kernel
auto-selection), JIT compilation, weight loading from cold cache,
NCCL communicator initialization, and disk-cache contention between
multiple processes can each individually take several minutes
without printing anything during the wait. A worker that has not
emitted a log line in 5 minutes is in an *unknown* state, not a
known-bad one.

The agent's job when it sees an unexplained long-running operation
is **not** to wait it out indefinitely (that risks burning a session
on a real deadlock), and **not** to declare it blocked (the agent
hasn't earned that yet — see the strict criteria above). It is to
*investigate*, narrow the unknown to either "this is making slow
forward progress and will finish" or "this is genuinely stuck," and
act accordingly. Concrete moves:

1. **Check whether the process is doing work.** `nvidia-smi` for GPU
   utilization (compiling kernels usually shows <100% but >0%);
   `ps -p <pid> -o stat,wchan,cmd` and `cat /proc/<pid>/wchan` for
   the kernel-side wait state (`futex_wait_queue_me` is a userspace
   lock — could be a real deadlock or could be inductor's internal
   parallel autotune queue — *not* itself proof of either); `py-spy
   dump --pid <pid>` (or `gdb -p <pid>` + `py-bt`) for a Python
   stack trace. A process actively in a triton autotune kernel
   shows a live frame in `torch._inductor.select_algorithm` or
   similar; a deadlocked process shows everyone parked in
   `acquire`/`wait` calls with no progress between snapshots.
2. **Check whether the disk cache is growing.**
   `watch -n 5 'du -sh /tmp/torchinductor_*'` (or wherever the
   inductor cache lives in the active environment). Cache size that
   increases between samples means autotune is making forward
   progress and the right move is to wait. Cache size that stays
   constant for many minutes while the process burns CPU points at
   the process being in some non-cache-writing phase (or stuck).
3. **Check whether the issue is multi-process contention.** Inductor
   uses file locks under the cache directory; N processes
   first-touching the same shape can serialize on the cache
   directory and look like a hang from the outside. If the same
   code path runs quickly in a single-process variant on the same
   hardware, the issue is contention, not a deadlock — the fix is
   to pre-warm the cache (run the single-process variant once to
   populate it) or to reduce concurrency
   (`TORCHINDUCTOR_COMPILE_THREADS=1`, one rank at a time during
   init), not to give up.
4. **If the wait is genuinely slow but progressing, document and
   wait.** If a multi-process first launch needs 20 minutes to
   populate the inductor + cuDNN caches, that is a one-time cost,
   not a blocker. Subsequent launches with a warm cache will be
   fast. The agent records the cold-launch wall time in the
   experiment log and continues.
5. **If snapshots prove the process is genuinely deadlocked,
   localising the deadlock to a particular library or layer is
   diagnostic — it is not a blocker on its own.** A futex held
   inside agent-written code is the agent's own bug; the agent
   fixes it. A futex held inside a shared-runtime or third-party
   call (verified via the criterion-(2c) isolated reproduction)
   tells the agent which library is misbehaving, but it does
   **not** by itself license a `blocked` status — the agent's
   options include vendoring the offending file and patching it
   (per the "Read-only areas are not a blocker — vendor the file"
   rule), re-implementing the affected call against a different
   shared-runtime primitive, working around the failure path
   (e.g., a different process-launch model, a different sharding
   strategy, a different concurrency level), or designing the
   variant so it does not exercise the broken path at all. Only
   when **every** such workaround has been attempted and itself
   produces an on-disk failure that satisfies criterion (2c) does
   the variant become eligible for `blocked` — and even then, the
   adversarial-self-framing paragraph must point at the *specific*
   failure log of each workaround, not at the original
   shared-runtime hang.

The discriminator between "slow but working" and "stuck" is
*evidence the agent collected* — `nvidia-smi` snapshots, cache-size
deltas, py-spy stacks, isolated reproductions of each candidate
workaround. Time spent staring at a stalled log file is not
evidence. Until that evidence exists, the variant is
`in_progress`, the agent keeps investigating, and a `blocked`
declaration premised on "it hung" is invalid.

**Forbidden session-handoff language.** The experiment log, the
README, and the final status message must describe the state *as
of this session*, not plans for future sessions. The following
phrases and every technical rephrasing of them may not appear
anywhere in those artifacts:

- "next session" / "future session" / "follow-on session"
- "pick this up later" / "I'll come back to this" / "come back to
  it"
- "the next session's first target" / "beyond this session's scope"
- "deferred to a later session" / "a future agent can"
- "out of scope for this session" / "left for follow-up"

There is no mechanism by which an unmeasured queue entry gets
"picked up later". Every variant is the current session's problem
until it reaches a terminal status.

**Adversarial self-framing (required).** Every `blocked` or
`discarded` queue row — and every log block for a variant that
was removed without being measured — must contain a dedicated
paragraph in the experiment log of the form:

> A skeptic reading this could argue it is implementation-cost
> skipping re-dressed. The concrete evidence it is not is
> <specific on-disk pointer>.

Acceptable fillings for `<specific on-disk pointer>` are narrow
and category-specific:

- For `discarded` entries (reason: IR was wrong): a pointer to the
  specific IR file path that was updated in the same edit, *plus*
  a pointer to a failing transformation-legality test checked into
  `flashrt/apps/<app>/backend/tests/` that shows the original IR-legal
  transformation would corrupt output.
- For `blocked` entries: a pointer to the specific on-disk artifact
  required under criterion (3) of "What `blocked` means (strict)"
  above — an `nvidia-smi` / `dmesg` log, a filesystem / HTTP error
  log, or a reproduced third-party library bug test under
  `flashrt/apps/<app>/backend/tests/`.

If you cannot write a convincing version of that paragraph whose
"concrete evidence" section is a **specific on-disk pointer** of
one of the above forms (not prose), the entry does not qualify
for its current status — go back and fix that first. If you find
yourself writing general justifications ("because the work required
to complete this is substantial", "because the integration with X
is complicated", "because a related measured variant probably
covers this"), stop — that *is* the failure mode this paragraph is
designed to catch, and writing a more eloquent version of it only
makes the failure more entrenched.

**Maintain a visible, evolving variant queue.** Before you implement
any variant, commit your initial variant queue to disk in the
experiment log as an explicit, ordered list. Each queue entry must
include: the variant name, a one-sentence hypothesis, the specific
IR analysis result motivating it, predicted impact (which active
optimization target, rough magnitude), and its status (`planned` /
`in_progress` / `measured` / `blocked` / `discarded`). The initial
queue is derived from the IR analysis — every parallelism opportunity,
pipelining / streaming edge, and co-location decision the analysis
surfaced is a candidate, plus any below-IR-granularity levers you
identified during Phase 2, plus combinations of the above.

The queue is a **live document** and must be updated **between
every variant you measure**. After each measurement: (1) update that
variant's row with its result, (2) append any new variant ideas the
result suggests — if combining two winning levers looks promising,
add the combined variant; if a variant produced a surprising
regression, add a variant that isolates the regression's source; if
a measured result invalidates an earlier hypothesis, add the
follow-up hypothesis — and (3) review the ordering of remaining
`planned` items in light of what you just learned. This is an
explicit evolutionary-search loop: the queue grows and re-orders as
evidence accumulates, and the 10-variant floor applies to the final
state of the queue (not to whatever you wrote down on day one).

You may not silently drop a `planned` queue entry. In fact you may
almost never drop a `planned` entry at all — the default is that
every queued variant gets measured. The **single** acceptable
reason for removing a variant without measuring it is:

(a) the IR analysis was wrong about the dependency making the
    transformation legal — the variant is not a valid
    transformation of the reference computation to begin with.
    This removal must be backed by a specific concrete demonstration
    on disk: the precise dependency the IR missed, a minimal
    reproduction showing the transformation would corrupt output
    (even when implemented correctly), and a pointer to the IR
    file / operator declaration that was wrong. Update the IR and
    re-run validation as part of the same edit; "the IR is wrong"
    without a corresponding IR fix is not sufficient evidence.

Reasons that are **explicitly not** acceptable — each is a specific
failure mode this section exists to prevent:

- "Would take too long to implement" / "the integration is
  non-trivial" / "I would need to vendor and rewrite a large file"
  — cost-based reordering. Forbidden. You work on the variant until
  it is `measured` or provably satisfies the strict `blocked`
  criteria above.
- "A different measured variant plausibly dominates this one" /
  "the combined version already covers this idea" / "Y likely beats
  X by construction" — predict-don't-measure. Forbidden. Strict
  domination is **only** ever a claim *between two measured
  variants*; an unmeasured variant cannot be dominated because its
  performance is unknown. If you believe X is dominated by
  measured Y, the way to show it is to measure X; until then X is
  not removable on that basis.
- "I expected it not to help" / "the prediction is that this
  regresses" / "this axis of parallelism is known not to matter" —
  predict-don't-measure. Forbidden.
- "The user would say stop if asked" / "this is obviously out of
  scope" — forbidden substitutes for measurement. The agent
  decides autonomously what to measure; the decision is "measure
  everything in the queue" unless reason (a) applies.

Every `discarded` entry must include the adversarial self-framing
paragraph described under "Terminal statuses" above, whose
"concrete evidence" pointer must be the specific IR fix / failing
transformation test under reason (a).

**No variant may stand in for another.** Every queued variant must
have its own independent implementation, correctness check, and
benchmark. Measuring one variant never satisfies a different queue
entry, even when the two variants look like they differ only in
"scale" or "implementation detail". Reasoning of the form "variant A
is just a bigger / smaller / more-general / shortcut version of
variant B, so the measurement of B also tells us about A" is the
predict-don't-measure failure mode applied to a variant you have
not built yet — you are using a related measurement as a proxy for
one you have not actually run. The combined cost / memory /
contention / synchronization / implementation-effort profile of any
variant is the whole reason the queue exists; sibling or
simpler-version measurements are useful evidence to inform
hypotheses, not substitutes for the measurements the queue demands.
If you find yourself tempted to drop one queue entry because a
related one was already measured, that temptation is the signal
that the related entry should stay live.

**Follow-ups must be enqueued, not narrated.** Every variant you
measure surfaces new ideas — combining the winner with another
lever, isolating a surprising regression, probing a new axis the
result made interesting. Those ideas belong in the variant queue as
`planned` rows, not in free-text "Follow-ups:" lines scattered
through log entries. Any follow-up you name in a log entry must
appear as a queue row inside the same edit that finalises that
entry, or be explicitly removed per the single acceptable removal
reason above. "Noted for later," "the obvious next step is X,"
"deferring to a follow-on session," and "scoped but not built" are
silent-drop failure modes wearing different clothes: they produce
unmeasured ideas with a paper trail that looks like they were
considered. Every idea the log mentions either lives in the queue
as `planned` / `in_progress` / `measured` / `blocked` /
`discarded`, or has a written removal reason.

**Prioritize the queue by IR-analysis-ranked expected impact, not
by implementation cost.** If the IR flagged pipelining / streaming
opportunities, model-parallelism splits, or co-location decisions,
the highest-ranked items go at the top of the queue regardless of
how much code they take to write. A queue ordered "easy variants
first, hard variants later" is a warning sign: since every variant
must be measured regardless (see "Implementation and debugging
cost are never valid rationale ..." above), the cost of
mis-ordering is not that hard variants get skipped — they will
still be measured — but that the agent may be tempted to interpret
reaching 10 measured items as progress toward being done, when in
fact the unmeasured tail is all of the hardest variants. Ordering
the hard variants first forces the session to do the actual work
before it feels "close to done."

**Watch for early-win anchoring.** If one variant delivers an
outsized speedup early in the loop, the correct response is to
explore *more* aggressively, not less. An early big win is a signal
that the optimization space is richer than you thought, not a
signal that you can coast. The expensive / ambitious variants you
might be tempted to skip once you are already "ahead" are exactly
the ones that reveal whether the early win is the ceiling or only
a local optimum; skipping them means you never find out.

**Isolate levers first, then combine.** Prefer variants that
exercise a single optimization lever at a time before you stack
them, and benchmark those isolated variants against the reference
(not against a previously-stacked variant). Isolated variants give
clean attribution — you know exactly which lever produced which
part of the gain on the active target(s) — and they make
interaction bugs much easier to spot. Combining the winning levers
into a stacked combined variant afterwards (named for what it stacks,
not `combined_best`) is legitimate and
is often where the best overall result on the active target(s)
actually comes from; just do the isolated
measurements first so you know what each lever is actually
contributing, and do not cram multiple untested optimizations into
one variant at the start. Both "orthogonal single-lever" variants
and "carefully composed combined" variants are valid end-states of
the exploration.

**Non-dominated winning pairs obligate a combination.** If two
measured variants each earn a `promote` verdict on *different*
optimization axes (e.g., one wins latency-to-first-output, the
other wins sustainable-output-rate; or, under a per-app override,
one wins the user's primary target while the other wins a
secondary target the variant must not regress) and neither
strictly dominates the other across the axes the session is
ranked against, the queue must gain a `combined(A, B)` entry as
`planned` before either isolated variant is closed out. Stopping
with two co-equal winners that each attack a different axis and
never trying the stack is exactly the failure mode this rule
exists to prevent: the stacked variant is the one most likely to
beat both isolated winners on the axis each one of them is already
best on. The combined entry
remains live in the queue until it is itself measured or removed
per the single acceptable removal reason — it is not optional and
cannot be deferred by citing predicted numbers for it (see the
transitive "measure, don't predict" rule later in this section).

**Audit but don't dismiss.** You may never write a candidate
direction off as "too much new implementation" or "would require
building X from scratch". Grep the shared runtime (`flashrt/*`) for
anything related to X and read at least the most promising file you
find. The user has gone to some effort to port reusable abstractions
into this repo; a direction that looks expensive on first read may
already be partially supported. If manual implementation is required,
you must proceed to do so. Documenting a direction as deferred due to
implementation cost is a failure mode — the experiment log entry can
never say "audited `flashrt/<path>` and it does not cover this case
because …" or "no relevant support found in the shared runtime after
grepping for …", or "too complex".

**Measure, do not predict efficacy.** Closely related to the audit
rule, but about a different kind of dismissal: do not write an
optimization axis off on the basis of a plausible-sounding prediction
that it "won't help anyway". Interactions between compute cost,
communication cost, kernel-launch overhead, memory-bandwidth
pressure, and overlap with other stages are hard to reason about
precisely ahead of time, and intuitions about "this will be too
small to benefit from distribution", "the communication cost will
dominate", "this stage isn't on the critical path so parallelizing
it is pointless", and similar are frequently wrong once actually
benchmarked. The only way to know whether a given axis of
deployment / parallelization helps is to build a minimal variant
for it and measure. Treat every axis you can reasonably scope as
something to be tested — including axes you personally expect to
underperform — rather than as something to rule in or out a priori.
An idea you only dismissed with a back-of-the-envelope argument
belongs in your variant queue until a measurement says otherwise;
"I expected this not to help" is an acceptable hypothesis entry, but
only if followed by "measured result: …", not by "therefore skipped".

The "measure, do not predict" rule extends transitively. When a log
entry quotes expected numbers for a *different* variant — e.g.,
"variant Y would give X on metric Z", "the combined version would
reach N", "the larger-scale version is predicted to …" — variant Y
must appear in the queue as `planned` before the current entry is
considered finalised. Predicting the numbers of an unbuilt variant
and then not building it is the same failure mode as predicting
efficacy of the current variant; burying the prediction inside a
different variant's log block does not excuse it. Predictions are
useful as hypotheses to motivate queue entries; they are not
substitutes for measurements, and writing them down is not a way to
close an axis of the search.

Each iteration does exactly these five steps, in order:

1. **Decide.** Write a short hypothesis in your experiment log: what
   transformation you are about to try, which IR analysis result
   made it legal (cite the specific state independence, streaming
   overlap, or pipeline stage), and what you expect the impact to be
   on each active optimization target (under defaults, latency-to-
   first-output and the sustainable-output-rate / smoothness pair;
   under a per-app override, the user's named target). If this is
   not the first iteration, explicitly state what the previous
   iteration's results taught you and how that informed this
   hypothesis.
2. **Implement.** Write or edit code *only* under
   `flashrt/apps/<app>/backend/`. Each variant you intend to benchmark
   should live in its own subpackage / own config, not as an
   in-place mutation of a previously-good variant.
3. **Verify correctness (mandatory — see details below).** Run your
   correctness harness. It must launch both the reference and
   candidate backends as subprocesses, drive them through the
   adapter with simulated input, collect outputs, **cleanly
   terminate both backends**, and compare outputs. You must read the
   harness output and confirm it reports PASS. A variant that fails
   correctness does not get promoted — document the failure in the
   experiment log and move on. **Do not skip this step.**
4. **Benchmark (mandatory — see details below).** Run your benchmark
   harness, which launches the candidate backend as a subprocess,
   drives it through the adapter, produces numbers for every active
   optimization target (timestamps for latency-to-first-output, a
   steady-rate sweep for sustainable-output-rate / smoothness, and
   any per-app override target), **cleanly terminates the backend**,
   and reports the results. Read the harness output and record the
   actual numbers. Persist results under `benchmarks/results/`. **Do
   not skip this step.**
5. **Iterate.** Update the experiment log with the result, decide
   whether the variant becomes the new baseline, and plan the next
   hypothesis. A regression is a signal, not a failure — record it.
   Then go back to step 1 with a new idea.

   **Gate before starting the next hypothesis:** before writing the
   next variant's hypothesis, verify on disk that the current variant
   has both (a) a recorded correctness verdict (PASS or FAIL, with the
   correctness harness log persisted) and (b) a benchmark JSON under
   `flashrt/apps/<app>/backend/benchmarks/results/`. If either artifact is
   missing, the current iteration is not complete and no new variant
   may be started. "I'll come back and benchmark this after I try the
   next idea" is not allowed — the current variant is the next
   variant until both artifacts exist.

**The "baseline" is always the user's reference backend.** The very
first entry in the deployment-iteration section of the experiment log
must be a benchmark of the user's reference backend under
`flashrt/apps/<app>/reference/`, run through your benchmark harness on the same
hardware you will use for variants. Every variant's Δ is measured
against that entry. An agent-authored variant may **not** serve as
the anchor — in this repo, "baseline" always refers to the user's
reference. This keeps the whole experiment log anchored in a single,
stable comparison point instead of drifting as you promote
intermediate variants.

**Stopping criterion.** You may stop when every variant in your
queue has reached a terminal *status* (`measured` / `discarded` /
`blocked` — see "Terminal statuses" above) and the definition-of-
done checklist is satisfied. `planned` and `in_progress` are not
stopping states. A `measured` variant may carry a verdict of
`needs-followup` (meaning the measurement itself surfaced a new
question that became its own queue entry), but a verdict of
`needs-followup` on an unmeasured variant is never valid — the
variant must be measured first.

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
