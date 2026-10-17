# Review — Domain (CUDA / kernel optimization)

**Recommendation:** Accept with one substantive concern.

## Accurate representation of the source

I checked the quoted chain against both the arXiv version and the technical blog. The four phrases attributed to the source — "less than 0.01% of pretraining data", "severe domain distribution mismatch", "$\pi_\theta(a_t|s_t) \approx 10^{-9}$", "even small train-rollout precision differences can explode the importance ratio" — all appear verbatim. The conditions ($\gamma=1$, $\lambda=0.95$, $\varepsilon_{lower}=0.2$, $\varepsilon_{higher}=0.28$, 150 steps) match too.

The paper is also right that what is *observed* and what is *claimed* differ. The entropy rise is an observation; the 10⁻⁹ is part of the argument. §2.1 attributes that reading to a third-party overview rather than asserting it as the authors' own framing, which is the correct way to handle a distinction the source itself does not draw.

## On the task families: a real gap the paper understates

Family B uses KernelBench Level 2. Those tasks are *operator chains*, and the fusion decision is: how many of the listed operations go into one kernel. That is a real decision, and the paper is right that the reduction tasks lacked it.

**But Level 2 chains have a property that weakens them as a test of "implementation difficulty": most of them are elementwise contracts glued to one heavy op.** `Conv2d → InstanceNorm → divide` is a convolution followed by two cheap elementwise passes. The hard part of that task is writing a correct convolution; the fusion decision is nearly trivial once the convolution is correct — you either fuse the normalisation into the epilogue or you do a second pass, and both are easy to write. The genuinely difficult fusion decisions in kernel optimization are different: choosing a tiling that keeps the intermediate in shared memory, deciding whether to transpose to change the access pattern, picking between split-K and a two-pass reduction for a reduction that does not fit one block.

**So the paper's null result on RQ3 may be a null result about the easy kind of fusion.** The claim "raising difficulty does not move the code-side rate" would be much stronger if at least one task required a non-obvious organisational choice — a FlashAttention-style online softmax, a split-K GEMM, or a fused layernorm with a two-pass variance. None of the five tasks has that property.

I would not require new data for acceptance, but §6.3 should say this plainly rather than the current "reduction and operator-chain tasks only". The distinction is not "reduction vs chain", it is "**chains with a trivial organisational choice vs chains with a non-obvious one**".

## Something the paper gets right that reviewers often miss

The manual reading (§4.7) finds that `y = torch.empty_like(x)` repeatedly produces a low-probability `y`. That observation is correct and important: at that position nothing determines the name, and the model spreads probability across `y`, `out`, `result`. The paper uses it to draw the naming-vs-implementation distinction, which is exactly the right use.

I would add one more instance the paper misses: the `' static'` case at −6.20, `for kh in static:`. That is not a naming choice — it is an **incomplete expression**: the model wrote `static` where `tl.static_range(KH)` was needed. This is a distinct category from naming and from slips: it is a *structural* token whose continuation was under-determined because the model had not settled on which loop construct to use. If the paper wants a category for "syntactic choices", this is its clearest example and §4.7 currently lists it under that heading without noting that it is the strongest case.

## On the proposed alternative (failure → prose → instability)

Domain-wise, this is plausible and consistent with practice. Anyone who has watched an agent loop knows that the token volume after a failed compile dwarfs the token volume of the code itself, and that the model's diagnostic text is far less constrained than the code. The account is coherent.

What is missing is any statement of *what would disconfirm it*. Two candidates: (a) a failure whose message is identical in content but whose formatting differs sharply (a compiler with a different diagnostic style) should change the prose distribution but not the code; (b) a success followed by a request to rewrite should produce prose of similar length but *without* the diagnostic content, letting one separate "more prose" from "bad news". Neither is run, but naming them would turn §5.4 from a plausible story into a testable hypothesis.

## Judgment

Accept with the task-scope concern: the paper's own limitation §6.3 should distinguish "chains with trivial organisational choice" from "chains with non-obvious ones", and say that only the former was tested.
