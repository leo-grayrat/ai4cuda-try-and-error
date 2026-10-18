# Review — Devil's Advocate

**Recommendation:** Publish, but the paper's central negative claim is weaker than it reads, and one of its positive claims is not supported at all.

## What the paper proves

The paper proves that **on this model, on these tasks, at these positions**, CUDA vocabulary is not where the low-probability mass concentrates. That is a solid, reproducible, well-documented finding, and the distributional evidence (quantiles, minima, mass concentration) is much harder to argue with than v1's threshold rates.

## Objection 1: the negative claim is about a measurement, not about a mechanism

The paper's headline is "the second link does not hold." But the second link, as the source states it, is about **a base model during rollout at 10⁻⁹**. The paper measures an aligned model at inference and finds ≈1.0.

The authors handle this honestly in §5.2 — they call it a tension, not a refutation, and name the control. **But the abstract and §7 both say "does not hold", not "does not hold here".** Those are different claims. "Does not hold here" is proven; "does not hold" requires the base-model control.

This is not nitpicking. The paper's own §5.2 concedes that alignment raises the probability of common code patterns. If alignment can move a token from 10⁻⁹ to 1.0, then it can move it by nine orders of magnitude, and the base-model condition is exactly the one that has not been tested.

**What the paper actually establishes** is narrower and still valuable: the claim cannot be about *CUDA syntax being intrinsically rare*, because if it were, alignment would not fix it — alignment does not change what is in the pretraining corpus. So the paper establishes that **whatever the source's 10⁻⁹ measures, it is not a property of the domain's vocabulary**. That is the defensible claim; "the second link does not hold" is not.

## Objection 2: the positive claim (§5.4) is not supported

§5.4 proposes that instability arises from failure-driven prose. Three observations are cited as consistent.

**Consistency is not support.** Each of the three admits at least one competing explanation:

- *Entropy rise with reward collapse*: equally consistent with "policy drifting off-manifold for any reason", including the reward being sparse and the advantage estimate being bad. No prose required.
- *RFT rescues training*: RFT also reduces trajectory variance, shortens episodes, and provides a supervised signal. Any of those could be the operative mechanism; "removes high-cost prose" is one hypothesis among several.
- *DR. KERNEL's MRS stabilises without raising the ceiling*: consistent with mismatch mattering, but mismatch is between *engines*, not between *prose and code*.

The paper says "this is an inference, not a demonstration" once, after three paragraphs of confident framing. That placement is wrong — the framing does the persuading, the disclaimer does not undo it.

**And there is a competing account the paper does not consider**: the low-probability prose may be a *symptom* rather than a cause. A model that is confused writes uncertain text **because** it is confused; the uncertainty in the text reflects an epistemic state, and the epistemic state is what destabilises training. Under that reading, prose cost is a *measurement* of confusion, not a *source* of instability, and "reduce prose" would be exactly the wrong intervention — it would suppress the symptom while leaving the state.

The paper's RFT argument does not distinguish these. RFT removes unsuccessful trajectories, not prose specifically.

## Objection 3: the code-side rate may be understated by the region split

Every low-probability token that sits inside a comment is counted as code. From Family A we know 75% of low-probability tokens were comments. The paper notes this biases code's rate upward — but if the true "pure code" rate is, say, 0.15%, then the contrast with prose is 90×, not 22×, and the paper is *understating its own finding* while drawing a weaker conclusion than the data support.

The methodology reviewer raises this too. It is a one-pass fix and it should be done before publication, because the current number is a blend.

## Objection 4: the Family B corpus is mostly failure text

87.5% `after_failure`, success rate 17.5%, and a third of runs invalidated by a driver bug. So the corpus that establishes "code 0.60%" is mostly a corpus of a model struggling. Whether a model that succeeds has a different code-side density is unknown — and the paper's whole thesis is about what happens in the region where the cost lives, which is disproportionately the failure region.

This does not invalidate the code-prose contrast (both regions are measured in the same trajectories), but it does mean the paper cannot say "the code-side rate is 0.60%" as a general property.

## What survives all four objections

1. **The distribution shape** (§4.4, §4.5). Code has a "template plus collapse" distribution, prose has a "spread" distribution, and the mass concentration table is a robust signature. This is the paper's real contribution and it does not depend on the disputed framing.
2. **The minimum: −13.02 (2.2×10⁻⁶) over 322,375 code tokens.** Four hundred times above the claimed floor. Whatever 10⁻⁹ is, it is not what CUDA code tokens do.
3. **The manual reading** (§4.7). The observation that no low-probability token corresponds to an implementation choice is the most direct evidence in the paper, and it does not depend on any statistical claim.

## What I would require

- Change "the second link does not hold" to "the claim is not about vocabulary" — the defensible version.
- Recompute the code rate with comments excluded.
- Move the "inference" disclaimer to the *start* of §5.4, and add the symptom-vs-cause alternative.
- Label the Family B corpus as a failure-dominated sample wherever its numbers are quoted.

None of these changes the findings. They change what the paper is entitled to say about them.
