# Review — Devil's Advocate

**Recommendation:** The paper should be published, but its central claim is weaker than it presents itself. My job here is to state the strongest case against it.

## The strongest objection: you did not measure the model the claim is about

CUDA Agent's assertion is about **an un-finetuned base model during training rollout**. The authors measured **an aligned commercial model through an inference API**. These differ in at least three ways that all push in the same direction: alignment raises the probability of common code patterns; a proprietary model has been trained on vastly more CUDA than the base model under discussion; and inference-time decoding applies its own filtering.

The authors know this. §5.2 says so. But the knowledge does not neutralise the problem. Consider what the paper can actually claim:

> In a condition that is not the one the source describes, and on a model that is not the source's model, the phenomenon the source describes does not appear.

That is not a test of the source's claim. It is a test of a *different* claim — that aligned models don't show the effect — and the paper has no way to distinguish "alignment removes the effect" from "the effect was never there".

The authors propose the base-model control as future work. Fine. Until it is run, the headline "Is CUDA syntax rare enough to break PPO?" has no answer, and the paper should not have a title that implies it might.

## Second objection: the tasks are the easy case, and the paper knows it

Four kernels. All reductions or elementwise maps. In such kernels the Triton vocabulary occurs in near-fixed templates, so of course it is high-probability. The source's motivating examples — fusion, BatchNorm folding, algebraic simplification — require *choosing a kernel organisation*, and the token sequences that realise different organisations share almost no surface form.

On a fusion task, "CUDA vocabulary is high-probability" may simply be false, and for a reason the paper's design cannot see: not token rarity but *structural under-determination*. The paper's limitation §6.3 admits this. But §5.1 then writes "the model knows what follows `tl.` in a Triton kernel" as if it were a fact about Triton rather than a fact about row-reduction.

## Third objection: token-level rates are the wrong level

The source speaks of sampling *sequences* at low probability. A sequence's probability is a product over thousands of tokens; a marginal token rate of 0.26% is compatible with an enormous range of sequence-level probabilities. The paper measures a proxy and reports it as if it were the quantity. It never computes, for example, the joint log-probability of a completed kernel, which would be the quantity that actually feeds $\rho_t(\theta)$.

This is the most fixable objection and also the most serious analytically: **the data to compute per-completion log-probabilities are already in `turns.jsonl`**, and the authors did not compute them.

## Fourth objection: the statistics are anti-conservative

Tokens within a completion are dependent. Fisher's exact test on 98,533 supposedly independent observations will produce p-values that are too small, possibly by a wide margin. The methodology reviewer flags this. The effect is large enough that the sign would not flip, but the reported $p = 0.0007$ should not be read as $p = 0.0007$.

## Fifth objection: the context effect has an alternative reading

`after_failure` at 4.30% is read as "failure moves the context off-distribution". An equally consistent reading: after a failure, the model *rewrites* the kernel, and rewriting means emitting tokens it has not just emitted — whereas after a success it emits small refinements. The elevated rate could reflect *generation of new content* rather than *response to bad news*. The truncation result is consistent with both readings, since truncating the message does not change whether the model must write new code.

Distinguishing these would require holding "must rewrite" constant while varying "told it failed" — for example, asking for a rewrite after a success.

## What survives

Even granting all five objections, two things stand:

1. **The negative intervention result.** Truncating feedback does not change the tail in this setting. That is a datum about agent pipelines that does not depend on which model or which task.
2. **The methodological lesson of §5.5.** A defensible classification choice silently removed 96% of the target class and produced a significant result in the wrong direction. The remedy — vary the boundary — is general.

Neither is the headline. The headline is conditional, and should be written as such.

## What I would require

- Compute per-completion joint log-probabilities from the existing data and report them. This turns a marginal proxy into the quantity the source actually discusses, and it costs no new collection.
- Add the base-model control or remove the question from the title.
- Address the "rewrite" alternative for the context effect with a third condition.
