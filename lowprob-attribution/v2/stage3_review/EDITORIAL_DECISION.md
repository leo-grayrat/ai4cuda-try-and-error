# Editorial Decision

**Manuscript:** Where Does the Probability Cost Live? A Distributional Test of the Claim That CUDA Syntax Is Rare Enough to Break PPO
**Date:** 2026-10-17

## Reviews

| Reviewer | Recommendation | Main thrust |
|---|---|---|
| EIC | Minor revision | Tighten subtitle; mark §5.4 as hypothesis earlier; note selection effect in abstract |
| Methodology | Major revision (two items) | Region split conflates comments; cross-family comparison has four confounds |
| Domain | Accept, one concern | Level 2 fusion is the *easy* kind; §6.3 should distinguish trivial vs non-obvious organisational choices |
| Perspective | Accept | Reframe headline as "region dominates vocabulary"; §5.4's direction of causation is open |
| Devil's Advocate | Publish, but weaken claims | "Does not hold" overreaches; §5.4 unsupported; symptom-vs-cause not considered |

**Consolidated: Major revision (bounded).** No reviewer disputes the measurement. What is disputed is (a) how strongly the negative claim can be stated, (b) whether §5.4 is entitled to its framing, and (c) whether the region split and the cross-family comparison are clean enough to carry the interpretive load placed on them.

## Points of agreement

**1. The defensible claim is narrower than "the second link does not hold."**

EIC and Devil's Advocate both land here. The source measures a base model during rollout; we measure an aligned model at inference. Alignment cannot change corpus statistics, so if the source's claim is "CUDA syntax is intrinsically rare in the corpus", alignment should not fix it — the fact that we measure ≈1.0 is evidence that **the 10⁻⁹ is not a property of the vocabulary**. That is the claim the data support. "The second link does not hold" requires the base-model control.

**2. §5.4 is a hypothesis, and its framing precedes its disclaimer.**

EIC, Perspective and Devil's Advocate all flag it. Devil's Advocate adds the alternative the paper omits: the prose may be a **symptom** of confusion rather than a **source** of instability. Under that reading the RFT argument does not apply — RFT removes unsuccessful trajectories, not prose — and the intervention implied by §5.4 would be wrong.

**3. The code-side rate is a blend, and the paper may be understating its own finding.**

Methodology and Devil's Advocate independently raise this: comments inside code blocks count as code, and comments were 75% of Family A's low-probability tokens. Recomputing the rate with comments excluded will either strengthen the contrast (if the pure-code rate is much lower) or leave it roughly unchanged. Either way it must be reported.

**4. The Family B corpus is failure-dominated and must be labelled as such wherever its numbers appear.**

87.5% `after_failure`, 17.5% success, a third of runs invalidated by a driver bug. The code-prose contrast is measured within the same trajectories so it is not threatened, but the *level* 0.60% is not a general property.

## Points of conflict, and how they were resolved

**Domain vs Perspective on what the headline should say.** Domain wants the task-scope limitation sharpened: Level 2 chains have a *trivial* organisational choice (fuse the elementwise tail or don't), whereas the genuinely hard fusion decisions (FlashAttention-style online softmax, split-K, two-pass variance) are not represented. Perspective wants the headline reframed as "region dominates vocabulary" rather than as specific percentages.

These are compatible: **the re-framed headline should carry the domain caveat.** "Within the region-vs-vocabulary contrast, the tasks tested exercise only the easy kind of fusion" is a stronger and more honest statement than either reviewer's version alone.

**Methodology's cross-family confound vs the paper's RQ3 claim.** Methodology lists four confounds between Family A and B (task construction, turn cap, success rate, context composition) and offers two fixes: weaken the claim, or compare within Family B. **The second is better and is feasible with existing data** — five tasks with different chain lengths, different success rates, and per-task rates already computed (0.25%–0.93%). If the code-side rate is flat within Family B across that variation, the invariance claim survives without the cross-family confounds.

## Required revisions

1. **Change the claim, not just the caveat.** "The second link does not hold" → "whatever 10⁻⁹ measures, it is not a property of the domain's vocabulary". Applies to abstract, §5.1, §5.2 and §7.
2. **Recompute the code-side rate with comments excluded** and report both numbers.
3. **Add the within-Family-B comparison** of code-side rate across tasks, and state the four cross-family confounds if the cross-family comparison is retained at all.
4. **Move the "inference, not demonstration" disclaimer to the start of §5.4**, and add the symptom-vs-cause alternative along with the two disconfirming experiments (different diagnostic formatting; rewrite-after-success).
5. **Label the Family B corpus as failure-dominated** at every point its numbers are quoted, including the abstract.
6. **Sharpen §6.3**: not "reduction vs chain" but "chains with a trivial organisational choice vs chains with a non-obvious one"; only the former was tested.
7. **Tighten the subtitle** so it names what was tested (proximity to the probability floor) rather than sufficiency for collapse.
8. **Trim §7** by a third (it restates §5.1).

## Not required

- New collection. Every required revision is computational or textual.
- Figures. The paper carries no figures; all numbers are in tables and the absence is not a weakness here.

## Note on §5.5 and §5.6

Three reviewers independently asked that these stay. §5.5 explains why the paper reports two vocabularies and two boundaries; §5.6 is the only reason the rest of the numbers can be trusted. Both are unusual and both are correct to include.
