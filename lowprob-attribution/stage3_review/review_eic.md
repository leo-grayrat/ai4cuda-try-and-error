# Review — Editor-in-Chief

**Manuscript:** Is CUDA Syntax Rare Enough to Break PPO? A Conditional-Probability Test of a Stated Root Cause
**Recommendation:** Minor revision

## Summary of the submission

The paper tests the first link of a widely cited causal chain. CUDA Agent attributes PPO collapse at step 17 to near-floor token probabilities caused by CUDA's low share of pretraining data. The authors argue that this first link is empirically checkable at generation time, and that closed APIs have prevented the check because tool-call arguments carry no logprobs. They remove the agent framework, obtain 98,533 content tokens across 128 runs, and find that CUDA vocabulary has a *lower* low-probability rate than the surrounding text, while context — specifically the turn after a failed evaluation — carries the real tail.

## Assessment

**Contribution.** The work is a targeted replication-and-critique rather than a new method, and it should be evaluated as such. Its value is that the first link is load-bearing for a much larger claim, and no one has measured it. The negative result on the intervention (truncated feedback) is a genuine addition, because truncating tool output is a common efficiency measure and the paper shows it does not touch the tail.

**Presentation.** Clear structure, honest about the nine-order-of-magnitude gap. The decision to state the result as "under aligned-model inference" rather than as a refutation is the right call; a weaker paper would have overclaimed.

**Concern 1 — the load-bearing control is missing.** The single experiment that would settle the question is a repeat on an un-finetuned base model. The paper acknowledges this in §5.2 and Appendix A, which is the right handling, but it means the central claim is conditional in a way the abstract only partly conveys. I would ask the authors to state in the abstract that the measurement is on an aligned model and that the base-model control is outstanding.

**Concern 2 — the title is a question that the paper answers only conditionally.** "Is CUDA Syntax Rare Enough to Break PPO?" invites a yes/no. The paper's answer is "not in the condition we measured, and we cannot measure the condition that matters." That is defensible but the title should not imply closure.

**Concern 3 — the four tasks are all of one kind.** Elementwise and reduction kernels. The vocabulary exercised is the mechanical layer of Triton. The authors acknowledge this in §6.3. It does not invalidate the measurement but it does bound the generalisation more tightly than the abstract suggests.

**Minor.** §5.5 (the retracted first pass) is unusual to include, and I think correct to include: it explains why the paper reports two vocabularies, and it is the kind of methodological transparency that reviewers usually have to extract. Keep it.

## Decision

Minor revision: adjust the abstract to state the model condition and the outstanding control in the first paragraph, and soften the title's implication of closure.
