# Review — Methodology

**Recommendation:** Major revision (two items), otherwise the strongest methodological work in this series.

## What is done well

**The pre-flight gate.** Before spending on 168 runs, the authors confirmed in one API call that 448/448 tokens inside a reply-body code block carry a logprob. This is the correct order: the entire design rests on one assumption, and the assumption is cheap to check.

**The observable is stated as a distribution.** v1 reported a threshold rate and nothing else; v2 reports quantiles, minima, and probability mass. This is what makes the comparison against 10⁻⁹ possible at all — a rate cannot be compared to a probability, a minimum can.

**The manual reading.** §4.7 says, in effect: aggregate rates cannot distinguish "chose an implementation" from "chose a word", so we read all 1,005 tokens. That is the right move and it is rare.

**The two-boundary rule.** Every headline checked under a second vocabulary. This rule exists because the first attempt violated it (§5.5), which is disclosed.

## Required revision 1: the region split is doing too much work

The paper's central contrast is `code` 0.60% vs `prose` 13.56%, where `code` means "an odd number of ``` delimiters precede the token". Three issues:

**(a) Comments inside code blocks count as code.** The paper acknowledges this in §6.8 and argues it biases the code rate *upward*, which is correct — comments are high-entropy, so including them inflates code's rate. **But the size of the effect is not reported.** From Family A's manual reading we know 75% of low-probability tokens were comments; if a similar fraction holds in Family B, then the "pure code" rate is well below 0.60%. The authors should compute and report the rate with comments excluded from the code region — it strengthens their claim and costs one pass.

**(b) Fenced blocks are used for non-code content.** The paper mentions "a few" instances. How are they handled? If a block contains a traceback or a shell command, those tokens are counted as code. Report the count or state the exclusion rule.

**(c) The prose region is heterogeneous.** It contains (i) the model's own reasoning, (ii) evaluator feedback quoted back into the conversation, and (iii) the model's summary to the user. These are three different writing tasks. The paper's mechanism in §5.3 ("composing vs templating") applies mainly to (i). A breakdown by these subtypes — or at least the argument that they are homogeneous — is needed, because otherwise "prose is high-entropy" conflates the model's thinking with text it is quoting.

## Required revision 2: the RQ3 comparison is confounded

The claim "raising difficulty does not move the code-side rate" compares Family A (0.67%) with Family B (0.60%). **But the two families differ in more than difficulty:**

- Different task construction (hand-written `run(x)` vs wrapped KernelBench `Model`)
- Different turn caps (8 vs 6)
- Different success rates (nearly all vs 17.5%)
- Different context composition (Family A has little `after_failure`; Family B has 87.5%)

So the equality of 0.67% and 0.60% could be a genuine invariance, **or** the result of difficulty pushing one way and the other four factors pushing the opposite way. The paper's conclusion — "difficulty does not move the code-side rate" — is stronger than the comparison supports.

**Minimum fix**: state the four confounds and weaken the claim to "the code-side rate is in the same range under both settings". **Better fix**: compare within Family B, where five tasks of different chain length (3 to 5 operators) and different success rates exist. If the code-side rate is flat *within* Family B across tasks, that supports the invariance claim without the cross-family confounds.

## Smaller points

- **The independence caveat is stated but the consequence is not.** §3.7 says p-values are anti-conservative. Since the paper's central claims are *negative* (no difference where one was predicted) and *directional* (code lower than prose), the anti-conservatism matters less than it would for a positive claim — but this should be said, because a reader may over-penalise.
- **`top_logprobs=20` is a bound.** If the true best alternative is outside the top 20, the reported probability for the sampled token is unaffected (it is a direct logprob), but any analysis of "what the model almost said" would be truncated. The paper does not do such analysis, so this is fine — but §3.2 should note that the 20-candidate list is not used in any reported statistic.
- **`temperature=1.0`** is stated. Note that logprobs are reported for the *sampled* sequence, so the distribution is the model's, not the sampler's; this is worth one sentence to pre-empt confusion.

## Judgment

The core measurement is sound and the disclosure is exemplary. The two required revisions are about over-claiming: the region split carries more interpretation than it can bear without a comment-excluded recomputation, and the cross-family comparison carries four confounds. Neither threatens the headline; both affect how strongly it can be stated.
