# Review — Methodology

**Recommendation:** Major revision (one item), otherwise sound.

## What is done well

**Pre-flight verification before committing to the design.** The 448/448 result on the reply-body probe is exactly the right gate: the entire study rests on one assumption, and the authors tested it in one API call before spending on 128 runs. Many papers in this space would have discovered the assumption was false after collecting.

**Two boundaries, reported both.** Given that the first pass failed precisely because of a single-boundary classification, reporting loose and strict side by side is the correct remedy, and the fact that they agree (0.49/0.26 against 0.84) is the strongest evidence in the paper. This is what makes the RQ1 result credible rather than another artefact.

**Exact tests rather than asymptotic.** Fisher's exact test on unbalanced cells is right; a χ² would have been wrong here.

**The intervention was implemented before collection, not after.** The plan's two conditions exist in code and in the data. The negative result is therefore informative.

## Required revision

**Contamination of `after_failure` and task identity.** `after_failure` is 3.2% of tokens, and those tokens are not distributed evenly across tasks: some tasks fail more often than others, so "after_failure rate is 4.30%" may partly be "the failing tasks have a higher rate for other reasons". The paper reports per-task rates (§4.4) and per-context rates (§4.2) but not the cross-tabulation. The authors should either report the context × task table or note that task composition of the failure cells is not controlled.

**Independence of tokens within a sequence.** Fisher's exact test assumes independent observations. Tokens inside one completion are plainly not independent — a low-probability token early in a regenerated block may raise or lower the next. The reported p-values are therefore anti-conservative. The direction and magnitude of the effect are large enough that this likely does not flip the conclusion, but the paper should either (a) report a cluster-level test with run as the unit, or (b) state the assumption explicitly as a limitation.

**Threshold sensitivity.** Only τ = −2 is used for the headline. The plan specified τ ∈ {−2, −5}. A sentence on τ = −5 would cost nothing and removes the appearance of threshold shopping.

## Smaller points

- The "structural tokens" comparison in §4.1 (1.54% vs 0.81%) is interesting and underdeveloped — it suggests "no choice" positions carry more uncertainty than "domain-specific name" positions, which cuts against the paper's own framing in a productive way. Either develop it or move it to the discussion.
- §3.2's formalisation is good but the termination rule "correctness after at least one revision" means every successful run has ≥ 2 turns by construction; this should be stated, since it affects the turn distribution.
- Report the run that failed to produce a fenced block. Currently it is mentioned in §3.8 only as a count.

## Judgment

The core measurement is sound and the robustness check is better than average. The p-values need an independence caveat, and the context × task cross-tabulation should be added. Neither threatens the headline.
