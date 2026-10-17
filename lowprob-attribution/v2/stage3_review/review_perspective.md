# Review — Perspective (cross-domain)

**Recommendation:** Accept; the transferable finding is the one about where cost lives, not the one about CUDA.

## What generalises

Strip the domain and the paper says: inside a multi-turn agent loop, conditional token probabilities are governed by *which kind of text is being produced* — templated output versus composed reasoning — and by an order of magnitude more than by any lexical property of the domain.

That is a statement about agentic RL as a text-composition problem. It should transfer to SQL generation, LaTeX authoring, CAD scripting, infrastructure-as-code — any setting where an agent alternates between writing a constrained artefact and explaining to itself why the last attempt failed. In all of them, the artefact is templated and the self-explanation is not.

The paper's strongest single number is not the 22.4× density ratio; it is the **probability-mass table** (§4.5):

```
code :  worst 5% of tokens carry 98.1% of the region's cost
prose:  worst 5% of tokens carry 30.3%
```

That is a signature difference in kind, not degree. Code has a "template plus occasional collapse" distribution; prose has a "every word costs a little" distribution. Any theory of why agentic RL is unstable on long trajectories has to explain why the cost concentrates the way it does, and this table is the constraint.

## What does not generalise, and the paper should say so

The specific claim "0.60% vs 13.56%" is one model on one API. The paper says this. But it does not say the more useful thing: **the ratio, not the level, is the finding.** A different model would give different absolute rates; the 22× ratio between regions is the thing that would have to hold for the account in §5.3 to be about something real.

I would encourage the authors to state the finding as "the cost is dominated by region, not by vocabulary" rather than as specific percentages. The percentages will not replicate; the ordering might.

## On §5.4 (the alternative account)

This is the part with the most reach and the least support. As a cross-domain reviewer I find it plausible: the failure-text-dominates-token-volume observation is common experience, and the RFT explanation is elegant — RFT keeps successes, successes have little debug prose, so RFT removes the high-cost region.

But I want to flag a specific weakness. **The account predicts that prose volume and prose cost co-vary with failure. The data show that they do — but they do not show direction.** It is equally possible that (i) failure → more prose → cost, or (ii) an unstable policy → more prose regardless of success → cost. The paper's `after_success` sample (12,147 tokens) is too small to distinguish them. This should be named as the account's weakest joint, not left implicit.

## On the methodological disclosures

§5.5 and §5.6 are the most transferable prose in the paper. The bias mechanism in §5.5 — a defensible exclusion ("don't mislabel English prose as code") silently removing 96% of the target class — is a failure mode that any measurement paper can fall into. The remedy ("vary the definition boundary, see if the conclusion survives") is one sentence and generalises completely.

§5.6 is a different kind of disclosure: a driver bug that made a third of the runs meaningless, reported with the corrected number. Most papers would report 28% and never mention 17.5%. Reporting both is what makes the rest credible.

## Style

Dense but readable. §4.7 is the best-written section; §5.1 and §7 are the weakest (they repeat each other). The abstract packs too much into one paragraph — the selection effect (87.5% after-failure) is buried there without comment and should either be qualified or moved.

## Judgment

Accept. Reframe the headline as "region dominates vocabulary" and mark §5.4's direction-of-causation as open.
