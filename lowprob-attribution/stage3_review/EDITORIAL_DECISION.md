# Editorial Decision

**Manuscript:** Is CUDA Syntax Rare Enough to Break PPO? A Conditional-Probability Test of a Stated Root Cause
**Date:** 2026-10-17

## Reviews received

| Reviewer | Recommendation |
|---|---|
| EIC | Minor revision |
| Methodology | Major revision (one item) |
| Domain | Accept, minor revision |
| Perspective | Accept |
| Devil's Advocate | Publish, but the headline is weaker than presented |

**Consolidated decision: Minor revision.** No reviewer disputes the core measurement or its direction. The required changes are about scope of claim, one statistical caveat, and one analysis that the existing data already supports.

## Points of agreement across reviewers

**1. The base-model control is the load-bearing gap.** The EIC, Domain and Devil's Advocate all raise it independently. The measurement is on an aligned model; the claim is about a base model during rollout. The paper acknowledges the gap in §5.2 but the abstract does not carry it.

**2. The task set bounds the generalisation.** Domain: reduction kernels have one structure, so the vocabulary is templated *by construction*; the source's motivating examples (fusion, BN folding) require choosing an organisation, and there the token sequences share little surface form. The paper's §6.3 says this; the abstract does not.

**3. The p-values need an independence caveat.** Methodology and Devil's Advocate agree. Tokens within a completion are dependent; Fisher's exact test on 98,533 observations is anti-conservative.

## Points of disagreement, and how they were resolved

**Devil's Advocate's third objection — token-level rates are a proxy for sequence-level quantities.** This was the strongest analytical objection, and it turned out to be answerable from data already collected, without new runs. Computed over all 257 completions:

```
CUDA tokens   k=2,344   Σ log π = −31.6     per token −0.0135
other tokens  k=96,189  Σ log π = −5,978.0  per token −0.0621

CUDA share of tokens:              2.38%
CUDA share of negative log-prob:   0.53%
```

**CUDA vocabulary accounts for 2.38% of tokens but only 0.53% of the total negative log-probability.** At the level that actually feeds $\rho_t(\theta)$, the CUDA vocabulary is not the drag; the surrounding text is. The objection was correct that the original paper reported a proxy — and resolving it strengthens rather than weakens the conclusion. This is now §4.7.

**Perspective's suggestion — reframe as a general claim about agentic context distribution.** Adopted in the abstract's closing sentence, with CUDA named as the measured instance rather than as the scope of the claim. Note this cuts against the Domain reviewer's preference for a narrower CUDA-specific framing; we follow Perspective because the evidence (context effect ≫ vocabulary effect) is itself domain-general, and the CUDA-specific framing would understate what was found.

## Required revisions

1. **Abstract**: state in the first three sentences that the measurement is on an aligned model at inference, and that the base-model control is outstanding. Remove the implication that the question is closed.
2. **Title**: the current title asks a yes/no question the paper answers conditionally. Either add a subtitle marking the condition, or state the answer in the title.
3. **Independence caveat**: add to §3.7 that the exact test assumes independence, that tokens within a completion are dependent, and that the p-values are therefore anti-conservative — with the note that the effect size is large enough that the sign is not in question.
4. **§4.7 (new)**: sequence-level attribution, per the numbers above.
5. **Context × task cross-tabulation** (Methodology): report or explicitly note as uncontrolled, since `after_failure` is only 3.2% of tokens and task composition within it is unverified.
6. **Promote the limitation** that task scope bounds the claim: elementwise and reduction only, no fusion or tiling, so the measured vocabulary is the mechanical layer.
7. **Develop or relocate** the structural-token comparison (§4.1): "no-choice" positions showing a *higher* rate than domain-name positions cuts against the paper's own framing and deserves a paragraph rather than a sentence.

## Not required, but noted as the highest-value follow-up

The base-model control (§5.2) and the "does failure matter, or does rewriting matter" third condition (Perspective, Devil's Advocate). Neither is feasible here; both are named in the paper as the experiments that would settle the open questions.

## Note on §5.5

Two reviewers flagged the inclusion of the retracted first pass as unusual. Both concluded it should stay: it explains why the paper reports two vocabulary boundaries, and the failure — a defensible exclusion that silently narrowed the measured class by 96% — is transferable to other measurement work.
