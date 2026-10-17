# Stage 4 — Revision Notes

**Source of requirements:** `stage3_review/EDITORIAL_DECISION.md` (consolidated: Minor revision).

| # | Requirement | Status | Where |
|---|---|---|---|
| 1 | Abstract states the model condition and the outstanding control in the opening | **Done** | Abstract, sentence 4 (bolded) |
| 2 | Title marked as conditional | **Done** | Subtitle now reads "…on an Aligned Model, With the Base-Model Control Outstanding" |
| 3 | Independence caveat on the exact tests | **Done** | §3.7, new paragraph |
| 4 | Sequence-level attribution (§4.7, new) | **Done** | §4.7, computed from existing data |
| 5 | Context × task cross-tabulation | **Done** | §4.2, new table |
| 6 | Task-scope limitation promoted | **Done** | §6.3 rewritten; abstract now says the claim is conditional; §1.5 already flagged it |
| 7 | Structural-token comparison developed | **Done** | §4.1, two new paragraphs + a testable prediction |

## Notes on specific items

**Item 4 required no new collection.** The reviewer was right that a per-token rate is a proxy for the sequence-level quantity, and right that the data to compute the latter were already in `turns.jsonl`. Over 257 completions:

```
CUDA tokens   k=2,344   Σ log π = −31.6     per token −0.0135
other tokens  k=96,189  Σ log π = −5,978.0  per token −0.0621
CUDA share of tokens 2.38%;  share of negative log-prob 0.53%
```

The result strengthens the paper's position: at the level the causal chain operates on, the vocabulary contributes roughly one fifth of its proportional share to the probability cost.

**Item 5 produced the strongest single piece of evidence in the revision.** The context × task table shows `after_failure > initial > after_success` in **all four tasks**:

| Task | initial | after_success | after_failure |
|---|---|---|---|
| `layernorm_rows` | 0.49% | 0.17% | 4.54% |
| `row_reduce` | 1.22% | 0.65% | 10.69% |
| `softmax_rows` | 1.37% | 0.75% | 3.79% |
| `vector_affine` | 0.54% | 0.30% | 2.71% |

The failure cells are unevenly distributed (`softmax_rows` 43.7%, `row_reduce` 4.2%), but the ordering is invariant. This closes the "task composition may explain it" objection.

**Item 7 was adopted as an interpretation, flagged as post-hoc.** The structural-token inversion (`for` at 1.54% versus the domain vocabulary at 0.26%) is the single most interesting number in the paper after the headline, and it cuts against the paper's own framing in a productive way. It is now given a paragraph, an explanation (the uncertainty sits in the decision the token realises, not in the token), and a prediction (positions following a structural keyword should be more variable than positions following a module prefix). It is explicitly marked as a post-hoc reading of a comparison not designed to test it.

## Items not addressed, with reasons

**Cluster-level reanalysis (§3.7 caveat).** Acknowledged in text rather than performed. Doing it properly means resampling at the run level with a statistic that respects the nested structure; a quick permutation over runs would be easy to compute badly and would add a number without adding evidence. The caveat is stated instead.

**Base-model control.** Requires weights we do not have. Named in the title, the abstract, §5.2 and Appendix A.

**Third condition to separate "told it failed" from "must rewrite".** Requires new collection. Named in §6.9 and §5.4.

## One requirement declined

The Domain reviewer preferred a narrower, CUDA-specific framing; the Perspective reviewer preferred a broader claim about agentic RL context distribution. These conflict. We follow Perspective, because the evidence itself is domain-general (the context effect is an order of magnitude larger than the vocabulary effect, and the vocabulary effect is *negative*), and because the CUDA-specific framing would understate what was measured. The Domain reviewer's concern is preserved as an explicit limitation (§6.3) rather than as a framing choice.

## Diff summary

- Title: subtitle added
- Abstract: rewritten; model condition stated in sentence 4; sequence-level figure added; closing sentence generalised
- §3.7: independence caveat (new paragraph)
- §4.1: structural-token paragraph expanded into a discussion + prediction
- §4.2: context × task table added
- §4.7: new section (sequence-level attribution)
- §6: expanded from 7 to 9 items (task scope rewritten; statistical dependence and the rewrite-alternative added)
- References: unchanged (11 entries)
