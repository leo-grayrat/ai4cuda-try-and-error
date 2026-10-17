# Stage 4.5 — Final Integrity Check

**Method**: independent recomputation of every number in the revised manuscript, from the raw `turns.jsonl` records, with a fresh script that does not reuse any intermediate output from earlier stages. The point is to catch errors that revision may have introduced, not to confirm what Stage 2.5 already found.

## Token totals

| Quantity | Recomputed | In paper | Status |
|---|---|---|---|
| Family A tokens | 98,533 | 98,533 | OK |
| Family B code tokens (all) | 322,375 | 322,375 | OK |
| Family B prose tokens | 220,236 | 220,236 | OK |
| Family B pure code (comments excluded) | 310,935 | 310,935 | OK |

## Rates

| Quantity | Recomputed | In paper | Status |
|---|---|---|---|
| code $R_{-2}$ (all code) | 0.60% | 0.60% | OK |
| **pure code $R_{-2}$** | **0.33%** | **0.33%** | OK |
| prose $R_{-2}$ | 13.56% | 13.56% | OK |
| **contrast (pure code)** | **40.7×** | **41×** | OK (rounded) |

## Per-task pure-code rates

| Task | Recomputed | In paper §4.2 / §4.3 |
|---|---|---|
| `12_gemm_multiply_leakyrelu` | 0.18% | 0.18% |
| `1_conv2d_relu_biasadd` | 0.37% | 0.37% |
| `14_gemm_divide_sum_scaling` | 0.44% | 0.44% |
| `17_conv2d_instancenorm_divide` | 0.26% | 0.26% |
| `19_convtranspose2d_gelu_groupnorm` | 0.42% | 0.42% |

All five match, and the two tables in the paper (§4.2 ordered by name, §4.3 ordered by chain length with success counts) carry the same values.

## Minima

| Quantity | Recomputed | In paper | Status |
|---|---|---|---|
| min code logprob | −13.016 | −13.02 | OK |
| min prose logprob | −16.647 | −16.65 | OK |
| min code as probability | 2.23×10⁻⁶ | 2.2×10⁻⁶ | OK |
| min prose as probability | 5.89×10⁻⁸ | 5.9×10⁻⁸ | OK |
| 10⁻⁹ as logprob | — | −20.7 | OK (identity) |

## Structural checks on the revision

- The claim "whatever 10⁻⁹ measures, it is not a property of the domain's vocabulary" appears in the abstract, §5.1 and §7, and nowhere is the older formulation ("the second link does not hold") still asserted as a conclusion.
- §4.3's four confounds are named **before** the within-family table, so a reader encounters the justification for the change of comparison before the comparison.
- §5.4's disclaimer is the section's first sentence; the competing account and both disconfirming experiments follow.
- The failure-dominance label (87.5%) appears in the abstract and §6.4, and §4.2's table sits directly after the sentence that introduces Family B's composition.
- No dangling reference: the two experiments named in §5.4 are not referenced anywhere as if run.

## One thing this check cannot cover

The manual classifications in §4.7 (816 + 189 tokens) are not recomputable by script — that is the point of them. Their correctness rests on the reading being re-checkable by a human: the sample files (`results/lowprob-contexts.txt`, `results/kb2-code-noncomment-extremes.txt`) are committed alongside so that any reader can draw 30 tokens at random and compare against the classification table. We recommend a journal do exactly that.

## Verdict

**PASS — zero discrepancies.** Every recomputable number in the revised manuscript matches an independent recomputation. The revision's two new figures (0.33%, 41×) are correct, and the per-task values are consistent between the two tables that carry them.
