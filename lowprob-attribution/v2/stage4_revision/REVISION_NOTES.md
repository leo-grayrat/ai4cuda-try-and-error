# Stage 4 — Revision Notes

Source: `stage3_review/EDITORIAL_DECISION.md` (8 required items, all computational or textual).

| # | Requirement | Status | Where |
|---|---|---|---|
| 1 | Narrow the claim: "does not hold" → "not a property of the vocabulary" | Done | Abstract, §5.1, §7 |
| 2 | Recompute the code rate with comments excluded | Done | §4.2 (new table) |
| 3 | Replace the cross-family comparison with a within-Family-B one | Done | §4.3 (rewritten) |
| 4 | Move the §5.4 disclaimer to the start; add symptom-vs-cause | Done | §5.4 |
| 5 | Label the Family B corpus as failure-dominated wherever quoted | Done | Abstract, §4.2, §6.4 |
| 6 | Sharpen §6.3: trivial vs non-obvious organisational choice | Done | §6.3 (rewritten) |
| 7 | Tighten the subtitle | Done | Title |
| 8 | Trim §7 by a third | Done | §7 (rewritten, ~300 words) |

## Item 2 was the most consequential

Recomputing with in-block comments excluded:

| Task | all code | pure code |
|---|---|---|
| `12_gemm_multiply_leakyrelu` | 0.25% | 0.18% |
| `14_gemm_divide_sum_scaling` | 0.93% | 0.44% |
| `17_conv2d_instancenorm_divide` | 0.58% | 0.26% |
| `19_convtranspose2d_gelu_groupnorm` | 0.73% | 0.42% |
| `1_conv2d_relu_biasadd` | 0.52% | 0.37% |
| **total (310,935)** | 0.60% | **0.33%** |

Every task drops. The contrast against prose rises from **22.4× to 41×**. The Devil's Advocate predicted this outcome and was right: the paper was understating its own finding by reporting a blend.

## Item 3 changed what can be claimed

The cross-family comparison had four confounds (task construction, turn cap, success rate, context composition). The within-family table removes them:

| Task | success | pure-code rate |
|---|---|---|
| `12_gemm_multiply_leakyrelu` | 5/8 | 0.18% |
| `1_conv2d_relu_biasadd` | 0/8 | 0.37% |
| `14_gemm_divide_sum_scaling` | 1/8 | 0.44% |
| `17_conv2d_instancenorm_divide` | 1/8 | 0.26% |
| `19_convtranspose2d_gelu_groupnorm` | 0/8 | 0.42% |

Success rate ranges from 0/8 to 5/8 and the code-side density does not track it. **This is a cleaner statement of invariance than the original cross-family claim**, and it rests entirely on data already in hand.

## Item 4 added a competing account the paper had omitted

The Devil's Advocate pointed out that prose may be a **symptom** of confusion rather than a **source** of instability. Added, together with the two experiments that would separate them: vary diagnostic formatting while holding content fixed; and request a rewrite after a success so that prose length is held constant without the bad-news content.

The §5.4 disclaimer was also moved from the end of the section to its first sentence, since a disclaimer placed after three confident paragraphs does not undo their framing.

## Item 6 is a limitation the paper should have had from the start

The Domain reviewer observed that KernelBench Level 2 chains are the *easy* kind of fusion: a heavy op plus elementwise passes, where the decision is "epilogue or second pass" and both are straightforward. The hard organisational decisions — shared-memory tiling, a transpose that changes access pattern, split-K, online softmax — are absent from all five tasks. §6.3 now says this, and says that the RQ3 null result is therefore about the easy kind of fusion.

## Items not changed, with reasons

**Comments inside fenced blocks are still counted as code in the primary table.** §4.2 now reports both numbers rather than replacing one with the other, because the "all code" number is the one comparable to v1's Family A figure (which used the same rule). Reporting both avoids a silent definition change between papers.

**No new collection.** Every required item was computational or textual. The two open questions the reviews raised — base-model control, and cause-vs-symptom for prose — require data we cannot collect here, and both are named in the text rather than papered over.

## Diff summary

- Title: subtitle rewritten to name what was tested
- Abstract: narrowed claim, added 0.33%/41×, labelled failure dominance
- §4.2: new pure-code table
- §4.3: rewritten as within-family comparison
- §5.1: added the alignment-cannot-change-corpus argument
- §5.4: disclaimer moved to first sentence; symptom-vs-cause added; two disconfirming experiments named
- §6.3: rewritten around trivial vs non-obvious organisational choice
- §7: rewritten, ~300 words, no restatement of §5.1
