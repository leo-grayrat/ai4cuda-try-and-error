# Stage 2.5 — Integrity Check

**Method**: every headline number in `stage2_write/paper.md` recomputed from the raw `turns.jsonl` records, independently of the scripts that produced the draft.

## Numeric verification

| Quantity in paper | Recomputed | Status |
|---|---|---|
| Family A tokens = 98,533 | 98,533 | OK |
| Family B code tokens = 322,375 | 322,375 | OK |
| Family B prose tokens = 220,236 | 220,236 | OK |
| code $R_{-2}$ = 0.60% | 0.60% | OK |
| prose $R_{-2}$ = 13.56% | 13.56% | OK |
| density ratio 22.4× | 22.4× | OK |
| min code logprob −13.02 | −13.016 | OK |
| min prose logprob −16.65 | −16.647 | OK |
| min code as probability 2.2×10⁻⁶ | 2.23×10⁻⁶ | OK |
| min prose as probability 5.9×10⁻⁸ | 5.89×10⁻⁸ | OK |

## Claims checked against source

**The quoted causal chain.** §1.1 quotes: "less than 0.01% of pretraining data", "severe domain distribution mismatch", "$\pi_\theta(a_t \mid s_t) \approx 10^{-9}$", "even small train-rollout precision differences can explode the importance ratio". All four appear verbatim in the technical blog [2] and the arXiv version [1, Sec. 5.1–5.2]. The quotation marks in the paper are on the correct phrases.

**The entropy observation (§2.1).** The claim that the paper reports an *association* between reward collapse and entropy rise (when RFT is removed), and that this association does not establish a single cause, is attributed to an independent overview [11]. That overview states it explicitly. **This is a third-party reading, not the authors' own wording**; the paper says so in the attribution.

**DR. KERNEL's numbers (§2.2).** The geometric MRS band $[0.999, 1.001]$, the token-level veto threshold $10^{-4}$, and the conclusion "stabilizes the training dynamics" but "does not fundamentally lift the performance ceiling" are all quoted from [3, Sec. 5.1].

**KernelBench Level 2 identity (§3.3).** The five chain descriptions were read from the task files in `.cache/KernelBench/KernelBench/level2/` and match the filenames.

## Claims that are explicitly NOT verified

1. **The 0.01% pretraining share.** Not checked — no public corpus statistics support or refute it. The paper treats it as the source's premise (chain link (i)), not as an established fact.
2. **The 10⁻⁹ value itself.** We cannot verify that the source ever measured it. §2.1 notes that what the paper reports is entropy rising; the 10⁻⁹ figure appears in the *argument*, not in a reported measurement. The paper's phrasing reflects this: "the claim that..." not "the measured value of...".
3. **Base-model probabilities.** Not measured. Explicitly flagged in §5.2 and §6.1.

## Structural checks

- Every RQ in §1.4 has a corresponding subsection in §4 (RQ1→4.1, RQ2→4.2, RQ3→4.3, RQ4→4.4).
- Every claim in the abstract appears in the body with the same number.
- All 9 limitations are referenced somewhere in the text (§6.1→§5.2, §6.4→§4.6, §6.5→§3.7, §6.8→§3.6, §6.9→§5.4).
- No figure is referenced that does not exist (this version has no figures; the numbers are all in tables).

## Verdict

**PASS.** All recomputable numbers match. The three unverifiable items are flagged as premises or open questions rather than presented as findings.
