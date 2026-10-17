# Review — Domain (CUDA / RL for code generation)

**Recommendation:** Accept with minor revision.

## Does the paper represent the source correctly?

Yes, and this matters because a critique that misstates its target is worthless. The paper quotes the actual mechanism — "less than 0.01% of pretraining data", "$\pi_\theta(a_t\mid s_t) \approx 10^{-9}$", "even small train-rollout precision differences can explode the importance ratio" — and correctly identifies BF16-vs-FP16 as the amplifier. It also correctly notes that links (iii) and (iv) are outside its scope. I checked the arXiv version and the technical blog; both match the quoted text.

One precision point: the source's claim is about *sampled* token sequences, and it frames the problem as distribution mismatch rather than as a property of individual tokens. The paper's RQ1 phrasing ("is CUDA vocabulary more likely to be low-probability") is a fair operationalisation, but §1.1 could note that the source is describing a *sequence-level* sampling phenomenon, and that measuring marginal token-level rates is a proxy for it. It is a good proxy, but it is a proxy.

## Is the alternative explanation (DR. KERNEL) represented correctly?

Yes. The geometric MRS formulation, the $[0.999, 1.001]$ band, the $10^{-4}$ token-level veto, and the "stabilises but does not lift the ceiling" conclusion are all accurate. The paper's use of this is legitimate: it does not claim DR. KERNEL supports its specific finding, only that both point away from vocabulary.

## Substantive domain concern

**The tasks do not exercise the part of CUDA where the interesting decisions live.** All four tasks are elementwise or reduction. A Triton kernel for row-reduction has essentially one structure: load a block, reduce, store. The vocabulary is templated *by construction* — which is exactly what the paper finds, and exactly what one would predict from the tasks rather than from the domain.

The source paper's motivating examples are different in kind: fusing two passes, folding BatchNorm into convolution weights, algebraic simplification of a diagonal matmul. Those kernels require choosing among genuinely different structures, and the token sequences that realise them are far less templated. It is entirely possible that on a fusion task, CUDA token probabilities are much lower — not because the tokens are rare, but because the *structure* is under-determined and there are many valid kernel organisations.

**This does not undermine the RQ1 result as stated**, because the result is about the vocabulary on the tasks measured. But the paper's framing ("CUDA vocabulary is not the low-probability tail") is broader than its evidence. The limitation §6.3 says this; the abstract and conclusion should carry it too.

## Where the paper is strongest domain-wise

The observation that `tl` is at logprob ≈ 0 while `.load` / `.zeros` / `.store` occasionally dip is the most valuable single detail. It says that the model is certain about *being in Triton* and occasionally uncertain about *which member of the API to call*. That distinction — module-level certainty, member-level variability — is a real and useful characterisation, and it is more informative than the aggregate rate.

I would encourage the authors to develop it. It is currently one sentence in §5.1 and it deserves a paragraph: what fraction of API-name occurrences are followed by a member selection, and does the variability concentrate there?

## Judgment

Accurate representation of the source, correct use of the nearest neighbour, one framing overreach that the authors already acknowledge in limitations but should propagate upward. Accept with minor revision.
