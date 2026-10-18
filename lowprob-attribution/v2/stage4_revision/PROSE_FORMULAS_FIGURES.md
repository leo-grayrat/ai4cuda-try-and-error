# Prose, formulas and figures revision

Date: 2026-10-18

The manuscript has been reread and rewritten from the abstract through the conclusion. The revision replaces fragmentary labels, emphatic standalone verdicts, conversational instructions to the reader and repeated rebuttal framing with connected explanations. It also removes the exchange about an assistant's later zero-code search from the scientific results; the detailed reconstruction remains in the local evidence report.

Definitions of the importance ratio, geometric mismatch ratio, generated-token log-probability, conversation update, threshold rate, regional cost and concentration are displayed separately. The probability conversions now use an aligned mathematical display instead of a code block. The document contains 12 displayed equation blocks, with short variables and scalar values retained inline.

Four two-panel figures are integrated into the Results section, with captions and text references:

1. Vocabulary comparisons and the empirical cumulative distribution for Family A.
2. Per-task code/prose rates and regional cumulative distributions for Family B.
3. Concentration of negative log-probability and observed regional minima.
4. Context composition and the corresponding low-probability rates.

The figure workflow uses Python and the nature-figure skill. The existing Python plotting source was adapted for style while retaining the raw-record calculations. The visual contract is a quantitative comparison layout with one principal comparison and a supporting distribution or composition panel. The restrained palette consistently maps code to indigo and prose to muted rose; line styles and marker shapes provide an additional distinction. Exports include editable SVG, vector PDF, PNG and compressed 600-dpi TIFF.

The plotting script reads all recorded content tokens with a log-probability from the two saved task collections. It reproduces the manuscript's vocabulary and fence-state rules, task categories and context grouping. Source file hashes and the computed plot metrics are saved in `../figures/figure-data.json`. No sampling or smoothing is used. Empirical CDF axes use strictly positive cumulative shares; minima are exponentiated recorded log-probabilities. The figures describe pooled token distributions and plot their observed values directly.

The old manuscript is retained as `paper.before-prose-revision-2026-10-18.md`. The current editable source is `../stage2_write/paper.md`; the reading copy, LaTeX PDF and Word document are regenerated in `../stage5_finalize/`.
