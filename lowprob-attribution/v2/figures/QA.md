# Figure and document verification

Date: 2026-10-18

The four figures were reviewed as exported images and within the manuscript pages. They use a white background, a restrained indigo/rose palette, Arial-compatible sans-serif text, small lowercase panel labels and consistent regional colour mapping. Code and prose are additionally distinguished by solid/dashed curves and filled-circle/open-square marks. Panels use unequal widths where the supporting comparison needs less space.

The input record totals reproduce 98,533 tokens for Family A and 542,611 for Family B. The latter comprises 322,375 code tokens and 220,236 prose tokens under the fence rule. Figure rates, regional minima, context counts and concentration at the 5% cutoff reproduce the reported table values. `figure-data.json` records these outputs and input hashes. The plots use all recorded token probabilities, with no random sampling, artificial values or fitted smoothing.

The figure generator passed the nature-figure source preflight with 14 passes, zero warnings and zero failures. `preflight.json` retains the report. SVG exports contain editable text elements, and PDF plot text can be extracted. Raster exports use 600 dpi; TIFF files use lossless LZW compression.

The final manuscript PDF has 21 pages, four numbered figures and 12 displayed equation blocks. Its compilation log has no undefined references, missing characters or overfull boxes. The Word document contains four embedded images and 12 native display-equation blocks; its final headless rendering has 22 pages after the reference revision. Its table grids and cell widths were explicitly set within the page margins and inspected after rendering.

The full PDF page layout and selected equation, figure and table pages were inspected during revision. Final PDF and Word figure pages were checked again after the final caption update. The machine-readable document check is saved as `../stage5_finalize/prose-figure-verification.json`.
