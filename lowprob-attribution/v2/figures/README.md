# Manuscript figure sources and exports

Regenerate all four figures from the saved response records:

```powershell
python lowprob-attribution/scripts/make_v2_figures.py
```

The generator writes SVG with editable text, vector PDF, 600-dpi PNG and compressed TIFF. The manuscript PDF uses the vector plots; Word embeds the PNG versions.

`figure-data.json` contains the computed rates, minima, context counts, concentration values and SHA-256 hashes of the original input files. The figures retain all recorded tokens with a log-probability in the two main collections. Their statistical definitions follow the manuscript tables, with complete empirical distributions used for the cumulative curves.

The panel layout, typography, colour family and export checks follow the nature-figure skill. The editable plotting source is `../../scripts/make_v2_figures.py`.
