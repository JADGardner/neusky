# Static figure bases

Hand-drawn static diagram bases for the NeuSky method figures, copied
2026-07-02 from the thesis assets at
`latex/9_Chapter3/paper_assets/neusky_tpami/figures/` (phd repo). The thesis
originally overlaid the label text on these bases with TikZ; the
`scripts/figures/fig_*.py` scripts bake that label text into generated
`publication/figures/<name>_labeled.{png,pdf}` outputs instead, so LaTeX can
use a plain `\includegraphics`.

| File | Used by | Notes |
|------|---------|-------|
| `technical_overview.pdf` | `fig_technical_overview.py` | Vector base of the method overview diagram (thesis `fig:technical_overview`). |
| `technical_overview_base.png` | `fig_technical_overview.py` | Pre-rasterized fallback of `technical_overview.pdf` (pymupdf, 4x zoom, 3590x1359 px, ~590 dpi at thesis display width). Used when neither pymupdf nor `pdftoppm` is available (e.g. the research container). |
| `implicit_visibility_explainer.svg` | `fig_implicit_visibility.py` | Active vector base of the outside-in visibility explainer (thesis `fig:implicit_visibility`); only the 600x600 rabbit rendering is embedded as raster data. |
| `implicit_visibility_explainer.png` | -- | Legacy fully rasterised base retained for comparison; no longer used by the figure script. |

Label positions/sizes in the scripts were verified against the compiled
thesis (`latex/build/thesis.pdf`): TikZ coordinates are in cm from the image
centre (y up), 1 cm = 28.3465 bp, thesis text width = 435.84 bp, `\tiny` at
the 12pt class = 5.98 bp.
