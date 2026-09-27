# Figure production standard

These diagrams follow Nature's figure-preparation guidance while retaining Novogaia branding in the presentation frame. Journal-sized figure exports and presentation-sized artwork serve different viewing distances; check typography at each actual output size. The receiving journal's editors and production team decide the final layout and any remaining changes.

The current collection contains **27 figures: 26 TikZ exports and one analytic plot**. TikZ pages are **183 × 60.047 mm**; the analytic plot is **89 × 43 mm**. Figure 9 is reserved for that plot (source slide 15); all other figure numbers follow the source-slide order recorded by `export_figures.py` and `figures/manifest.json`. The [figure index and legends](figures/README.md) distinguish explanatory diagrams, proposed tests and the measured retrieval result.

## Standalone artwork

- Use 89 mm or 183 mm width, choosing the smallest readable layout; keep height at or below 170 mm. Arrange related panels in reading order with balanced spacing. These dimensions follow the current [Nature figure-building guide](https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/).
- Set ordinary labels to 5–7 pt at final size; use 8 pt bold upright lowercase letters for panel labels. Use Liberation Sans, an Arial-compatible substitute, without identifying it as Arial. Keep mathematical glyphs selectable and inspect their embedded fonts separately. Nature prefers standard sans-serif text and editable, embedded fonts. [Typography specifications](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/).
- Export editable vector PDF, with source TikZ/Python and SVG when available. Embed TrueType fonts; use Matplotlib `pdf.fonttype = 42`, `ps.fonttype = 42`, and `svg.fonttype = "none"`. Do not convert text into paths or use Type 3 lettering. [Export specifications](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/).
- Set strokes to 0.25–1 pt at final size, including scaled arrow shafts and molecular bonds. This range is explicit in Nature's [Extended Data guide, page 4](https://www.nature.com/documents/Extended_Data_guide.pdf).
- Use white standalone backgrounds, black labels, and an accessible, restrained palette. Preserve distinctions with labels, symbols or line styles. Avoid background gridlines, decorative icons, shadows, busy fills and overlapping text. A semantic node boundary is permitted; “do not outline text” means retain characters as text, not remove every diagram box. [Graph and text specifications](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/).

## Scientific interpretation

- Give quantitative axes tick marks, quantity names and units. Label normalized quantities “dimensionless” and constructed scales “arbitrary units” where appropriate. Spectral axes identify mass-to-charge ratio and relative intensity; schematic spectra remain explicitly illustrative.
- Define every arrow: computation, sampling, comparison or parameter update. Frozen teachers must not appear to receive learning updates. A pair representation is a directed index relation, not evidence of a physical fragmentation arrow or a chemical bond.
- Define matrix rows and columns. A bond matrix needs a categorical bond-type key, not a continuous intensity scale. A scalar compatibility heatmap needs its own scale and a statement that the example is synthetic. A grid of vector-valued pair states must not imply measured scalar values through unexplained shading.
- Make coarse probability bins equal the sums of their fine bins. Use consistent sign conventions for mass differences and show the corresponding endpoint order.
- Preserve the distinction between a probability distribution and illustrative diagram shading. The revised pair-distogram example uses eight synthetic probabilities, **0.02, 0.03, 0.08, 0.25, 0.40, 0.15, 0.05, 0.02**, summing to one, with vertical ticks at **0, 0.2, 0.4**. Its horizontal bins are schematic; they do not change the 2,000-bin production head. The diagnostics grid denotes independent proposed experiments, not quantitative scores.
- Describe analytic plots as analytic solutions, with assumptions and parameters. They have no experimental sample size or error bars. For measured results, report the unit of observation, sample count, evaluation protocol and any uncertainty definition. Define statistics and error bars in the legend. [Nature formatting and figure-legend guidance](https://www.nature.com/nature/for-authors/formatting-guide).
- Keep the result scope explicit: **36.94% Hit@1**, **67.07% Hit@5**, **78.61% Hit@10**, **88.89% Hit@20**, **0.50814 MRR**, **6.14830 mean MCES@1**, and **median rank 2** on **17,556 test spectra** and supplied truth-containing formula pools. Show only the latest Global context + GLACIER results. The pair-aware extension remains an unevaluated dashed path.

## Production checks

1. Rebuild from the saved source using the repository's `.venv/bin/python` for Python scripts and the documented LaTeX command.
2. Check PDF dimensions and actual label sizes at the intended 89/183 mm width; measure the exported strokes after scaling.
3. Inspect `pdffonts`: fonts embedded, no Type 3 diagram lettering. Extract text with `pdftotext` and confirm Greek letters, subscripts, labels and minus signs remain readable. Inspect SVG for live text where SVG is supplied.
4. Inspect `pdfimages -list`: scientific diagrams remain vector artwork. Render every slide at 180 dpi or higher, then inspect labels, equation baselines, box edges, connector endpoints and clearance above the takeaway. Inspect each standalone export separately because its font metrics differ. Page-boundary checks alone do not detect internal overlaps or misalignment.
5. Verify legend semantics, axis units, matrix orientation, probability aggregation and synthetic-versus-measured labels against the equations and cited source.
6. Keep standalone exports, their editable sources, citations and any data used to generate them under `presentations/`.
7. Check that `figures/README.md` lists all 27 figures in manifest order, including the reserved analytic figure 9. Recheck legends and probability values when simplifying slide text or replacing prose with diagrams; shorter slides still need complete scientific definitions in their standalone legends.

The [Nature final-submission page](https://www.nature.com/nature/for-authors/final-submission) redirected to the publisher's cookie service during this review. The directly accessible official figure guides above supply the technical specifications used here; this document records a production standard rather than journal acceptance.
