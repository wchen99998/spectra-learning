# Page review: where formula-candidate retrieval fails

Reviewed at 180 dpi after compilation, using the contact sheets for the narrative and individual renders for dense pages. The deck uses the shared unbranded theme and fonts and 16:9 grid. Charts are vector matplotlib PDFs with live Liberation Sans text and transparent backgrounds. Molecules are TikZ generated from RDKit coordinates. Every chart shows measured values.

| Page | Visual and content check |
|---|---|
| 1 | Title and subtitle separated after shortening the title. 47.33% is labelled full-test, with validation and the clustered interval beside it. Four headline numbers share one baseline. |
| 2 | Three protocol columns, then a clustering explanation. The takeaway was qualified after review to "replicate on validation; single bins can differ". |
| 3 | Four panels share the 0–90 axis and the dashed overall line. Counts moved from inside the bars to tick labels to avoid colliding with error bars. Validation shown as open circles. |
| 4 | Two 2×2 tables. Shading tracks accuracy; counts sit in each cell. The instrument-missing count was corrected to 409 after checking. |
| 5 | Forest plot on a log axis with a reference line at 1. The legend was moved to clear the lowest row. Text on the right matches the macros. |
| 6 | Train and test distributions on shared bins. The text was shortened to two lines to clear the takeaway, and the reweighting result moved to the source line. |
| 7 | Decoy-similarity and train-similarity panels share axes. Tick labels carry counts. |
| 8 | Two measured bars on a shared scale, a large 0.4% figure and the chance rate in the source. |
| 9 | Truth-rank bars and a top-1 similarity histogram; bins at 0.7 and above are black. The x-axis end label was fixed. |
| 10 | Three columns of truth and wrong top-1 structures, each fitted to a 4.3 × 1.62 cm box. Captions reflowed to three lines at 6.8 pt to stop them running into the next column. The fixed selection rule is in the source. |
| 11 | Grouped bars plus the "≥ truth" share. The takeaway gives the strict and tie split, and the source was shortened to two lines. The right-hand title was shortened to avoid clipping. |
| 12 | Cap-curve panels labelled directly instead of with legends, which had collided with the enumeration reference line. Figure height reduced; text fits in two lines. |
| 13 | Horizontal Δ bars for validation and test around zero; the control is quoted with its absolute values. |
| 14 | Agreement stacked bar and counterfactual bars with the current and 50% lines. The 50% label moved inside the plot after being clipped. |
| 15 | Taxonomy table with a manual break in "Fragment-indistinguishable" so it clears the share column. Levers are marked as proposals in the source. |
| 16 | Four numbered conclusions. The source states the uncertainty caveat on the 50% target. |
| 17 | Definitions list with source files. |

Automated audit (`build/layout-audit.json`): 17 pages, all fonts embedded, no raster images, no words outside the page, no overfull boxes or missing glyphs. Remaining LaTeX warnings are underfull boxes in ragged text.

Factual checks: every slide number is a generated macro. The hand-written statements are verified in the [evidence sheet](research/retrieval_failures_evidence.md). The recomputed ensemble reproduces 48.26% validation and 47.33% test exactly. No model or training code was changed for this deck.
