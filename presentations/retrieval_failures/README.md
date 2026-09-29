# Where formula-candidate retrieval fails

A **17-slide, 16:9 Beamer presentation** (16 main slides, one definitions appendix) on the failure cases of the validation-selected 15-member retrieval ensemble on MassSpecGym formula-candidate retrieval: **47.33% test Hit@1** (48.26% validation). It reuses the shared unbranded theme and fonts, page grid and takeaway/source layout of the earlier retrieval decks.

- [Presentation PDF](retrieval_failures.pdf)
- [Editable deck](retrieval_failures.tex)
- [Speaker notes](retrieval_failures_notes.md)
- [Evidence sheet: every number and its source](research/retrieval_failures_evidence.md)
- [Page review](retrieval_failures_page_review.md)
- [Detailed written analysis](../../results/retrieval_failure_analysis_20260929/ANALYSIS.md)
- [Portable source bundle](retrieval_failures_source.zip)

## Story

1. **Protocol.** One fixed ensemble; spectrum-level Hit@1 with molecule-clustered bootstrap intervals (17,556 test spectra come from 2,998 molecules, so the 95% interval is 43.6–51.2%).
2. **Failure map.** Accuracy falls from 67.6% at 300–400 Da to 35.3% above 500 Da; [M+Na]⁺ 33.6%, QTOF 32.6%, 1–5 peaks 32.3%. Two-way tables and a logistic model show these effects are independent.
3. **Shift.** Half of test lies above 500 Da, where training has 17% of its spectra; QTOF is over-represented in training, so its weakness is not a shift.
4. **Pool difficulty.** 51% of test spectra have a decoy with Tanimoto ≥0.7 to the truth; they hold 72% of the misses. The drop for molecules with close training analogues is a decoy effect, not memorization (0.4% of misses pick a training molecule).
5. **Anatomy.** 56% of misses rank the truth in the top 5; 29% pick a near-isomer. Three rule-selected examples.
6. **Fragment evidence.** In 78% of misses the wrong top-1 explains at least as much intensity as the truth. More proposals or adduct-conditioned GLACIER proposals do not change this (validation bounds).
7. **What was tried.** Ten single-head interventions stay within ±0.6 test points.
8. **Headroom and summary.** 32% of spectra defeat all members; a taxonomy of three failure modes with matching levers.

Test is analysed post hoc for diagnosis; models were selected on validation beforehand. Every chart plots measured values; the molecule drawings on slide 10 are the actual truth and wrong top-1 structures.

## Rebuild

From the repository root:

```sh
.venv/bin/python results/retrieval_failure_analysis_20260929/analyze_failures.py      # ~15 min
.venv/bin/python results/retrieval_failure_analysis_20260929/decompose_failures.py
.venv/bin/python results/retrieval_failure_analysis_20260929/memorization_check.py
.venv/bin/python results/retrieval_failure_analysis_20260929/write_report.py
.venv/bin/python presentations/retrieval_failures/make_failure_figures.py
presentations/retrieval_failures/build.sh
.venv/bin/python presentations/retrieval_failures/inspect_retrieval_failures.py
```

The analysis scripts need the ensemble score dumps under `results/retrieval_ensemble_20260927/` and the local fragment-store extract used by `fragment_coverage.py` (path in `analyze_failures.py`). `make_failure_figures.py` writes the vector charts, the RDKit-coordinate TikZ molecules and `figures/numbers.tex`; every number on the slides is a macro from that file, generated from the analysis JSON. Build auxiliaries and 180 dpi previews stay under `build/`.

## Verification

`inspect_retrieval_failures.py` writes `build/layout-audit.json`: 17 pages, all fonts embedded, no raster images, no text outside the page, no overfull boxes or missing glyphs. Every page was also reviewed at 180 dpi; see the [page review](retrieval_failures_page_review.md).
