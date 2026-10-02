# Page and artwork review

Reviewed 30 September 2026 after the final text/caption corrections.

## Build and export checks

- 22 slides, using the shared 16:9 presentation theme.
- Every slide rendered at 180 dpi; contact sheets and full-resolution pages inspected.
- All 30 figure PDFs inspected: 24 measured exports and six diagram exports. Slide and standalone variants are separate artwork layouts.
- All fonts embedded; no Type 3 fonts; no raster images in the deck or scientific figure PDFs.
- No text outside PDF page bounds. No overfull boxes or missing-character warnings. Four underfull footnote-paragraph diagnostics remain; visual review found no clipping or overlap.
- Compact diagrams fit the shared plot area: 183 × 60.5 mm exports, displayed at approximately 143 × 47.28 mm. Full standalone diagrams retain the detailed protocol annotations.
- Measured figures: 143 × 47 mm slide PDFs and 183 × 60 mm standalone PDFs, with editable SVG text/vector artwork.
- Scientific claims independently checked against the saved study artifacts. The evidence index includes exact JSON selectors, values and source hashes.

Machine-readable inspection is in `build/layout-audit.json`; standalone measured export checks are in `research/export_audit.json`; diagram checks are in `build/diagrams/qa.json`. These generated checks do not replace visual review of internal labels and connector endpoints.

## Page review

| Page | Review |
|---|---|
| 1 | Main question, qualified conclusion and primary AUROC readable; parent/latest order explicit. |
| 2 | Compact lineage diagram clear; teacher initialization, fresh student and scheduled copies distinct. |
| 3 | Train/validation/test columns shortened to avoid footer overlap; fold counts and selection scope preserved. |
| 4 | All three panels fit; paired-difference axis clipping corrected; signs and target support explicit. |
| 5 | Selected-state geometry readable; N/O normalization/output abbreviations defined in source note. |
| 6 | Residual-preserving SAE edit diagram clear; controls and frozen downstream readouts distinct. |
| 7 | All six SAE fits and full-reconstruction loss damage visible; fixed-budget qualification retained. |
| 8 | Both frozen features per concept included; failed nitrile association visible. |
| 9 | Replication controls and two-feature averaging identified; shared hydroxyl replica caveat retained. |
| 10 | All selected features and both control directions visible; no omitted test failures. |
| 11 | Reconstruction heading shortened; both columns and loss values fit without overlap. |
| 12 | Halide positive/false-positive structures and spectra readable; measured peaks versus motif interpretation distinguished. |
| 13 | Full and restricted query populations identified; difference sign defined as latest minus parent. |
| 14 | Vector head heatmaps use a shared signed scale; validation-first branches and tested head subset explicit. |
| 15 | Curves defined as selected-minus-random recovery; logit rescue separated from inconclusive chemistry-loss rescue. |
| 16 | Peak perturbations readable; artificial m/z perturbation asterisk defined; ordinary BCE scope explicit. |
| 17 | CE-routing diagram readable; fixed physical input and raw25 patches including CLS retained. |
| 18 | Observed repeats and metadata paths distinct; final-path mean loss distinguished from prediction movement. |
| 19 | Summary table labels shortened; no wrapped row collision. Each endpoint retains its separate conclusion. |
| 20 | Native execution and numerical gates readable; archived numerical functions excluded explicitly. |
| 21 | Established/unresolved/untested/next-test sections shortened to clear the takeaway and source note. |
| 22 | Artifact path, build command and method citations fit; overlapping source blocks removed. |

## Interpretation checks

AUROC ensembles mean the three seed logits. Intervention score ensembles average sigmoid outputs. Input/metadata ordinary BCE is not the class-balanced causal metric. Feature control norms match before BF16 casting and only approximately afterward. No report page claims calibrated chemical probabilities, chemical sufficiency, head-to-feature mediation, validated fragment assignment or training-recipe causality.

The molecular case examples are post-confirmation illustrations from the frozen feature atlas. The source bundle contains final report sources and compact plotted evidence, not a claim that every bundled file is the exact historical source of every model run.
