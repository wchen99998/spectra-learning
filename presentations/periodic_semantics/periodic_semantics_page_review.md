# Page review — periodic teacher semantics

Reviewed every page at 180 dpi, including all 12 standalone figures. The main deck has 21 pages; the overview has 5. Scientific art is vector, text is selectable, all fonts are embedded and there are no Type 3 fonts or raster images. Page-bound checks found no text outside the slide area. Final LuaLaTeX passes have no overfull boxes, missing characters or LaTeX warnings.

The first visual pass found body/takeaway collisions on the definitions and protocol slides, overly dense controls text, and plot-value labels near category labels. These were fixed by shortening text, separating rows and adding plot margins. The overview interpretation slide was shortened. No scientific qualifications were removed to make the slides fit.

| Main page | Review |
|---:|---|
| 1 | Mixed verdict appears before the headline AUROC decline; positive findings remain visible. |
| 2 | Three definitions and their diagnostics now occupy separate rows without overlap. |
| 3 | Initial JEPA teacher, MAE ancestor and fresh student clearly distinguished; copy count and two late steps labeled. |
| 4 | Actual logs; log scale explicit; promotion markers apply only to the enlarged late window. |
| 5 | Supporting legacy protocol labeled; embedder, block depth and final norm distinguished. |
| 6 | Model colors consistent; three different metric scales labeled; numeric labels clear of axes. |
| 7 | Corrected ±5 Th mass window; raw versus train-transformed geometry explicit. |
| 8 | Paired 95% intervals, zero reference and negative direction readable; training-seed limit in source. |
| 9 | Mean and CLS concentration shown together; no claim of uniform model collapse. |
| 10 | Positive global-correlation evidence receives its own slide; raw and scaled CLS separated. |
| 11 | Label-budget log axis readable; one subsample seed disclosed. |
| 12 | All seven property scores shown; no fingerprint score confused with structure-recovery accuracy. |
| 13 | Dropout, intensity and metadata interventions separated; real repeat-spectrum result separately labeled. |
| 14 | Two actual downstream seeds visible; no spurious confidence intervals. |
| 15 | Conditional-mean assumptions stated; mathematical fixed point distinguished from observed collapse. |
| 16 | Observed result and unidentified causal claims in separate columns. |
| 17 | New training experiment explicitly proposed, never presented as completed. |
| 18 | Fresh and reused evidence separated; legacy evaluation limitations retained; no text at takeaway baseline. |
| 19 | Single-spectrum control uses one row per molecule and corrected physical precursor units. |
| 20 | Controls, identity-averaging limit and scope fit with clear separation. |
| 21 | Live reference links; complete source paths in the report; no unsupported literature inference. |

Overview pages 1–5 were independently rendered and reviewed: mixed result, chemical probes, geometry, paired uncertainty and the causal follow-up are readable at the same page size. Figures retain the same measured data and labels as the main deck.

Standalone figures are 183 × 68 mm with ordinary text at 6–7 pt, 8 pt panel titles, and embedded Liberation Sans. Math superscripts on logarithmic ticks are approximately 4.55 pt. White standalone exports and transparent slide exports are separate files. The figure inventory records protocol and uncertainty for each panel.

The native editor remains open with editable source. Its standalone compiler cannot load the repository's `presentation-theme.sty` and other project files. The distributed PDFs were successfully built with the repository's existing LuaLaTeX toolchain; this limitation does not affect those exports.
