# Page review — periodic teacher semantics

Reviewed every page at 180 dpi, including 12 measured plots and the training schematic. The main deck has 22 pages; the overview has 6. Scientific art is vector, text is selectable, all fonts are embedded and there are no Type 3 fonts or raster images. Page-bound checks found no text outside the slide area. Final LuaLaTeX passes have no overfull boxes, missing characters or LaTeX warnings.

The original visual pass found body/takeaway collisions on the definitions and protocol slides, overly dense controls text, and plot-value labels near category labels. These were fixed by shortening text, separating rows and adding plot margins. The overview interpretation slide was shortened. No scientific qualifications were removed to make the slides fit.

| Main page | Review |
|---:|---|
| 1 | MAE → initial JEPA → periodic student scores are shown together; the distinction between chemical readouts and geometry remains explicit. |
| 2 | Three definitions and their diagnostics now occupy separate rows without overlap. |
| 3 | Initial JEPA teacher, MAE ancestor and fresh student clearly distinguished; copy count and two late steps labeled. |
| 4 | Side-by-side training paths; masking, detached teacher targets, gradient updates and scheduled encoder copies are visually distinct. Teacher receives no optimizer gradient. |
| 5 | Actual logs; log scale explicit; promotion markers apply only to the enlarged late window. |
| 6 | Supporting legacy protocol labeled; embedder, block depth and final norm distinguished. |
| 7 | All four states appear with consistent colors and symbols; three metric scales and exact checkpoint roles are labeled. |
| 8 | Corrected ±5 Th mass window; raw versus train-transformed geometry explicit. |
| 9 | Three paired stage comparisons against MAE and initial JEPA; zero references and sparse ticks are readable. JEPA–MAE neighbor interval crosses zero. |
| 10 | Mean and CLS concentration shown together; no claim of uniform model collapse. |
| 11 | Positive global-correlation evidence receives its own slide; raw and scaled CLS separated. |
| 12 | Label-budget log axis readable; one subsample seed disclosed. |
| 13 | All seven property scores shown; no fingerprint score confused with structure-recovery accuracy. |
| 14 | MAE already has strong dropout robustness; repeat-spectrum retrieval decreases across the three main stages. Interventions and repeat spectra remain separate. |
| 15 | Two actual downstream seeds visible; no spurious confidence intervals. |
| 16 | Conditional-mean assumptions stated; mathematical fixed point distinguished from observed collapse. |
| 17 | Observed result and unidentified causal claims in separate columns. |
| 18 | New training experiment explicitly proposed, never presented as completed. |
| 19 | Fresh and reused evidence separated; legacy evaluation limitations retained; no text at takeaway baseline. |
| 20 | Single-spectrum control uses one row per molecule and corrected physical precursor units. |
| 21 | Controls, identity-averaging limit and scope fit with clear separation. |
| 22 | Live reference links; complete source paths in the report; no unsupported literature inference. |

Overview pages 1–6 were independently rendered and reviewed: four-checkpoint scorecard, training diagram, chemical probes, geometry, paired stage differences and the causal follow-up are readable at the same page size. Figures retain the same measured data and labels as the main deck.

Standalone measured plots are 183 × 68 mm with ordinary text at 6–7 pt, 8 pt panel titles, and embedded Liberation Sans. Math superscripts on logarithmic ticks are approximately 4.55 pt. White standalone exports and transparent slide exports are separate files. The figure inventory records protocol and uncertainty for each panel.

The native editor remains open with editable source. Its standalone compiler cannot load the repository's `presentation-theme.sty` and other project files. The distributed PDFs were successfully built with the repository's existing LuaLaTeX toolchain; this limitation does not affect those exports.

## MAE extension review

All 26 slides and 12 standalone figures were regenerated and visually checked after adding the fresh MAE measurements. The three earlier model records and their existing paired intervals are unchanged byte-for-value in the summary. A lineage-arrow label was moved clear of its boxes, and the uncertainty chart now uses fewer x-axis ticks. The attention plot includes the existing MAE observations and automatically covers their range. No branding, clipped text, raster artwork, missing fonts or compiler warnings were found.

## Training schematic review

Added full-deck page 4 and overview page 2, plus the 183 × 61 mm standalone TikZ schematic. Both inserted slides and the standalone export were rendered and visually inspected at 180 dpi. Teacher labels fit without hyphenation; the stop-gradient labels are clear of the copy loop. Black forward paths, teal gradient paths and ochre weight-copy paths are independently labeled. Both decks compile with 28 pages total; all 13 vector figures pass the font and page-bound checks. Measurements are unchanged. Native preview cannot resolve the shared theme/assets; repository LuaLaTeX builds pass.
