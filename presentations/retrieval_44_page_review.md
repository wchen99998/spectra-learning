# Page review: the 44.10% retrieval recipe

Reviewed at 180dpi after compilation, using contact sheets for the complete narrative and individual page renders for diagrams, dense equations and results. The presentation uses the existing Novogaia theme, fonts, logo and 16:9 geometry. All scientific artwork is vector TikZ; explanatory values are distinguished from measured results.

| Page | Visual and content check |
|---|---|
| 1 | Clear title/result separation; 44.10% labelled full-test Hit@1, 46.02% labelled validation in the source note. |
| 2 | Three historical test bars share the same zero-based axis and spectrum denominator; retained foundations and new changes are distinct. Historical progression is not presented as a component ablation. |
| 3 | Encoder arrows terminate at operation/token boundaries; peak and molecule streams converge on the sole candidate score. No PCA or pair-stream encoder is implied. |
| 4 | Real-atom pooling, MLP and broadcast residual are explicit; local captions and equation have separate space. Graph distinguishability is not guaranteed. |
| 5 | Intact/member/boundary drawings have a semantic legend. Representation and gate branches are distinct; predicted intensity reaches only the gate. Feature dimensions and captions sit below their boxes. |
| 6 | Peak/fragment matrix axes and evidence sources are labelled. Heatmaps are expressly schematic, not measured activations or a numerical matrix sum. |
| 7 | Fragment attention feeds candidate context, then a learned residual changes the compatibility matrix. Q/K256 and the 32D adapter remain distinct. |
| 8 | Forward log-mixture and reverse maximum have separate arrows/boxes. Their 0.65/0.35 weights and exact combined equation agree with the code. |
| 9 | Both pool diagrams select eight black dots. Only sampling support changes; the schematic pool size and actual 0–255 eligible-negative range are explicit. |
| 10 | Validation curve uses the eight recorded joint epochs. Warm initialization, fresh optimizer moments, learning rates and the small head-only/joint control difference are labelled. |
| 11 | All full-test metrics match the canonical result. Cumulative Hit@k bars share a 0–100% scale. Mean MCES and the threshold-15 convention are identified. |
| 12 | Selected checkpoint → maintained pipeline → exact full-test rank reproduction. The method-document line stays in the left column and clears the limitation text. |
| 13 | Exact ion and cross-attention equations fit within the page; hydrogen/adduct constants, masks and dimensions are defined. |
| 14 | Fixed settings use aligned rows with readable spacing; the selected validation subset is distinguished from the full validation split. |

An independent factual review checked equations, historical baselines, training values and denominators against [the evidence record](research/retrieval_44_evidence.md). An independent visual review checked the title, historical/results charts, sampling slide and appendices. Revisions addressed wrapped title/body text, equation clearance, branch/arrow endpoints, fragment-box labels and one two-column text overlap.

The final automated report is `build/retrieval_44/layout-audit.json`: 14 pages, embedded fonts, no raster artwork, no text outside the page margins, and no missing-glyph or overfull-box warnings. The final document is a presentation artifact; no model/training code was changed for this request.
