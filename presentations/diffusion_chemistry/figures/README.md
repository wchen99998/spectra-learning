# Figure index

All figures retain selectable vector text and embedded fonts. `_slide.pdf` variants fit the shared slide figure area; standalone PDFs have more generous labels. The editable TikZ and Python sources are included.

| Stem | Meaning and uncertainty |
|---|---|
| `native_token_sae` | Computational schematic of native m/z preprocessing, frozen encoder, final-token readout and raw19 sparse decomposition. Arrows show computation/export, not chemical fragmentation or causal effects. |
| `final_token_auroc` | AUROC of mean three-seed logits on 1,024 test molecular connectivities. Shared support: 145 MACCS / 13 functional-group targets. Descriptive paired molecule 95% intervals. |
| `paired_readout_differences` | Diffusion minus each reference using the same resampled molecules. Positive favors diffusion. Fixed MACCS support retains 980/1,000 defined resamples. |
| `sae_quality_and_associations` | Left: held-out centered variance explained for all nine fitted dictionaries, with no error bars. Right: average test AUROC of all frozen candidates per concept; compact slide adds paired molecule 95% intervals, absent selections remain gaps. Different dictionary IDs are not matched features. |

Exact plotted values and source hashes are in `../research/plot_inputs.json`; regenerate using `../make_figures.py`. Original-unit reconstruction error and centered variance explained have different denominators and are not interchangeable.
