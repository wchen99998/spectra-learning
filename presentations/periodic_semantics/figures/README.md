# Periodic teacher semantics — figure inventory

All plots show measured data. Each has editable Matplotlib source in `../make_periodic_semantics_figures.py`, vector PDF and live-text SVG. `_slide.pdf` files have transparent backgrounds. Standalone PDFs are 183 × 68 mm, white, with embedded Liberation Sans. No low-dimensional embedding projection is used as evidence.

Initial teacher: frozen-teacher JEPA encoder step 1,190,000. Promoted teacher: periodic student at 1,200,000. Latest student: periodic checkpoint 1,209,000. Black, gray and teal colors are consistent across figures.

| Figure | Measurement and legend |
|---|---|
| `training_loss` | Actual training and validation masked-latent MSE from W&B, 4,835 / 120 non-null observations. Left logarithmic y axis; right enlarged late interval with scheduled promotion steps. Changing teachers changes the prediction task. |
| `legacy_depth` | Reused spectrum-weighted 27-depth mean-pool ridge MACCS AUROC and standardized-cosine neighbor chemistry; 17,556 test spectra. Random and MAE references included. No layer selected on test; sweep is exploratory. |
| `chemical_probes` | Fresh final-mean molecule-centroid probes: MACCS macro AUROC, 15-group macro AUROC and mean seven-property R². 22,746 / 3,185 / 2,998 train / validation / test molecules. Penalties selected on validation. |
| `geometry` | Molecule-centroid final-mean raw, centered, standardized and whitened cosine. Nearest train-neighbor Tanimoto; partial Spearman with chemical similarity; nearest neighbor within ±5 Th of mean precursor m/z. Train-only transforms. |
| `paired_uncertainty` | Student minus initial teacher, fixed fitted models. 95% paired molecule percentile intervals; 500 resamples for macro AUROC and 2,000 for neighbor chemistry. No correction for multiple secondary comparisons. |
| `covariance` | Train-centroid covariance, separately mean and CLS: effective rank exp(H(eigenvalue fractions)), leading-PC fraction, average raw cosine including the negligible self-pair contribution. Covariance concentration is not chemical accuracy. |
| `cls_exception` | 99,955 distinct-test-molecule pairs: Spearman and partial Spearman with radius-2 4,096-bit Morgan Tanimoto. Partial ranks control spectral cosine and precursor gap. Shared molecule endpoints mean pairs are dependent; no pair-based confidence intervals. |
| `label_efficiency` | Final-mean ridge at 227 / 2,274 / 22,746 training molecules, identical deterministic subsets across models. Validation selection and train-only preprocessing per budget. No subset-seed uncertainty. |
| `properties` | Seven RDKit property R² scores on 2,998 test centroids. One penalty per model chosen by mean validation R²; targets include logP, rings, TPSA, fraction sp³, aromatic rings and hydrogen-bond donors/acceptors. |
| `robustness` | Centered embedding cosine and self-retrieval for 2,048 fixed test spectra under 20% peak dropout, lognormal intensity jitter and acquisition-field zeroing. Right: retrieval of a second measured spectrum among 2,659 molecule candidates. Interventions are diagnostics, not calibrated physical acquisition models. |
| `attentive` | Reused spectrum-weighted mean-linear and attention-pooling MACCS probes at three depths. Lines show mean of two seed observations; short marks show the two actual seed scores. Epoch selected on validation with 12-epoch cap. |
| `single_spectrum` | One measured spectrum per molecule (first file row), 2,998 test queries versus 22,746 training candidates. Raw, train-standardized and train-whitened final-mean cosine. Unrestricted and actual precursor ±5 Th windows; no identity-centroid averaging. |

Fingerprints have no units; Tanimoto, cosine, AUROC, correlations, variance fractions and R² are dimensionless. Effective rank is an effective number of feature dimensions. Absolute covariance scale depends on checkpoint coordinates and is reported separately in the full results.

The corrected physical-window result is used throughout. The serialized metric name ending in `_5Da` is retained from the first script version; its computation and all displayed labels use **±5 Th precursor m/z** after undoing the /1,000 input normalization.
