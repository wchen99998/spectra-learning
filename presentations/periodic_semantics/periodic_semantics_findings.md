# Periodic teacher promotion: chemical semantics audit

**Verdict: chemical structure is present, but promotion did not deliver a broad improvement over the initial teacher.** The endpoint trades stronger global chemical-similarity ordering and peak-dropout stability for weaker chemical probes, local neighbors and repeat-spectrum recognition. This is an observational comparison of the specified run, not an isolated estimate of promotion's causal effect.

Evaluated 29 September 2026. [Requested W&B run](https://wandb.ai/iclac/jepa-finalrun/runs/1p8b-ptjepa-t1190k-p50k-b1024-v6e32-20260922-062250). W&B reports `crashed`, with last logged step 1,208,750. GCS contains complete checkpoint metadata through **1,209,000**. The W&B and latest checkpoint configurations match exactly. The snapshot contains the student and its frozen teacher from periodic step **1,200,000** (24 promotions). The **initial teacher** is the `encoder` of frozen-teacher JEPA checkpoint **1,190,000**, not that checkpoint's own MAE teacher.

## Final mean-pooled representation

Fresh molecule-balanced audit. All 231,104 spectra were encoded; molecular centroids use the arithmetic mean of spectra sharing a full InChIKey. There are 22,746 training, 3,185 validation and 2,998 test molecules. The candidate database consists only of training molecules. Centroid results use molecular identity for averaging and are not single-spectrum deployment scores.

| Measure | Initial teacher | Promoted teacher 1,200k | Student 1,209k |
|---|---:|---:|---:|
| MACCS macro AUROC | 0.8339 | 0.8138 | 0.8141 |
| Functional-group AUROC | 0.8288 | 0.8052 | 0.8045 |
| Mean property R² | 0.6464 | 0.6180 | 0.6175 |
| Raw-cosine neighbor Tanimoto | 0.1756 | 0.1657 | 0.1655 |
| Standardized neighbor Tanimoto | 0.1864 | 0.1698 | 0.1707 |
| Whitened neighbor Tanimoto | 0.1972 | 0.1821 | 0.1836 |
| Raw pair Spearman | 0.1914 | 0.3003 | 0.2985 |
| Raw partial Spearman | 0.1528 | 0.2449 | 0.2434 |
| Final-mean effective rank | 28.3256 | 78.9217 | 78.8013 |
| Final-CLS effective rank | 12.1735 | 1.2779 | 1.2764 |
| Repeat-spectrum mean-pool retrieval | 0.3603 | 0.2614 | 0.2621 |
| 20% peak-dropout self-retrieval | 0.3242 | 0.9863 | 0.9883 |
| Intensity-jitter self-retrieval | 0.9990 | 0.9980 | 0.9985 |
| Zero-metadata self-retrieval | 0.9512 | 0.1899 | 0.2036 |
| Single-spectrum raw neighbor Tanimoto | 0.1727 | 0.1610 | 0.1612 |
| Single-spectrum ±5 Th neighbor Tanimoto | 0.1415 | 0.1370 | 0.1373 |

All seven final-mean property R² values and all three label-budget MACCS scores are lower for the latest student. The student nevertheless exceeds the freshly evaluated binned-spectrum baseline (MACCS AUROC 0.6538; raw nearest-neighbor Tanimoto 0.1369). Random-molecule neighbor Tanimoto is 0.0951. Training-label permutation gives MACCS AUROC approximately 0.50 for all three encoders. Chemical information was not erased.

## Uncertainty

Paired differences below are latest student minus initial teacher. Rows are molecules, not spectra or fingerprint bits. Percentile intervals condition on the trained encoders and probe fits; they do not include training-seed variability. There is no multiple-testing correction. The final-mean comparison is the main endpoint; selected internal-layer and CLS checks are supporting analyses.

| Endpoint | Difference | 95% paired molecule-bootstrap interval |
|---|---:|---:|
| mean_final_maccs_auc | -0.01986 | [-0.02454, -0.01553] |
| mean_19_maccs_auc | -0.01753 | [-0.02165, -0.01339] |
| cls_final_maccs_auc | -0.02161 | [-0.02681, -0.01636] |
| mean_final_raw_knn1 | -0.01006 | [-0.01324, -0.00694] |
| mean_final_standardized_knn1 | -0.01569 | [-0.01898, -0.01240] |
| single_raw_knn1 | -0.01149 | [-0.01467, -0.00834] |
| single_raw_knn_mass5 | -0.00415 | [-0.00615, -0.00220] |

AUROC uses average ranks for ties and excludes targets with only one test class: 151 of 166 MACCS bits are test-evaluable. The bootstrap uses 500 paired molecule resamples, retaining this set and excluding any bit undefined within a resample. Neighbor means use 2,000 paired resamples. Correlation uses 99,955 distinct-molecule pairs sampled with seed 20260929; pairs share molecules and are not treated as independent observations. No misleading pair-bootstrap intervals are reported.

## What improved, and what did not

- **Recoverable chemical information deteriorates modestly.** Final-mean MACCS 0.8339 → 0.8141, functional groups 0.8288 → 0.8045, and mean property R² 0.6464 → 0.6175. Block-19 and final-CLS probes also decline. Label budgets of 227, 2,274 and 22,746 molecules all favor the initial teacher.
- **Global chemical ordering improves.** Raw mean-pool cosine versus Morgan Tanimoto Spearman rises 0.1914 → 0.2985; after rank-linear adjustment for binned spectral cosine and precursor-mass gap, 0.1528 → 0.2434. This is positive evidence for one meaning of semantic organization. Partial correlation does not establish complete independence from acquisition or mass.
- **Local chemical neighborhoods deteriorate.** Raw, centered, standardized and whitened final-mean nearest-neighbor chemistry is weaker. Global ordering and local nearest-neighbor quality measure different parts of the geometry; neither implies the other. Train-only whitening improves neighbors for both models but lowers random-pair rank correlation and does not reverse their neighbor ordering.
- **Covariance changes depend on pooling.** Final-mean effective rank rises 28.3 → 78.8, while total centered variance falls 77.8 → 23.8 in checkpoint-specific units. Final-CLS effective rank falls 12.2 → 1.28; its leading component carries 97.2% of variance. This is strong CLS concentration, not complete collapse of the whole model. Effective rank alone is not a chemical-quality metric.
- **Robustness is selective.** Under 20% independent peak dropout (strongest peak preserved), mean-centered self-retrieval rises 32.4% → 98.8%. Under acquisition-field zeroing, it falls 95.1% → 20.4%; intensity-jitter retrieval stays near 99.9%. This suggests greater sensitivity to acquisition conditioning, but the intervention is out of distribution and does not by itself prove a learned shortcut. Dropout retains original peak slots; it is not a physical remeasurement.
- **Actual repeat spectra become harder to match with mean pooling.** First-versus-last measured spectrum retrieval among 2,659 distinct molecule candidates falls 36.0% → 26.2%; CLS rises 8.3% → 11.2%. There are 366 pairs with different instrument-family IDs and 2,055 with different collision energies. Row order is deterministic and these are distinct rows, not necessarily independent acquisitions.
- **A stronger supervised readout does not rescue the endpoint ordering.** The existing two-seed attentive probes favor the initial teacher at depths 13, 19 and final. They are supporting evidence under a spectrum-weighted protocol; their scores must not be subtracted from the new molecule-centroid scores.
- **The latest teacher and student are close.** Only 9,000 optimizer updates separate them, within one 50,000-step generation. This does not establish stability across the preceding 24 generations.

## Exact evaluation protocol

Frozen inference reuses the repository's staged JAX-to-PyTorch encoder exports, with strict state loading and recorded SHA-256s. The original JAX training code, optimizer, checkpoints and run were not changed. State-dictionary conversion and promotion lifecycle tests passed. This audit did not independently restore and hash every original GCS tensor against the pre-existing exports; recorded URI lineage and export hashes make that provenance limitation explicit.

Input preprocessing is the existing 127-peak m/z-ordered MassSpecGym path. Readouts are valid-peak means after blocks 13 and 19 and the final norm, plus the final CLS token. Frozen inference uses bfloat16 autocast; pooled features are stored in bfloat16, analysis uses float32 and float64 covariance/ridge solves. GPU aggregation can introduce very small floating-point differences. All geometry variants are reported. Layers 13 and 19 are exploratory choices informed by prior work on this benchmark; the final encoder output is the main readout. This is a reused benchmark, not a newly untouched confirmation set.

Ridge probes use train-only feature means and standard deviations; eight penalties from 1e-4 to 1e3 times the mean Gram eigenvalue. One binary penalty is chosen by validation MACCS AUROC; one regression penalty by mean validation R². Chemical targets are 166 MACCS bits, 7 element-presence targets, 15 functional groups and 7 RDKit properties. The decoder is supervised; the encoder is frozen. For 1% and 10% label budgets, normalization and regression fits use only the selected training molecules; validation stays fixed. These subsets use one deterministic seed.

Geometry is cosine without preprocessing, train-centered cosine, train-standardized cosine, or train-whitened cosine. Whitening uses C + 0.01 tr(C)/D I. Chemical similarity is binary Morgan radius-2, 4,096-bit Tanimoto. k=1 and mean k=10 neighbors are evaluated against train only. Precursors are converted from the preprocessing representation (m/z divided by 1,000) back to Th before any physical window is applied. Centroid matching uses mean precursor m/z; the separate single-spectrum audit uses actual per-row precursor m/z and the first file row per molecule, without averaging. No formula-matched candidate pool is used.

Robustness uses the first measured row from each of the first 2,048 test molecules in sorted InChIKey order, not a random sample of all test spectra. Perturbation and reference embeddings are centered using train-spectrum means. Dropout preserves at least the strongest peak. Intensity noise is multiplicative lognormal with log-standard-deviation 0.2, then maximum-normalized. Metadata zeroing sets energy, presence flags, charge, polarity and instrument fields to zero while retaining precursor mass. These are diagnostics, not physically calibrated augmentations.

## Why promotion alone is not a guarantee

For a fixed teacher and unrestricted squared-error predictor, the best prediction is the conditional mean of the target code given the predictor's inputs. Promoting the student changes the target code; it does not directly reward molecular similarity or retention of all chemical distinctions. Predictable mass or acquisition cues can coexist with chemistry. A constant-code student/teacher is a possible zero-loss fixed point of the unconstrained formulation, which is not a claim that this run converged to that point.

The empirical importance of design choices is consistent with [I-JEPA](https://arxiv.org/abs/2301.08243), whose image results depend on masking, and [VICReg](https://arxiv.org/abs/2105.04906), which explicitly controls variance and covariance. [DINO](https://arxiv.org/abs/2104.14294) demonstrates semantic behavior in another self-distillation recipe. Those papers do not establish the behavior of this mass-spectrometry promotion schedule.

## Causal limits and next experiment

The initial teacher and latest student differ in training history and supervision sequence. The periodic student started fresh. There is no matched fresh-student run with this exact initial teacher, data order, masks, optimizer and compute budget but promotion disabled. Therefore **do not attribute the endpoint difference solely to promotion**. Nearby short control/regularization runs in the workspace use different teacher conditions and are not substitutes for that counterfactual.

One training seed and late endpoints cannot establish a generation-by-generation trend. The benchmark's supplied folds are used: both full InChIKeys and the first 14-character connectivity keys are disjoint across train, validation and test. MCES split provenance and overlap with the much larger pretraining corpus were not reconstructed. Molecular fingerprints and RDKit properties are proxies for semantics. There is no full structure-generation experiment or controlled downstream retraining here.

The decisive follow-up is a matched never-promoted versus 50k-promoted versus slower/EMA experiment, at least three paired fresh-student seeds, identical minibatches/masks/optimizer/token budgets, and saved pre/post-promotion snapshots. Select design choices on validation and confirm on a new untouched test set. That expensive training experiment is proposed, not claimed as completed.
