# MassSpecGym MACCS benchmark: 1.8B step 650,000

Completed 2026-08-31 UTC. The 1.8B checkpoint sets a new best result in the
stored fixed-width MACCS benchmark for token-level covariance features:
**0.786030 ± 0.003710 macro ROC-AUC** and
**0.608240 ± 0.002265 macro average precision** with the covariance-linear
head. The covariance MLP is slightly worse, while the CLS probes remain below
the prior 1B mix2x8 step-250k CLS results.

Values below are test-set mean ± sample standard deviation over seeds 66, 67,
and 68. The primary metric is macro ROC-AUC over the 151 non-degenerate MACCS
bits.

## Primary results

| Representation | Head | Macro ROC-AUC | Macro AP | Weighted ROC-AUC | Micro ROC-AUC | Weighted AP | Micro AP | BCE | Brier | Best epochs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Covariance | linear | **0.786030 ± 0.003710** | **0.608240 ± 0.002265** | 0.811858 ± 0.001876 | 0.923929 ± 0.000072 | 0.818391 ± 0.001090 | 0.861990 ± 0.000239 | 0.348486 ± 0.004434 | 0.106812 ± 0.000448 | 4/6/6 |
| Covariance | MLP-256 | 0.783915 ± 0.000678 | 0.605911 ± 0.003062 | **0.813177 ± 0.001466** | **0.924604 ± 0.000933** | **0.818922 ± 0.001432** | **0.862903 ± 0.002102** | **0.344455 ± 0.006443** | **0.105975 ± 0.000859** | 7/6/5 |
| CLS | linear | 0.733101 ± 0.002283 | 0.530135 ± 0.000996 | 0.762074 ± 0.000648 | 0.811475 ± 0.002934 | 0.768712 ± 0.000615 | 0.739166 ± 0.004308 | 0.579600 ± 0.006555 | 0.193033 ± 0.003107 | 20/16/21 |
| CLS | MLP-256 | 0.762226 ± 0.001675 | 0.554252 ± 0.001235 | 0.786422 ± 0.000475 | 0.915149 ± 0.000120 | 0.786472 ± 0.001057 | 0.842828 ± 0.000393 | 0.362022 ± 0.000476 | 0.113052 ± 0.000100 | 27/28/30 |

The covariance `linear` result is not a pure linear probe of the frozen
encoder. It includes the benchmark's trainable 2,048-to-64 left and right
projections followed by a linear MACCS head. The MLP result changes only that
head to `Linear(4096, 256), SiLU, Linear(256, 166)`.

## Thresholded metrics

`@val` uses the independently selected per-bit validation-F1 threshold.
Tanimoto is averaged over test spectra.

| Representation | Head | F1 macro @0.5 | F1 micro @0.5 | Tanimoto @0.5 | Hamming @0.5 | F1 macro @val | F1 micro @val | Tanimoto @val | Hamming @val |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Covariance | linear | 0.488616 ± 0.002466 | 0.763802 ± 0.000887 | 0.626614 ± 0.000967 | 0.150391 ± 0.000093 | **0.530608 ± 0.001798** | **0.700628 ± 0.002696** | **0.543665 ± 0.002963** | **0.247258 ± 0.003337** |
| Covariance | MLP-256 | 0.485150 ± 0.001812 | **0.767312 ± 0.000685** | **0.631258 ± 0.001007** | **0.149145 ± 0.000729** | 0.529772 ± 0.001720 | 0.699797 ± 0.002449 | 0.542865 ± 0.003036 | 0.248349 ± 0.003517 |
| CLS | linear | 0.469759 ± 0.001911 | 0.699717 ± 0.007773 | 0.547299 ± 0.008400 | 0.222367 ± 0.009112 | 0.489505 ± 0.000645 | 0.664605 ± 0.001300 | 0.500657 ± 0.001609 | 0.290791 ± 0.001676 |
| CLS | MLP-256 | 0.441426 ± 0.003096 | 0.750312 ± 0.000363 | 0.608679 ± 0.000528 | 0.160886 ± 0.000113 | 0.504453 ± 0.000658 | 0.685060 ± 0.001953 | 0.526483 ± 0.002064 | 0.267435 ± 0.002856 |

## Comparison with prior fixed-width results

The matching 1B comparison is the corrected mix2x8 step-250k rerun. The DreaMS
comparison uses its published embedding model and native preprocessing. All
three use the same MassSpecGym rows, folds, MACCS targets, probe widths, seeds,
training rule, model-selection rule, and test metrics.

| Model | Representation | Head | Macro ROC-AUC | Macro AP |
|---|---|---|---:|---:|
| **1.8B step 650k** | Covariance | linear | **0.786030 ± 0.003710** | **0.608240 ± 0.002265** |
| 1B mix2x8 step 250k | Covariance | linear | 0.765734 ± 0.000995 | 0.583438 ± 0.001508 |
| DreaMS | Covariance | linear | 0.712646 ± 0.003488 | 0.521125 ± 0.003427 |
| **1.8B step 650k** | Covariance | MLP-256 | **0.783915 ± 0.000678** | **0.605911 ± 0.003062** |
| 1B mix2x8 step 250k | Covariance | MLP-256 | 0.769713 ± 0.001831 | 0.585865 ± 0.001940 |
| DreaMS | Covariance | MLP-256 | 0.719405 ± 0.003140 | 0.524366 ± 0.001844 |
| 1.8B step 650k | CLS | linear | 0.733101 ± 0.002283 | 0.530135 ± 0.000996 |
| **1B mix2x8 step 250k** | CLS | linear | **0.740071 ± 0.001930** | **0.538424 ± 0.001388** |
| DreaMS official embedding | CLS | linear | 0.696797 ± 0.005420 | 0.503546 ± 0.002345 |
| 1.8B step 650k | CLS | MLP-256 | 0.762226 ± 0.001675 | 0.554252 ± 0.001235 |
| **1B mix2x8 step 250k** | CLS | MLP-256 | **0.771380 ± 0.000740** | **0.567480 ± 0.002005** |
| DreaMS official embedding | CLS | MLP-256 | 0.732256 ± 0.002880 | 0.528122 ± 0.002633 |

Relative to 1B mix2x8 step 250k, the 1.8B checkpoint improves covariance
linear ROC-AUC/AP by +0.020296/+0.024802 and covariance MLP ROC-AUC/AP by
+0.014202/+0.020046. Its CLS linear and MLP ROC-AUC are lower by 0.006970 and
0.009154. Relative to DreaMS, the gains are +0.073384/+0.087115 for covariance
linear ROC-AUC/AP and +0.064510/+0.081545 for covariance MLP.

Within the 1.8B result, covariance minus CLS is +0.052929 ± 0.002644 ROC-AUC
and +0.078105 ± 0.002713 AP for the linear heads; for the MLP heads it is
+0.021689 ± 0.002351 ROC-AUC and +0.051659 ± 0.003925 AP. The token-level
representation is therefore carrying substantially more useful structure than
the checkpoint's CLS token under this benchmark.

This is not a controlled parameter-count ablation: the comparison also changes
embedding width (2,048 versus 1,536), visible peak count (127 versus 63),
optimizer/training recipe, and checkpoint progress. The result establishes the
quality of this checkpoint's embeddings under the fixed probe contract, not
the causal effect of scaling alone.

## Checkpoint and input contract

- W&B run: `1p8b-sm-hd256-adam-v6e64-b6144-ga1-666k-3d-mr6-20260828-024511`.
- Selected checkpoint: step 650,000, the newest fully committed checkpoint at
  2026-08-30 08:03:07 UTC. W&B had reached step 652,500.
- Checkpoint URI:
  `gs://metal-repeater-411410-spectra-checkpoints/skypilot/1p8b-sm-hd256-adam-v6e64-b6144-ga1-666k-3d-mr6-20260828-024511/checkpoints/orbax/650000`.
- The commit contained 76 objects and 12,022,075,832 bytes and had all four
  expected commit markers.
- Checkpoint-native encoder input: 127 m/z-ordered peak tokens, 2,048 embedding
  dimensions, CLS enabled, pair path disabled, and corrected 28-field
  massive-v2 acquisition metadata.
- Dataset: MassSpecGym 1.5 revision
  `d2e86d0c3bd905a6d578c0dd6053ed2bd41f9c2a`; TSV SHA-256
  `50cfdd1d22f79543c59555f9ce43c6893bd788a19a42b21fb4e2e3a54673c06a`.
- Split: 194,119 train / 19,429 validation / 17,556 test spectra; 166 MACCS
  bits, of which 151 are non-degenerate in the test set.
- Probe training: seeds 66/67/68, unweighted BCE, AdamW at 3e-4 with no weight
  decay, half-epoch linear warmup then cosine decay to 1%, maximum 100 epochs,
  validation macro-AUROC selection, minimum 20 epochs, and patience 5.
- Batch sizes: 1,024 for covariance and 4,096 for CLS. The 112-GiB token array
  was streamed from host-local ephemeral storage because it cannot fit in an
  80-GiB GPU; this changes storage placement only, not examples, ordering,
  arithmetic, or the probe definition.

Extraction used JAX 0.11.0 and took 1,810 seconds. Probe fitting used PyTorch
2.13.0 on A100-80GB GPUs and took 5,154 seconds for covariance and 731 seconds
for CLS.

## Artifacts, validation, and cleanup

The compact bundle is in
`artifacts/massspecgym_maccs_covariance_1p8b-sm-hd256-adam-v6e64-b6144-ga1-666k-3d-mr6-20260828-024511_step650000_fullvisible128_metadata-v2/`.
It contains 39 non-empty files: provenance and aggregate JSON, 12 probe
weights, 12 compressed prediction bundles, and 12 per-bit reports.

Independent validation recomputed every default- and validation-threshold
metric and every aggregate from all 12 saved prediction bundles. It also
confirmed finite `[17556, 166]` probabilities, 166 finite validation
thresholds, seeds 66/67/68, identical test indices and targets across every
head and representation, 151 valid test bits, the dataset SHA, checkpoint
step, and embedding shape `[231104, 127, 2048]`.

After local validation, the newly generated remote embedding/result root, its
staged checkpoint metadata, and the temporary Modal credential were deleted
and verified absent. The shared prepared MassSpecGym input was retained for
future benchmarks.
