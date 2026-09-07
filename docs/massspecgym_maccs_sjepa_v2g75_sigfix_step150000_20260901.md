# MassSpecGym MACCS benchmark: S-JEPA v2g75 sigfix step 150,000

Completed 2026-09-01 UTC. At this early checkpoint, the token-level covariance
representation reaches **0.659070 ± 0.001097 macro ROC-AUC** and
**0.452413 ± 0.001082 macro average precision** with the covariance-linear
head. The larger covariance MLP changes this only to
**0.660312 ± 0.000716 ROC-AUC** and **0.452763 ± 0.001181 AP**. Both are
below DreaMS and the stronger earlier spectra-learning checkpoints under the
same probe contract.

Values below are test-set mean ± sample standard deviation over seeds 66, 67,
and 68. The primary metric is macro ROC-AUC over the 151 non-degenerate MACCS
bits.

## Primary results

| Representation | Head | Macro ROC-AUC | Macro AP | Weighted ROC-AUC | Micro ROC-AUC | Weighted AP | Micro AP | BCE | Brier | Best epochs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Covariance | linear | 0.659070 ± 0.001097 | 0.452413 ± 0.001082 | 0.680636 ± 0.002444 | **0.879317 ± 0.000909** | 0.695849 ± 0.001279 | **0.773402 ± 0.001497** | **0.419186 ± 0.001349** | **0.134900 ± 0.000422** | 12/16/18 |
| Covariance | MLP-256 | **0.660312 ± 0.000716** | **0.452763 ± 0.001181** | **0.681322 ± 0.002777** | 0.878716 ± 0.001042 | **0.696180 ± 0.001303** | 0.773047 ± 0.001721 | 0.420001 ± 0.003077 | 0.135259 ± 0.000700 | 12/16/18 |
| CLS | linear | 0.628659 ± 0.001106 | 0.435539 ± 0.001044 | 0.637140 ± 0.001568 | 0.740273 ± 0.007130 | 0.676080 ± 0.001411 | 0.595411 ± 0.008343 | 0.664534 ± 0.014555 | 0.223785 ± 0.005519 | 17/11/18 |
| CLS | MLP-256 | **0.636796 ± 0.000833** | **0.439592 ± 0.001221** | **0.643096 ± 0.000609** | **0.868281 ± 0.000552** | **0.680558 ± 0.000942** | **0.753162 ± 0.001349** | **0.427125 ± 0.000944** | **0.141649 ± 0.000453** | 9/13/6 |

Bold values compare the two heads within each representation. The covariance
`linear` result is not a pure linear probe of the frozen encoder: it includes
the benchmark's trainable 1,536-to-64 left and right projections, constructs
an uncentered 64-by-64 cross-covariance over all valid peak tokens, and then
applies a linear MACCS head. The MLP changes the final head to
`Linear(4096, 256), SiLU, Linear(256, 166)`.

## Thresholded metrics

`@val` uses the independently selected per-bit validation-F1 threshold.
Tanimoto is averaged over test spectra.

| Representation | Head | F1 macro @0.5 | F1 micro @0.5 | Tanimoto @0.5 | Hamming @0.5 | F1 macro @val | F1 micro @val | Tanimoto @val | Hamming @val |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Covariance | linear | 0.323449 ± 0.001732 | 0.673806 ± 0.001019 | **0.517342 ± 0.001049** | **0.199496 ± 0.000698** | 0.454003 ± 0.000638 | 0.615506 ± 0.002288 | 0.451080 ± 0.002391 | 0.379607 ± 0.003770 |
| Covariance | MLP-256 | **0.325515 ± 0.001123** | **0.673902 ± 0.001084** | 0.516920 ± 0.000890 | 0.199878 ± 0.000644 | **0.454419 ± 0.000137** | **0.619773 ± 0.003096** | **0.455168 ± 0.003167** | **0.372185 ± 0.006199** |
| CLS | linear | **0.411731 ± 0.003725** | 0.642660 ± 0.010389 | 0.476360 ± 0.010843 | 0.266145 ± 0.015483 | 0.444937 ± 0.000694 | 0.594211 ± 0.001218 | 0.427867 ± 0.001419 | 0.409447 ± 0.002215 |
| CLS | MLP-256 | 0.299894 ± 0.002191 | **0.644530 ± 0.002425** | **0.481530 ± 0.002675** | **0.216935 ± 0.000703** | **0.446754 ± 0.000502** | **0.605946 ± 0.004357** | **0.439328 ± 0.003602** | **0.388841 ± 0.005508** |

## Comparison with prior fixed-width results

The DreaMS row uses its published embedding model and native preprocessing.
All rows use the same MassSpecGym examples and folds, MACCS targets, probe
widths, seeds, optimization and model-selection rules, and test metrics.

| Model | Representation | Head | Macro ROC-AUC | Macro AP |
|---|---|---|---:|---:|
| S-JEPA v2g75 sigfix step 150k | Covariance | linear | 0.659070 ± 0.001097 | 0.452413 ± 0.001082 |
| DreaMS | Covariance | linear | 0.712646 ± 0.003488 | 0.521125 ± 0.003427 |
| 1B mix2x8 step 250k | Covariance | linear | 0.765734 ± 0.000995 | 0.583438 ± 0.001508 |
| 1.8B step 650k | Covariance | linear | **0.786030 ± 0.003710** | **0.608240 ± 0.002265** |
| S-JEPA v2g75 sigfix step 150k | Covariance | MLP-256 | 0.660312 ± 0.000716 | 0.452763 ± 0.001181 |
| DreaMS | Covariance | MLP-256 | 0.719405 ± 0.003140 | 0.524366 ± 0.001844 |
| 1B mix2x8 step 250k | Covariance | MLP-256 | 0.769713 ± 0.001831 | 0.585865 ± 0.001940 |
| 1.8B step 650k | Covariance | MLP-256 | **0.783915 ± 0.000678** | **0.605911 ± 0.003062** |
| S-JEPA v2g75 sigfix step 150k | CLS | linear | 0.628659 ± 0.001106 | 0.435539 ± 0.001044 |
| DreaMS official embedding | CLS | linear | 0.696797 ± 0.005420 | 0.503546 ± 0.002345 |
| 1B mix2x8 step 250k | CLS | linear | **0.740071 ± 0.001930** | **0.538424 ± 0.001388** |
| 1.8B step 650k | CLS | linear | 0.733101 ± 0.002283 | 0.530135 ± 0.000996 |
| S-JEPA v2g75 sigfix step 150k | CLS | MLP-256 | 0.636796 ± 0.000833 | 0.439592 ± 0.001221 |
| DreaMS official embedding | CLS | MLP-256 | 0.732256 ± 0.002880 | 0.528122 ± 0.002633 |
| 1B mix2x8 step 250k | CLS | MLP-256 | **0.771380 ± 0.000740** | **0.567480 ± 0.002005** |
| 1.8B step 650k | CLS | MLP-256 | 0.762226 ± 0.001675 | 0.554252 ± 0.001235 |

Relative to DreaMS, S-JEPA is lower by 0.053576 ROC-AUC and 0.068712 AP for
the covariance-linear probe; the covariance MLP gap is 0.059093 ROC-AUC and
0.071603 AP. Relative to 1B mix2x8 step 250k, the covariance-linear gaps are
0.106664 ROC-AUC and 0.131025 AP.

Within this checkpoint, covariance exceeds CLS by 0.030412 ROC-AUC and
0.016874 AP for the linear heads, and by 0.023516 ROC-AUC and 0.013172 AP for
the MLP heads. Token-level embeddings therefore contain more useful MACCS
signal than CLS, but the absolute representation quality is still weak at
step 150k. The covariance MLP's gain over covariance-linear is only 0.001241
ROC-AUC and 0.000350 AP, so probe capacity is not the limiting factor in this
result.

This checkpoint is only 150,000 of the configured 2,000,000 training steps
(7.5%). The result diagnoses this point in training; it does not establish
that the final S-JEPA recipe will remain below the baselines.

## Checkpoint and input contract

- W&B run:
  `1b-sjepa-adam-v6e32-b2048-v2g75-v3l20-sigfix-s9-20260901-033604`.
- Selected checkpoint: step 150,000, the newest fully committed checkpoint at
  2026-09-01 12:31:07 UTC. W&B also reported step 150,000 and the run was still
  active at selection time.
- Checkpoint URI:
  `gs://metal-repeater-411410-spectra-checkpoints/skypilot/1b-sjepa-adam-v6e32-b2048-v2g75-v3l20-sigfix-s9-20260901-033604/checkpoints/orbax/150000`.
- The checkpoint contained 52 objects and 9,542,444,333 bytes and had all four
  expected commit markers.
- Checkpoint-native encoder input: 127 m/z-ordered peak tokens, 1,536 embedding
  dimensions, CLS enabled, corrected 28-field massive-v2 acquisition metadata,
  and the pair path disabled. Covariance excludes CLS and uses all valid final
  student-encoder peak embeddings.
- Dataset: MassSpecGym 1.5 revision
  `d2e86d0c3bd905a6d578c0dd6053ed2bd41f9c2a`; TSV SHA-256
  `50cfdd1d22f79543c59555f9ce43c6893bd788a19a42b21fb4e2e3a54673c06a`.
- Split: 194,119 train / 19,429 validation / 17,556 test spectra; 166 MACCS
  bits, of which 151 are non-degenerate in the test set.
- Probe training: seeds 66/67/68, unweighted BCE, AdamW at 3e-4 with no weight
  decay, half-epoch linear warmup then cosine decay to 1%, maximum 100 epochs,
  validation macro-AUROC selection, minimum 20 epochs, and patience 5.
- Batch sizes: 1,024 for covariance and 4,096 for CLS. The approximately
  90-GiB token array was streamed from host-local ephemeral storage; this
  changes storage placement only, not examples, ordering, arithmetic, or the
  probe definition.

Extraction used JAX 0.11.0 on an H100, produced an embedding array with shape
`[231104, 127, 1536]`, and took 1,517 seconds (1,562 seconds total including
setup). Probe fitting used PyTorch 2.13.0 on A100-80GB GPUs and took 8,078
seconds for covariance and 497 seconds for CLS.

## Artifacts, validation, and cleanup

The compact bundle is in
`artifacts/massspecgym_maccs_covariance_1b-sjepa-adam-v6e32-b2048-v2g75-v3l20-sigfix-s9-20260901-033604_step150000_fullvisible128_metadata-v2/`.
It contains 39 non-empty files: provenance and aggregate JSON, 12 probe
weights, 12 compressed prediction bundles, and 12 per-bit reports.

Independent validation recomputed all 880 saved per-run metric and aggregate
values and all 35,856 saved per-bit fields from the 12 prediction bundles. It
also confirmed finite `[17556, 166]` probabilities, 166 finite validation
thresholds, seeds 66/67/68, identical test indices and targets across every
head and representation, 151 valid test bits, the dataset SHA, checkpoint
step, and embedding shape `[231104, 127, 1536]`.

After local validation, the newly generated remote embedding/result root, its
staged checkpoint metadata, and the temporary Modal credential were deleted
and verified absent. The shared prepared MassSpecGym input was retained for
future benchmarks.
