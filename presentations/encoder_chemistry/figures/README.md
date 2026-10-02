# Figures and diagrams

The report follows `presentations/shared/figure_standards.md`. Each measured figure has a standalone vector PDF, an editable SVG and a slide PDF. Each explanatory diagram has editable TikZ source and standalone/slide vector PDFs. The measured figure generator uses the frozen `../research/plot_inputs.json`; exact plot legends and provenance are recorded with the inputs and in `../research/summary.json`.

| Slide | Figure | Meaning and scope |
|---|---|---|
| 2 | [Checkpoint lineage](checkpoint_lineage.pdf) · [TikZ](checkpoint_lineage.tex) | Explanatory training lineage. Parent 1,190k initializes the teacher; periodic student starts fresh. Arrows distinguish initialization, scheduled copying and compared states. The stored 1,200k teacher is not a third measured encoder here. |
| 4 | [Token attention](token_attention.pdf) | Primary held-out macro AUROC, raw block 19 and final tokens. Physical tokens only; three readout seeds; mean-logit ensemble. Parent/latest comparisons use paired molecule uncertainty and fixed support. |
| 5 | [Token geometry](token_geometry.pdf) | Eight selected states; rank from a fixed 256-dimensional projection over the full 6,144-molecule panel. Raw states, normalization and final affine are distinct. Rank and token similarity are not chemical-label metrics. |
| 6 | [SAE intervention pipeline](token_sae_interventions.pdf) · [TikZ](token_sae_interventions.tex) | Explanatory computation/intervention. Raw-19 physical-token features; sparse code; original-residual-preserving edits; remaining encoder and frozen readouts. Controls share requested feature amplitudes. |
| 7 | [SAE reconstruction](sae_reconstruction.pdf) | Six fitted dictionaries: k32 seeds 0/1 and k64 seed 0 per checkpoint. Held-out explained variance; equal molecule weights. Individual fits are not pretraining-seed replicates. |
| 8 | [Chemical features](chemical_features.pdf) | Train/validation-frozen feature associations on test. All failed selected candidates are retained. Molecular maximum feature activation predicts a structural label; it does not identify a fragment. |
| 9 | [Feature stability](feature_stability.pdf) | Train-only, label-free feature alignment followed by held-out replication. Dictionary-local feature IDs do not identify common axes across models. |
| 10 | [SAE causal effects](sae_causal.pdf) | Half/full feature removal and matched controls on 512 molecules. Effects are on frozen chemical decoders. Only the family of 40 full-dose feature-versus-control contrasts receives Holm correction. |
| 12 | [Halide illustration](halide_case.pdf) | Exact frozen latest-f7363 positive/false-positive examples, selected after confirmation. Measured spectra and actual feature activations; molecular motifs do not validate peak-to-fragment assignments. |
| 13 | [Chemical neighbors](chemical_neighbors.pdf) | Full versus nuisance-restricted test-to-train banks. The matched population contains 472 queries with at least two candidates. Bank restriction changes both candidates and coverage. |
| 14 | [Component heads](component_heads.pdf) | Heads from validation-selected attention layers, tested on 512 molecules. Effects are non-additive computational ablations. Test-best labels are descriptive rankings within the fixed tested set. |
| 15 | [Activation rescue](activation_rescue.pdf) | Clean-state restoration after top-20% intensity attenuation, 256 molecules. Original readout-vector recovery is distinct from correct-chemistry recovery. Full-state restoration is exact at every tested depth. |
| 16 | [Input robustness](input_robustness.pdf) | Ordinary BCE over 177 train-active outputs on 1,024 molecules. Lower damage must be interpreted alongside worse latest clean error. |
| 17 | [Metadata paths](metadata_paths.pdf) · [TikZ](metadata_paths.tex) | Explanatory 2×2 path intervention with observed same-molecule collision energy. Raw-25 state patching separates block-conditioned state and final modulation; physical input stays fixed. |
| 18 | [Metadata effects](metadata_effects.pdf) | 689 shared eligible molecules. Ordinary BCE and prediction drift under observed-value swaps. Near-zero average loss change can coexist with substantial individual responses. |

Axes and interval definitions are included in the figures and source footnotes. Unless otherwise stated, intervals are descriptive paired molecule-bootstrap intervals conditional on the fixed checkpoints and fitted readouts. No plot introduces new experimental measurements, simulated results or GPU inference.
