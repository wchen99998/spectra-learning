# Encoder chemistry: token-level and mechanistic checkpoint comparison

- [22-slide report](encoder_chemistry.pdf)
- [Editable LaTeX source](encoder_chemistry.tex)
- [Speaker notes and interpretation](encoder_chemistry_notes.md)
- [Figure index](figures/README.md), [plotting source](make_encoder_chemistry_figures.py) and [saved plot inputs](research/plot_inputs.json)
- [Slide-to-evidence index](research/evidence_index.json)
- [Page and artwork review](encoder_chemistry_page_review.md)
- [Source bundle](encoder_chemistry_source.zip)

This report uses the existing shared 16:9 presentation theme, typography, source footnotes and vector-figure conventions. It summarizes the native JAX study in `artifacts/encoder_chemistry_20260930/`, comparing parent checkpoint 1,190,000 with periodic student checkpoint 1,209,000. It is a separate report from `periodic_semantics/`: the native execution, selected molecular panel and token readouts differ, so numerical results from the two studies must not be interchanged.

The main finding is qualified: token-attention chemical readouts favor the parent, while the latest encoder is more sparsely reconstructable and more resistant to peak removal. Nuisance-matched chemical-neighbor differences are inconclusive. Six token-level SAEs, matched feature interventions, component/head ablations, activation rescue and metadata-path swaps distinguish chemical association from causal influence on a fixed readout.

## Diagrams

Three editable TikZ diagrams explain the checkpoint lineage, token-SAE intervention pipeline and observed collision-energy path experiment. They distinguish weight initialization/copying from forward computation and computational intervention. The stored periodic teacher at 1,200,000 is contextual lineage information, not a third encoder measured in this study.

All measured charts are rebuilt from saved inputs. The halide illustration uses the exact frozen positive/false-positive examples from the feature atlas; highlighted spectrum peaks are measured token locations, not validated fragment assignments.

## Build

From the repository root:

```sh
bash presentations/encoder_chemistry/build.sh
```

The build regenerates measured vector figures, compiles the TikZ diagrams, compiles the deck twice with the shared LuaLaTeX helper, renders and inspects the PDF, and packages the sources. It does not run model inference, fit probes or SAEs, or allocate GPUs. Use the repository's existing `.venv`; LuaLaTeX, Poppler utilities, `rsvg-convert` and the installed shared font assets are required. All were already available for this build; no packages were installed.

To regenerate only the charts:

```sh
.venv/bin/python presentations/encoder_chemistry/make_encoder_chemistry_figures.py
```

To inspect an existing build:

```sh
.venv/bin/python presentations/encoder_chemistry/inspect_encoder_chemistry.py
```

The source bundle preserves `presentations/encoder_chemistry/` and the required `presentations/shared/` layout, with compact measured inputs and font assets. It expects the repository's existing Python environment. Individual model-run manifests remain the authority for historical source and checkpoint hashes; the presentation bundle contains the final report sources.

## Statistical scope

Primary readout AUROC uses the mean of three seed logits and fixed shared target support. Intervention loss/AUROC uses the mean of three sigmoid scores; these scores are not calibrated chemical probabilities. Ordinary BCE in the input/metadata experiments differs from class-balanced BCE in the feature/component experiments. Most confidence intervals are descriptive paired molecule-bootstrap intervals; only the predeclared family of 40 full-dose SAE feature-versus-control contrasts uses Holm correction.

Geometry is descriptive over all 6,144 panel molecules at eight selected depths, with a 256-dimensional projection for effective rank. All-block pooled chemistry covers 28 states. Feature and component selection uses training/validation data before test; the highest-activation molecular illustrations were selected after confirmation and do not supply independent confirmatory evidence.

There is no positive-steering sufficiency experiment, head-to-feature mediation test, measured fragment ground truth or matched training-recipe control. The report does not claim an exhaustive chemical circuit or a causal effect of teacher promotion.
