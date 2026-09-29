# Periodic teacher promotion: measured findings

- [Five-slide overview](../periodic_semantics_overview/periodic_semantics_overview.pdf)
- [Full 21-slide presentation](periodic_semantics.pdf)
- [Detailed findings and limitations](periodic_semantics_findings.md)
- [Speaker notes](periodic_semantics_notes.md)
- [Main editable source](periodic_semantics.tex) and [overview source](../periodic_semantics_overview/periodic_semantics_overview.tex)
- [Machine-readable summary](research/summary.json) and [metric table](research/metrics.csv)
- [Vector figures](figures/) and [plotting source](make_periodic_semantics_figures.py)

The presentation follows the existing shared unbranded theme and figure standard. Main conclusions are deliberately mixed: chemical information survives; global similarity and dropout robustness improve; chemical probes, local neighbors and mean-pool repeat-spectrum retrieval weaken relative to the initial teacher. A matched nonpromotion run was not conducted, so promotion's isolated causal effect remains unmeasured.

## Experiments

Three states were evaluated: the initial frozen-teacher JEPA encoder at step 1,190,000, the teacher stored in the latest periodic checkpoint (student at step 1,200,000), and the latest student at step 1,209,000. All 231,104 MassSpecGym 1.5 spectra were processed. The main fresh protocol balances by 28,929 molecule identities, with an additional single-spectrum control. The broad suite includes chemical probes, depths and pooling variants, four coordinate treatments, local and global chemical geometry, label budgets, shuffled labels, covariance, repeat spectra, perturbations and paired uncertainty.

Raw JSON and per-molecule predictions are under `results/periodic_semantics_20260929/`. The supplied source bundle includes these predictions. The existing 28 September depth and attention results were copied into the evidence directory with `legacy_` prefixes and are clearly distinguished from new measurements.

The first physical-mass-window implementation used normalized precursor units. That error was detected during analysis, corrected, and the geometry tests rerun. Only corrected window results belong in the final report. The rerun also adds first-spectrum-per-molecule comparisons and caches centroids remotely. Other chemical-probe predictions are unchanged. The original and corrected source hashes are retained for provenance.

## Reproduce

Use the repository's existing `.venv`. Frozen encoders and the dataset must already be staged on the existing `spectra-layerwise-semantic-probe` Modal volume. Access to the existing Modal and GCS resources is required for inference; regenerating plots and PDFs needs only the saved evidence and theme assets.

```sh
.venv/bin/python -m modal run scripts/modal_periodic_semantics.py::main --output-dir results/periodic_semantics_20260929
.venv/bin/python -m modal run scripts/modal_periodic_semantics.py::physical_mass_windows --output-dir results/periodic_semantics_20260929
.venv/bin/python -m scripts.analyze_periodic_semantics
.venv/bin/python presentations/periodic_semantics/write_periodic_semantics_report.py
.venv/bin/python presentations/periodic_semantics/make_periodic_semantics_figures.py
bash presentations/periodic_semantics/build.sh
```

The second inference command includes the single-spectrum extension. For the recorded geometry-only correction, `analyze_periodic_semantics --reuse-probe-bootstrap` retains the verified, unchanged chemical-probe intervals. The standard command recomputes everything from the saved predictions.

Build uses LuaLaTeX and the repository's multi-file `presentation-theme.sty`, licensed local PP Neue Montreal fonts, bundled Liberation Sans and vector figures. No TeX or Python packages were installed for this work. The source is opened in the native editor; the exported PDFs are compiled and inspected using the repository's existing toolchain. The native standalone compiler was attempted and cannot resolve `presentation-theme.sty`; it does not support this project's additional theme/assets. The editable source remains available and the exported PDFs compile successfully.

## Statistical and scientific scope

Full protocol, assumptions, uncertainty definitions and caveats are in the findings document. In particular: centroids use identity labels; no supervised metric is trained for geometry; raw and postprocessed distances are separate; pairs share molecules; internal-layer comparisons are exploratory; training-seed uncertainty and pretraining-overlap audits are absent. The report does not claim exact structure recovery, a universally optimal layer, or a causal promotion effect.

## Verification

Focused statistics tests verify ties, true nearest-mass selection, paired resampling and weighted AUROC bootstrap against explicit resampling. Existing promotion and JAX/PyTorch state-conversion checks passed. The presentation review checks every page and every standalone figure, fonts, vector artwork, page bounds and compiler diagnostics. See [page review](periodic_semantics_page_review.md) and `research/artifact_audit.json` for the final recorded audit.
