"""Generate the findings table and numbers from saved experimental evidence."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "research"
ARMS = ("mae_666k", "ftjepa_1190k", "periodic_student_1200k", "periodic_student_1209k")


def main():
    s = json.loads((DATA / "summary.json").read_text())
    mae, initial, promoted, latest = [s["models"][arm] for arm in ARMS]
    states = (mae, initial, promoted, latest)
    fs = [d["features"]["mean_final"] for d in states]
    delta = s["paired_differences"][ARMS[3]]
    rows = []
    def row(label, values):
        rows.append("| " + label + " | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    for label,key in (("MACCS macro AUROC","maccs_auc"),("Functional-group AUROC","fg_auc"),("Mean property R²","r2_mean")):
        row(label,[f["ridge"]["1.0"][key] for f in fs])
    for label,v,key in (("Raw-cosine neighbor Tanimoto","raw","knn1_tanimoto"),
                        ("Standardized neighbor Tanimoto","standardized","knn1_tanimoto"),
                        ("Whitened neighbor Tanimoto","whiten_0.01","knn1_tanimoto"),
                        ("Raw pair Spearman","raw","pair_spearman"),
                        ("Raw partial Spearman","raw","pair_partial_spearman")):
        row(label,[f["geometry"][v][key] for f in fs])
    row("Final-mean effective rank",[f["collapse"]["effective_rank"] for f in fs])
    row("Final-CLS effective rank",[d["features"]["cls_final"]["collapse"]["effective_rank"] for d in states])
    row("Repeat-spectrum mean-pool retrieval",[f["replicate_retrieval"]["top1"] for f in fs])
    for name,desc in (("drop20","20% peak-dropout self-retrieval"),("intensity_jitter","Intensity-jitter self-retrieval"),("metadata_zero","Zero-metadata self-retrieval")):
        row(desc,[d["robustness"][name]["self_retrieval_top1"] for d in states])
    if "single_spectrum" in latest:
        row("Single-spectrum raw neighbor Tanimoto",[d["single_spectrum"]["raw"]["knn1_tanimoto"] for d in states])
        row("Single-spectrum ±5 Th neighbor Tanimoto",[d["single_spectrum"]["raw"]["knn1_mass_window_5Da"] for d in states])
    intervals=[]
    for key in ("mean_final_maccs_auc","mean_19_maccs_auc","cls_final_maccs_auc","mean_final_raw_knn1","mean_final_standardized_knn1", "single_raw_knn1", "single_raw_knn_mass5"):
        d=delta[key]
        intervals.append(f"| {key} | {d['delta']:+.5f} | [{d['ci95'][0]:+.5f}, {d['ci95'][1]:+.5f}] |")
    stage_intervals = []
    for label, effects in (("JEPA initial − MAE", s["stage_differences"]["jepa_minus_mae"]),
                           ("Periodic student − JEPA initial", delta),
                           ("Periodic student − MAE", s["stage_differences"]["latest_minus_mae"])):
        for key, metric in (("mean_final_maccs_auc", "MACCS AUROC"), ("mean_final_raw_knn1", "Raw neighbor Tanimoto")):
            d = effects[key]
            stage_intervals.append(f"| {label} | {metric} | {d['delta']:+.5f} | [{d['ci95'][0]:+.5f}, {d['ci95'][1]:+.5f}] |")
    stage_text = f"""
## Original MAE → JEPA initial teacher → periodic endpoint

All four checkpoints now have the same fresh molecule-balanced protocol, including the original MAE encoder at step 666,000. MAE supplied the frozen targets for the JEPA student at step 1,190,000, which then became the periodic run's initial teacher. Each student's training step count is local to its run; the periodic student started from random weights. These are checkpoint comparisons, not one continuously fine-tuned model or measurements of every promotion.

Final-mean MACCS AUROC is **{fs[0]['ridge']['1.0']['maccs_auc']:.4f} → {fs[1]['ridge']['1.0']['maccs_auc']:.4f} → {fs[3]['ridge']['1.0']['maccs_auc']:.4f}** for MAE, JEPA initial teacher and later periodic student. The intervening promoted teacher at 1,200,000 scores **{fs[2]['ridge']['1.0']['maccs_auc']:.4f}**. Raw neighbor Tanimoto is **{fs[0]['geometry']['raw']['knn1_tanimoto']:.4f} → {fs[1]['geometry']['raw']['knn1_tanimoto']:.4f} → {fs[3]['geometry']['raw']['knn1_tanimoto']:.4f}**. The full table above also compares properties, geometry, covariance, repeat spectra and all three perturbations.

The initial JEPA checkpoint improves final-mean chemical probes over MAE, but the periodic endpoint falls below both on MACCS, functional groups and mean property R². Repeat-spectrum retrieval decreases from 43.7% (MAE) to 36.0% (initial JEPA) to 26.2% (periodic student). MAE already has 98.3% dropout self-retrieval, so the periodic endpoint's 98.8% mainly restores stability lost at the initial JEPA checkpoint (32.4%). Raw partial global chemical correlation instead moves 0.201 → 0.153 → 0.243.

JEPA minus MAE MACCS AUROC is +0.01038 (95% CI +0.00720 to +0.01343); periodic student minus MAE is −0.00948 (−0.01406 to −0.00489). The JEPA–MAE raw-neighbor Tanimoto difference is −0.00127 (−0.00434 to +0.00161), which includes zero. Thus the initial JEPA gain is in chemical decodability, not an established raw-neighborhood gain.

The initial JEPA phase and later periodic phase must be assessed separately. A gain over the MAE ancestor would not by itself establish a gain from periodic promotion. Paired confidence intervals below directly compare both transitions and the endpoint against MAE, using identical test molecules and labels. Dataset SHA-256, test identities and target arrays are checked before pairing.

| Comparison | Endpoint | Difference | 95% paired molecule-bootstrap interval |
|---|---|---:|---:|
""" + "\n".join(stage_intervals) + "\n"
    body = """# Periodic teacher promotion: chemical semantics audit

**Verdict: chemical structure is present, but promotion did not deliver a broad improvement over the initial teacher.** The endpoint trades stronger global chemical-similarity ordering and peak-dropout stability for weaker chemical probes, local neighbors and repeat-spectrum recognition. This is an observational comparison of the specified run, not an isolated estimate of promotion's causal effect.

Evaluated 29 September 2026. [Requested W&B run](https://wandb.ai/iclac/jepa-finalrun/runs/1p8b-ptjepa-t1190k-p50k-b1024-v6e32-20260922-062250). W&B reports `crashed`, with last logged step 1,208,750. GCS contains complete checkpoint metadata through **1,209,000**. The W&B and latest checkpoint configurations match exactly. The snapshot contains the student and its frozen teacher from periodic step **1,200,000** (24 promotions). The **initial teacher** is the `encoder` of frozen-teacher JEPA checkpoint **1,190,000**, not that checkpoint's own MAE teacher.

The original MAE encoder at **666,000** is also measured with the full fresh audit. It is the teacher ancestor, not the periodic run's initial teacher.

## Final mean-pooled representation

Fresh molecule-balanced audit. All 231,104 spectra were encoded; molecular centroids use the arithmetic mean of spectra sharing a full InChIKey. There are 22,746 training, 3,185 validation and 2,998 test molecules. The candidate database consists only of training molecules. Centroid results use molecular identity for averaging and are not single-spectrum deployment scores.

| Measure | Original MAE 666k | JEPA initial teacher 1,190k | Promoted teacher 1,200k | Student 1,209k |
|---|---:|---:|---:|---:|
""" + "\n".join(rows) + """

All seven final-mean property R² values and all three label-budget MACCS scores are lower for the latest student than for the initial JEPA teacher. The student nevertheless exceeds the freshly evaluated binned-spectrum baseline (MACCS AUROC 0.6538; raw nearest-neighbor Tanimoto 0.1369). Random-molecule neighbor Tanimoto is 0.0951. Training-label permutation gives MACCS AUROC approximately 0.50 for all four encoders. Chemical information was not erased.

""" + stage_text + """
## Uncertainty

Paired differences below are latest student minus initial teacher. Rows are molecules, not spectra or fingerprint bits. Percentile intervals condition on the trained encoders and probe fits; they do not include training-seed variability. There is no multiple-testing correction. The final-mean comparison is the main endpoint; selected internal-layer and CLS checks are supporting analyses.

| Endpoint | Difference | 95% paired molecule-bootstrap interval |
|---|---:|---:|
""" + "\n".join(intervals) + """

AUROC uses average ranks for ties and excludes targets with only one test class: 151 of 166 MACCS bits are test-evaluable. The bootstrap uses 500 paired molecule resamples, retaining this set and excluding any bit undefined within a resample. Neighbor means use 2,000 paired resamples. Correlation uses 99,955 distinct-molecule pairs sampled with seed 20260929; pairs share molecules and are not treated as independent observations. No misleading pair-bootstrap intervals are reported.

## What improved, and what did not

The following changes compare the periodic endpoint to its initial JEPA teacher. The preceding section separately compares both to the original MAE.

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
"""
    (ROOT / "periodic_semantics_findings.md").write_text(body)
    numbers = {"MaeAUC":f"{fs[0]['ridge']['1.0']['maccs_auc']:.3f}",
               "PromotedAUC":f"{fs[2]['ridge']['1.0']['maccs_auc']:.3f}",
               "InitialAUC":f"{fs[1]['ridge']['1.0']['maccs_auc']:.3f}","LatestAUC":f"{fs[3]['ridge']['1.0']['maccs_auc']:.3f}",
               "MaccsDelta":f"{delta['mean_final_maccs_auc']['delta']:.3f}","RepeatCount":"2,659",
               "RandomTanimoto":f"{initial['baselines']['random_molecule_tanimoto']:.3f}",
               "MassTanimoto":f"{initial['baselines']['nearest_mass_tanimoto']:.3f}",
               "PermutationAUC":f"{fs[1]['permuted_label_ridge']['maccs_auc']:.3f}"}
    (DATA / "numbers.tex").write_text("\n".join("\\newcommand{\\"+k+"}{"+v+"}" for k,v in numbers.items())+"\n")


if __name__ == "__main__":
    main()
