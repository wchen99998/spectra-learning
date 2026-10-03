"""Paired molecular analysis of frozen, train/validation-selected SAE edits.

No model is run here. All ten selected features per arm are retained, including
failed association confirmations. Interventions test a frozen decoder's response
to residual directions, not chemically realizable edits or fragment identities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import ttest_1samp
from sklearn.metrics import roc_auc_score

SEED = 20260930
ARMS = ("parent_1190k", "periodic_1209k")
SUFFIX = "causal/sae_depth19_k32_seed0_test_512"
KINDS = ("feature", "matched_feature_direction", "random_unit_direction")
METRICS = ("target_logit_change", "target_truth_aligned_logit_change", "target_score_change",
           "target_balanced_bce_change", "target_unweighted_bce_change",
           "nontarget_fg_mean_absolute_logit_change", "nontarget_fg_balanced_bce_change",
           "all_supported_balanced_bce_change")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def holm_adjust(pvalues: list[float]) -> list[float]:
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adjusted = np.minimum(1., np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    result = np.empty_like(p)
    result[order] = adjusted
    return result.tolist()


def paired_summary(values: np.ndarray, mask: np.ndarray, counts: np.ndarray,
                   names: tuple[str, ...] = METRICS) -> dict:
    """One bootstrap draw per molecule, shared across variants and metrics."""
    n = int(mask.sum())
    if values.ndim == 1:
        values = values[:, None]
    if values.shape != (len(mask), len(names)) or counts.shape[1] != len(mask):
        raise ValueError("Bootstrap values, names, masks and molecular rows must align")
    if not n:
        return {"molecules": 0, "small_support": True, "metrics": {name: None for name in names}}
    cohort_counts = counts[:, mask]
    denominators = cohort_counts.sum(axis=1)
    valid = denominators > 0
    samples = (cohort_counts[valid] @ values[mask]) / denominators[valid, None]
    means = values[mask].mean(axis=0)
    intervals = np.quantile(samples, [.025, .975], axis=0)
    return {"molecules": n, "small_support": n < 20, "metrics": {
        name: {"mean": float(means[i]), "ci95": intervals[:, i].tolist()}
        for i, name in enumerate(names)}}


def sparse_code_audit(saved: dict, export: dict, selected_features: list[int]) -> dict:
    """Compare codes on identical physical row/slot coordinates; ordering is irrelevant."""
    valid = saved["valid_peak_mask"]
    sample_rows, sample_slots = np.nonzero(valid)
    panel_rows = saved["rows"][sample_rows]
    source_keys = export["rows"].astype(np.int64) * valid.shape[1] + export["slots"]
    target_keys = panel_rows.astype(np.int64) * valid.shape[1] + sample_slots
    if np.any(np.diff(source_keys) <= 0):
        raise ValueError("Exported sparse physical coordinates must be unique and sorted")
    positions = np.searchsorted(source_keys, target_keys)
    if np.any(positions >= len(source_keys)) or not np.array_equal(source_keys[positions], target_keys):
        raise ValueError("Causal physical row/slot absent from exported SAE codes")
    old_i = export["feature_indices"][positions]
    old_v = export["feature_values"][positions].astype(np.float64)
    new_i = saved["sparse_code_indices"][valid]
    new_v = saved["sparse_code_values"][valid].astype(np.float64)
    match = old_i[:, :, None] == new_i[:, None, :]
    aligned_new = np.where(match, new_v[:, None, :], 0).sum(axis=-1)
    missing_new = np.where(match.any(axis=1), 0., new_v)
    square_error = np.square(old_v - aligned_new).sum(axis=1) + np.square(missing_new).sum(axis=1)
    positive_matched = (match & (old_v[:, :, None] > 0) & (new_v[:, None, :] > 0)).sum((1, 2))
    set_changed = (positive_matched != (old_v > 0).sum(1)) | (positive_matched != (new_v > 0).sum(1))
    result = {"valid_physical_tokens": len(positions),
              "ordered_indices_equal": bool(np.array_equal(old_i, new_i)),
              "ordered_values_bitwise_equal": bool(np.array_equal(old_v, new_v)),
              "positive_membership_changed_tokens": int(set_changed.sum()),
              "positive_membership_changed_fraction": float(set_changed.mean()),
              "dense_code_relative_l2": float(np.sqrt(square_error.sum() / np.maximum(np.square(old_v).sum(), 1e-30))),
              "maximum_token_code_l2_difference": float(np.sqrt(square_error.max())),
              "maximum_aligned_old_feature_amplitude_difference": float(np.abs(old_v - aligned_new).max()),
              "selected_features": []}
    for index, feature in enumerate(selected_features):
        before = np.where(old_i == feature, old_v, 0).sum(1)
        after = np.where(new_i == feature, new_v, 0).sum(1)
        np.testing.assert_array_equal(saved["selected_feature_activations"][index][valid], after)
        result["selected_features"].append({"feature": feature,
            "firing_membership_flips": int(np.sum((before > 0) != (after > 0))),
            "max_absolute_amplitude_difference": float(np.max(np.abs(after - before))),
            "amplitude_relative_l2": float(np.linalg.norm(after - before) / max(np.linalg.norm(before), 1e-30))})
    return result


def analyze_arm(root: Path, arm: str, resamples: int) -> dict:
    directory = root / arm / SUFFIX
    manifest = json.loads((directory / "manifest.json").read_text())
    if not manifest["complete"] or manifest["fold"] != "test" or len(manifest["rows"]) != 512:
        raise ValueError("Expected the complete prespecified 512-molecule test cohort")
    if sha256(directory / "predictions.npz") != manifest["predictions_sha256"]:
        raise ValueError("Causal prediction file differs from the verified remote result")
    with np.load(directory / "predictions.npz", allow_pickle=False) as archive:
        saved = dict(archive)
    with np.load(root / "panel/inputs.npz", allow_pickle=False) as archive:
        panel = dict(archive)
    with np.load(root / "panel/labels.npz", allow_pickle=False) as labels:
        truth_all = np.concatenate([labels["maccs"], labels["binary"]], axis=1)
        names = [f"maccs_{i:03d}" for i in range(1, 167)] + labels["binary_names"].tolist()
    for filename, expected in manifest["panel_sha256"].items():
        if sha256(root / "panel" / filename) != expected:
            raise ValueError("Panel no longer matches the pinned causal experiment")
    rows, raw_logits = saved["rows"], saved["logits"].astype(np.float64)
    np.testing.assert_array_equal(rows, manifest["rows"])
    np.testing.assert_array_equal(panel["rows"][rows], saved["source_rows"])
    np.testing.assert_array_equal(panel["peak_valid_mask"][rows], saved["valid_peak_mask"])
    if len(np.unique(panel["molecule"][rows])) != len(rows) or not np.all(panel["fold"][rows] == "test"):
        raise ValueError("Bootstrap needs one independent test row per molecule")
    if raw_logits.shape != (len(manifest["variants"]), len(rows), 3, 188) or not np.isfinite(raw_logits).all():
        raise ValueError("Expected finite intervention × molecule × three fixed heads × target logits")
    np.testing.assert_array_equal(raw_logits[0], raw_logits[1], err_msg="Raw19 sham must be bitwise identical")
    np.testing.assert_array_equal(saved["actual_l2_dose"][1], 0)
    gates = [manifest["baseline_pilot_parity"], *manifest["raw19_sham_parity"]]
    gates += [value for batch in manifest["baseline_cache_parity"] for value in batch.values()]
    if not all(gate["bitwise_equal"] and gate["finite"] for gate in gates):
        raise ValueError("Native, cache and physical-token sham parity gates must all pass")
    truth = truth_all[rows].astype(np.float64)
    positive, negative = truth.sum(0), len(rows) - truth.sum(0)
    train = truth_all[panel["fold"] == "train"]
    support = (positive >= 5) & (negative >= 5) & (train.sum(0) >= 5) & ((1 - train).sum(0) >= 5)
    weights = np.where(truth > 0, len(rows) / (2 * np.maximum(positive, 1)),
                       len(rows) / (2 * np.maximum(negative, 1)))
    probability = expit(raw_logits).mean(axis=2)
    clipped = np.clip(probability, 1e-12, 1 - 1e-12)
    bce = -(truth[None] * np.log(clipped) + (1 - truth[None]) * np.log1p(-clipped))
    delta_logits = raw_logits.mean(axis=2) - raw_logits[0].mean(axis=1)[None]
    delta_scores = probability - probability[0]
    delta_bce = bce - bce[0]
    weighted_delta = delta_bce * weights[None]
    rng = np.random.default_rng(SEED)
    counts = rng.multinomial(len(rows), np.full(len(rows), 1 / len(rows)), size=resamples).astype(np.float64)
    groups = {"MACCS": np.arange(166), "elements": np.arange(166, 173),
              "functional_groups": np.arange(173, 188), "all": np.arange(188)}
    reconstruction = {}
    for name, indices in groups.items():
        indices = indices[support[indices]]
        values = weighted_delta[2][:, indices].mean(1)
        summary = paired_summary(values, np.ones(len(rows), bool), counts, ("balanced_bce_change",))
        summary.update(targets=len(indices), baseline_balanced_bce=float((bce[0][:, indices] * weights[:, indices]).mean()),
                       reconstruction_balanced_bce=float((bce[2][:, indices] * weights[:, indices]).mean()),
                       baseline_auroc=float(np.mean([roc_auc_score(truth[:, j], probability[0, :, j]) for j in indices])),
                       reconstruction_auroc=float(np.mean([roc_auc_score(truth[:, j], probability[2, :, j]) for j in indices])))
        reconstruction[name] = summary
    sae_path = root / arm / "sae/depth19/k32/seed0"
    sae_record = json.loads((sae_path / "trained_sae.json").read_text())
    if sha256(sae_path / "trained_sae.json") != manifest["sae_metadata_sha256"]:
        raise ValueError("SAE metadata differs from the causal source")
    sparse_file = sae_path / "sparse_tokens.npz"
    if sha256(sparse_file) != sae_record["artifact_files"]["sparse_tokens.npz"]["sha256"]:
        raise ValueError("Sparse export hash differs from the fitted dictionary artifact")
    records = manifest["selection"]["records"]
    feature_ids = [record["feature"] for record in records]
    np.testing.assert_array_equal(saved["selected_feature_ids"], feature_ids)
    with np.load(sparse_file, allow_pickle=False) as export:
        code_audit = sparse_code_audit(saved, dict(export), feature_ids)
    confirmation = json.loads((sae_path / "chemical_features_test_confirmation.json").read_text())
    confirmations = {record["feature"]: record for record in confirmation["families"]["sae"]}
    result = {"encoder": arm, "molecules": len(rows), "rows": rows.tolist(), "target_names": names,
              "quality": sae_record["quality"], "reconstruction": reconstruction, "features": [],
              "sparse_code_audit": code_audit, "parity": {"gates": len(gates), "all_bitwise_equal": True,
                    "sham_logits_bitwise_equal": True, "sham_realized_l2": 0.},
              "source": {"manifest_sha256": sha256(directory / "manifest.json"),
                         "predictions_sha256": manifest["predictions_sha256"],
                         "selection_sha256": manifest["selection_sha256"],
                         "sae_metadata_sha256": manifest["sae_metadata_sha256"],
                         "readout_weights_sha256": manifest["readout_weights_sha256"]}}
    profile_logits, profile_bce = [], []
    variant_lookup = {(v.get("feature"), v.get("dose"), v["kind"]): i for i, v in enumerate(manifest["variants"])}
    primary_tests = []
    for number, selected in enumerate(records):
        feature, target = selected["feature"], 166 + selected["concept_index"]
        if names[target] != selected["concept"]:
            raise ValueError("Selected concept does not match target order")
        active = saved["selected_feature_activations"][number].max(axis=1) > 0
        pos = truth[:, target] > 0
        cohorts = {"all": np.ones(len(rows), bool), "positive": pos, "negative": ~pos,
                   "active": active, "inactive": ~active, "active_positive": active & pos,
                   "active_negative": active & ~pos}
        other_fg = np.asarray([j for j in groups["functional_groups"] if j != target and support[j]])
        item = {"feature": feature, "concept": selected["concept"], "target_index": target,
                "validation_auroc": selected["val_auc"], "test_association": confirmations[feature]["test"],
                "test_matched_donor_ordering": confirmations[feature]["test_matched_donor_ordering"],
                "nuisance_flags": confirmations[feature]["test_nuisance_flags"],
                "support": {key: int(mask.sum()) for key, mask in cohorts.items()},
                "random_control_feature": selected["random_control_feature"], "doses": {}}
        for dose in (.5, 1.):
            metrics_by_kind, entries = {}, {}
            for kind in KINDS:
                index = variant_lookup[(feature, dose, kind)]
                np.testing.assert_array_equal(saved["supported"][index], active)
                metrics = np.stack([delta_logits[index, :, target],
                    delta_logits[index, :, target] * (2 * truth[:, target] - 1),
                    delta_scores[index, :, target], weighted_delta[index, :, target], delta_bce[index, :, target],
                    np.abs(delta_logits[index][:, other_fg]).mean(1), weighted_delta[index][:, other_fg].mean(1),
                    weighted_delta[index][:, support].mean(1)], axis=1)
                metrics_by_kind[kind] = metrics
                actual, requested = saved["actual_l2_dose"][index], saved["requested_l2_dose"][index]
                ratios = actual[active] / requested[active]
                entries[kind] = {"variant": manifest["variants"][index]["name"],
                    "cohorts": {key: paired_summary(metrics, mask, counts) for key, mask in cohorts.items()},
                    "realized_dose": {"active_molecules": int(active.sum()),
                        "mean_requested_l2_active": float(requested[active].mean()),
                        "mean_actual_l2_active": float(actual[active].mean()),
                        "mean_relative_l2_active": float(saved["actual_relative_l2_dose"][index, active].mean()),
                        "actual_over_requested_quantiles_active": np.quantile(ratios, [0, .05, .5, .95, 1]).tolist(),
                        "inactive_maximum_actual_l2": float(actual[~active].max(initial=0)),
                        "inactive_maximum_absolute_logit_change": float(np.abs(delta_logits[index, ~active]).max(initial=0))}}
            contrasts, dose_comparisons = {}, {}
            for control in KINDS[1:]:
                difference = metrics_by_kind["feature"] - metrics_by_kind[control]
                contrasts[control] = {key: paired_summary(difference, mask, counts) for key, mask in cohorts.items()}
                feature_index = variant_lookup[(feature, dose, "feature")]
                control_index = variant_lookup[(feature, dose, control)]
                np.testing.assert_array_equal(saved["requested_l2_dose"][feature_index], saved["requested_l2_dose"][control_index])
                actual_ratio = (saved["actual_l2_dose"][feature_index, active]
                                / saved["actual_l2_dose"][control_index, active] - 1)
                dose_comparisons[control] = {"requested_l2_bitwise_equal": True,
                    "realized_feature_over_control_minus1_quantiles": np.quantile(actual_ratio, [0, .01, .5, .99, 1]).tolist(),
                    "absolute_realized_relative_mismatch_median_p99": np.quantile(np.abs(actual_ratio), [.5, .99]).tolist()}
                if dose == 1.:
                    target_difference = difference[:, 0]
                    # Approximate paired t test across molecules; zeros remain.
                    pvalue = (float(ttest_1samp(target_difference, 0.).pvalue)
                              if np.std(target_difference) > 0 else 1.)
                    primary_tests.append({"encoder": arm, "feature": feature, "concept": selected["concept"],
                        "dose": dose, "control": control, "cohort": "all", "metric": METRICS[0],
                        "mean": float(target_difference.mean()), "unadjusted_two_sided_p": pvalue,
                        "ci95": contrasts[control]["all"]["metrics"][METRICS[0]]["ci95"]})
            item["doses"][str(dose)] = {"variants": entries, "feature_minus_control": contrasts,
                                        "control_dose_comparison": dose_comparisons}
        index = variant_lookup[(feature, 1., "feature")]
        profile_logits.append(delta_logits[index].mean(0))
        profile_bce.append(weighted_delta[index].mean(0))
        item["target_profile"] = [{"target": name, "positive": int(positive[j]), "negative": int(negative[j]),
            "supported": bool(support[j]), "mean_logit_change": float(profile_logits[-1][j]),
            "mean_balanced_bce_change": float(profile_bce[-1][j])} for j, name in enumerate(names)]
        ranked_fg = sorted(item["target_profile"][173:], key=lambda entry: -abs(entry["mean_logit_change"]))
        item["functional_group_effect_profile"] = {
            "ranking": "Absolute mean logit change across15 functional groups; descriptive magnitude, not concept specificity proof",
            "nominated_target_rank": 1 + [entry["target"] for entry in ranked_fg].index(item["concept"]),
            "largest_three_effects": [{key: entry[key] for key in ("target", "mean_logit_change")}
                                      for entry in ranked_fg[:3]]}
        result["features"].append(item)
    result["primary_tests"] = primary_tests
    result["full_reconstruction_dose"] = {"mean_actual_l2": float(saved["actual_l2_dose"][2].mean()),
        "mean_relative_l2": float(saved["actual_relative_l2_dose"][2].mean()),
        "mean_requested_l2": float(saved["requested_l2_dose"][2].mean())}
    np.savez_compressed(directory / "sae_causal_target_profiles.npz", feature_ids=np.asarray(feature_ids),
                        target_names=np.asarray(names), mean_logit_change=profile_logits,
                        mean_balanced_bce_change=profile_bce)
    return result


def make_figures(summary: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    colors = {"feature": "#1c6e8c", "matched_feature_direction": "#ce7c32", "random_unit_direction": "#7c6c9d"}
    labels = {"feature": "Feature − baseline", "matched_feature_direction": "Feature − matched direction",
              "random_unit_direction": "Feature − orthogonal direction"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), layout="constrained")
    for ax, arm in zip(axes, summary["arms"]):
        for i, item in enumerate(arm["features"]):
            for offset, kind in zip((-.2, 0, .2), KINDS):
                dose = item["doses"]["1.0"]
                stats = (dose["variants"][kind]["cohorts"]["all"] if kind == "feature"
                         else dose["feature_minus_control"][kind]["all"])["metrics"][METRICS[0]]
                mean, (lo, hi) = stats["mean"], stats["ci95"]
                ax.errorbar(mean, i + offset, xerr=[[mean - lo], [hi - mean]], fmt="o", ms=4,
                            capsize=2, color=colors[kind], label=labels[kind] if i == 0 else None)
        ax.axvline(0, color=".6", lw=.8)
        ax.set(yticks=np.arange(len(arm["features"])),
               yticklabels=[f"{r['concept'].removeprefix('fg_')} · {r['feature']}" for r in arm["features"]],
               xlabel="Target logit change at full dose (paired 95% CI)", title=arm["encoder"])
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle("All frozen SAE features · negative change means less target evidence\n512 test molecules; three fixed heads; intervals are descriptive, Holm tests reported separately")
    fig.savefig(output / "sae_feature_effects.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    groups = ("MACCS", "functional_groups", "all")
    for offset, arm, color in zip((-.16, .16), summary["arms"], ("#64748b", "#16817a")):
        for ax, field in zip(axes, ("balanced_bce_change", "auroc")):
            if field == "balanced_bce_change":
                metrics = [arm["reconstruction"][g]["metrics"][field] for g in groups]
                mean = np.asarray([m["mean"] for m in metrics]); lo, hi = np.asarray([m["ci95"] for m in metrics]).T
                ax.errorbar(np.arange(3) + offset, mean, yerr=[mean - lo, hi - mean], fmt="o", capsize=4,
                            color=color, label=arm["encoder"])
                ax.set(ylabel="Change in class-balanced BCE (paired 95% CI)")
            else:
                mean = [arm["reconstruction"][g]["reconstruction_auroc"] - arm["reconstruction"][g]["baseline_auroc"] for g in groups]
                ax.plot(np.arange(3) + offset, mean, "o", color=color)
                ax.set(ylabel="Change in macro AUROC (point estimates)")
            ax.axhline(0, color=".6", lw=.8)
            ax.set(xticks=np.arange(3), xticklabels=("MACCS", "Functional groups", "All supported"))
            ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle("Full SAE reconstruction is a separate downstream damage baseline")
    fig.savefig(output / "sae_reconstruction_damage.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    for column, arm in enumerate(summary["arms"]):
        x = np.arange(len(arm["features"]))
        text = [f"{r['concept'].removeprefix('fg_')}\n{r['feature']}" for r in arm["features"]]
        positive = [r["support"]["active_positive"] for r in arm["features"]]
        negative = [r["support"]["active_negative"] for r in arm["features"]]
        axes[0, column].bar(x, positive, color="#16817a", label="Active and concept-positive")
        axes[0, column].bar(x, negative, bottom=positive, color="#b9c1cd", label="Active and concept-negative")
        axes[0, column].set(ylabel="Molecules among the causal 512", title=f"{arm['encoder']} · SAE test FVE {arm['quality']['test']['fve_vs_train_mean']:.3f}")
        axes[1, column].plot(x, [r["validation_auroc"] for r in arm["features"]], "o-", color="#64748b", label="Selection validation AUROC")
        axes[1, column].plot(x, [r["test_association"]["auroc"] for r in arm["features"]], "s-", color="#ce7c32", label="Independent 1024 test association")
        axes[1, column].axhline(.5, color=".5", ls="--", lw=.8)
        axes[1, column].set(ylabel="Single-feature concept AUROC", ylim=(.2, 1.))
        for ax in axes[:, column]:
            ax.set(xticks=x, xticklabels=text)
            ax.tick_params(axis="x", labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Support and independent confirmation · weak and failed features remain included")
    fig.savefig(output / "sae_support_quality.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), layout="constrained")
    bound = max(abs(entry["mean_logit_change"]) for arm in summary["arms"] for r in arm["features"] for entry in r["target_profile"][173:])
    for ax, arm in zip(axes, summary["arms"]):
        values = np.asarray([[entry["mean_logit_change"] for entry in r["target_profile"][173:]] for r in arm["features"]])
        im = ax.imshow(values, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
        for i, item in enumerate(arm["features"]):
            ax.add_patch(Rectangle((item["target_index"] - 173 - .5, i - .5), 1, 1, fill=False, ec="black", lw=1.5))
        ax.set(xticks=np.arange(15), xticklabels=[name.removeprefix("fg_") for name in arm["target_names"][173:]],
               yticks=np.arange(10), yticklabels=[f"{r['concept'].removeprefix('fg_')} · {r['feature']}" for r in arm["features"]], title=arm["encoder"])
        ax.tick_params(axis="x", rotation=70, labelsize=8)
    fig.colorbar(im, ax=axes, label="Mean logit change after feature removal, dose1", shrink=.7)
    fig.suptitle("Functional-group response profiles · black squares mark the selected target")
    fig.savefig(output / "sae_target_profiles.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for ax, arm in zip(axes, summary["arms"]):
        for kind in KINDS:
            for dose, marker in (("0.5", "o"), ("1.0", "s")):
                actual = [item["doses"][dose]["variants"][kind]["realized_dose"] for item in arm["features"]]
                ax.scatter([v["mean_requested_l2_active"] for v in actual],
                           [v["mean_actual_l2_active"] for v in actual], color=colors[kind], marker=marker,
                           label=f"{kind.replace('_', ' ')} · {dose}", alpha=.8)
        maximum = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, maximum], [0, maximum], color=".6", ls="--", lw=.8)
        ax.set(xlabel="Requested L2, active molecules", ylabel="Realized bf16 L2, active molecules", title=arm["encoder"])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Direction controls share requested dose; native bf16 rounding is measured")
    fig.savefig(output / "sae_dose_realization.png", dpi=180)
    plt.close(fig)


def make_cases(summary: dict, root: Path, output: Path) -> None:
    """Three concepts fixed before outcomes; feature choice uses validation only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rdkit import Chem
    from rdkit.Chem import Draw

    arm = next(item for item in summary["arms"] if item["encoder"] == "periodic_1209k")
    source = root / arm["encoder"] / "sae/depth19/k32/seed0"
    with np.load(root / "panel/inputs.npz", allow_pickle=False) as data:
        panel = dict(data)
    panel_manifest = json.loads((root / "panel/manifest.json").read_text())
    mz_scale = panel_manifest["physical_peak_mz_scale"]
    with np.load(source / "feature_statistics.npz", allow_pickle=False) as data:
        examples = dict(data)
    with np.load(root / "panel/labels.npz", allow_pickle=False) as data:
        targets, target_names = data["binary"], data["binary_names"].tolist()
    cases = []
    chosen = [max((item for item in arm["features"] if item["concept"] == concept),
                  key=lambda item: (item["validation_auroc"], -item["feature"]))
              for concept in ("fg_sulfonyl", "fg_halide", "fg_nitrile")]
    # Both already-frozen sulfonyl features are illustrated. The fourth case was
    # added descriptively after confirmation; it does not add a new tested feature.
    chosen.insert(1, next(item for item in arm["features"] if item["concept"] == "fg_sulfonyl"
                          and item["feature"] != chosen[0]["feature"]))
    fig, axes = plt.subplots(4, 3, figsize=(14, 12), gridspec_kw={"width_ratios": [1.1, 1.8, 1.4]}, layout="constrained")
    for row_number, feature in enumerate(chosen):
        concept = feature["concept"]
        fid = feature["feature"]
        candidates = [(int(row), int(slot), float(value)) for row, slot, value in zip(
            examples["top_rows"][fid], examples["top_slots"][fid], examples["top_values"][fid])
            if row >= 0 and panel["fold"][row] == "train"]
        if not candidates:
            raise ValueError("Frozen top-activation export lacks a training example for a case feature")
        row, slot, value = candidates[0]
        info = {"encoder": arm["encoder"], "feature": fid, "concept": concept,
                "selection": ("Second already-frozen sulfonyl feature; descriptive case added after confirmation" if row_number == 1 else
                              "Highest validation AUROC among frozen features for sulfonyl/halide/nitrile")
                              + "; top activation TRAIN example from existing export",
                "row": row, "source_row": int(panel["rows"][row]), "slot": slot,
                "molecule": str(panel["molecule"][row]), "smiles": str(panel["smiles"][row]),
                "label": int(targets[row, target_names.index(concept)]), "activation": value,
                "peak_mz": float(panel["peak_mz"][row, slot] * mz_scale), "peak_mz_normalized": float(panel["peak_mz"][row, slot]),
                "peak_intensity_processed": float(panel["peak_intensity"][row, slot]),
                "validation_auroc": feature["validation_auroc"], "test_association": feature["test_association"],
                "causal_support": feature["support"], "causal_dose1": feature["doses"]["1.0"],
                "interpretation": "Example peak carries a molecular concept-associated activation; no fragment identity label is asserted"}
        cases.append(info)
        mol = Chem.MolFromSmiles(info["smiles"])
        axes[row_number, 0].imshow(Draw.MolToImage(mol, size=(480, 320)))
        axes[row_number, 0].axis("off")
        axes[row_number, 0].set_title(f"{concept.removeprefix('fg_')} · feature {fid}\nTraining example; label={info['label']}", fontsize=11)
        valid = panel["peak_valid_mask"][row]
        mz, intensity = panel["peak_mz"][row] * mz_scale, panel["peak_intensity"][row]
        axes[row_number, 1].vlines(mz[valid], 0, intensity[valid], color="#a8b2bf", lw=1)
        axes[row_number, 1].vlines(mz[slot], 0, intensity[slot], color="#b95640", lw=2)
        axes[row_number, 1].annotate(f"m/z {mz[slot]:.3f}\nSAE a={value:.2f}", (mz[slot], intensity[slot]),
                                   xytext=(8, 12), textcoords="offset points", fontsize=8)
        axes[row_number, 1].set(xlabel="m/z", ylabel="Processed intensity")
        axes[row_number, 1].set_ylim(top=float(intensity[valid].max()) * 1.35)
        axes[row_number, 1].spines[["top", "right"]].set_visible(False)
        effect = feature["doses"]["1.0"]
        direct = effect["variants"]["feature"]["cohorts"]["all"]["metrics"][METRICS[0]]
        lines = [f"Validation / test AUC: {feature['validation_auroc']:.3f} / {feature['test_association']['auroc']:.3f}",
                 f"Active in causal cohort: {feature['support']['active']}/512",
                 f"Target logit shift: {direct['mean']:+.4f}", f"95% CI [{direct['ci95'][0]:+.4f}, {direct['ci95'][1]:+.4f}]"]
        for kind, title in (("matched_feature_direction", "vs matched"), ("random_unit_direction", "vs orthogonal")):
            metric = effect["feature_minus_control"][kind]["all"]["metrics"][METRICS[0]]
            lines += [f"{title}: {metric['mean']:+.4f}", f"  [{metric['ci95'][0]:+.4f}, {metric['ci95'][1]:+.4f}]"]
        axes[row_number, 2].text(0, .95, "\n".join(lines), va="top", fontsize=10, linespacing=1.5)
        axes[row_number, 2].axis("off")
    fig.suptitle("Fixed concept cases · representation association and computational intervention remain separate")
    fig.savefig(output / "sae_chemical_cases.png", dpi=180)
    plt.close(fig)
    write_json(output / "sae_chemical_cases.json", {
        "case_rule": "Primary three cases: validation-best frozen sulfonyl/halide/nitrile features. Fourth descriptive case adds the other already-frozen sulfonyl feature after confirmation. No new feature is tested or omitted from full analysis.",
        "examples_use_train_only": True, "physical_peak_mz_scale": mz_scale, "cases": cases})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/encoder_chemistry_20260930"))
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()
    if args.resamples < 1000:
        raise ValueError("Use at least 1000 paired molecule bootstrap draws")
    output = args.root / "sae_causal_analysis"
    output.mkdir(parents=True, exist_ok=True)
    arms = [analyze_arm(args.root, arm, args.resamples) for arm in ARMS]
    np.testing.assert_array_equal(arms[0]["rows"], arms[1]["rows"])
    tests = [test for arm in arms for test in arm["primary_tests"]]
    for test, adjusted in zip(tests, holm_adjust([test["unadjusted_two_sided_p"] for test in tests])):
        test["holm_adjusted_p"] = adjusted
        test["family_comparisons"] = len(tests)
    result = {"arms": arms, "primary_tests": tests,
              "protocol": {"bootstrap_resamples": args.resamples, "bootstrap_seed": SEED,
                  "bootstrap_unit": "One test molecule; shared multinomial resamples across variants, heads fixed; subgroup denominators recomputed within each bootstrap",
                  "head_ensemble": "Logit shifts average three frozen heads; loss/AUROC use mean of three sigmoid scores",
                  "bce": "Class-balanced BCE using empirical test-cohort class weights fixed across variants and bootstrap; probabilities are uncalibrated readout scores",
                  "support": "Macro targets need at least5 positive and5 negative molecules in both train and this test cohort; all selected feature targets and cohorts are reported",
                  "primary_family": "Dose1 target-logit feature-minus-control differences on all512 molecules, two controls, ten features, two checkpoints; paired t approximations with Holm correction",
                  "multiplicity": "Bootstrap95% intervals are unadjusted descriptive intervals. Dose0.5, subgroup, reconstruction and188-target profiles are secondary, not simultaneous discovery claims.",
                  "selection": "All train/validation-frozen features retained; no feature exclusion or new feature selection on test outcomes",
                  "causal_scope": "Frozen-readout response to computational dictionary-direction edits; does not establish monosemanticity, fragment identity, or a chemically realizable molecule intervention"},
              "analysis_source_sha256": sha256(Path(__file__))}
    write_json(output / "sae_causal_analysis.json", result)
    for arm in arms:
        write_json(args.root / arm["encoder"] / SUFFIX / "analysis.json", arm)
    make_figures(result, output)
    make_cases(result, args.root, output)
    print(json.dumps({"complete": True, "output": str(output), "features": sum(len(arm["features"]) for arm in arms),
                      "primary_tests": len(tests), "holm_below_0_05": sum(test["holm_adjusted_p"] < .05 for test in tests)}), flush=True)


if __name__ == "__main__":
    main()
