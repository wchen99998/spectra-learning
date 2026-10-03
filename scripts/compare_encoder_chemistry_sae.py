"""Compare frozen SAE chemistry features and label-free geometry across checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.analyze_encoder_chemistry_readouts import interval, weighted_auroc
from scripts.analyze_encoder_chemistry_sae import CONCEPTS, SEED, load_panel, sha256, write_json

ARMS = ("parent_1190k", "periodic_1209k")


def compare(root: Path) -> dict:
    output = root / "sae_comparison"
    output.mkdir(exist_ok=True)
    panel, labels, _ = load_panel(root / "panel")
    test = np.flatnonzero(panel["fold"] == "test")
    n = len(test)
    weights = np.random.default_rng(SEED).multinomial(n, np.full(n, 1/n), size=1000)
    directories = {arm: root / arm / "sae/depth19/k32/seed0" for arm in ARMS}
    report = {"schema": "sae_checkpoint_chemistry_comparison_v1", "arms": list(ARMS),
        "panel_sha256": {f: sha256(root / "panel" / f) for f in ("inputs.npz", "labels.npz", "manifest.json")},
        "bootstrap": {"replicates": 1000, "seed": SEED, "paired_unit": "test molecule"},
        "scope": "Conditional on two fixed encoders, fitted primary dictionaries and train/validation-selected features. No pretraining/dictionary-selection variance or multiple-comparison significance claims.",
        "feature_comparison": "Per-concept mean AUROC of all frozen selected features, never test-best feature; counts and missing concepts explicit. IDs are dictionary-specific, not aligned between encoders.",
        "feature_models": {}, "geometry_models": {}, "geometry_checkpoint_contrasts": {}, "feature_checkpoint_contrasts": {}}
    geometries, feature_bootstrap, feature_points = {}, {}, {}
    for arm, directory in directories.items():
        selection = json.loads((directory / "chemical_features.json").read_text())
        confirmation = json.loads((directory / "chemical_features_test_confirmation.json").read_text())
        if confirmation["selection_sha256"] != sha256(directory / "chemical_features.json"):
            raise ValueError("Test confirmation does not reference frozen feature selection")
        report["feature_models"][arm] = {"selection_sha256": confirmation["selection_sha256"], "families": {}}
        report["geometry_models"][arm] = json.loads((directory / "chemical_geometry.json").read_text())
        with np.load(directory / "chemical_geometry_rows.npz") as data:
            geometries[arm] = {key: data[key] for key in data.files}
        for family, records in confirmation["families"].items():
            source = "sae" if family == "sae_equal_dense_search_budget" else family
            path = Path(selection["baseline_files"][source]["path"])
            if sha256(path) != selection["baseline_files"][source]["sha256"]:
                raise ValueError("Chemical features changed after selection")
            features = np.load(path, mmap_mode="r")
            stats, boot_by_concept, points_by_concept = [], {}, {}
            for record in records:
                feature, column = record["feature"], record["concept_index"]
                values = weighted_auroc(features[test, feature], labels["binary"][test, column], weights)
                point = record["test"]["auroc"]
                stats.append({"feature": feature, "concept": record["concept"],
                    "auroc": interval(values, point), "test": record["test"],
                    "nuisance_flags_train_val": record["nuisance_flags"],
                    "nuisance_flags_test": record["test_nuisance_flags"],
                    "matched_donor_ordering": record["test_matched_donor_ordering"]})
                boot_by_concept.setdefault(record["concept"], []).append(values)
                points_by_concept.setdefault(record["concept"], []).append(point)
            grouped = {}
            for concept in CONCEPTS:
                if concept not in boot_by_concept:
                    grouped[concept] = {"selected_features": 0, "auroc": None}
                    continue
                values = np.mean(boot_by_concept[concept], axis=0)
                point = float(np.mean(points_by_concept[concept]))
                grouped[concept] = {"selected_features": len(boot_by_concept[concept]), "auroc": interval(values, point)}
                feature_bootstrap[arm, family, concept] = values
                feature_points[arm, family, concept] = point
            report["feature_models"][arm]["families"][family] = {"per_feature": stats, "concepts": grouped}
    for representation in ("sae", "random_dictionary", "dense_channel", "sae_equal_dense_search_budget"):
        values = {}
        for metric in ("morgan_tanimoto", "functional_group_jaccard", "precursor_mass_gap_da"):
            key = representation + "/" + metric
            for arm in ARMS:
                np.testing.assert_array_equal(geometries[arm][representation + "/test_rows"], test)
            a, b = geometries[ARMS[1]][key], geometries[ARMS[0]][key]
            finite = np.isfinite(a) & np.isfinite(b)
            if not finite.any():
                values[metric] = {"paired_test_molecules": 0, "difference": None}
                continue
            difference = a[finite] - b[finite]
            samples = weights[:, finite] @ difference / weights[:, finite].sum(1)
            values[metric] = {"paired_test_molecules": int(finite.sum()), "difference": interval(samples, difference.mean())}
        report["geometry_checkpoint_contrasts"][representation] = values
    common = sorted(set((family, concept) for arm, family, concept in feature_bootstrap if arm == ARMS[0]) &
                    set((family, concept) for arm, family, concept in feature_bootstrap if arm == ARMS[1]))
    for family, concept in common:
        a, b = (ARMS[1], family, concept), (ARMS[0], family, concept)
        report["feature_checkpoint_contrasts"].setdefault(family, {})[concept] = interval(
            feature_bootstrap[a] - feature_bootstrap[b], feature_points[a] - feature_points[b])
    report["contrast_direction"] = "periodic_1209k minus parent_1190k; positive favors periodic chemical similarity/AUROC, negative favors periodic mass proximity"
    write_json(output / "comparison.json", report)
    plot(report, output)
    return report


def plot(report: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("#3c6e8f", "#bd6849")
    representations = ("sae", "random_dictionary", "dense_channel", "sae_equal_dense_search_budget")
    names = ("Learned SAE (8192)", "Untrained dictionary (8192)", "Residual channels (2048)", "Fixed SAE subset (2048)")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    for axis, metric, title in zip(axes, ("morgan_tanimoto", "functional_group_jaccard"),
                                   ("Morgan fingerprint similarity", "Functional-group overlap"), strict=True):
        for i, arm in enumerate(ARMS):
            records = [report["geometry_models"][arm]["representations"][r]["metrics"][metric] for r in representations]
            points = np.array([r["mean"] for r in records])
            ci = np.array([r["ci95"] for r in records]).T
            axis.errorbar(points, np.arange(4) + (i-.5)*.18, xerr=np.abs(ci-points), fmt="o", capsize=3,
                          color=colors[i], label=("Parent" if i == 0 else "Latest"))
        axis.set(yticks=np.arange(4), yticklabels=names, xlabel="Mean held-out nearest-train similarity", title=title)
        axis.invert_yaxis(); axis.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False)
    fig.suptitle("Token-max geometry · raw cosine retrieval · paired molecule uncertainty")
    for extension in ("png", "pdf"):
        fig.savefig(output / f"chemical_retrieval.{extension}", dpi=180)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(9, 4.7), layout="constrained")
    concepts = [c for c in CONCEPTS if any(report["feature_models"][arm]["families"]["sae"]["concepts"][c]["selected_features"] for arm in ARMS)]
    for i, arm in enumerate(ARMS):
        for position, concept in enumerate(concepts):
            record = report["feature_models"][arm]["families"]["sae"]["concepts"][concept]
            if not record["selected_features"]:
                continue
            stat = record["auroc"]; value = stat["estimate"]; low, high = stat["ci95"]
            axis.errorbar(value, position + (i-.5)*.18, xerr=[[value-low], [high-value]], fmt="o", capsize=3,
                          color=colors[i], label=("Parent" if i == 0 else "Latest") if position == 0 else None)
    axis.axvline(.5, color=".6", linewidth=1)
    axis.set(yticks=np.arange(len(concepts)), yticklabels=[c.removeprefix("fg_") for c in concepts],
             xlabel="Test AUROC · mean of both fixed train/validation-selected features",
             title="SAE chemical candidates · no test-best feature selection")
    axis.invert_yaxis(); axis.spines[["top", "right"]].set_visible(False); axis.legend(frameon=False)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"selected_feature_auroc.{extension}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/encoder_chemistry_20260930"))
    compare(parser.parse_args().root)
