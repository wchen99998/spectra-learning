"""Align fixed primary SAE features using train activations, then audit val/test.

Alignment never uses chemical labels. Train rank correlation is the primary
matching rule; nearest decoder cosine is a separate prespecified diagnostic.
This compares dictionaries of one checkpoint, never feature IDs across models.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from scripts.analyze_encoder_chemistry_sae import (
    choose_threshold, classification, donor_ordering, load_panel,
    rank_correlation, sha256, write_json,
)


def normalized_rank_columns(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(values, axis=0, method="average")
    centered = ranks - ranks.mean(axis=0)
    norms = np.linalg.norm(centered, axis=0)
    return np.divide(centered, norms, out=np.zeros_like(centered), where=norms > 0)


def align_features(primary: np.ndarray, other: np.ndarray, rows: np.ndarray,
                   selected: list[int], primary_decoder: np.ndarray,
                   other_decoder: np.ndarray) -> list[dict]:
    """Use only supplied train rows; exact ties prefer the smaller feature ID."""
    if primary.shape[0] != other.shape[0] or primary_decoder.shape[1] != other_decoder.shape[1]:
        raise ValueError("Dictionaries must share molecules and residual coordinates")
    left = normalized_rank_columns(np.asarray(primary[rows][:, selected]))
    right = normalized_rank_columns(np.asarray(other[rows]))
    correlations = left.T @ right
    left_decoder = primary_decoder / np.linalg.norm(primary_decoder, axis=1, keepdims=True)
    right_decoder = other_decoder / np.linalg.norm(other_decoder, axis=1, keepdims=True)
    cosines = left_decoder[selected] @ right_decoder.T
    matches = []
    for position, feature in enumerate(selected):
        if not np.any(left[:, position]):
            raise ValueError("A selected primary feature has no training activation variation")
        # Constant target features cannot be activated or meaningfully correlated.
        eligible = np.any(right != 0, axis=0)
        if not eligible.any():
            raise ValueError("The control dictionary has no variable training features")
        score = np.where(eligible, correlations[position], -np.inf)
        activation_feature = int(np.argmax(score))
        decoder_feature = int(np.argmax(cosines[position]))
        record = {"feature": feature,
            "activation_alignment": {"aligned_feature": activation_feature,
                "train_spearman": float(correlations[position, activation_feature]),
                "decoder_cosine": float(cosines[position, activation_feature])},
            "decoder_alignment": {"aligned_feature": decoder_feature,
                "train_spearman": float(correlations[position, decoder_feature]),
                "decoder_cosine": float(cosines[position, decoder_feature])}}
        # Mutual matching is descriptive; it never changes the selected mapping.
        for key in ("activation_alignment", "decoder_alignment"):
            target = record[key]["aligned_feature"]
            if key == "decoder_alignment":
                reverse = int(np.argmax(left_decoder @ right_decoder[target]))
                record[key]["mutual_nearest_decoder"] = reverse == feature
        matches.append(record)
    return matches


def analyze(primary_path: Path, control_path: Path, panel_path: Path) -> dict:
    selection = json.loads((primary_path / "chemical_features.json").read_text())
    original = selection["contract"]
    metadata = json.loads((control_path / "trained_sae.json").read_text())
    contract = metadata["contract"]
    if not metadata["complete"] or any(contract[k] != original[k] for k in ("encoder", "depth")):
        raise ValueError("Control must be a completed dictionary for the same checkpoint/depth")
    if original["test_used_for_selection"] or original["selection_splits"] != ["train", "val"]:
        raise ValueError("Primary feature selection has invalid splits")
    for name, expected in original["panel_sha256"].items():
        if sha256(panel_path / name) != expected:
            raise ValueError("Panel differs from frozen feature selection")
    for key in ("inputs.npz", "labels.npz", "manifest.json", "matched_controls.npz"):
        if contract["panel_sha256"][key] != original["panel_sha256"][key]:
            raise ValueError("Control dictionary uses a different panel")
    panel, labels, controls = load_panel(panel_path)
    rows = {fold: np.flatnonzero(panel["fold"] == fold) for fold in ("train", "val", "test")}
    arrays, decoders = [], []
    for directory, weight_hash, activation_hash in (
        (primary_path, original["params_sha256"], original["molecule_max_sha256"]),
        (control_path, metadata["weights_sha256"], metadata["artifact_files"]["molecule_max.npy"]["sha256"]),
    ):
        if sha256(directory / "params.npz") != weight_hash or sha256(directory / "molecule_max.npy") != activation_hash:
            raise ValueError("Dictionary parameters or activations differ from completed metadata")
        arrays.append(np.load(directory / "molecule_max.npy", mmap_mode="r"))
        with np.load(directory / "params.npz") as params:
            decoders.append(params["decoder"])
    primary, other = arrays
    features = [record["feature"] for record in selection["records"]]
    aligned = align_features(primary, other, rows["train"], features, *decoders)
    # Freeze all dictionary mappings and control thresholds before reading test.
    validation_records = []
    for record, match in zip(selection["records"], aligned, strict=True):
        feature, column = record["feature"], record["concept_index"]
        result = {**match, "concept": record["concept"], "concept_index": column}
        for rule in ("activation_alignment", "decoder_alignment"):
            target = result[rule]["aligned_feature"]
            train, val = rows["train"], rows["val"]
            threshold = choose_threshold(other[train, target], other[val, target], labels["binary"][val, column])
            result[rule]["threshold"] = threshold["selected"]["threshold"]
            result[rule]["validation"] = classification(other[val, target], labels["binary"][val, column], result[rule]["threshold"])
            result[rule]["validation_spearman"] = rank_correlation(primary[val, feature], other[val, target])
            result[rule]["validation_matched_donors"] = donor_ordering(other[val, target], labels["binary"][val, column], val, controls, column)
        validation_records.append(result)
    identifier = f"k{contract['k']}_seed{contract['seed']}"
    output = primary_path / "stability"
    output.mkdir(exist_ok=True)
    report = {"schema": "sae_feature_stability_v1", "primary_selection_sha256": sha256(primary_path / "chemical_features.json"),
        "primary_params_sha256": original["params_sha256"], "control_params_sha256": metadata["weights_sha256"],
        "control_metadata_sha256": sha256(control_path / "trained_sae.json"), "control_contract": contract,
        "alignment_split": "train", "test_used_for_alignment": False,
        "chemical_labels_used_for_alignment": False,
        "primary_rule": "Highest positive-direction training molecule-max Spearman across all control features; no chemistry-based remapping",
        "secondary_rule": "Highest positive decoder cosine, with mutual nearest check; no absolute/sign-flipped matching",
        "threshold_rule": "Primary frozen concept, control train quantiles and validation F1; test never selects thresholds",
        "variation": "seed1 changes initialization, optimization order and within-molecule token sampling; k64/seed0 shares sampled tokens and normalization with k32/seed0",
        "limits": "Many-feature matching inflates training agreement; validation and test correlations diagnose replication. Sparse features, reconstruction quality, and different k can limit one-to-one correspondence. Duplicate control matches are retained and reported.",
        "source_sha256": sha256(Path(__file__)), "records": validation_records,
        "duplicate_control_matches": {rule: {str(k): v for k, v in Counter(r[rule]["aligned_feature"] for r in validation_records).items() if v > 1}
                                      for rule in ("activation_alignment", "decoder_alignment")},
        "control_reconstruction": metadata["quality"]}
    frozen_path = output / f"{identifier}_alignment.json"
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if frozen_path.exists() and frozen_path.read_text() != serialized:
        raise ValueError("Frozen alignment differs; use a new experiment output")
    frozen_path.write_text(serialized)
    confirmation = {"alignment_sha256": sha256(frozen_path), "test_molecules": len(rows["test"]), "records": []}
    test = rows["test"]
    for record in validation_records:
        result = {"feature": record["feature"], "concept": record["concept"]}
        column, feature = record["concept_index"], record["feature"]
        for rule in ("activation_alignment", "decoder_alignment"):
            target = record[rule]["aligned_feature"]
            result[rule] = {"aligned_feature": target,
                "test_spearman": rank_correlation(primary[test, feature], other[test, target]),
                "test": classification(other[test, target], labels["binary"][test, column], record[rule]["threshold"]),
                "test_matched_donors": donor_ordering(other[test, target], labels["binary"][test, column], test, controls, column)}
        confirmation["records"].append(result)
    write_json(output / f"{identifier}_test_confirmation.json", confirmation)
    return confirmation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--panel", type=Path, default=Path("artifacts/encoder_chemistry_20260930/panel"))
    args = parser.parse_args()
    result = analyze(args.primary, args.control, args.panel)
    print("Aligned and confirmed", len(result["records"]), "fixed primary features", flush=True)


if __name__ == "__main__":
    main()
