"""Supplemental nuisance-matched nearest-train chemistry with fixed panel calipers.

The candidate bank is identical for every representation and checkpoint. Chemical
labels only score neighbors and the uniform-bank expectation; they never select
candidates or distance transforms. At least two candidates are required so that
the representation can affect which molecule is retrieved.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.analyze_encoder_chemistry_readouts import interval
from scripts.analyze_encoder_chemistry_sae import SEED, load_panel, sha256, write_json


def candidate_banks(panel: dict) -> tuple[np.ndarray, list[np.ndarray]]:
    train = np.flatnonzero(panel["fold"] == "train")
    test = np.flatnonzero(panel["fold"] == "test")
    counts = panel["peak_valid_mask"].sum(1)
    present = panel["collision_energy_present"].astype(bool)
    banks = []
    for row in test:
        mask = ((panel["instrument_type"][train] == panel["instrument_type"][row]) &
                (panel["adduct"][train] == panel["adduct"][row]) & (present[train] == present[row]) &
                (np.abs(panel["precursor_mz_raw"][train] - panel["precursor_mz_raw"][row]) <= 5.) &
                (np.abs(counts[train] - counts[row]) <= 5) & (panel["molecule"][train] != panel["molecule"][row]))
        if present[row]:
            mask &= np.abs(panel["collision_energy_raw"][train] - panel["collision_energy_raw"][row]) <= 10.
        banks.append(train[mask])
    return test, banks


def jaccard(bits: np.ndarray, row: int, neighbors: np.ndarray) -> np.ndarray:
    both = np.logical_and(bits[row], bits[neighbors]).sum(1)
    either = np.logical_or(bits[row], bits[neighbors]).sum(1)
    return np.divide(both, either, out=np.ones(len(neighbors)), where=either > 0)


def retrieve(features: np.ndarray, test: np.ndarray, banks: list[np.ndarray],
             panel: dict, labels: dict) -> dict:
    x = np.asarray(features, np.float32)
    norms = np.linalg.norm(x, axis=1)
    x = np.divide(x, norms[:, None], out=np.zeros_like(x), where=norms[:, None] > 0)
    fg = labels["binary"][:, len(labels["element_names"]):].astype(bool)
    values = {metric: np.full(len(test), np.nan) for metric in ("morgan_tanimoto", "functional_group_jaccard", "precursor_mass_gap_da")}
    values["nearest_train_row"] = np.full(len(test), -1, int)
    for i, (row, bank) in enumerate(zip(test, banks, strict=True)):
        if len(bank) < 2 or norms[row] == 0 or np.any(norms[bank] == 0):
            continue  # never change the shared bank according to a representation
        similarities = x[bank] @ x[row]
        nearest = bank[similarities == similarities.max()]
        values["nearest_train_row"][i] = nearest[0]
        values["morgan_tanimoto"][i] = jaccard(labels["morgan"], row, nearest).mean()
        values["functional_group_jaccard"][i] = jaccard(fg, row, nearest).mean()
        values["precursor_mass_gap_da"][i] = np.abs(panel["precursor_mz_raw"][nearest] - panel["precursor_mz_raw"][row]).mean()
    return values


def analyze(root: Path) -> dict:
    panel, labels, _ = load_panel(root / "panel")
    test, banks = candidate_banks(panel)
    count = np.array([len(bank) for bank in banks])
    common = count >= 2
    data = {}
    for arm in ("parent_1190k", "periodic_1209k"):
        directory = root / arm / "sae/depth19/k32/seed0"
        selection = json.loads((directory / "chemical_features.json").read_text())
        for representation, metadata in selection["baseline_files"].items():
            path = Path(metadata["path"])
            if sha256(path) != metadata["sha256"]:
                raise ValueError("Features differ from the frozen primary analysis")
            features = np.load(path, mmap_mode="r")
            data[arm + "/" + representation] = retrieve(features, test, banks, panel, labels)
            common &= np.isfinite(data[arm + "/" + representation]["morgan_tanimoto"])
    if not common.any():
        raise ValueError("No common test molecules have at least two matched training candidates")
    fg = labels["binary"][:, len(labels["element_names"]):].astype(bool)
    uniform = {name: np.full(len(test), np.nan) for name in ("morgan_tanimoto", "functional_group_jaccard", "precursor_mass_gap_da")}
    for i in np.flatnonzero(common):
        row, bank = test[i], banks[i]
        uniform["morgan_tanimoto"][i] = jaccard(labels["morgan"], row, bank).mean()
        uniform["functional_group_jaccard"][i] = jaccard(fg, row, bank).mean()
        uniform["precursor_mass_gap_da"][i] = np.abs(panel["precursor_mz_raw"][bank] - panel["precursor_mz_raw"][row]).mean()
    n = int(common.sum())
    weights = np.random.default_rng(SEED).multinomial(n, np.full(n, 1/n), size=1000)
    report = {"schema": "nuisance_matched_geometry_v1", "supplemental": True,
        "calipers": {"precursor_mass_da": 5, "raw_collision_energy": 10, "processed_peak_count": 5},
        "exact_matches": ["instrument_type", "adduct", "collision_energy_present"],
        "policy": "Reuse panel donor calipers unchanged; train candidates/test queries, distinct connectivity. At least2 candidates, identical pool for all representations; raw L2 cosine, exact score ties averaged. No chemical-label candidate filtering or test tuning.",
        "support": {"test_molecules": len(test), "no_candidate": int((count == 0).sum()), "one_candidate_only": int((count == 1).sum()),
            "at_least_two_candidates": int((count >= 2).sum()), "common_nonzero_representation_molecules": n,
            "candidate_count_quantiles_supported": np.quantile(count[common], [0, .25, .5, .75, 1]).tolist()},
        "bootstrap": {"replicates": 1000, "seed": SEED, "unit": "shared supported test molecule", "paired": True},
        "limits": "Restricted-support observational control; residual nuisance and scaffold differences remain. No causal inference and no claim of population-wide improvement.",
        "models": {}, "checkpoint_contrasts": {}, "uniform_candidate_baseline": {}, "source_sha256": sha256(Path(__file__))}
    for metric in uniform:
        scores = uniform[metric][common]
        report["uniform_candidate_baseline"][metric] = interval(weights @ scores/n, scores.mean())
    for key, values in data.items():
        report["models"][key] = {}
        for metric in uniform:
            scores = values[metric][common]
            delta = scores - uniform[metric][common]
            report["models"][key][metric] = {"absolute": interval(weights @ scores/n, scores.mean()),
                "minus_uniform_candidate_expectation": interval(weights @ delta/n, delta.mean())}
    for representation in ("sae", "random_dictionary", "dense_channel"):
        report["checkpoint_contrasts"][representation] = {}
        for metric in uniform:
            delta = data["periodic_1209k/" + representation][metric][common] - data["parent_1190k/" + representation][metric][common]
            report["checkpoint_contrasts"][representation][metric] = interval(weights @ delta/n, delta.mean())
    report["contrast_direction"] = "Latest minus parent"
    output = root / "sae_comparison"
    write_json(output / "matched_geometry.json", report)
    np.savez_compressed(output / "matched_geometry_rows.npz", test_rows=test, candidate_counts=count, common=common,
                        **{key + "/" + metric: scores for key, values in data.items() for metric, scores in values.items()})
    return report


if __name__ == "__main__":
    result = analyze(Path("artifacts/encoder_chemistry_20260930"))
    print(json.dumps(result["support"]))
