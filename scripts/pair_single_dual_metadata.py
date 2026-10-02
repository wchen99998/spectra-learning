"""MassSpecGym acquisition metadata for the pair/single dual pilots.

``prepare`` writes per-row metadata aligned with the pilot panel (precursor
m/z alignment is checked). ``condition_vectors`` builds the production
``massive_v2_acquisition_v1`` layout (28 values, see
``spectra_learning.models.spectrum_metadata``) from a chosen subset of fields,
optionally followed by a three-value adduct extension that MassIVE pretraining
data does not carry. ``drop_fields`` mirrors the production per-field dropout,
and ``blind`` gives the all-absent vector used to test reliance on metadata.
Run from the repository root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from spectra_learning.data.spectra import COLLISION_ENERGY_MAX, PRECURSOR_CHARGE_MAX
from spectra_learning.models.spectrum_metadata import (
    MASSIVE_V2_CONDITION_DIM, POLARITY_POSITIVE, instrument_family_id,
)

FIELDS = ("precursor", "ce", "instrument", "adduct")
ADDUCTS = ("[M+H]+", "[M+Na]+")
ADDUCT_MASS_DA = (1.007276, 22.989218)
PRECURSOR_SCALE_DA = 1000.0  # panel precursor_mz is divided by PEAK_MZ_MAX


def parse_fields(text: str) -> tuple[str, ...]:
    fields = tuple(f for f in text.split("+") if f)
    unknown = set(fields) - set(FIELDS)
    if unknown:
        raise ValueError(f"unknown metadata fields {sorted(unknown)}; choose from {FIELDS}")
    return fields


def condition_dim(fields: tuple[str, ...]) -> int:
    return MASSIVE_V2_CONDITION_DIM + 3 * ("adduct" in fields)


def condition_vectors(meta: dict, precursor_mz: np.ndarray, fields: tuple[str, ...]) -> np.ndarray:
    """Full-information condition vectors, one row per panel spectrum."""
    n = len(precursor_mz)
    out = np.zeros((n, condition_dim(fields)), np.float32)
    if "precursor" in fields:
        out[:, 0], out[:, 1] = precursor_mz, 1.0
    if "ce" in fields:
        present = meta["collision_energy_present"].astype(np.float32)
        out[:, 2] = np.clip(meta["collision_energy"], 0.0, COLLISION_ENERGY_MAX) / COLLISION_ENERGY_MAX * present
        out[:, 3] = present
    # MassSpecGym spectra are all singly charged positive-mode adducts.
    out[:, 4], out[:, 5] = 1.0 / PRECURSOR_CHARGE_MAX, 1.0
    out[:, 12 + POLARITY_POSITIVE] = 1.0
    out[:, 15] = 1.0  # acquisition type unknown
    family = meta["instrument_family_id"] if "instrument" in fields else np.zeros(n, np.int64)
    out[np.arange(n), 21 + family] = 1.0
    if "adduct" in fields:
        out[:, 28] = 1.0
        out[np.arange(n), 29 + meta["adduct_id"]] = 1.0
    return out


def blind(vectors: np.ndarray) -> np.ndarray:
    out = np.zeros_like(vectors)
    out[:, 12] = out[:, 15] = out[:, 21] = 1.0  # unknown polarity, acquisition, instrument
    return out


def drop_fields(vectors: np.ndarray, probability: float, rng: np.random.Generator) -> np.ndarray:
    """Per-field dropout with the production semantics (each field independently)."""
    out = vectors.copy()
    if probability <= 0:
        return out
    dropped = rng.random((len(out), 6)) < probability
    for column, (value, present) in enumerate(((0, 1), (2, 3), (4, 5))):
        out[dropped[:, column], value] = 0.0
        out[dropped[:, column], present] = 0.0
    out[dropped[:, 3], 12:15] = (1.0, 0.0, 0.0)
    out[dropped[:, 4], 21:28] = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if out.shape[1] > MASSIVE_V2_CONDITION_DIM:
        out[dropped[:, 5], MASSIVE_V2_CONDITION_DIM:] = 0.0
    return out


def prepare(args):
    import pyarrow.csv as csv

    columns = ["precursor_mz", "adduct", "instrument_type", "collision_energy"]
    rows = csv.read_csv(args.tsv, parse_options=csv.ParseOptions(delimiter="\t"),
                        convert_options=csv.ConvertOptions(include_columns=columns)).to_pydict()
    panel = np.load(args.panel)
    precursor = np.asarray(rows["precursor_mz"], np.float64)
    if len(precursor) != len(panel["precursor_mz"]) or np.abs(
            precursor / PRECURSOR_SCALE_DA - panel["precursor_mz"]).max() > 1e-6:
        raise ValueError("TSV rows are not aligned with the panel")
    adduct = np.array([ADDUCTS.index(a) for a in rows["adduct"]], np.int64)
    energy = np.array([np.nan if v is None else float(v) for v in rows["collision_energy"]])
    data = {
        "collision_energy": np.nan_to_num(energy).astype(np.float32),
        "collision_energy_present": ~np.isnan(energy),
        "instrument_family_id": np.array([instrument_family_id(v) for v in rows["instrument_type"]],
                                         np.int64),
        "adduct_id": adduct,
        "neutral_mass_da": precursor - np.asarray(ADDUCT_MASS_DA)[adduct],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **data)
    print(json.dumps({
        "rows": len(adduct), "ce_present": float(data["collision_energy_present"].mean()),
        "instrument_families": np.bincount(data["instrument_family_id"]).tolist(),
        "adducts": np.bincount(adduct).tolist()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", default="data/massspecgym/data/MassSpecGym1.5.tsv")
    parser.add_argument("--panel", required=True)
    parser.add_argument("--output", required=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
