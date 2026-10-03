"""Do frozen encoders already treat one compound at two collision energies as one?

Positives are MassIVE MS2 spectra from the same file and polarity with the same
precursor (|dm/z| <= 0.01), retention times within 12 s, and collision energies
at least 5 eV apart; their binned cosine is ~0.5, so they are non-trivial views.
Negatives share the file and polarity, have a different precursor 0.05-5 Da
away, and also differ by >= 5 eV. For each encoder layer the score is the AUROC
of cosine similarity separating positives from negatives.

    .venv/bin/modal run scripts/modal_cross_ce_invariance.py::build
    .venv/bin/modal run scripts/modal_cross_ce_invariance.py::main --output-dir DIR
"""

from __future__ import annotations

import glob
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import modal
import numpy as np

from scripts.modal_layerwise_semantic_probe import (
    PROJECT_ROOT,
    VOLUME_ROOT,
    _auroc,
    _load_encoder,
    _pooled_layer_features,
    image,
    volume,
)

SHARDS = ("shard_00002.hdf5", "shard_00004.hdf5")
LOCAL_DATA = "data/massive_v2_ms2_t095_l080_sharded_10gb/gems/*/train"
ROWS_PER_SHARD = 4_000_000
PAIRS_PER_SHARD = 6_000
PAIRS_FILE = "cross_ce_pairs.npz"
ENCODERS = ("mae_666k", "ftjepa_1190k", "ctrl_100k", "sjepa_sigreg_2m")
RT_WINDOW_S = 12.0
MIN_CE_GAP = 5.0

app = modal.App("spectra-cross-ce-invariance")


def _shard_pairs(path: str, rng: np.random.Generator) -> dict[str, np.ndarray]:
    import h5py

    with h5py.File(path, "r") as h5:
        rows = slice(0, ROWS_PER_SHARD)
        file_id = h5["file_id"][rows]
        polarity = h5["polarity_id"][rows]
        precursor = h5["precursor_mz"][rows]
        rt = h5["RT"][rows]
        ce = h5["collision_energy"][rows]
        usable = (
            h5["training_eligible"][rows]
            & h5["collision_energy_present"][rows]
            & h5["precursor_mz_present"][rows]
        )
        index = np.flatnonzero(usable)
        order = np.lexsort((rt[index], precursor[index], polarity[index], file_id[index]))
        index = index[order]
        same_run = (np.diff(file_id[index]) == 0) & (np.diff(polarity[index]) == 0)
        ce_gap = np.abs(np.diff(ce[index])) >= MIN_CE_GAP

        # Positives: adjacent rows in (file, polarity, precursor, RT) order.
        positive = np.flatnonzero(
            same_run
            & (np.abs(np.diff(precursor[index])) <= 0.01)
            & (np.abs(np.diff(rt[index])) <= RT_WINDOW_S)
            & ce_gap
        )
        positive = rng.choice(positive, min(PAIRS_PER_SHARD, len(positive)), replace=False)
        positive_pairs = np.stack([index[positive], index[positive + 1]], 1)

        # Negatives: a random row a few positions later in precursor order within
        # the same file, kept when its precursor is 0.05-5 Da away and CE differs.
        offsets = rng.integers(2, 200, size=len(index))
        partner = np.minimum(np.arange(len(index)) + offsets, len(index) - 1)
        mz_gap = np.abs(precursor[index[partner]] - precursor[index])
        negative = np.flatnonzero(
            (file_id[index[partner]] == file_id[index])
            & (polarity[index[partner]] == polarity[index])
            & (mz_gap >= 0.05)
            & (mz_gap <= 5.0)
            & (np.abs(ce[index[partner]] - ce[index]) >= MIN_CE_GAP)
        )
        negative = rng.choice(negative, min(PAIRS_PER_SHARD, len(negative)), replace=False)
        negative_pairs = np.stack([index[negative], index[partner[negative]]], 1)

        pairs = np.concatenate([positive_pairs, negative_pairs])
        members, inverse = np.unique(pairs, return_inverse=True)
        fields = {
            name: h5[name][members]
            for name in (
                "spectrum",
                "precursor_mz",
                "collision_energy",
                "collision_energy_present",
                "charge",
                "charge_present",
                "polarity_id",
                "instrument_family_id",
            )
        }
    fields["pairs"] = inverse.reshape(pairs.shape)
    fields["label"] = np.concatenate(
        [np.ones(len(positive_pairs), bool), np.zeros(len(negative_pairs), bool)]
    )
    return fields


@app.local_entrypoint()
def build() -> None:
    rng = np.random.default_rng(0)
    parts = []
    for shard in SHARDS:
        path = sorted(glob.glob(f"{LOCAL_DATA}/{shard}"))[0]
        parts.append(_shard_pairs(path, rng))
        print(shard, "positives", int(parts[-1]["label"].sum()), "negatives", int((~parts[-1]["label"]).sum()))
    offset = 0
    merged: dict[str, list[np.ndarray]] = {key: [] for key in parts[0]}
    for part in parts:
        for key, value in part.items():
            merged[key].append(value + offset if key == "pairs" else value)
        offset += len(part["spectrum"])
    arrays = {key: np.concatenate(values) for key, values in merged.items()}
    directory = Path(tempfile.mkdtemp())
    np.savez_compressed(directory / PAIRS_FILE, **arrays)
    with volume.batch_upload(force=True) as batch:
        batch.put_file(str(directory / PAIRS_FILE), f"/{PAIRS_FILE}")
    print("spectra", len(arrays["spectrum"]), "pairs", len(arrays["pairs"]))


def _encoder_inputs(archive: dict[str, np.ndarray], config: Any) -> dict[str, np.ndarray]:
    from spectra_learning.data.spectra import (
        COLLISION_ENERGY_MAX,
        canonicalize_precursor_charge_numpy,
        preprocess_peak_batch_numpy,
    )

    batch = preprocess_peak_batch_numpy(
        archive["spectrum"].astype(np.float32),
        archive["precursor_mz"].astype(np.float32),
        num_peaks=int(config.num_peaks),
        peak_drop_min_intensity=float(config.peak_drop_min_intensity),
        peak_ordering=str(config.peak_ordering),
        max_precursor_mz=float(config.max_precursor_mz),
        precursor_peak_exclusion_window_da=float(config.precursor_peak_exclusion_window_da),
        min_peak_intensity=float(config.min_peak_intensity),
    )
    return {
        "peak_mz": batch["peak_mz"],
        "peak_intensity": batch["peak_intensity"],
        "peak_valid_mask": batch["peak_valid_mask"],
        "precursor_mz": batch["precursor_mz"],
        "collision_energy": (
            np.clip(archive["collision_energy"], 0.0, COLLISION_ENERGY_MAX)
            / COLLISION_ENERGY_MAX
        ).astype(np.float32),
        "collision_energy_present": archive["collision_energy_present"].astype(np.float32),
        "charge": canonicalize_precursor_charge_numpy(archive["charge"]),
        "charge_present": archive["charge_present"].astype(np.float32),
        "polarity_id": archive["polarity_id"].astype(np.int64),
        "instrument_family_id": archive["instrument_family_id"].astype(np.int64),
    }


def _pair_auroc(features, pairs, label) -> float:
    import torch

    x = torch.nn.functional.normalize(features.float(), dim=-1)
    cosine = (x[pairs[:, 0]] * x[pairs[:, 1]]).sum(-1)
    return float(_auroc(cosine[:, None], torch.from_numpy(label[:, None]).float())[0])


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=64 * 1024,
    timeout=2 * 60 * 60,
    volumes={str(VOLUME_ROOT): volume},
)
def invariance(name: str) -> dict:
    os.chdir(PROJECT_ROOT)

    import torch
    from ml_collections import config_dict

    started = time.time()
    archive = dict(np.load(VOLUME_ROOT / PAIRS_FILE))
    pairs = torch.from_numpy(archive["pairs"])
    label = archive["label"]
    config = config_dict.ConfigDict(
        json.loads((VOLUME_ROOT / name / "config.json").read_text())
    )
    data = _encoder_inputs(archive, config)

    # Binned sqrt-intensity 1 Da baseline on the same preprocessed peaks.
    mz = data["peak_mz"] * 1000.0
    bins = np.zeros((len(mz), 1001), dtype=np.float32)
    rows, cols = np.nonzero(data["peak_valid_mask"])
    np.add.at(
        bins,
        (rows, np.clip(np.round(mz[rows, cols]).astype(int), 0, 1000)),
        np.sqrt(np.maximum(data["peak_intensity"][rows, cols], 0.0)),
    )
    results: dict[str, Any] = {
        "encoder": name,
        "positives": int(label.sum()),
        "negatives": int((~label).sum()),
        "binned_auroc": _pair_auroc(torch.from_numpy(bins), pairs, label),
    }
    encoder = _load_encoder(VOLUME_ROOT / name, config)
    # "ce_hidden" removes collision energy from the metadata conditioning of
    # both spectra, isolating how much of the gap comes from conditioning.
    hidden = {
        **data,
        "collision_energy": np.zeros_like(data["collision_energy"]),
        "collision_energy_present": np.zeros_like(data["collision_energy_present"]),
    }
    for variant, inputs in (("layers", data), ("layers_ce_hidden", hidden)):
        pooled = _pooled_layer_features(encoder, inputs, config, 256)
        num_layers = pooled["mean"].shape[0]
        results[variant] = {}
        for layer in range(num_layers):
            label_name = (
                "embed" if layer == 0 else "final" if layer == num_layers - 1 else str(layer)
            )
            results[variant][label_name] = {
                pool: _pair_auroc(pooled[pool][layer], pairs, label)
                for pool in ("mean", "cls")
            }
    results["seconds"] = time.time() - started
    return results


@app.local_entrypoint()
def main(encoders: str = ",".join(ENCODERS), output_dir: str = "") -> None:
    output = Path(output_dir) if output_dir else None
    for result in invariance.map(encoders.split(",")):
        summary = {"binned": round(result["binned_auroc"], 4)}
        for variant in ("layers", "layers_ce_hidden"):
            best = max(
                (value, layer, pool)
                for layer, pools in result[variant].items()
                for pool, value in pools.items()
            )
            summary[variant] = {
                "final_mean": round(result[variant]["final"]["mean"], 4),
                "best": (round(best[0], 4), best[1], best[2]),
            }
        print(result["encoder"], summary)
        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            (output / f"{result['encoder']}.json").write_text(json.dumps(result, indent=2))
