"""Unsupervised native JAX SAE study on parity-qualified physical peak tokens.

Each explicitly launched job fits one dictionary. Primary: depth19, 8192 atoms,
k32, seed0. No chemistry label enters sampling, normalization, training, or
checkpoint selection. Source hashes and source snapshots accompany artifacts.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import time

import modal

from scripts.modal_encoder_chemistry import CHECKPOINTS, LOCAL, PROJECT, ROOT, error_metrics, image, sha256, volume, write_json

app = modal.App("spectra-encoder-chemistry-sae")


def _top_examples(indices, values, rows, slots, features: int, count: int = 10):
    import numpy as np

    flat_feature, flat_value = indices.ravel(), values.ravel()
    positive = np.flatnonzero(flat_value > 0)
    ordering = positive[np.lexsort((-flat_value[positive], flat_feature[positive]))]
    sorted_features = flat_feature[ordering]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_features)) + 1]
    ranks = np.arange(len(ordering)) - np.repeat(starts, np.diff(np.r_[starts, len(ordering)]))
    selected = ranks < count
    position, feature, rank = ordering[selected], sorted_features[selected], ranks[selected]
    result = {"top_rows": np.full((features, count), -1, np.int32),
              "top_slots": np.full((features, count), -1, np.int16),
              "top_values": np.zeros((features, count), np.float32)}
    result["top_rows"][feature, rank] = rows[position // indices.shape[1]]
    result["top_slots"][feature, rank] = slots[position // indices.shape[1]]
    result["top_values"][feature, rank] = flat_value[position]
    return result


@app.function(image=image, gpu="H100", cpu=8, memory=65536, timeout=14400,
              volumes={"/results": volume})
def fit(name: str, depth: int = 19, k: int = 32, seed: int = 0, features: int = 8192,
        epochs: int = 40) -> dict:
    os.chdir(PROJECT)
    import jax
    import jax.numpy as jnp
    import ml_dtypes
    import numpy as np
    from scripts.encoder_chemistry_sae import (
        dictionary_redundancy, encode_sae, load_sae, reconstruction_metrics,
        sample_physical_tokens, save_sae, train_sae,
    )

    if name not in CHECKPOINTS or depth != 19:
        raise ValueError("Use a pinned encoder's raw depth19 physical tokens")
    started = time.monotonic()
    source, panel = ROOT / name, ROOT / "panel"
    extraction = json.loads((source / "extraction.json").read_text())
    if not extraction["complete"] or not extraction["extraction_vs_native_pilot"]["bitwise_equal"]:
        raise ValueError("SAE training requires completed extraction with exact native pilot parity")
    hashes = {filename: sha256(panel / filename) for filename in (
        "inputs.npz", "labels.npz", "manifest.json", "matched_controls.npz",
    )}
    if hashes["inputs.npz"] != extraction["inputs_sha256"]:
        raise ValueError("Panel differs from extracted encoder inputs")
    with np.load(panel / "inputs.npz", allow_pickle=False) as data:
        valid, folds, molecules, original_rows = [data[key] for key in ("peak_valid_mask", "fold", "molecule", "rows")]
    if len(np.unique(molecules)) != len(molecules):
        raise ValueError("SAE panel requires unique molecules and no identity transfer across folds")
    if {fold: int(np.sum(folds == fold)) for fold in np.unique(folds)} != {"train": 4096, "val": 1024, "test": 1024}:
        raise ValueError("Expected the pinned 4096/1024/1024 molecule panel")
    # Labels are inspected only for identity alignment, never supplied to SAE fit.
    with np.load(panel / "labels.npz", allow_pickle=False) as labels:
        for key, expected in (("rows", original_rows), ("fold", folds), ("molecule", molecules)):
            np.testing.assert_array_equal(labels[key], expected)
    token_file = source / "tokens_bf16.npy"
    storage = np.load(token_file, mmap_mode="r", allow_pickle=False)
    if (storage.dtype != np.uint16 or list(storage.shape) != extraction["shapes"]["tokens_bf16"]
            or token_file.stat().st_size != extraction["files"]["tokens_bf16"]["bytes"]):
        raise ValueError("Token storage differs from the completed extraction contract")
    with np.load(source / "pilot_tokens.npz", allow_pickle=False) as pilot:
        cached = storage[extraction["token_depths"].index(27), pilot["rows"]].view(ml_dtypes.bfloat16).astype(np.float32)
        pilot_mask = np.concatenate([valid[pilot["rows"]], np.ones((len(pilot["rows"]), 1), bool)], axis=1)
        np.testing.assert_array_equal(pilot["valid"], pilot_mask)
        parity = {"native": error_metrics(cached, pilot["native"], pilot_mask),
                  "trace": error_metrics(cached, pilot["instrumented"], pilot_mask)}
    if any(not entry["bitwise_equal"] or not entry["finite"] for entry in parity.values()):
        raise ValueError("Final token cache must equal the native and full-trace pilots bitwise")
    tokens = storage[extraction["token_depths"].index(depth)].view(ml_dtypes.bfloat16)
    if tokens.shape[:2] != (len(valid), valid.shape[1] + 1):
        raise ValueError("Physical token validity and trailing CLS do not align")
    destination = source / "sae" / f"depth{depth}" / f"k{k}" / f"seed{seed}"
    contract = {
        "encoder": name, "checkpoint": extraction["checkpoint"], "depth": depth, "k": k, "seed": seed,
        "features": features, "token_scope": "physical_peaks_only; trailing CLS and padding excluded",
        "extraction_sha256": sha256(source / "extraction.json"),
        "token_source_sha256": extraction["files"]["tokens_bf16"]["sha256"], "panel_sha256": hashes,
        "cache_final_parity": parity, "pilot_tokens_sha256": sha256(source / "pilot_tokens.npz"),
        "token_dtype": "bfloat16 decoded losslessly from uint16; SAE parameters and arithmetic float32",
        "sampling": {"maximum_per_molecule": 32, "seed": seed, "within_molecule": "uniform without replacement",
                     "weight": "1 / sampled physical tokens in this molecule; total weight 1 per molecule",
                     "export": "all valid physical tokens; maxima and firing counts are not sampling-limited"},
        "source_sha256": {filename: sha256(PROJECT / "scripts" / filename) for filename in (
            "encoder_chemistry_sae.py", "modal_encoder_chemistry_sae.py",
        )}, "epochs_max": epochs, "selection": "minimum validation reconstruction MSE; no chemical labels",
    }
    if (destination / "trained_sae.json").exists():
        _, saved = load_sae(destination)
        if saved["contract"] != contract:
            raise ValueError("Existing SAE belongs to a different data/code contract")
        return saved
    destination.mkdir(parents=True, exist_ok=True)
    snapshot = destination / "source"
    snapshot.mkdir(exist_ok=True)
    for filename in contract["source_sha256"]:
        shutil.copyfile(PROJECT / "scripts" / filename, snapshot / filename)
    sampled = sample_physical_tokens(valid, maximum=32, seed=seed)
    sampled["fold"] = folds[sampled["rows"]]
    np.savez_compressed(destination / "sampled_tokens.npz", **sampled)
    sampled_x = np.asarray(tokens[sampled["rows"], sampled["slots"]], np.float32)
    subset = {fold: sampled["fold"] == fold for fold in ("train", "val", "test")}
    jax.config.update("jax_default_matmul_precision", "highest")
    result = train_sae(sampled_x[subset["train"]], sampled_x[subset["val"]],
                       train_weights=sampled["weights"][subset["train"]],
                       validation_weights=sampled["weights"][subset["val"]],
                       features=features, k=k, seed=seed, epochs=epochs, patience=5,
                       learning_rate=3e-4, batch_size=1024)
    params = jax.tree.map(jnp.asarray, result["params"])
    initial = jax.tree.map(jnp.asarray, result["initial_params"])
    quality, sampled_counts = {}, {}
    for fold in ("train", "val", "test"):
        metrics = reconstruction_metrics(params, sampled_x[subset[fold]], sampled["weights"][subset[fold]], k=k)
        sampled_counts[fold] = metrics.pop("firing_counts")
        quality[fold] = metrics
    initial_validation = reconstruction_metrics(initial, sampled_x[subset["val"]], sampled["weights"][subset["val"]], k=k)
    initial_validation.pop("firing_counts")
    redundancy = dictionary_redundancy(params)
    np.savez(destination / "dictionary_statistics.npz", nearest_feature=redundancy.pop("nearest_feature"),
             max_absolute_cosine=redundancy.pop("max_absolute_cosine"))
    np.savez(destination / "random_dictionary_params.npz", **result["initial_params"])
    np.savez(destination / "sampled_firing_counts.npz", **sampled_counts)
    del sampled_x
    all_rows, all_slots = np.nonzero(valid)
    all_rows, all_slots = all_rows.astype(np.int32), all_slots.astype(np.int16)
    indices = np.empty((len(all_rows), k), np.int32)
    values = np.empty((len(all_rows), k), np.float32)
    maxima = np.zeros((len(valid), features), np.float32)
    random_maxima = np.zeros_like(maxima)
    encode = jax.jit(lambda parameters, batch: encode_sae(parameters, batch, k))
    batch_size = 1024
    for first in range(0, len(all_rows), batch_size):
        last = min(first + batch_size, len(all_rows))
        rows, slots = all_rows[first:last], all_slots[first:last]
        batch = jnp.asarray(tokens[rows, slots], jnp.float32)
        code = encode(params, batch)
        ix, val = np.asarray(code.indices), np.asarray(code.values)
        indices[first:last], values[first:last] = ix, val
        np.maximum.at(maxima, (rows[:, None], ix), val)
        random_code = encode(initial, batch)
        np.maximum.at(random_maxima, (rows[:, None], np.asarray(random_code.indices)), np.asarray(random_code.values))
    np.save(destination / "molecule_max.npy", maxima)
    np.save(destination / "random_dictionary_molecule_max.npy", random_maxima)
    dense = np.load(source / "max.npy", mmap_mode="r", allow_pickle=False)
    np.save(destination / "dense_channel_molecule_max.npy", np.asarray(dense[depth]))
    np.savez(destination / "sparse_tokens.npz", feature_indices=indices, feature_values=values,
             rows=all_rows, slots=all_slots, fold=folds[all_rows])
    statistics = _top_examples(indices, values, all_rows, all_slots, features)
    for fold in ("train", "val", "test"):
        selected = folds[all_rows] == fold
        counts = np.bincount(indices[selected][values[selected] > 0], minlength=features)
        statistics[f"firing_counts_{fold}"] = counts
        statistics[f"firing_molecules_{fold}"] = (maxima[folds == fold] > 0).sum(axis=0)
        statistics[f"tokens_{fold}"] = np.asarray(selected.sum())
    statistics["max_activation"] = maxima.max(axis=0)
    np.savez(destination / "feature_statistics.npz", **statistics)
    artifact_hashes = {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
                       for path in destination.iterdir() if path.is_file()}
    result["metadata"].update({"quality": quality, "initial_dictionary_validation": initial_validation,
                               "dictionary_redundancy": redundancy, "artifact_files": artifact_hashes,
                               "valid_physical_tokens": len(all_rows), "molecules": len(valid),
                               "sampling_counts": {fold: int(subset[fold].sum()) for fold in subset},
                               "device": str(jax.devices()[0]), "seconds": time.monotonic() - started,
                               "feature_semantics": "Unsupervised directions; monosemanticity and chemical causality require independent tests",
                               "top_examples": "Global top10 physical peaks per feature, includes all folds; selection must use train/validation only"})
    save_sae(result, destination, contract=contract)
    volume.commit()
    record = json.loads((destination / "trained_sae.json").read_text())
    print(name, "SAE complete", json.dumps(quality), flush=True)
    return record


@app.local_entrypoint()
def main(name: str = "both", k: int = 32, seed: int = 0, epochs: int = 40):
    names = ("parent_1190k", "periodic_1209k") if name == "both" else (name,)
    jobs = [(encoder, fit.spawn(encoder, k=k, seed=seed, epochs=epochs)) for encoder in names]
    write_json(LOCAL / f"sae_launch_{name}_k{k}_seed{seed}.json",
               {"jobs": [{"name": encoder, "call_id": job.object_id} for encoder, job in jobs]})
    for encoder, job in jobs:
        result = job.get()
        write_json(LOCAL / encoder / "sae" / "depth19" / f"k{k}" / f"seed{seed}" / "trained_sae.json", result)
        print(encoder, "SAE artifact ready", flush=True)
