"""Frozen-token readouts for the native encoder chemistry study; launch explicitly.

Default sweep prioritizes final output (27) and raw block 19. The wrapper never
changes or refits encoders. Other exported depths can be requested later.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import modal

from scripts.modal_encoder_chemistry import (
    CHECKPOINTS, LOCAL, PROJECT, ROOT, error_metrics, image, sha256, volume, write_json,
)


app = modal.App("spectra-encoder-chemistry-readouts")


@app.function(image=image, gpu="H100", cpu=8, memory=65536, timeout=10800,
              volumes={"/results": volume})
def fit(
    name: str, depth: int = 27, kind: str = "attention",
    seeds: tuple[int, ...] = (0, 1, 2), include_cls: bool = False,
) -> dict:
    os.chdir(PROJECT)
    import jax
    import ml_dtypes
    import numpy as np
    from scripts.encoder_chemistry_readouts import load_readout, save_readout, train_readout

    if name not in CHECKPOINTS or kind not in ("mean", "attention"):
        raise ValueError("Use a pinned study encoder and mean/attention readout")
    source = ROOT / name
    extraction = json.loads((source / "extraction.json").read_text())
    if not extraction["complete"] or depth not in extraction["token_depths"]:
        raise ValueError("Completed token extraction at the requested depth is required")
    panel = ROOT / "panel"
    hashes = {filename: sha256(panel / filename) for filename in (
        "inputs.npz", "labels.npz", "manifest.json", "matched_controls.npz",
    )}
    if hashes["inputs.npz"] != extraction["inputs_sha256"]:
        raise ValueError("Readout panel differs from the extracted token inputs")
    with np.load(panel / "inputs.npz", allow_pickle=False) as saved:
        valid = saved["peak_valid_mask"]
        folds, molecules, rows = saved["fold"], saved["molecule"], saved["rows"]
        analysis_rows = {key: saved[key] for key in (
            "rows", "molecule", "identifier", "fold", "scaffold", "unseen_nonempty_train_scaffold",
            "exact_mass", "adduct", "instrument_type", "collision_energy_raw", "precursor_mz_raw",
        )}
    with np.load(panel / "labels.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["rows"], rows)
        np.testing.assert_array_equal(saved["molecule"], molecules)
        np.testing.assert_array_equal(saved["fold"], folds)
        labels = np.concatenate([saved["maccs"], saved["binary"]], axis=1)
        target_names = [f"maccs_{index:03d}" for index in range(1, 167)] + saved["binary_names"].tolist()
    token_file = source / "tokens_bf16.npy"
    if token_file.stat().st_size != extraction["files"]["tokens_bf16"]["bytes"]:
        raise ValueError("Token file size differs from completed extraction")
    storage = np.load(token_file, mmap_mode="r", allow_pickle=False)
    if list(storage.shape) != extraction["shapes"]["tokens_bf16"] or storage.dtype != np.uint16:
        raise ValueError("Unexpected token storage shape or dtype")
    if storage.shape[1:3] != (len(rows), valid.shape[1] + 1) or labels.shape[1] != 188:
        raise ValueError("Tokens, physical masks and 188 chemistry labels do not align")
    with np.load(source / "pilot_tokens.npz", allow_pickle=False) as pilot:
        cache_final = storage[extraction["token_depths"].index(27), pilot["rows"]].view(ml_dtypes.bfloat16).astype(np.float32)
        expected_valid = np.concatenate([valid[pilot["rows"]], np.ones((len(pilot["rows"]), 1), bool)], axis=1)
        np.testing.assert_array_equal(pilot["valid"], expected_valid)
        cache_parity = {
            "cache_vs_pilot_trace": error_metrics(cache_final, pilot["instrumented"], expected_valid),
            "cache_vs_pilot_native": error_metrics(cache_final, pilot["native"], expected_valid),
            "tolerance_relative_l2": 0.,
            "rationale": "Training requires bitwise identity to batch8 native and full-trace pilots; initial batch16 caches are excluded",
        }
    if any(not cache_parity[key]["finite"] or not cache_parity[key]["bitwise_equal"]
           for key in ("cache_vs_pilot_trace", "cache_vs_pilot_native")):
        raise ValueError(f"Final token cache differs from verified pilot: {cache_parity}")
    tokens = storage[extraction["token_depths"].index(depth)].view(ml_dtypes.bfloat16)
    mask = np.concatenate([valid, np.full((len(valid), 1), include_cls, dtype=bool)], axis=1)
    readout_root = source / "readouts"
    readout_root.mkdir(parents=True, exist_ok=True)
    # Small row-addressable sidecar preserves scaffold and acquisition controls;
    # matched molecule pairs remain the original panel artifact referenced below.
    sidecar = readout_root / "panel_rows.npz"
    if not sidecar.exists():
        np.savez_compressed(sidecar, **analysis_rows)
    contract = {
        "encoder": name, "checkpoint": extraction["checkpoint"], "depth": depth,
        "token_scope": "physical_peaks_and_cls" if include_cls else "physical_peaks_only",
        "mask_policy": "original physical validity plus explicitly declared trailing CLS validity",
        "token_shape": list(tokens.shape), "token_dtype": "bfloat16; decoded losslessly from uint16 storage",
        "token_source": str(token_file), "token_source_sha256": extraction["files"]["tokens_bf16"]["sha256"],
        "extraction_sha256": sha256(source / "extraction.json"), "panel_sha256": hashes,
        "cache_final_parity": cache_parity, "pilot_tokens_sha256": sha256(source / "pilot_tokens.npz"),
        "panel_rows": str(sidecar), "matched_controls": str(panel / "matched_controls.npz"),
        "source_sha256": {filename: sha256(PROJECT / "scripts" / filename) for filename in (
            "encoder_chemistry_readouts.py", "modal_encoder_chemistry_probes.py",
        )}, "selection_protocol": "separate predeclared seed fits; LR and epoch selected on validation only",
    }
    jax.config.update("jax_default_matmul_precision", "highest")
    tokens = jax.device_put(tokens)
    output_kind = kind + ("_with_cls" if include_cls else "")
    results = []
    for seed in seeds:
        destination = readout_root / f"depth{depth}" / output_kind / f"seed{seed}"
        if (destination / "trainedprobe.json").exists():
            _, metadata = load_readout(destination)
            if metadata["contract"] != contract or metadata["seed"] != seed or metadata["kind"] != kind:
                raise ValueError("Existing readout artifact belongs to a different data/code contract")
            results.append(metadata)
            continue
        started = time.monotonic()
        result = train_readout(tokens, mask, labels, folds, molecules, target_names,
                              kind=kind, seed=seed, learning_rates=(3e-4, 1e-3),
                              epochs=40, patience=6, batch_size=128)
        result["metadata"]["seconds"] = time.monotonic() - started
        result["metadata"]["device"] = str(jax.devices()[0])
        save_readout(result, destination, contract=contract)
        metadata = json.loads((destination / "trainedprobe.json").read_text())
        results.append(metadata)
        volume.commit()
        print(name, depth, kind, seed, "selected", metadata["selection"], "test",
              metadata["metrics"]["test"]["macro_auroc"], "seconds", round(metadata["seconds"], 1), flush=True)
    return {"name": name, "depth": depth, "kind": output_kind, "readouts": results}


@app.local_entrypoint()
def main(name: str = "both", depths: str = "27,19", kind: str = "both"):
    names = ("parent_1190k", "periodic_1209k") if name == "both" else (name,)
    kinds = ("mean", "attention") if kind == "both" else (kind,)
    requested_depths = tuple(int(value) for value in depths.split(","))
    jobs, handles = [], []
    for encoder in names:
        for depth in requested_depths:
            for readout in kinds:
                job = fit.spawn(encoder, depth, readout)
                jobs.append(job)
                handles.append({"name": encoder, "depth": depth, "kind": readout, "call_id": job.object_id})
    write_json(LOCAL / f"readout_launch_{name}_{depths.replace(',', '-')}_{kind}.json", {"jobs": handles})
    print(json.dumps({"launched": handles}), flush=True)
    for job in jobs:
        result = job.get()
        for metadata in result["readouts"]:
            destination = (LOCAL / result["name"] / "readouts" / f"depth{result['depth']}"
                           / result["kind"] / f"seed{metadata['seed']}" / "trainedprobe.json")
            write_json(destination, metadata)
        print(result["name"], result["depth"], result["kind"], "complete", flush=True)
