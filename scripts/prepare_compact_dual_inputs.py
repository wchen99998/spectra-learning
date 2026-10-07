"""Compact the recipe's 127-slot spectra to a dual student's pretraining slots.

Each spectrum keeps its top ``num_peaks`` peaks by intensity (ties to the
lower slot), in the original m/z order, packed into the first ``num_peaks``
slots; padding follows, exactly the dual student's pretraining layout. Every
per-peak array (prepared m/z, intensity, validity and the legacy retained-mass
fractions) is gathered with the same index, so rows, values and the physical
ion inputs stay aligned. Intensities keep the spectrum-maximum scaling: the
maximum peak is always retained, so pretraining's renormalization is a no-op.

Writes the new ``jepa_data`` root, an ``input_root`` without token cache
(online encoding only) and the compact fractions named in ``--inputs``.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts.export_salt_glacier_tokens import atomic_json, file_sha256
from spectra_learning.data.gems.mask_schedule import jepa_mask_stage


def compact_index(mask: np.ndarray, intensity: np.ndarray, num_peaks: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-np.where(mask, intensity, -np.inf), axis=1, kind="stable")[:, :num_peaks]
    keep = np.zeros_like(mask)
    np.put_along_axis(keep, order, True, axis=1)
    keep &= mask
    index = np.argsort(~keep, axis=1, kind="stable")[:, :num_peaks]
    return index, np.take_along_axis(keep, index, axis=1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="New input JSON (outputs named here)")
    parser.add_argument("--reference-inputs", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args(argv)
    paths = {k: Path(v) for k, v in json.loads(args.inputs.read_text()).items()}
    reference = {k: Path(v) for k, v in json.loads(args.reference_inputs.read_text()).items()}
    for name in ("prepared", "fragments"):
        if paths[name] != reference[name]:
            raise ValueError(f"Shared input changed: {name}")
    config = json.loads(paths["jepa_metadata"].read_text())["task_contract"]["config"]
    original_config = json.loads(reference["jepa_metadata"].read_text())["task_contract"]["config"]
    for name in ("peak_ordering", "peak_drop_min_intensity", "min_peak_intensity",
                 "precursor_peak_exclusion_window_da", "max_precursor_mz", "encoder_metadata_schema"):
        if config[name] != original_config[name]:
            raise ValueError(f"Preprocessing differs from the recipe: {name}")
    # The peak-count stage the student was training in at this checkpoint
    # (e.g. 31 peaks before the 63-peak stage), not the final num_peaks.
    num_peaks = jepa_mask_stage(config, args.step, int(config["training_max_steps"])).num_peaks

    source = reference["jepa_data"] / "prepared_massspecgym.npz"
    with np.load(source, allow_pickle=False) as saved:
        data = {name: saved[name] for name in saved.files}
    mask = data["peak_valid_mask"]
    index, compact_mask = compact_index(mask, data["peak_intensity"], num_peaks)
    for name in ("peak_mz", "peak_intensity"):
        data[name] = np.where(compact_mask, np.take_along_axis(data[name], index, axis=1), 0).astype(np.float32)
    data["peak_valid_mask"] = compact_mask
    if not compact_mask.any(axis=1).all():
        raise ValueError("Every spectrum must keep at least one peak")
    out = paths["jepa_data"]
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "prepared_massspecgym.partial.npz", **data)
    (out / "prepared_massspecgym.partial.npz").replace(out / "prepared_massspecgym.npz")
    subset = {"source": str(source), "source_sha256": file_sha256(source), "num_peaks": num_peaks,
              "selection": "top intensity among valid slots (ties to lower slot), m/z order, packed to the front",
              "rows_truncated": int((mask.sum(1) > num_peaks).sum()),
              "valid_peaks_before": int(mask.sum()), "valid_peaks_after": int(compact_mask.sum())}
    preparation = json.loads((reference["jepa_data"] / "preparation_metadata.json").read_text())
    atomic_json(out / "preparation_metadata.json", {**preparation, "physical_peak_subset": subset})

    fractions = np.load(reference["peak_fractions"])
    fractions = np.where(compact_mask, np.take_along_axis(fractions, index, axis=1), 0).astype(np.float32)
    paths["peak_fractions"].parent.mkdir(parents=True, exist_ok=True)
    np.save(paths["peak_fractions"], fractions)

    root = paths["input_root"]
    (root / "data").mkdir(parents=True, exist_ok=True)
    if not (root / "msg15").is_symlink():
        (root / "msg15").symlink_to((reference["input_root"] / "msg15").resolve(strict=True), target_is_directory=True)
    if not (root / "data/fold.npy").is_symlink():
        (root / "data/fold.npy").symlink_to((reference["input_root"] / "data/fold.npy").resolve(strict=True))
    np.save(root / "data/token_valid_mask.npy",
            np.concatenate([compact_mask, np.ones((len(mask), 1), bool)], axis=1))
    original = json.loads((reference["input_root"] / "embedding_metadata.json").read_text())
    embedding = {**original, "run_id": args.run_id,
                 "run_url": f"https://wandb.ai/iclac/jepa-finalrun/runs/{args.run_id}",
                 "checkpoint": str(paths["jepa_checkpoint"]), "checkpoint_role": "student",
                 "checkpoint_step": args.step, "encoder": "student",
                 "config_metadata_path": str(paths["jepa_metadata"]), "data_root": str(out),
                 "batch_size": 128, "embedding_dim": int(config["model_dim"]),
                 "embedding_shape": [len(mask), num_peaks, int(config["model_dim"])],
                 "encoder_uses_pair_path": True, "physical_peak_subset": subset,
                 "source_embedding_root": str(reference["input_root"]), "embedding_complete": False,
                 "token_cache": "none: online encoding"}
    for name in ("also_saved_final_cls", "configured_pairmixer_eval_max_visible_tokens",
                 "encoder_eval_max_visible_tokens", "extraction_seconds", "total_seconds", "jax_devices"):
        embedding.pop(name, None)
    atomic_json(root / "embedding_metadata.json", embedding)
    print(json.dumps({"compact": subset, "fractions": str(paths["peak_fractions"])}), flush=True)


if __name__ == "__main__":
    main()
