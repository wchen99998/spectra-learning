"""Build per-experiment .npz files from GeMS HDF5, grouped by experiment name.

Usage:
    python scripts/prepare_gems_grouped.py \
        --source-hdf5 data/data/GeMS_A/GeMS_A10.hdf5 \
        --output-dir data/gems_grouped \
        --max-precursor-mz 1000.0 \
        --val-fraction 0.05 \
        --split-seed 42

Strategy: Single sequential read of all arrays into RAM (~25 GB for GeMS_A10),
then argsort by experiment name for contiguous slicing, iterate experiments, and
write compressed .npz files.  Much faster than per-experiment HDF5 fancy indexing
on non-contiguous data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

log = logging.getLogger(__name__)

_NUM_PEAKS_INPUT = 128


def _experiment_hash(name: bytes) -> str:
    return hashlib.sha256(name).hexdigest()[:12]


def split_experiments(
    names: list[bytes],
    val_fraction: float,
    seed: int,
) -> tuple[list[bytes], list[bytes]]:
    """Split experiment names into train/val sets."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(names))
    val_size = max(1, int(len(names) * val_fraction))
    val_indices = perm[:val_size]
    train_indices = perm[val_size:]
    return [names[i] for i in train_indices], [names[i] for i in val_indices]


def build_manifest(
    *,
    train_files: list[dict],
    val_files: list[dict],
    source_hdf5: str,
    max_precursor_mz: float,
    val_fraction: float,
    split_seed: int,
    total_experiments: int,
    filtered_experiments: int,
) -> dict:
    def _stats(files: list[dict]) -> dict:
        counts = [f["num_spectra"] for f in files]
        if not counts:
            return {"num_files": 0, "total_spectra": 0}
        arr = np.array(counts)
        return {
            "num_files": len(counts),
            "total_spectra": int(arr.sum()),
            "min_spectra": int(arr.min()),
            "max_spectra": int(arr.max()),
            "mean_spectra": float(arr.mean()),
            "median_spectra": float(np.median(arr)),
        }

    return {
        "version": 1,
        "source_hdf5": source_hdf5,
        "max_precursor_mz": max_precursor_mz,
        "val_fraction": val_fraction,
        "split_seed": split_seed,
        "total_experiments_in_hdf5": total_experiments,
        "filtered_experiments": filtered_experiments,
        "num_peaks_input": _NUM_PEAKS_INPUT,
        "train": {
            "files": train_files,
            **_stats(train_files),
        },
        "validation": {
            "files": val_files,
            **_stats(val_files),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build per-experiment .npz files from GeMS HDF5."
    )
    parser.add_argument("--source-hdf5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-precursor-mz", type=float, default=1000.0)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()

    source_hdf5 = args.source_hdf5.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    # ---- Step 1: Sequential bulk read ----
    import h5py

    log.info("Loading all arrays from %s (sequential read)...", source_hdf5)
    with h5py.File(source_hdf5, "r") as f:
        log.info("  Reading name column...")
        names_raw: np.ndarray = f["name"][:]
        log.info("  Reading RT column...")
        rt_all: np.ndarray = np.asarray(f["RT"][:], dtype=np.float32)
        log.info("  Reading precursor_mz column...")
        precursor_all: np.ndarray = np.asarray(f["precursor_mz"][:], dtype=np.float32)
        log.info("  Reading spectrum column (this is the big one)...")
        spectrum_all: np.ndarray = f["spectrum"][:]  # (N, 2, 128) float64

    n_total = len(names_raw)
    log.info("Loaded %d spectra into RAM", n_total)

    # Split spectrum into mz and intensity, cast to float32 to halve memory
    log.info("Casting spectrum to float32...")
    mz_all = spectrum_all[:, 0, :].astype(np.float32)
    intensity_all = spectrum_all[:, 1, :].astype(np.float32)
    del spectrum_all

    # ---- Step 2: Sort by experiment name for contiguous slicing ----
    log.info("Sorting %d rows by experiment name...", n_total)
    names_bytes = np.array([bytes(n) for n in names_raw], dtype=object)
    del names_raw

    sort_order = np.argsort(names_bytes, kind="stable")
    names_sorted = names_bytes[sort_order]
    mz_all = mz_all[sort_order]
    intensity_all = intensity_all[sort_order]
    rt_all = rt_all[sort_order]
    precursor_all = precursor_all[sort_order]
    del sort_order, names_bytes

    # Find contiguous experiment boundaries
    change_mask = np.empty(n_total + 1, dtype=bool)
    change_mask[0] = True
    change_mask[-1] = True
    change_mask[1:-1] = names_sorted[1:] != names_sorted[:-1]
    boundaries = np.nonzero(change_mask)[0]

    experiment_names: list[bytes] = []
    experiment_slices: list[tuple[int, int]] = []
    for i in range(len(boundaries) - 1):
        experiment_names.append(names_sorted[boundaries[i]])
        experiment_slices.append((int(boundaries[i]), int(boundaries[i + 1])))

    total_experiments = len(experiment_names)
    log.info("Found %d unique experiments", total_experiments)
    del names_sorted, change_mask, boundaries

    # ---- Step 3: Split experiments ----
    train_names, val_names = split_experiments(
        experiment_names, args.val_fraction, args.split_seed
    )
    train_names_set = set(train_names)
    log.info("Split: %d train, %d validation", len(train_names), len(val_names))

    # ---- Step 4: Create output directories ----
    train_dir = output_dir / "train"
    val_dir = output_dir / "validation"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 5: Write .npz files ----
    train_files: list[dict] = []
    val_files: list[dict] = []
    filtered_count = 0
    max_prec = args.max_precursor_mz
    seen_hashes: set[str] = set()

    for name, (start, end) in tqdm(
        zip(experiment_names, experiment_slices),
        total=total_experiments,
        desc="Writing .npz",
    ):
        mz = mz_all[start:end]
        intensity = intensity_all[start:end]
        rt = rt_all[start:end]
        precursor = precursor_all[start:end]

        # Filter
        keep = (
            np.isfinite(rt)
            & (rt > 0.0)
            & np.isfinite(precursor)
            & (precursor <= max_prec)
        )
        if not keep.any():
            filtered_count += 1
            continue

        mz = mz[keep]
        intensity = intensity[keep]
        rt = rt[keep]
        precursor = precursor[keep]

        # Sort by RT ascending
        order = np.argsort(rt)
        mz = mz[order]
        intensity = intensity[order]
        rt = rt[order]
        precursor = precursor[order]

        is_train = name in train_names_set
        out_dir = train_dir if is_train else val_dir
        h = _experiment_hash(name)
        if h in seen_hashes:
            raise RuntimeError(
                f"Hash collision: {h} already used. "
                f"Experiment: {name!r}"
            )
        seen_hashes.add(h)
        fname = f"{h}.npz"
        out_path = out_dir / fname

        np.savez_compressed(
            out_path,
            mz=mz,
            intensity=intensity,
            rt=rt,
            precursor_mz=precursor,
        )

        entry = {
            "filename": fname,
            "experiment_name": name.decode("utf-8", errors="replace"),
            "num_spectra": len(rt),
        }
        if is_train:
            train_files.append(entry)
        else:
            val_files.append(entry)

    log.info(
        "Wrote %d train + %d validation files (%d experiments filtered out)",
        len(train_files),
        len(val_files),
        filtered_count,
    )

    # ---- Step 6: Write manifest ----
    manifest = build_manifest(
        train_files=train_files,
        val_files=val_files,
        source_hdf5=str(source_hdf5),
        max_precursor_mz=max_prec,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        total_experiments=total_experiments,
        filtered_experiments=filtered_count,
    )
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Manifest written to %s", manifest_path)

    for split_name in ("train", "validation"):
        info = manifest[split_name]
        log.info(
            "%s: %d files, %d total spectra, "
            "min/max/mean/median spectra per experiment: %s/%s/%.1f/%.1f",
            split_name,
            info["num_files"],
            info["total_spectra"],
            info.get("min_spectra", "N/A"),
            info.get("max_spectra", "N/A"),
            info.get("mean_spectra", 0),
            info.get("median_spectra", 0),
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
