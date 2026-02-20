"""Unified TF input pipeline and DataModule for GeMS_A peak lists."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
from pathlib import Path
from typing import Any, Callable, Optional

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import torch
from huggingface_hub import hf_hub_download
from ml_collections import config_dict
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

tf.config.set_visible_devices([], "GPU")

logger = logging.getLogger(__name__)

_DEFAULT_BATCH_SIZE = 512
_DEFAULT_SHUFFLE_BUFFER = 10_000
_DEFAULT_VALIDATION_FRACTION = 0.05
_DEFAULT_TFRECORD_DIR = Path("data/gems_peaklist_tfrecord")
_DEFAULT_TFRECORD_BUFFER_SIZE = 250_000
_DEFAULT_SPLIT_SEED = 42
_DEFAULT_NUM_SHARDS = 4
_NUM_PEAKS_INPUT = 128
_NUM_PEAKS_OUTPUT = 60
_DEFAULT_MAX_PRECURSOR_MZ = 1000.0
_FINGERPRINT_BITS = 1024
_FINGERPRINT_RADIUS = 2
_PEAK_MZ_MIN = 20.0
_PEAK_MZ_MAX = 1000.0
_PRECURSOR_MZ_WINDOW = 2.5
_INTENSITY_EPS = 1e-4
_DEFAULT_MIN_PEAK_INTENSITY = _INTENSITY_EPS
_DEFAULT_SIGREG_FILL_INVALID_WITH_NOISE = True
_DEFAULT_SIGREG_NOISE_MZ_MIN = 0.0
_DEFAULT_SIGREG_NOISE_MZ_MAX = 1.0
_DEFAULT_SIGREG_NOISE_INTENSITY_MEAN = 7e-4
_DEFAULT_SIGREG_NOISE_INTENSITY_STD = 2e-4
_METADATA_FILENAME = "metadata.json"

_GEMS_HF_REPO = "roman-bushuiev/GeMS"
_GEMS_HDF5_PATH = "data/GeMS_A/GeMS_A.hdf5"
_MASSSPEC_HF_REPO = "roman-bushuiev/MassSpecGym"
_MASSSPEC_TSV_PATH = "data/MassSpecGym.tsv"
_MASSSPEC_METADATA_VERSION = 2


# -----------------------------------------------------------------------------
# TFRecord creation and dataset preparation
# -----------------------------------------------------------------------------


def _download_hf_file(repo_id: str, filename: str, local_dir: Path) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=str(local_dir),
    )
    return Path(path)



def _load_gems_arrays(hdf5_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import h5py

    with h5py.File(hdf5_path, "r") as f:
        spectra = f["spectrum"][:]
        retention = np.asarray(f["RT"], dtype=np.float32)
        precursor = np.asarray(f["precursor_mz"], dtype=np.float32)
    return spectra, retention, precursor


def _load_massspec_tsv(
    tsv_path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    import csv

    spectra: list[np.ndarray] = []
    precursor: list[float] = []
    fold: list[str] = []
    smiles: list[str] = []
    adduct: list[str] = []
    instrument_type: list[str] = []
    collision_energy: list[float] = []
    collision_energy_present: list[int] = []

    with tsv_path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            mz = np.fromstring(row["mzs"], sep=",", dtype=np.float32)
            intensity = np.fromstring(row["intensities"], sep=",", dtype=np.float32)
            if mz.size > _NUM_PEAKS_INPUT:
                idx = np.argpartition(intensity, -_NUM_PEAKS_INPUT)[-_NUM_PEAKS_INPUT:]
                idx = idx[np.argsort(intensity[idx])[::-1]]
                mz = mz[idx]
                intensity = intensity[idx]
            elif mz.size < _NUM_PEAKS_INPUT:
                pad = _NUM_PEAKS_INPUT - mz.size
                mz = np.pad(mz, (0, pad))
                intensity = np.pad(intensity, (0, pad))
            spectra.append(np.stack([mz, intensity], axis=0))
            precursor.append(float(row["precursor_mz"]))
            fold.append(row["fold"])
            smiles.append(row["smiles"])
            adduct.append(row["adduct"] if row["adduct"] != "" else "unknown")
            instrument_type.append(
                row["instrument_type"] if row["instrument_type"] != "" else "unknown"
            )
            if row["collision_energy"] == "":
                collision_energy.append(0.0)
                collision_energy_present.append(0)
            else:
                collision_energy.append(float(row["collision_energy"]))
                collision_energy_present.append(1)

    spectra_array = np.stack(spectra, axis=0)
    retention = np.full(len(spectra_array), 392.3146, dtype=np.float32)
    precursor_array = np.asarray(precursor, dtype=np.float32)
    fold_array = np.asarray(fold)
    smiles_array = np.asarray(smiles)
    adduct_array = np.asarray(adduct)
    instrument_type_array = np.asarray(instrument_type)
    collision_energy_array = np.asarray(collision_energy, dtype=np.float32)
    collision_energy_present_array = np.asarray(collision_energy_present, dtype=np.int32)
    return (
        spectra_array,
        retention,
        precursor_array,
        fold_array,
        smiles_array,
        adduct_array,
        instrument_type_array,
        collision_energy_array,
        collision_energy_present_array,
    )



def _compute_morgan_fingerprints(smiles: np.ndarray) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs

    fps = np.zeros((len(smiles), _FINGERPRINT_BITS), dtype=np.int8)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(s))
        fp = AllChem.GetMorganFingerprintAsBitVect(  # type: ignore[attr-defined]
            mol,
            _FINGERPRINT_RADIUS,
            nBits=_FINGERPRINT_BITS,
        )
        arr = np.zeros((_FINGERPRINT_BITS,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps[i] = arr
    return fps


def _encode_categorical_ids(values: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    normalized = np.asarray(
        [str(v) if str(v) != "" else "unknown" for v in values],
        dtype=object,
    )
    categories = sorted(set(normalized.tolist()))
    if "unknown" in categories:
        categories = ["unknown"] + [c for c in categories if c != "unknown"]
    else:
        categories = ["unknown"] + categories
    vocab = {category: i for i, category in enumerate(categories)}
    ids = np.asarray([vocab[str(v)] for v in normalized], dtype=np.int32)
    return ids, vocab


def _write_tfrecords(
    spectra: np.ndarray,
    retention: np.ndarray,
    precursor: np.ndarray,
    output_path: Path,
    num_shards: int,
    desc: str,
) -> tuple[list[str], list[int]]:
    n = len(spectra)
    num_shards = max(1, min(num_shards, n))
    shard_size = math.ceil(n / num_shards)

    output_path.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    lengths: list[int] = []

    for shard_id in range(num_shards):
        start = shard_id * shard_size
        end = min(start + shard_size, n)
        if start >= end:
            break

        shard_file = output_path / f"shard-{shard_id:05d}-of-{num_shards:05d}.tfrecord"
        options = tf.io.TFRecordOptions(compression_type="GZIP")

        with tf.io.TFRecordWriter(str(shard_file), options=options) as writer:  # type: ignore[attr-defined]
            for i in tqdm(range(start, end), desc=f"{desc} [{shard_id + 1}/{num_shards}]"):
                mz = spectra[i, 0].astype(np.float32)
                intensity = spectra[i, 1].astype(np.float32)

                features = {
                    "mz": tf.train.Feature(float_list=tf.train.FloatList(value=mz)),
                    "intensity": tf.train.Feature(
                        float_list=tf.train.FloatList(value=intensity)
                    ),
                    "rt": tf.train.Feature(
                        float_list=tf.train.FloatList(value=[retention[i]])
                    ),
                    "precursor_mz": tf.train.Feature(
                        float_list=tf.train.FloatList(value=[precursor[i]])
                    ),
                }

                example = tf.train.Example(
                    features=tf.train.Features(feature=features)
                )
                writer.write(example.SerializeToString())

        files.append(shard_file.name)
        lengths.append(end - start)

    return files, lengths


def _write_tfrecords_with_fingerprint(
    spectra: np.ndarray,
    retention: np.ndarray,
    precursor: np.ndarray,
    fingerprint: np.ndarray,
    adduct_id: np.ndarray,
    instrument_type_id: np.ndarray,
    collision_energy: np.ndarray,
    collision_energy_present: np.ndarray,
    output_path: Path,
    num_shards: int,
    desc: str,
) -> tuple[list[str], list[int]]:
    n = len(spectra)
    num_shards = max(1, min(num_shards, n))
    shard_size = math.ceil(n / num_shards)

    output_path.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    lengths: list[int] = []

    for shard_id in range(num_shards):
        start = shard_id * shard_size
        end = min(start + shard_size, n)
        if start >= end:
            break

        shard_file = output_path / f"shard-{shard_id:05d}-of-{num_shards:05d}.tfrecord"
        options = tf.io.TFRecordOptions(compression_type="GZIP")

        with tf.io.TFRecordWriter(str(shard_file), options=options) as writer:  # type: ignore[attr-defined]
            for i in tqdm(range(start, end), desc=f"{desc} [{shard_id + 1}/{num_shards}]"):
                mz = spectra[i, 0].astype(np.float32)
                intensity = spectra[i, 1].astype(np.float32)
                fp = fingerprint[i].astype(np.int64)
                adduct = int(adduct_id[i])
                instrument = int(instrument_type_id[i])
                ce = float(collision_energy[i])
                ce_present = int(collision_energy_present[i])

                features = {
                    "mz": tf.train.Feature(float_list=tf.train.FloatList(value=mz)),
                    "intensity": tf.train.Feature(
                        float_list=tf.train.FloatList(value=intensity)
                    ),
                    "rt": tf.train.Feature(
                        float_list=tf.train.FloatList(value=[retention[i]])
                    ),
                    "precursor_mz": tf.train.Feature(
                        float_list=tf.train.FloatList(value=[precursor[i]])
                    ),
                    "fingerprint": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=fp)
                    ),
                    "adduct_id": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[adduct])
                    ),
                    "instrument_type_id": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[instrument])
                    ),
                    "collision_energy": tf.train.Feature(
                        float_list=tf.train.FloatList(value=[ce])
                    ),
                    "collision_energy_present": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[ce_present])
                    ),
                }

                example = tf.train.Example(
                    features=tf.train.Features(feature=features)
                )
                writer.write(example.SerializeToString())

        files.append(shard_file.name)
        lengths.append(end - start)

    return files, lengths



def _process_gems(
    output_dir: Path,
    validation_fraction: float,
    split_seed: int,
    num_shards: int,
) -> dict[str, Any]:
    logger.info("Downloading GeMS HDF5...")
    hdf5_path = _download_hf_file(_GEMS_HF_REPO, _GEMS_HDF5_PATH, output_dir.parent)

    logger.info("Loading GeMS data...")
    spectra, retention, precursor = _load_gems_arrays(hdf5_path)

    mask = np.isfinite(retention) & (retention > 0.0)
    spectra = spectra[mask]
    retention = retention[mask]
    precursor = precursor[mask]

    n = len(spectra)
    logger.info("Valid GeMS spectra: %d", n)

    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n)
    train_size = int(n * (1.0 - validation_fraction))

    train_idx = perm[:train_size]
    val_idx = perm[train_size:]

    train_files, train_lengths = _write_tfrecords(
        spectra[train_idx],
        retention[train_idx],
        precursor[train_idx],
        output_dir / "train",
        num_shards,
        desc="Train",
    )

    val_files, val_lengths = _write_tfrecords(
        spectra[val_idx],
        retention[val_idx],
        precursor[val_idx],
        output_dir / "validation",
        max(1, num_shards // 4),
        desc="Validation",
    )

    return {
        "train_files": train_files,
        "train_lengths": train_lengths,
        "validation_files": val_files,
        "validation_lengths": val_lengths,
        "train_size": int(train_size),
        "validation_size": int(n - train_size),
    }


def _process_massspec(output_dir: Path, num_shards: int) -> dict[str, Any]:
    logger.info("Downloading MassSpecGym TSV...")
    tsv_path = _download_hf_file(_MASSSPEC_HF_REPO, _MASSSPEC_TSV_PATH, output_dir.parent)

    logger.info("Loading MassSpecGym data...")
    (
        spectra,
        retention,
        precursor,
        fold,
        smiles,
        adduct,
        instrument_type,
        collision_energy,
        collision_energy_present,
    ) = _load_massspec_tsv(tsv_path)
    fingerprints = _compute_morgan_fingerprints(smiles)
    adduct_id, adduct_vocab = _encode_categorical_ids(adduct)
    instrument_type_id, instrument_type_vocab = _encode_categorical_ids(instrument_type)

    train_mask = fold == "train"
    val_mask = fold == "val"
    test_mask = fold == "test"
    train_size = int(np.count_nonzero(train_mask))
    val_size = int(np.count_nonzero(val_mask))
    test_size = int(np.count_nonzero(test_mask))
    logger.info(
        "MassSpecGym spectra: %d (train=%d, val=%d, test=%d)",
        len(spectra),
        train_size,
        val_size,
        test_size,
    )

    train_files, train_lengths = _write_tfrecords_with_fingerprint(
        spectra[train_mask],
        retention[train_mask],
        precursor[train_mask],
        fingerprints[train_mask],
        adduct_id[train_mask],
        instrument_type_id[train_mask],
        collision_energy[train_mask],
        collision_energy_present[train_mask],
        output_dir / "massspec_train",
        max(1, num_shards // 2),
        desc="MassSpec Train",
    )

    val_files, val_lengths = _write_tfrecords_with_fingerprint(
        spectra[val_mask],
        retention[val_mask],
        precursor[val_mask],
        fingerprints[val_mask],
        adduct_id[val_mask],
        instrument_type_id[val_mask],
        collision_energy[val_mask],
        collision_energy_present[val_mask],
        output_dir / "massspec_val",
        max(1, num_shards // 4),
        desc="MassSpec Val",
    )

    test_files, test_lengths = _write_tfrecords_with_fingerprint(
        spectra[test_mask],
        retention[test_mask],
        precursor[test_mask],
        fingerprints[test_mask],
        adduct_id[test_mask],
        instrument_type_id[test_mask],
        collision_energy[test_mask],
        collision_energy_present[test_mask],
        output_dir / "massspec_test",
        max(1, num_shards // 4),
        desc="MassSpec Test",
    )

    return {
        "massspec_train_files": train_files,
        "massspec_train_lengths": train_lengths,
        "massspec_train_size": train_size,
        "massspec_val_files": val_files,
        "massspec_val_lengths": val_lengths,
        "massspec_val_size": val_size,
        "massspec_test_files": test_files,
        "massspec_test_lengths": test_lengths,
        "massspec_test_size": test_size,
        "massspec_metadata_version": _MASSSPEC_METADATA_VERSION,
        "massspec_adduct_vocab": adduct_vocab,
        "massspec_instrument_type_vocab": instrument_type_vocab,
    }



def _ensure_processed(
    output_dir: Path,
    validation_fraction: float,
    split_seed: int,
    num_shards: int,
) -> dict[str, Any]:
    metadata_path = output_dir / _METADATA_FILENAME

    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

        train_ok = all(
            (output_dir / "train" / fn).exists()
            for fn in metadata.get("train_files", [])
        )
        val_ok = all(
            (output_dir / "validation" / fn).exists()
            for fn in metadata.get("validation_files", [])
        )
        massspec_train_files = metadata.get("massspec_train_files", [])
        massspec_train_ok = all(
            (output_dir / "massspec_train" / fn).exists()
            for fn in massspec_train_files
        )
        massspec_val_files = metadata.get("massspec_val_files", [])
        massspec_val_ok = all(
            (output_dir / "massspec_val" / fn).exists()
            for fn in massspec_val_files
        )
        massspec_test_files = metadata.get("massspec_test_files", [])
        massspec_test_ok = all(
            (output_dir / "massspec_test" / fn).exists()
            for fn in massspec_test_files
        )
        massspec_version_ok = (
            int(metadata.get("massspec_metadata_version", 0))
            == _MASSSPEC_METADATA_VERSION
        )
        massspec_vocab_ok = (
            "massspec_adduct_vocab" in metadata
            and "massspec_instrument_type_vocab" in metadata
        )

        if (
            train_ok
            and val_ok
            and massspec_train_files
            and massspec_train_ok
            and massspec_val_files
            and massspec_val_ok
            and massspec_test_files
            and massspec_test_ok
            and massspec_version_ok
            and massspec_vocab_ok
        ):
            logger.info("Found existing TFRecords at %s", output_dir)
            return metadata

        if train_ok and val_ok:
            logger.info("Updating MassSpecGym TFRecords at %s", output_dir)
            massspec_info = _process_massspec(output_dir, num_shards)
            metadata.update(massspec_info)
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            return metadata

    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = _process_gems(output_dir, validation_fraction, split_seed, num_shards)
    massspec_info = _process_massspec(output_dir, num_shards)
    metadata.update(massspec_info)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved metadata to %s", metadata_path)
    return metadata



# -----------------------------------------------------------------------------
# tf.data pipeline
# -----------------------------------------------------------------------------

def _augment_masked_view_tf(
    peak_mz: tf.Tensor,
    peak_intensity: tf.Tensor,
    peak_valid_mask: tf.Tensor,
    *,
    contiguous_mask_fraction: float,
    contiguous_mask_min_len: int,
    random_mask_prob: float,
    mz_jitter_std: float,
    intensity_jitter_std: float,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    batch_size = tf.shape(peak_mz)[0]
    num_peaks = tf.shape(peak_mz)[1]
    has_valid = tf.reduce_any(peak_valid_mask, axis=1)

    sort_keys = tf.where(
        peak_valid_mask,
        peak_mz,
        tf.fill(tf.shape(peak_mz), tf.cast(float("inf"), peak_mz.dtype)),
    )
    sorted_order = tf.argsort(sort_keys, axis=1, direction="ASCENDING", stable=True)
    sorted_valid = tf.gather(peak_valid_mask, sorted_order, batch_dims=1)
    valid_counts = tf.reduce_sum(tf.cast(sorted_valid, tf.int32), axis=1)

    raw_mask_len = tf.cast(
        tf.floor(tf.cast(valid_counts, tf.float32) * contiguous_mask_fraction),
        tf.int32,
    )
    mask_len = tf.maximum(
        raw_mask_len,
        tf.fill(tf.shape(raw_mask_len), tf.cast(contiguous_mask_min_len, tf.int32)),
    )
    mask_len = tf.minimum(mask_len, valid_counts)
    mask_len = tf.where(has_valid, mask_len, tf.zeros_like(mask_len))

    max_start_offset = valid_counts - mask_len + 1
    sampled_offset = tf.cast(
        tf.floor(
            tf.random.uniform([batch_size], dtype=tf.float32)
            * tf.cast(max_start_offset, tf.float32),
        ),
        tf.int32,
    )
    sampled_offset = tf.where(has_valid, sampled_offset, tf.zeros_like(sampled_offset))
    mask_start = sampled_offset
    mask_end = mask_start + mask_len

    positions = tf.range(num_peaks, dtype=tf.int32)[tf.newaxis, :]
    masked_sorted = (
        has_valid[:, tf.newaxis]
        & (positions >= mask_start[:, tf.newaxis])
        & (positions < mask_end[:, tf.newaxis])
        & (positions < valid_counts[:, tf.newaxis])
    )
    inverse_order = tf.argsort(sorted_order, axis=1, direction="ASCENDING", stable=True)
    masked = tf.gather(masked_sorted, inverse_order, batch_dims=1)
    masked = tf.logical_and(masked, peak_valid_mask)

    # Independent random masking: additionally mask each valid, non-contiguous-masked
    # peak with probability random_mask_prob. This dramatically increases mask
    # pattern diversity beyond the contiguous-only scheme.
    if random_mask_prob > 0.0:
        random_coins = tf.random.uniform(tf.shape(peak_mz), dtype=tf.float32)
        random_drop = random_coins < random_mask_prob
        random_drop = tf.logical_and(random_drop, peak_valid_mask)
        random_drop = tf.logical_and(random_drop, tf.logical_not(masked))
        masked = tf.logical_or(masked, random_drop)

    view_valid = tf.logical_and(peak_valid_mask, tf.logical_not(masked))

    # jitterable should be view_valid now (since masked are dropped)
    jitterable = view_valid

    mz_noise = tf.random.normal(tf.shape(peak_mz), stddev=mz_jitter_std, dtype=peak_mz.dtype)
    mz = tf.where(jitterable, peak_mz + mz_noise, tf.zeros_like(peak_mz))
    mz = tf.clip_by_value(mz, 0.0, 1.0)

    intensity_noise = tf.random.normal(
        tf.shape(peak_intensity),
        stddev=intensity_jitter_std,
        dtype=peak_intensity.dtype,
    )
    intensity = tf.where(
        jitterable,
        peak_intensity + intensity_noise,
        tf.zeros_like(peak_intensity),
    )
    intensity = tf.clip_by_value(intensity, 0.0, 1.0)
    intensity = tf.where(jitterable, intensity, tf.zeros_like(intensity))

    valid_counts_safe = tf.maximum(
        tf.reduce_sum(tf.cast(peak_valid_mask, tf.float32), axis=1),
        1.0,
    )
    masked_counts = tf.reduce_sum(tf.cast(masked, tf.float32), axis=1)
    masked_fraction = tf.reduce_mean(masked_counts / valid_counts_safe)

    return (
        mz,
        intensity,
        view_valid,
        masked,
        masked_fraction,
    )


def _augment_unmasked_view_tf(
    peak_mz: tf.Tensor,
    peak_intensity: tf.Tensor,
    peak_valid_mask: tf.Tensor,
    *,
    mz_jitter_std: float,
    intensity_jitter_std: float,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    view_valid = peak_valid_mask
    masked = tf.zeros_like(peak_valid_mask)

    mz_noise = tf.random.normal(tf.shape(peak_mz), stddev=mz_jitter_std, dtype=peak_mz.dtype)
    mz = tf.where(view_valid, peak_mz + mz_noise, tf.zeros_like(peak_mz))
    mz = tf.clip_by_value(mz, 0.0, 1.0)

    intensity_noise = tf.random.normal(
        tf.shape(peak_intensity),
        stddev=intensity_jitter_std,
        dtype=peak_intensity.dtype,
    )
    intensity = tf.where(
        view_valid,
        peak_intensity + intensity_noise,
        tf.zeros_like(peak_intensity),
    )
    intensity = tf.clip_by_value(intensity, 0.0, 1.0)
    intensity = tf.where(view_valid, intensity, tf.zeros_like(intensity))

    return mz, intensity, view_valid, masked


def _augment_sigreg_batch_tf(
    *,
    contiguous_mask_fraction: float,
    contiguous_mask_min_len: int,
    random_mask_prob: float,
    mz_jitter_std: float,
    intensity_jitter_std: float,
    fill_invalid_with_noise: bool = _DEFAULT_SIGREG_FILL_INVALID_WITH_NOISE,
    noise_mz_min: float = _DEFAULT_SIGREG_NOISE_MZ_MIN,
    noise_mz_max: float = _DEFAULT_SIGREG_NOISE_MZ_MAX,
    noise_intensity_mean: float = _DEFAULT_SIGREG_NOISE_INTENSITY_MEAN,
    noise_intensity_std: float = _DEFAULT_SIGREG_NOISE_INTENSITY_STD,
) -> Callable[[dict], dict]:
    def apply(batch: dict) -> dict:
        peak_mz = batch["peak_mz"]
        peak_intensity = batch["peak_intensity"]
        peak_valid_mask = batch["peak_valid_mask"]
        precursor_mz = batch["precursor_mz"]

        view1_mz, view1_int, view1_valid, view1_masked, view1_masked_fraction = _augment_masked_view_tf(
            peak_mz,
            peak_intensity,
            peak_valid_mask,
            contiguous_mask_fraction=contiguous_mask_fraction,
            contiguous_mask_min_len=contiguous_mask_min_len,
            random_mask_prob=random_mask_prob,
            mz_jitter_std=mz_jitter_std,
            intensity_jitter_std=intensity_jitter_std,
        )
        view2_mz, view2_int, view2_valid, view2_masked = _augment_unmasked_view_tf(
            peak_mz,
            peak_intensity,
            peak_valid_mask,
            mz_jitter_std=mz_jitter_std,
            intensity_jitter_std=intensity_jitter_std,
        )

        if fill_invalid_with_noise:
            view1_noise_mz = tf.random.uniform(
                tf.shape(view1_mz),
                minval=noise_mz_min,
                maxval=noise_mz_max,
                dtype=view1_mz.dtype,
            )
            view1_noise_intensity = tf.random.truncated_normal(
                tf.shape(view1_int),
                mean=noise_intensity_mean,
                stddev=noise_intensity_std,
                dtype=view1_int.dtype,
            )
            view1_noise_intensity = tf.clip_by_value(view1_noise_intensity, 0.0, 1.0)
            view1_mz = tf.where(view1_valid, view1_mz, view1_noise_mz)
            view1_int = tf.where(view1_valid, view1_int, view1_noise_intensity)

            view2_noise_mz = tf.random.uniform(
                tf.shape(view2_mz),
                minval=noise_mz_min,
                maxval=noise_mz_max,
                dtype=view2_mz.dtype,
            )
            view2_noise_intensity = tf.random.truncated_normal(
                tf.shape(view2_int),
                mean=noise_intensity_mean,
                stddev=noise_intensity_std,
                dtype=view2_int.dtype,
            )
            view2_noise_intensity = tf.clip_by_value(view2_noise_intensity, 0.0, 1.0)
            view2_mz = tf.where(view2_valid, view2_mz, view2_noise_mz)
            view2_int = tf.where(view2_valid, view2_int, view2_noise_intensity)

        out = dict(batch)
        out["fused_mz"] = tf.concat([view1_mz, view2_mz], axis=0)
        out["fused_intensity"] = tf.concat([view1_int, view2_int], axis=0)
        out["fused_precursor_mz"] = tf.concat([precursor_mz, precursor_mz], axis=0)
        out["fused_original_valid_mask"] = tf.concat([view1_valid, view2_valid], axis=0)
        out["fused_masked_positions"] = tf.concat([view1_masked, view2_masked], axis=0)

        if fill_invalid_with_noise:
            out["fused_valid_mask"] = tf.ones_like(out["fused_original_valid_mask"])
            # Re-sort by mz after noise fill: random mz values break the
            # original sort order, so we restore ascending-mz ordering.
            sort_idx = tf.argsort(
                out["fused_mz"], axis=1, direction="ASCENDING", stable=True,
            )
            out["fused_mz"] = tf.gather(out["fused_mz"], sort_idx, batch_dims=1)
            out["fused_intensity"] = tf.gather(
                out["fused_intensity"], sort_idx, batch_dims=1,
            )
            out["fused_original_valid_mask"] = tf.gather(
                out["fused_original_valid_mask"], sort_idx, batch_dims=1,
            )
            out["fused_masked_positions"] = tf.gather(
                out["fused_masked_positions"], sort_idx, batch_dims=1,
            )
            out["fused_valid_mask"] = tf.gather(
                out["fused_valid_mask"], sort_idx, batch_dims=1,
            )
        else:
            out["fused_valid_mask"] = out["fused_original_valid_mask"]
        out["view1_masked_fraction"] = view1_masked_fraction
        return out

    return apply



def _batched_parse_and_transform(
    *,
    max_precursor_mz: float,
    min_peak_intensity: float,
    num_peaks: int,
    mz_representation: str,
    peak_ordering: str,
    include_fingerprint: bool,
) -> Callable[[tf.Tensor], dict[str, tf.Tensor]]:
    """Return a tf.function that operates on a **batch** of serialized examples.

    Uses ``tf.io.parse_example`` (batch parse) and fully-vectorized ops on
    ``[B, num_peaks]`` tensors, avoiding per-element ``boolean_mask`` which
    creates variable-length intermediates that TF cannot parallelise.
    """
    peak_mz_min = tf.constant(_PEAK_MZ_MIN, tf.float32)
    peak_mz_max_c = tf.constant(_PEAK_MZ_MAX, tf.float32)
    precursor_window = tf.constant(_PRECURSOR_MZ_WINDOW, tf.float32)
    min_int = tf.constant(min_peak_intensity, tf.float32)
    max_prec = tf.constant(max_precursor_mz, tf.float32)


    if include_fingerprint:
        feature_spec: dict[str, tf.io.FixedLenFeature] = {
            "mz": tf.io.FixedLenFeature([_NUM_PEAKS_INPUT], tf.float32),
            "intensity": tf.io.FixedLenFeature([_NUM_PEAKS_INPUT], tf.float32),
            "rt": tf.io.FixedLenFeature([1], tf.float32),
            "precursor_mz": tf.io.FixedLenFeature([1], tf.float32),
            "fingerprint": tf.io.FixedLenFeature([_FINGERPRINT_BITS], tf.int64),
            "adduct_id": tf.io.FixedLenFeature([1], tf.int64),
            "instrument_type_id": tf.io.FixedLenFeature([1], tf.int64),
            "collision_energy": tf.io.FixedLenFeature([1], tf.float32),
            "collision_energy_present": tf.io.FixedLenFeature([1], tf.int64),
        }
    else:
        feature_spec = {
            "mz": tf.io.FixedLenFeature([_NUM_PEAKS_INPUT], tf.float32),
            "intensity": tf.io.FixedLenFeature([_NUM_PEAKS_INPUT], tf.float32),
            "rt": tf.io.FixedLenFeature([1], tf.float32),
            "precursor_mz": tf.io.FixedLenFeature([1], tf.float32),
        }

    @tf.function
    def transform(serialized_batch: tf.Tensor) -> dict[str, tf.Tensor]:
        parsed = tf.io.parse_example(serialized_batch, feature_spec)

        mz = parsed["mz"]                          # [B, 128]
        intensity = parsed["intensity"]             # [B, 128]
        rt = parsed["rt"][:, 0]                     # [B]
        precursor_mz_val = parsed["precursor_mz"][:, 0]  # [B]

        # ── Filter peak mz range (vectorized) ───────────────────────────────────
        upper = tf.where(
            precursor_mz_val > 0.0,
            precursor_mz_val - precursor_window,
            peak_mz_max_c,
        )  # [B]
        keep = (mz >= peak_mz_min) & (mz <= upper[:, tf.newaxis])
        mz = tf.where(keep, mz, 0.0)
        intensity = tf.where(keep, intensity, 0.0)

        # ── Filter min peak intensity (vectorized) ──────────────────────────────
        keep2 = intensity >= min_int
        mz = tf.where(keep2, mz, 0.0)
        intensity = tf.where(keep2, intensity, 0.0)

        # ── Top-k peaks (vectorized on last dim) ────────────────────────────────
        values, indices = tf.math.top_k(intensity, k=num_peaks, sorted=True)
        intensity = values
        mz = tf.gather(mz, indices, batch_dims=1)  # [B, num_peaks]

        # ── Fixed-size sort (replaces variable-length compact_sort) ──────────────
        # Invalid peaks get +inf sort key so they land at the end.
        valid = intensity > 0
        if peak_ordering == "mz":
            sort_key = tf.where(
                valid, mz, tf.fill(tf.shape(mz), float("inf")),
            )
            sorted_idx = tf.argsort(
                sort_key, axis=1, direction="ASCENDING", stable=True,
            )
        else:
            sort_key = tf.where(
                valid, intensity, tf.fill(tf.shape(intensity), float("-inf")),
            )
            sorted_idx = tf.argsort(
                sort_key, axis=1, direction="DESCENDING", stable=True,
            )
        mz = tf.gather(mz, sorted_idx, batch_dims=1)
        intensity = tf.gather(intensity, sorted_idx, batch_dims=1)
        valid_sorted = tf.gather(valid, sorted_idx, batch_dims=1)
        mz = tf.where(valid_sorted, mz, 0.0)
        intensity = tf.where(valid_sorted, intensity, 0.0)

        # ── Normalize ────────────────────────────────────────────────────────────
        valid_final = intensity > 0
        precursor_mz_norm = (
            tf.clip_by_value(precursor_mz_val, 0.0, max_prec) / max_prec
        )

        out: dict[str, tf.Tensor] = {
            "peak_mz": mz / peak_mz_max_c,
            "peak_intensity": tf.where(valid_final, intensity, 0.0),
            "peak_valid_mask": valid_final,
            "precursor_mz": precursor_mz_norm,
            "rt": rt,
            "mz": mz,
            "intensity": intensity,
        }
        if include_fingerprint:
            out["fingerprint"] = tf.cast(parsed["fingerprint"], tf.int32)
            out["adduct_id"] = tf.cast(parsed["adduct_id"][:, 0], tf.int32)
            out["instrument_type_id"] = tf.cast(
                parsed["instrument_type_id"][:, 0], tf.int32,
            )
            out["collision_energy"] = parsed["collision_energy"][:, 0]
            out["collision_energy_present"] = tf.cast(
                parsed["collision_energy_present"][:, 0], tf.int32,
            )
        return out

    return transform


def _build_dataset(
    filenames: list[str],
    batch_size: int,
    shuffle_buffer: int,
    seed: Optional[int],
    drop_remainder: bool,
    *,
    tfrecord_buffer_size: int,
    max_precursor_mz: float,
    include_fingerprint: bool,
    min_peak_intensity: float,
    mz_representation: str,
    include_sigreg_augmentation: bool,
    sigreg_contiguous_mask_fraction: float,
    sigreg_contiguous_mask_min_len: int,
    sigreg_random_mask_prob: float,
    sigreg_mz_jitter_std: float,
    sigreg_intensity_jitter_std: float,
    sigreg_fill_invalid_with_noise: bool = _DEFAULT_SIGREG_FILL_INVALID_WITH_NOISE,
    sigreg_noise_mz_min: float = _DEFAULT_SIGREG_NOISE_MZ_MIN,
    sigreg_noise_mz_max: float = _DEFAULT_SIGREG_NOISE_MZ_MAX,
    sigreg_noise_intensity_mean: float = _DEFAULT_SIGREG_NOISE_INTENSITY_MEAN,
    sigreg_noise_intensity_std: float = _DEFAULT_SIGREG_NOISE_INTENSITY_STD,
    peak_ordering: str = "intensity",
    num_parallel_reads: int | None = None,
) -> tf.data.Dataset:
    if num_parallel_reads is None:
        num_parallel_reads = tf.data.AUTOTUNE
    ds = tf.data.TFRecordDataset(
        filenames,
        compression_type="GZIP",
        buffer_size=int(tfrecord_buffer_size),
        num_parallel_reads=num_parallel_reads,
    )
    # Shuffle raw serialized bytes (lightweight), then batch, then batch-parse.
    # tf.io.parse_example + vectorized ops on [B, ...] is ~3x faster than
    # per-element parse_single_example + variable-length boolean_mask.
    if shuffle_buffer > 0:
        ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    batched_fn = _batched_parse_and_transform(
        max_precursor_mz=max_precursor_mz,
        min_peak_intensity=min_peak_intensity,
        num_peaks=_NUM_PEAKS_OUTPUT,
        mz_representation=mz_representation,
        peak_ordering=peak_ordering,
        include_fingerprint=include_fingerprint,
    )
    ds = ds.map(batched_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if include_sigreg_augmentation:
        ds = ds.map(
            _augment_sigreg_batch_tf(
                contiguous_mask_fraction=sigreg_contiguous_mask_fraction,
                contiguous_mask_min_len=sigreg_contiguous_mask_min_len,
                random_mask_prob=sigreg_random_mask_prob,
                mz_jitter_std=sigreg_mz_jitter_std,
                intensity_jitter_std=sigreg_intensity_jitter_std,
                fill_invalid_with_noise=sigreg_fill_invalid_with_noise,
                noise_mz_min=sigreg_noise_mz_min,
                noise_mz_max=sigreg_noise_mz_max,
                noise_intensity_mean=sigreg_noise_intensity_mean,
                noise_intensity_std=sigreg_noise_intensity_std,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    options = tf.data.Options()
    options.deterministic = True
    ds = ds.with_options(options)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds



# -----------------------------------------------------------------------------
# Info and step resolution
# -----------------------------------------------------------------------------


def _compute_info(
    metadata: dict[str, Any],
    *,
    output_dir: Path,
    max_precursor_mz: float,
    mz_representation: str,
) -> dict[str, Any]:
    massspec_adduct_vocab = metadata.get("massspec_adduct_vocab", {"unknown": 0})
    massspec_instrument_type_vocab = metadata.get(
        "massspec_instrument_type_vocab",
        {"unknown": 0},
    )
    return {
        "tfrecord_dir": str(output_dir),
        "train_size": metadata["train_size"],
        "validation_size": metadata["validation_size"],
        "massspec_train_size": metadata.get("massspec_train_size", 0),
        "massspec_val_size": metadata.get("massspec_val_size", 0),
        "massspec_test_size": metadata.get("massspec_test_size", 0),
        "massspec_metadata_version": int(metadata.get("massspec_metadata_version", 0)),
        "massspec_adduct_vocab": massspec_adduct_vocab,
        "massspec_instrument_type_vocab": massspec_instrument_type_vocab,
        "massspec_adduct_vocab_size": len(massspec_adduct_vocab),
        "massspec_instrument_type_vocab_size": len(massspec_instrument_type_vocab),
        "num_peaks": _NUM_PEAKS_OUTPUT,
        "mz_representation": mz_representation,
        "fingerprint_bits": _FINGERPRINT_BITS,
        "max_precursor_mz": max_precursor_mz,
        "peak_mz_min": _PEAK_MZ_MIN,
        "peak_mz_max": _PEAK_MZ_MAX,
    }



def _steps_from_size(size: int, batch_size: int, drop_remainder: bool) -> int:
    if drop_remainder:
        return int(size // batch_size)
    return int(math.ceil(size / batch_size))


# -----------------------------------------------------------------------------
# NumPy -> torch conversion
# -----------------------------------------------------------------------------


def _to_torch(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return _to_torch(value.tolist())
        # Zero-copy when possible: torch.from_numpy shares memory.
        # The downstream .to(device) creates a new tensor anyway.
        if not value.flags.c_contiguous or not value.flags.writeable:
            value = value.copy()
        return torch.from_numpy(value)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, list):
        return [_to_torch(item) for item in value]
    return value


def numpy_batch_to_torch(batch: dict[str, Any]) -> dict[str, Any]:
    return {key: _to_torch(value) for key, value in batch.items()}


def _identity_collate(batch: dict[str, Any]) -> dict[str, Any]:
    return batch


# -----------------------------------------------------------------------------
# Lightning DataModule and IterableDataset
# -----------------------------------------------------------------------------


class _TfIterableDataset(IterableDataset):
    def __init__(
        self,
        *,
        dataset_builder: Callable[[], tf.data.Dataset],
        steps_per_epoch: int,
    ) -> None:
        super().__init__()
        self._dataset_builder = dataset_builder
        self._dataset: tf.data.Dataset | None = None
        self.steps_per_epoch = int(steps_per_epoch)
        self._resume_from = 0
        self._num_yielded = 0

    def _get_dataset(self) -> tf.data.Dataset:
        if self._dataset is None:
            self._dataset = self._dataset_builder()
        return self._dataset

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self):
        self._num_yielded = self._resume_from if self._resume_from > 0 else 0
        resume_from = self._resume_from
        self._resume_from = 0
        dataset = self._get_dataset()
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            dataset = dataset.shard(
                num_shards=worker_info.num_workers, index=worker_info.id,
            )
        if resume_from > 0:
            dataset = dataset.skip(resume_from)
        iterator = dataset.as_numpy_iterator()
        for batch in iterator:
            yield numpy_batch_to_torch(batch)
            self._num_yielded += 1

    def state_dict(self) -> dict[str, Any]:
        return {"num_yielded": self._num_yielded}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._resume_from = int(state_dict["num_yielded"])
        self._num_yielded = self._resume_from


class _StatefulDataLoader(DataLoader):
    dataset: _TfIterableDataset  # type: ignore[assignment]

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._resume_from = 0
        self._num_yielded = 0

    def __iter__(self):
        state = {"num_yielded": self._resume_from}
        self.dataset.load_state_dict(state)
        self._num_yielded = self._resume_from
        self._resume_from = 0
        for batch in super().__iter__():
            self._num_yielded += 1
            yield batch

    def state_dict(self) -> dict[str, Any]:
        return {"num_yielded": self._num_yielded}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._resume_from = int(state_dict["num_yielded"])
        self._num_yielded = self._resume_from
        self.dataset.load_state_dict(state_dict)


class TfLightningDataModule:
    """DataModule that rebuilds tf.data pipelines each epoch."""

    def __init__(self, config: config_dict.ConfigDict, seed: int) -> None:
        self.config = config
        self.seed = int(seed)

        self.output_dir = (
            Path(config.get("tfrecord_dir", str(_DEFAULT_TFRECORD_DIR)))
            .expanduser()
            .resolve()
        )
        self.validation_fraction = float(
            config.get("validation_fraction", _DEFAULT_VALIDATION_FRACTION)
        )
        self.batch_size = int(config.get("batch_size", _DEFAULT_BATCH_SIZE))
        self.shuffle_buffer = int(config.get("shuffle_buffer", _DEFAULT_SHUFFLE_BUFFER))
        self.tfrecord_buffer_size = int(
            config.get("tfrecord_buffer_size", _DEFAULT_TFRECORD_BUFFER_SIZE)
        )
        self.split_seed = int(config.get("split_seed", _DEFAULT_SPLIT_SEED))
        self.num_shards = int(config.get("num_shards", _DEFAULT_NUM_SHARDS))
        self.drop_remainder = bool(config.get("drop_remainder", True))
        self.max_precursor_mz = float(
            config.get("max_precursor_mz", _DEFAULT_MAX_PRECURSOR_MZ)
        )
        self.min_peak_intensity = float(
            config.get("min_peak_intensity", _DEFAULT_MIN_PEAK_INTENSITY)
        )
        self.mz_representation = str(config.get("mz_representation", "mz"))
        self.peak_ordering = str(config.get("peak_ordering", "intensity"))
        self.sigreg_contiguous_mask_fraction = float(
            config.get("sigreg_contiguous_mask_fraction", 0.25)
        )
        self.sigreg_contiguous_mask_min_len = int(
            config.get("sigreg_contiguous_mask_min_len", 1)
        )
        self.sigreg_random_mask_prob = float(
            config.get("sigreg_random_mask_prob", 0.05)
        )
        self.sigreg_mz_jitter_std = float(config.get("sigreg_mz_jitter_std", 0.005))
        self.sigreg_intensity_jitter_std = float(
            config.get("sigreg_intensity_jitter_std", 0.05)
        )
        self.sigreg_fill_invalid_with_noise = bool(
            config.get(
                "sigreg_fill_invalid_with_noise",
                _DEFAULT_SIGREG_FILL_INVALID_WITH_NOISE,
            )
        )
        self.sigreg_noise_mz_min = float(
            config.get("sigreg_noise_mz_min", _DEFAULT_SIGREG_NOISE_MZ_MIN)
        )
        self.sigreg_noise_mz_max = float(
            config.get("sigreg_noise_mz_max", _DEFAULT_SIGREG_NOISE_MZ_MAX)
        )
        self.sigreg_noise_intensity_mean = float(
            config.get(
                "sigreg_noise_intensity_mean",
                _DEFAULT_SIGREG_NOISE_INTENSITY_MEAN,
            )
        )
        self.sigreg_noise_intensity_std = float(
            config.get(
                "sigreg_noise_intensity_std",
                _DEFAULT_SIGREG_NOISE_INTENSITY_STD,
            )
        )

        self.metadata = _ensure_processed(
            self.output_dir,
            self.validation_fraction,
            self.split_seed,
            self.num_shards,
        )

        self.gems_train_files = [
            str(self.output_dir / "train" / fn) for fn in self.metadata["train_files"]
        ]
        self.gems_val_files = [
            str(self.output_dir / "validation" / fn)
            for fn in self.metadata["validation_files"]
        ]
        self.gems_test_files = list(self.gems_val_files)
        self.massspec_train_files = [
            str(self.output_dir / "massspec_train" / fn)
            for fn in self.metadata.get("massspec_train_files", [])
        ]
        self.massspec_val_files = [
            str(self.output_dir / "massspec_val" / fn)
            for fn in self.metadata.get("massspec_val_files", [])
        ]
        self.massspec_test_files = [
            str(self.output_dir / "massspec_test" / fn)
            for fn in self.metadata.get("massspec_test_files", [])
        ]

        self.info = _compute_info(
            self.metadata,
            output_dir=self.output_dir,
            max_precursor_mz=self.max_precursor_mz,
            mz_representation=self.mz_representation,
        )
        self.train_splits = ["gems_train", "massspec_train"]
        self.eval_splits = ["gems_val", "massspec_val"]
        self.test_splits = ["gems_test", "massspec_test"]

        self.steps = {
            "gems_train": _steps_from_size(
                int(self.info["train_size"]),
                self.batch_size,
                self.drop_remainder,
            ),
            "gems_val": _steps_from_size(
                int(self.info["validation_size"]),
                self.batch_size,
                False,
            ),
            "gems_test": _steps_from_size(
                int(self.info["validation_size"]),
                self.batch_size,
                False,
            ),
            "massspec_train": _steps_from_size(
                int(self.info.get("massspec_train_size", 0)),
                self.batch_size,
                self.drop_remainder,
            ),
            "massspec_val": _steps_from_size(
                int(self.info.get("massspec_val_size", 0)),
                self.batch_size,
                False,
            ),
            "massspec_test": _steps_from_size(
                int(self.info.get("massspec_test_size", 0)),
                self.batch_size,
                False,
            ),
        }
        self.train_steps = self.steps["gems_train"]

        default_pin = torch.cuda.is_available()
        self.pin_memory = bool(config.get("dataloader_pin_memory", default_pin))
        self.dataloader_num_workers = int(config.get("dataloader_num_workers", 1))
        self.dataloader_prefetch_factor = int(config.get("dataloader_prefetch_factor", 2))
        self.dataloader_persistent_workers = bool(
            config.get("dataloader_persistent_workers", self.dataloader_num_workers > 0)
        )

        self._train_loader: DataLoader | None = None
        self._val_loader: DataLoader | None = None
        self._val_loader_built = False

    def _build_dataset_for_files(
        self,
        files: list[str],
        *,
        seed: int,
        shuffle: bool,
        drop_remainder: bool,
        include_fingerprint: bool,
    ) -> tf.data.Dataset:
        shuffle_buffer = self.shuffle_buffer if shuffle else 0
        return _build_dataset(
            files,
            self.batch_size,
            shuffle_buffer,
            seed,
            drop_remainder=drop_remainder,
            tfrecord_buffer_size=self.tfrecord_buffer_size,
            max_precursor_mz=self.max_precursor_mz,
            include_fingerprint=include_fingerprint,
            min_peak_intensity=self.min_peak_intensity,
            mz_representation=self.mz_representation,
            include_sigreg_augmentation=True,
            sigreg_contiguous_mask_fraction=self.sigreg_contiguous_mask_fraction,
            sigreg_contiguous_mask_min_len=self.sigreg_contiguous_mask_min_len,
            sigreg_random_mask_prob=self.sigreg_random_mask_prob,
            sigreg_mz_jitter_std=self.sigreg_mz_jitter_std,
            sigreg_intensity_jitter_std=self.sigreg_intensity_jitter_std,
            sigreg_fill_invalid_with_noise=self.sigreg_fill_invalid_with_noise,
            sigreg_noise_mz_min=self.sigreg_noise_mz_min,
            sigreg_noise_mz_max=self.sigreg_noise_mz_max,
            sigreg_noise_intensity_mean=self.sigreg_noise_intensity_mean,
            sigreg_noise_intensity_std=self.sigreg_noise_intensity_std,
            peak_ordering=self.peak_ordering,
        )

    def _build_gems_train_dataset(self, seed: int) -> tf.data.Dataset:
        return self._build_dataset_for_files(
            self.gems_train_files,
            seed=seed,
            shuffle=True,
            drop_remainder=self.drop_remainder,
            include_fingerprint=False,
        )

    def _build_gems_val_dataset(self, seed: int) -> tf.data.Dataset:
        return self._build_dataset_for_files(
            self.gems_val_files,
            seed=seed,
            shuffle=False,
            drop_remainder=True,
            include_fingerprint=False,
        )

    def _build_gems_test_dataset(self, seed: int) -> tf.data.Dataset:
        return self._build_dataset_for_files(
            self.gems_test_files,
            seed=seed,
            shuffle=False,
            drop_remainder=True,
            include_fingerprint=False,
        )

    def _build_massspec_train_dataset(self, seed: int) -> tf.data.Dataset:
        return self._build_dataset_for_files(
            self.massspec_train_files,
            seed=seed,
            shuffle=True,
            drop_remainder=self.drop_remainder,
            include_fingerprint=False,
        )

    def _build_massspec_val_dataset(self, seed: int) -> tf.data.Dataset:
        return self._build_dataset_for_files(
            self.massspec_val_files,
            seed=seed,
            shuffle=False,
            drop_remainder=True,
            include_fingerprint=False,
        )

    def _build_massspec_test_dataset(self, seed: int) -> tf.data.Dataset:
        return self._build_dataset_for_files(
            self.massspec_test_files,
            seed=seed,
            shuffle=False,
            drop_remainder=True,
            include_fingerprint=False,
        )

    def build_massspec_probe_dataset(
        self,
        split: str,
        seed: int,
        *,
        peak_ordering: str | None = None,
        shuffle: bool = False,
        drop_remainder: bool = True,
    ) -> tf.data.Dataset:
        if split == "massspec_train":
            files = self.massspec_train_files
        elif split == "massspec_val":
            files = self.massspec_val_files
        else:
            files = self.massspec_test_files
        if peak_ordering is None:
            peak_ordering = self.peak_ordering
        shuffle_buffer = self.shuffle_buffer if shuffle else 0
        return _build_dataset(
            files,
            self.batch_size,
            shuffle_buffer=shuffle_buffer,
            seed=seed,
            drop_remainder=drop_remainder,
            tfrecord_buffer_size=self.tfrecord_buffer_size,
            max_precursor_mz=self.max_precursor_mz,
            include_fingerprint=True,
            min_peak_intensity=self.min_peak_intensity,
            mz_representation=self.mz_representation,
            include_sigreg_augmentation=False,
            sigreg_contiguous_mask_fraction=self.sigreg_contiguous_mask_fraction,
            sigreg_contiguous_mask_min_len=self.sigreg_contiguous_mask_min_len,
            sigreg_random_mask_prob=self.sigreg_random_mask_prob,
            sigreg_mz_jitter_std=self.sigreg_mz_jitter_std,
            sigreg_intensity_jitter_std=self.sigreg_intensity_jitter_std,
            peak_ordering=peak_ordering,
        )

    def _make_loader(
        self,
        *,
        dataset_builder: Callable[[], tf.data.Dataset],
        steps: int,
    ) -> DataLoader:
        dataset = _TfIterableDataset(
            dataset_builder=dataset_builder,
            steps_per_epoch=steps,
        )
        loader_kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_size": None,
            "num_workers": self.dataloader_num_workers,
            "pin_memory": self.pin_memory,
            "collate_fn": _identity_collate,
        }
        if self.dataloader_num_workers > 0:
            loader_kwargs["persistent_workers"] = self.dataloader_persistent_workers
            loader_kwargs["prefetch_factor"] = self.dataloader_prefetch_factor
        return _StatefulDataLoader(**loader_kwargs)

    @property
    def train_loader(self) -> DataLoader:
        """GeMS train DataLoader (built once, cached)."""
        if self._train_loader is None:
            self._train_loader = self._make_loader(
                dataset_builder=lambda: self._build_gems_train_dataset(self.seed),
                steps=self.train_steps,
            )
        return self._train_loader

    @property
    def val_loader(self) -> DataLoader | None:
        """MassSpecGym val DataLoader (built once, cached).

        Returns ``None`` if the MassSpecGym val split has no steps.
        """
        if not self._val_loader_built:
            steps = self.steps["massspec_val"]
            if steps == 0:
                self._val_loader = None
            else:
                self._val_loader = self._make_loader(
                    dataset_builder=lambda: self._build_massspec_val_dataset(
                        self.seed + 10_000,
                    ),
                    steps=steps,
                )
            self._val_loader_built = True
        return self._val_loader


# -----------------------------------------------------------------------------
# CLI helpers
# -----------------------------------------------------------------------------


def _load_config(path: str) -> config_dict.ConfigDict:
    spec = importlib.util.spec_from_file_location("gems_dataset_config", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.get_config()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Create and inspect GeMS peak list datasets."
    )
    parser.add_argument("config", help="Path to a dataset config file (python).")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    datamodule = TfLightningDataModule(cfg, seed=int(cfg.seed))
    train_iter = datamodule._build_gems_train_dataset(int(cfg.seed)).as_numpy_iterator()
    info = datamodule.info

    print("\nDataset info:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    print("\nSample batch:")
    batch = next(train_iter)
    for k, v in batch.items():
        v_type = type(v)
        v_shape = getattr(v, "shape", None)
        v_dtype = getattr(v, "dtype", None)
        print(f"  {k}: type={v_type}, shape={v_shape}, dtype={v_dtype}")

    if "peak_mz" in batch:
        print("\nFirst sample peak_mz:", batch["peak_mz"][0][:10])
        print("First sample peak_intensity:", batch["peak_intensity"][0][:10])
        print("First sample peak_valid_mask:", batch["peak_valid_mask"][0][:10])
