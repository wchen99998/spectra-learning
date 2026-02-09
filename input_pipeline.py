"""Unified TF input pipeline and Lightning DataModule for GeMS_A peak lists."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Callable, Optional

import lightning.pytorch as pl
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
_DEFAULT_GEMS_FORMULA_TFRECORD_DIR = Path("data/gems_formula_tfrecord")
_DEFAULT_GEMS_FORMULA_RAW_CSV_PATH = Path(
    "data/gems_formula/raw/GeMS_2m_combined_formula_identifications.csv"
)
_DEFAULT_TFRECORD_BUFFER_SIZE = 250_000
_DEFAULT_SPLIT_SEED = 42
_DEFAULT_NUM_SHARDS = 4
_DEFAULT_GEMS_FORMULA_NUM_SHARDS = 16
_NUM_PEAKS_INPUT = 128
_NUM_PEAKS_OUTPUT = 60
_DEFAULT_MAX_PRECURSOR_MZ = 1000.0
_FINGERPRINT_BITS = 1024
_FINGERPRINT_RADIUS = 2
_PEAK_MZ_MIN = 20.0
_PEAK_MZ_MAX = 1000.0
_PRECURSOR_MZ_WINDOW = 2.5
_INTENSITY_BINS = 32
_INTENSITY_EPS = 1e-4
_DEFAULT_INTENSITY_SCALING = "log"
_SPECIAL_TOKENS = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3}
_NUM_SPECIAL_TOKENS = len(_SPECIAL_TOKENS)
_DEFAULT_PAIR_SEQUENCE_LENGTH = 128

_METADATA_FILENAME = "metadata.json"

_GEMS_HF_REPO = "roman-bushuiev/GeMS"
_GEMS_HDF5_PATH = "data/GeMS_A/GeMS_A.hdf5"
_MASSSPEC_HF_REPO = "roman-bushuiev/MassSpecGym"
_MASSSPEC_TSV_PATH = "data/MassSpecGym.tsv"
_MASSSPEC_METADATA_VERSION = 2
_GEMS_FORMULA_METADATA_VERSION = 1
_GEMS_FORMULA_GCS_URI = (
    "gs://main-novogaia-bucket/gems/GeMS_2m_combined_formula_identifications.csv"
)
_DEFAULT_GCP_KEY_PATH = Path("/home/wuhao/md4/key.json")
_DEFAULT_GEMS_FORMULA_COLUMN_NAME = "formula"
_DEFAULT_GEMS_ADDUCT_COLUMN_NAME = "adduct"


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


def _download_gcs_file(gcs_uri: str, local_path: Path, gcp_key_path: Path) -> Path:
    local_path = local_path.expanduser().resolve()
    if local_path.exists():
        return local_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(
        gcp_key_path.expanduser().resolve()
    )
    tf.io.gfile.copy(gcs_uri, str(local_path), overwrite=False)
    return local_path


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


def _load_gems_formula_csv(
    csv_path: Path,
    *,
    formula_column_name: str,
    adduct_column_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    import csv

    formula: list[str] = []
    adduct: list[str] = []

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            formula.append(row[formula_column_name])
            value = row[adduct_column_name]
            adduct.append(value if value != "" else "unknown")

    formula_array = np.asarray(formula, dtype=object)
    adduct_array = np.asarray(adduct, dtype=object)
    return formula_array, adduct_array


def _split_indices(
    n: int,
    split_seed: int,
    *,
    train_fraction: float,
    val_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n)
    train_size = int(n * train_fraction)
    val_size = int(n * val_fraction)
    train_idx = perm[:train_size]
    val_idx = perm[train_size : train_size + val_size]
    test_idx = perm[train_size + val_size :]
    return train_idx, val_idx, test_idx


def _compute_morgan_fingerprints(smiles: np.ndarray) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs

    fps = np.zeros((len(smiles), _FINGERPRINT_BITS), dtype=np.int8)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(s))
        fp = AllChem.GetMorganFingerprintAsBitVect(
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

        with tf.io.TFRecordWriter(str(shard_file), options=options) as writer:
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

        with tf.io.TFRecordWriter(str(shard_file), options=options) as writer:
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


def _write_formula_tfrecords(
    formula: np.ndarray,
    adduct_id: np.ndarray,
    output_path: Path,
    num_shards: int,
    desc: str,
) -> tuple[list[str], list[int]]:
    n = len(formula)
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

        with tf.io.TFRecordWriter(str(shard_file), options=options) as writer:
            for i in tqdm(range(start, end), desc=f"{desc} [{shard_id + 1}/{num_shards}]"):
                formula_bytes = str(formula[i]).encode("utf-8")
                adduct = int(adduct_id[i])
                features = {
                    "formula": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[formula_bytes])
                    ),
                    "adduct_id": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[adduct])
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


def _process_gems_formula(
    output_dir: Path,
    *,
    raw_csv_path: Path,
    gcs_uri: str,
    gcp_key_path: Path,
    formula_column_name: str,
    adduct_column_name: str,
    split_seed: int,
    num_shards: int,
) -> dict[str, Any]:
    if not raw_csv_path.exists():
        logger.info("Downloading GeMS formula CSV from GCS...")
        _download_gcs_file(gcs_uri, raw_csv_path, gcp_key_path)

    logger.info("Loading GeMS formula CSV...")
    formula, adduct = _load_gems_formula_csv(
        raw_csv_path,
        formula_column_name=formula_column_name,
        adduct_column_name=adduct_column_name,
    )
    adduct_id, adduct_vocab = _encode_categorical_ids(adduct)

    train_idx, val_idx, test_idx = _split_indices(
        len(formula),
        split_seed,
        train_fraction=0.90,
        val_fraction=0.05,
    )
    train_size = int(train_idx.size)
    val_size = int(val_idx.size)
    test_size = int(test_idx.size)
    logger.info(
        "GeMS formula entries: %d (train=%d, val=%d, test=%d)",
        len(formula),
        train_size,
        val_size,
        test_size,
    )

    train_files, train_lengths = _write_formula_tfrecords(
        formula[train_idx],
        adduct_id[train_idx],
        output_dir / "train",
        num_shards,
        desc="GeMS Formula Train",
    )
    val_files, val_lengths = _write_formula_tfrecords(
        formula[val_idx],
        adduct_id[val_idx],
        output_dir / "validation",
        max(1, num_shards // 4),
        desc="GeMS Formula Validation",
    )
    test_files, test_lengths = _write_formula_tfrecords(
        formula[test_idx],
        adduct_id[test_idx],
        output_dir / "test",
        max(1, num_shards // 4),
        desc="GeMS Formula Test",
    )

    return {
        "gems_formula_train_files": train_files,
        "gems_formula_train_lengths": train_lengths,
        "gems_formula_train_size": train_size,
        "gems_formula_val_files": val_files,
        "gems_formula_val_lengths": val_lengths,
        "gems_formula_val_size": val_size,
        "gems_formula_test_files": test_files,
        "gems_formula_test_lengths": test_lengths,
        "gems_formula_test_size": test_size,
        "gems_formula_adduct_vocab": adduct_vocab,
        "gems_formula_adduct_vocab_size": len(adduct_vocab),
        "gems_formula_metadata_version": _GEMS_FORMULA_METADATA_VERSION,
        "gems_formula_source_csv": str(raw_csv_path),
        "gems_formula_formula_column_name": formula_column_name,
        "gems_formula_adduct_column_name": adduct_column_name,
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


def _ensure_formula_processed(
    output_dir: Path,
    *,
    raw_csv_path: Path,
    gcs_uri: str,
    gcp_key_path: Path,
    formula_column_name: str,
    adduct_column_name: str,
    split_seed: int,
    num_shards: int,
) -> dict[str, Any]:
    metadata_path = output_dir / _METADATA_FILENAME
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        train_files = metadata.get("gems_formula_train_files", [])
        val_files = metadata.get("gems_formula_val_files", [])
        test_files = metadata.get("gems_formula_test_files", [])
        train_ok = all((output_dir / "train" / fn).exists() for fn in train_files)
        val_ok = all((output_dir / "validation" / fn).exists() for fn in val_files)
        test_ok = all((output_dir / "test" / fn).exists() for fn in test_files)
        version_ok = (
            int(metadata.get("gems_formula_metadata_version", 0))
            == _GEMS_FORMULA_METADATA_VERSION
        )
        vocab_ok = "gems_formula_adduct_vocab" in metadata
        if train_files and val_files and test_files and train_ok and val_ok and test_ok and version_ok and vocab_ok:
            logger.info("Found existing GeMS formula TFRecords at %s", output_dir)
            return metadata

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = _process_gems_formula(
        output_dir,
        raw_csv_path=raw_csv_path,
        gcs_uri=gcs_uri,
        gcp_key_path=gcp_key_path,
        formula_column_name=formula_column_name,
        adduct_column_name=adduct_column_name,
        split_seed=split_seed,
        num_shards=num_shards,
    )
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved formula metadata to %s", metadata_path)
    return metadata


# -----------------------------------------------------------------------------
# tf.data pipeline
# -----------------------------------------------------------------------------


def _parse_example(serialized: tf.Tensor) -> dict[str, tf.Tensor]:
    features = {
        "mz": tf.io.FixedLenFeature([_NUM_PEAKS_INPUT], tf.float32),
        "intensity": tf.io.FixedLenFeature([_NUM_PEAKS_INPUT], tf.float32),
        "rt": tf.io.FixedLenFeature([1], tf.float32),
        "precursor_mz": tf.io.FixedLenFeature([1], tf.float32),
    }
    parsed = tf.io.parse_single_example(serialized, features)
    return {
        "mz": parsed["mz"],
        "intensity": parsed["intensity"],
        "rt": parsed["rt"][0],
        "precursor_mz": parsed["precursor_mz"][0],
    }


def _parse_example_with_fingerprint(serialized: tf.Tensor) -> dict[str, tf.Tensor]:
    features = {
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
    parsed = tf.io.parse_single_example(serialized, features)
    return {
        "mz": parsed["mz"],
        "intensity": parsed["intensity"],
        "rt": parsed["rt"][0],
        "precursor_mz": parsed["precursor_mz"][0],
        "fingerprint": tf.cast(parsed["fingerprint"], tf.int32),
        "adduct_id": tf.cast(parsed["adduct_id"][0], tf.int32),
        "instrument_type_id": tf.cast(parsed["instrument_type_id"][0], tf.int32),
        "collision_energy": parsed["collision_energy"][0],
        "collision_energy_present": tf.cast(parsed["collision_energy_present"][0], tf.int32),
    }


def _parse_formula_example(serialized: tf.Tensor) -> dict[str, tf.Tensor]:
    features = {
        "formula": tf.io.FixedLenFeature([], tf.string),
        "adduct_id": tf.io.FixedLenFeature([1], tf.int64),
    }
    parsed = tf.io.parse_single_example(serialized, features)
    return {
        "formula": parsed["formula"],
        "adduct_id": tf.cast(parsed["adduct_id"][0], tf.int32),
    }


def _filter_max_precursor_mz(max_precursor_mz: float) -> Callable[[dict], tf.Tensor]:
    max_val = tf.constant(max_precursor_mz, tf.float32)

    def keep(example: dict) -> tf.Tensor:
        return tf.squeeze(example["precursor_mz"]) <= max_val

    return keep


def _filter_peak_mz_range(
    min_mz: float, max_mz: float, precursor_window: float
) -> Callable[[dict], dict]:
    min_val = tf.constant(min_mz, tf.float32)
    max_val = tf.constant(max_mz, tf.float32)
    window = tf.constant(precursor_window, tf.float32)

    def apply(example: dict) -> dict:
        mz = example["mz"]
        precursor_mz = tf.squeeze(example["precursor_mz"])
        upper = tf.where(precursor_mz > 0.0, precursor_mz - window, max_val)
        keep = (mz >= min_val) & (mz <= upper)
        example["mz"] = tf.where(keep, mz, 0.0)
        example["intensity"] = tf.where(keep, example["intensity"], 0.0)
        return example

    return apply


def _convert_to_neutral_loss() -> Callable[[dict], dict]:
    def apply(example: dict) -> dict:
        precursor_mz = tf.squeeze(example["precursor_mz"])
        example["mz"] = precursor_mz - example["mz"]
        return example

    return apply


def _compact_sort_peaks() -> Callable[[dict], dict]:
    def apply(example: dict) -> dict:
        mz = example["mz"]
        intensity = example["intensity"]
        keep = mz > 0
        kept_mz = tf.boolean_mask(mz, keep)
        kept_intensity = tf.boolean_mask(intensity, keep)
        sorted_idx = tf.argsort(kept_intensity, direction="DESCENDING")
        kept_mz = tf.gather(kept_mz, sorted_idx)
        kept_intensity = tf.gather(kept_intensity, sorted_idx)
        pad = _NUM_PEAKS_OUTPUT - tf.shape(kept_mz)[0]
        example["mz"] = tf.pad(kept_mz, [[0, pad]])
        example["intensity"] = tf.pad(kept_intensity, [[0, pad]])
        return example

    return apply


def _topk_peaks(num_peaks: int) -> Callable[[dict], dict]:
    def apply(example: dict) -> dict:
        intensity = example["intensity"]
        mz = example["mz"]
        values, indices = tf.math.top_k(intensity, k=num_peaks, sorted=True)
        example["intensity"] = values
        example["mz"] = tf.gather(mz, indices)
        return example

    return apply


def _strip_padding_and_tokenize(
    max_precursor_mz: float,
    intensity_scaling: str,
) -> Callable[[dict], dict]:
    eps = tf.constant(_INTENSITY_EPS, tf.float32)
    log_eps = tf.math.log(eps)
    denom = -log_eps
    linear_denom = tf.constant(1.0, tf.float32) - eps
    bins = tf.constant(_INTENSITY_BINS - 1, tf.float32)
    mz_bins = int(_PEAK_MZ_MAX) + 1
    precursor_bins = int(max_precursor_mz) + 1
    mz_offset = tf.constant(_NUM_SPECIAL_TOKENS, tf.int32)
    precursor_offset = mz_offset
    intensity_offset = tf.constant(_NUM_SPECIAL_TOKENS + mz_bins, tf.int32)

    if intensity_scaling == "linear":
        def scale(intensity: tf.Tensor) -> tf.Tensor:
            return (intensity - eps) / linear_denom
    else:
        def scale(intensity: tf.Tensor) -> tf.Tensor:
            return (tf.math.log(intensity) - log_eps) / denom

    def apply(example: dict) -> dict:
        mz = example["mz"]
        intensity = example["intensity"]
        keep = mz > 0
        mz = tf.boolean_mask(mz, keep)
        intensity = tf.boolean_mask(intensity, keep)
        mz_tokens = tf.cast(tf.floor(mz), tf.int32) + mz_offset
        precursor_tokens = tf.cast(
            tf.floor(tf.clip_by_value(example["precursor_mz"], 0.0, max_precursor_mz)),
            tf.int32,
        ) + precursor_offset
        intensity = tf.clip_by_value(intensity, eps, 1.0)
        s = scale(intensity)
        tokens = tf.floor(s * bins)
        example["mz"] = mz_tokens
        example["intensity"] = tf.cast(tokens, tf.int32) + intensity_offset
        example["precursor_mz"] = precursor_tokens
        return example

    return apply


def detokenize_spectrum(
    token_ids: np.ndarray | torch.Tensor,
    *,
    max_precursor_mz: float = _DEFAULT_MAX_PRECURSOR_MZ,
    intensity_scaling: str = _DEFAULT_INTENSITY_SCALING,
) -> dict[str, Any] | list[dict[str, Any]]:
    tokens = token_ids
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.detach().cpu().numpy()
    tokens = np.asarray(tokens, dtype=np.int32)

    if tokens.ndim == 2:
        return [
            detokenize_spectrum(
                row,
                max_precursor_mz=max_precursor_mz,
                intensity_scaling=intensity_scaling,
            )
            for row in tokens
        ]

    pad_id = _SPECIAL_TOKENS["[PAD]"]
    pad_positions = np.where(tokens == pad_id)[0]
    end = int(pad_positions[0]) if pad_positions.size > 0 else int(tokens.shape[0])
    content = tokens[1:end]
    peaks = content
    mz_tokens = peaks[0::2]
    intensity_tokens = peaks[1::2]

    mz = (mz_tokens - _NUM_SPECIAL_TOKENS).astype(np.float32)

    bins = _INTENSITY_BINS - 1
    intensity_offset = _NUM_SPECIAL_TOKENS + int(_PEAK_MZ_MAX) + 1
    intensity_idx = intensity_tokens - intensity_offset
    s = intensity_idx.astype(np.float32) / float(bins)
    if intensity_scaling == "linear":
        intensity = (s * (1.0 - _INTENSITY_EPS) + _INTENSITY_EPS).astype(np.float32)
    else:
        log_eps = math.log(_INTENSITY_EPS)
        denom = -log_eps
        intensity = np.exp(s * denom + log_eps).astype(np.float32)

    return {
        "precursor_mz": None,
        "mz": mz,
        "intensity": intensity,
    }


def _build_single_spectrum_input(max_len: int) -> Callable[[dict], dict]:
    cls_id = tf.constant(_SPECIAL_TOKENS["[CLS]"], tf.int32)
    pad_id = tf.constant(_SPECIAL_TOKENS["[PAD]"], tf.int32)
    max_peaks = tf.constant((max_len - 1) // 2, tf.int32)

    def interleave(mz: tf.Tensor, intensity: tf.Tensor) -> tf.Tensor:
        pair = tf.stack([mz, intensity], axis=1)
        return tf.reshape(pair, [-1])

    def apply(example: dict) -> dict:
        mz = example["mz"][:max_peaks]
        intensity = example["intensity"][:max_peaks]
        seq = interleave(mz, intensity)

        tokens = tf.concat([cls_id[None], seq], axis=0)
        seg = tf.zeros([tf.shape(tokens)[0]], tf.int32)

        tokens = tokens[:max_len]
        seg = seg[:max_len]
        pad_len = max_len - tf.shape(tokens)[0]
        tokens = tf.pad(tokens, [[0, pad_len]], constant_values=pad_id)
        seg = tf.pad(seg, [[0, pad_len]], constant_values=0)

        example["token_ids"] = tokens
        example["segment_ids"] = seg
        del example["mz"]
        del example["intensity"]
        return example

    return apply


def _apply_mlm_mask(mask_ratio: float, mask_token_id: int) -> Callable[[dict], dict]:
    cls_id = tf.constant(_SPECIAL_TOKENS["[CLS]"], tf.int32)
    pad_id = tf.constant(_SPECIAL_TOKENS["[PAD]"], tf.int32)
    mask_token = tf.constant(mask_token_id, tf.int32)
    mask_ratio_t = tf.constant(mask_ratio, tf.float32)

    def apply(batch: dict) -> dict:
        token_ids = batch["token_ids"]
        maskable = tf.logical_and(token_ids != cls_id, token_ids != pad_id)
        maskable_count = tf.reduce_sum(tf.cast(maskable, tf.int32), axis=1)
        mask_count = tf.cast(
            mask_ratio_t * tf.cast(maskable_count, tf.float32),
            tf.int32,
        )
        scores = tf.random.uniform(tf.shape(token_ids), dtype=tf.float32)
        scores = tf.where(maskable, scores, tf.constant(-1.0, tf.float32))
        order = tf.argsort(scores, direction="DESCENDING")
        rank = tf.argsort(order, direction="ASCENDING")
        mask = rank < mask_count[:, None]
        batch["masked_token_ids"] = tf.where(mask, mask_token, token_ids)
        batch["mlm_mask"] = mask
        return batch

    return apply


def _build_dataset(
    filenames: list[str],
    batch_size: int,
    shuffle_buffer: int,
    seed: Optional[int],
    drop_remainder: bool,
    *,
    tfrecord_buffer_size: int,
    max_precursor_mz: float,
    pair_sequence_length: int,
    mask_ratio: float,
    mask_token_id: int,
    include_fingerprint: bool,
    intensity_scaling: str,
    mz_representation: str,
) -> tf.data.Dataset:
    parse_fn = _parse_example_with_fingerprint if include_fingerprint else _parse_example
    ds = tf.data.TFRecordDataset(
        filenames,
        compression_type="GZIP",
        buffer_size=int(tfrecord_buffer_size),
        num_parallel_reads=tf.data.AUTOTUNE,
    )
    ds = ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.filter(_filter_max_precursor_mz(max_precursor_mz))
    ds = ds.map(
        _filter_peak_mz_range(_PEAK_MZ_MIN, _PEAK_MZ_MAX, _PRECURSOR_MZ_WINDOW),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.map(_topk_peaks(_NUM_PEAKS_OUTPUT), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(_compact_sort_peaks(), num_parallel_calls=tf.data.AUTOTUNE)
    if mz_representation == "neutral_loss":
        ds = ds.map(_convert_to_neutral_loss(), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(
        _strip_padding_and_tokenize(max_precursor_mz, intensity_scaling),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if shuffle_buffer > 0:
        ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)

    ds = ds.map(
        _build_single_spectrum_input(pair_sequence_length),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.map(
        _apply_mlm_mask(mask_ratio, mask_token_id),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    options = tf.data.Options()
    options.experimental_deterministic = True
    ds = ds.with_options(options)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _build_formula_dataset(
    filenames: list[str],
    batch_size: int,
    shuffle_buffer: int,
    seed: Optional[int],
    drop_remainder: bool,
    *,
    tfrecord_buffer_size: int,
) -> tf.data.Dataset:
    ds = tf.data.TFRecordDataset(
        filenames,
        compression_type="GZIP",
        buffer_size=int(tfrecord_buffer_size),
        num_parallel_reads=tf.data.AUTOTUNE,
    )
    ds = ds.map(_parse_formula_example, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle_buffer > 0:
        ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    options = tf.data.Options()
    options.experimental_deterministic = True
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
    pair_sequence_length: int,
    intensity_scaling: str,
    mz_representation: str,
) -> dict[str, Any]:
    mz_bins = int(_PEAK_MZ_MAX) + 1
    precursor_bins = int(max_precursor_mz) + 1
    mz_offset = _NUM_SPECIAL_TOKENS
    precursor_offset = mz_offset
    intensity_offset = mz_offset + mz_bins
    vocab_size = intensity_offset + _INTENSITY_BINS
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
        "intensity_bins": _INTENSITY_BINS,
        "intensity_eps": _INTENSITY_EPS,
        "mz_bins": mz_bins,
        "mz_offset": mz_offset,
        "precursor_bins": precursor_bins,
        "precursor_offset": precursor_offset,
        "intensity_offset": intensity_offset,
        "vocab_size": vocab_size,
        "special_tokens": dict(_SPECIAL_TOKENS),
        "pair_sequence_length": pair_sequence_length,
        "intensity_scaling": intensity_scaling,
        "mz_representation": mz_representation,
        "fingerprint_bits": _FINGERPRINT_BITS,
    }


def _compute_formula_info(
    metadata: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    adduct_vocab = metadata.get("gems_formula_adduct_vocab", {"unknown": 0})
    return {
        "gems_formula_tfrecord_dir": str(output_dir),
        "gems_formula_train_size": int(metadata["gems_formula_train_size"]),
        "gems_formula_val_size": int(metadata["gems_formula_val_size"]),
        "gems_formula_test_size": int(metadata["gems_formula_test_size"]),
        "gems_formula_adduct_vocab": adduct_vocab,
        "gems_formula_adduct_vocab_size": int(metadata["gems_formula_adduct_vocab_size"]),
        "gems_formula_metadata_version": int(metadata["gems_formula_metadata_version"]),
        "gems_formula_source_csv": str(metadata["gems_formula_source_csv"]),
        "gems_formula_formula_column_name": str(
            metadata["gems_formula_formula_column_name"]
        ),
        "gems_formula_adduct_column_name": str(metadata["gems_formula_adduct_column_name"]),
    }


def _steps_from_size(size: int, batch_size: int, drop_remainder: bool) -> int:
    if drop_remainder:
        return int(size // batch_size)
    return int(math.ceil(size / batch_size))


# -----------------------------------------------------------------------------
# NumPy -> torch conversion
# -----------------------------------------------------------------------------


def _torch_dtype(array: np.ndarray) -> torch.dtype:
    if np.issubdtype(array.dtype, np.bool_):
        return torch.bool
    if np.issubdtype(array.dtype, np.integer):
        return torch.long
    if np.issubdtype(array.dtype, np.floating):
        return torch.float32
    return torch.as_tensor(array).dtype


def _to_torch(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, list):
        return [_to_torch(item) for item in value]
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return _to_torch(value.tolist())
        array = np.ascontiguousarray(value)
        if not array.flags.writeable:
            array = array.copy()
        return torch.tensor(array, dtype=_torch_dtype(array))
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
        dataset: tf.data.Dataset,
        steps_per_epoch: int,
    ) -> None:
        super().__init__()
        self._dataset = dataset
        self.steps_per_epoch = int(steps_per_epoch)
        self._resume_from = 0
        self._num_yielded = 0

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self):
        self._num_yielded = self._resume_from if self._resume_from > 0 else 0
        resume_from = self._resume_from
        self._resume_from = 0
        dataset = self._dataset.skip(resume_from) if resume_from > 0 else self._dataset
        iterator = dataset.as_numpy_iterator()
        for batch in iterator:
            yield numpy_batch_to_torch(batch)
            self._num_yielded += 1

    def state_dict(self) -> dict[str, Any]:
        return {"num_yielded": self._num_yielded}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._resume_from = int(state_dict["num_yielded"])
        self._num_yielded = self._resume_from


class _RoundRobinTfIterableDataset(IterableDataset):
    def __init__(
        self,
        *,
        datasets: list[tf.data.Dataset],
        steps_per_epoch: list[int],
    ) -> None:
        super().__init__()
        self._datasets = datasets
        self._steps_per_epoch = [int(step) for step in steps_per_epoch]
        self.steps_per_epoch = int(sum(self._steps_per_epoch))
        self._resume_from = 0
        self._num_yielded = 0

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self):
        self._num_yielded = self._resume_from if self._resume_from > 0 else 0
        resume_from = self._resume_from
        self._resume_from = 0
        iterators = [ds.as_numpy_iterator() for ds in self._datasets]
        remaining = list(self._steps_per_epoch)
        num_iters = len(iterators)
        idx = 0
        while sum(remaining) > 0:
            while remaining[idx] == 0:
                idx = (idx + 1) % num_iters
            batch = next(iterators[idx])
            remaining[idx] -= 1
            idx = (idx + 1) % num_iters
            if resume_from > 0:
                resume_from -= 1
                continue
            yield numpy_batch_to_torch(batch)
            self._num_yielded += 1

    def state_dict(self) -> dict[str, Any]:
        return {"num_yielded": self._num_yielded}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._resume_from = int(state_dict["num_yielded"])
        self._num_yielded = self._resume_from


class _StatefulDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
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


class TfLightningDataModule(pl.LightningDataModule):
    """Lightning DataModule that rebuilds tf.data pipelines each epoch."""

    def __init__(self, config: config_dict.ConfigDict, seed: int) -> None:
        super().__init__()
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
        self.drop_remainder = bool(config.get("drop_remainder", False))
        self.max_precursor_mz = float(
            config.get("max_precursor_mz", _DEFAULT_MAX_PRECURSOR_MZ)
        )
        self.pair_sequence_length = int(
            config.get("pair_sequence_length", _DEFAULT_PAIR_SEQUENCE_LENGTH)
        )
        self.intensity_scaling = str(
            config.get("intensity_scaling", _DEFAULT_INTENSITY_SCALING)
        )
        self.mz_representation = str(config.get("mz_representation", "mz"))
        self.mask_ratio = float(config.get("mask_ratio", 0.15))
        self.mask_token_id = int(config.get("mask_token_id", _SPECIAL_TOKENS["[MASK]"]))

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
            pair_sequence_length=self.pair_sequence_length,
            intensity_scaling=self.intensity_scaling,
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
        self.train_steps = self.steps["gems_train"] + self.steps["massspec_train"]

        default_pin = torch.cuda.is_available()
        self.pin_memory = bool(config.get("dataloader_pin_memory", default_pin))
        self.dataloader_num_workers = int(config.get("dataloader_num_workers", 1))
        self.dataloader_prefetch_factor = int(config.get("dataloader_prefetch_factor", 2))
        self.dataloader_persistent_workers = bool(
            config.get("dataloader_persistent_workers", self.dataloader_num_workers > 0)
        )

    def state_dict(self) -> dict[str, Any]:
        return {"seed": self.seed}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.seed = int(state_dict["seed"])

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
            pair_sequence_length=self.pair_sequence_length,
            mask_ratio=self.mask_ratio,
            mask_token_id=self.mask_token_id,
            include_fingerprint=include_fingerprint,
            intensity_scaling=self.intensity_scaling,
            mz_representation=self.mz_representation,
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

    def build_massspec_probe_dataset(self, split: str, seed: int) -> tf.data.Dataset:
        files = self.massspec_train_files if split == "massspec_train" else self.massspec_test_files
        return _build_dataset(
            files,
            self.batch_size,
            shuffle_buffer=0,
            seed=seed,
            drop_remainder=False,
            tfrecord_buffer_size=self.tfrecord_buffer_size,
            max_precursor_mz=self.max_precursor_mz,
            pair_sequence_length=self.pair_sequence_length,
            mask_ratio=self.mask_ratio,
            mask_token_id=self.mask_token_id,
            include_fingerprint=True,
            intensity_scaling=self.intensity_scaling,
            mz_representation=self.mz_representation,
        )

    def _make_loader(
        self,
        *,
        dataset: tf.data.Dataset,
        steps: int,
    ) -> DataLoader:
        dataset = _TfIterableDataset(
            dataset=dataset,
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

    def _make_round_robin_loader(
        self,
        *,
        datasets: list[tf.data.Dataset],
        steps_per_epoch: list[int],
    ) -> DataLoader:
        dataset = _RoundRobinTfIterableDataset(
            datasets=datasets,
            steps_per_epoch=steps_per_epoch,
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

    def train_dataloader(self) -> DataLoader:
        base_seed = self.seed
        return self._make_round_robin_loader(
            datasets=[
                self._build_gems_train_dataset(base_seed),
                self._build_massspec_train_dataset(base_seed + 10_000),
            ],
            steps_per_epoch=[
                self.steps["gems_train"],
                self.steps["massspec_train"],
            ],
        )

    def val_dataloader(self):
        base_seed = self.seed + 1_000_000
        gems_loader = self._make_loader(
            dataset=self._build_gems_val_dataset(base_seed),
            steps=self.steps["gems_val"],
        )
        massspec_loader = self._make_loader(
            dataset=self._build_massspec_val_dataset(base_seed + 10_000),
            steps=self.steps["massspec_val"],
        )
        return [gems_loader, massspec_loader]

    def test_dataloader(self):
        base_seed = self.seed + 2_000_000
        gems_loader = self._make_loader(
            dataset=self._build_gems_test_dataset(base_seed),
            steps=self.steps["gems_test"],
        )
        massspec_loader = self._make_loader(
            dataset=self._build_massspec_test_dataset(base_seed + 10_000),
            steps=self.steps["massspec_test"],
        )
        return [gems_loader, massspec_loader]


class GeMSFormulaLightningDataModule(pl.LightningDataModule):
    """Lightning DataModule for GeMS formula/adduct records."""

    def __init__(self, config: config_dict.ConfigDict, seed: int) -> None:
        super().__init__()
        self.config = config
        self.seed = int(seed)

        self.output_dir = (
            Path(
                config.get(
                    "gems_formula_tfrecord_dir",
                    str(_DEFAULT_GEMS_FORMULA_TFRECORD_DIR),
                )
            )
            .expanduser()
            .resolve()
        )
        self.raw_csv_path = (
            Path(
                config.get(
                    "gems_formula_raw_csv_path",
                    str(_DEFAULT_GEMS_FORMULA_RAW_CSV_PATH),
                )
            )
            .expanduser()
            .resolve()
        )
        self.gcs_uri = str(config.get("gems_formula_gcs_uri", _GEMS_FORMULA_GCS_URI))
        self.gcp_key_path = (
            Path(config.get("gcp_key_path", str(_DEFAULT_GCP_KEY_PATH)))
            .expanduser()
            .resolve()
        )
        self.formula_column_name = str(
            config.get("gems_formula_column_name", _DEFAULT_GEMS_FORMULA_COLUMN_NAME)
        )
        self.adduct_column_name = str(
            config.get("gems_adduct_column_name", _DEFAULT_GEMS_ADDUCT_COLUMN_NAME)
        )

        self.batch_size = int(config.get("batch_size", _DEFAULT_BATCH_SIZE))
        self.shuffle_buffer = int(config.get("shuffle_buffer", _DEFAULT_SHUFFLE_BUFFER))
        self.tfrecord_buffer_size = int(
            config.get("tfrecord_buffer_size", _DEFAULT_TFRECORD_BUFFER_SIZE)
        )
        self.split_seed = int(config.get("gems_formula_split_seed", _DEFAULT_SPLIT_SEED))
        self.num_shards = int(
            config.get("gems_formula_num_shards", _DEFAULT_GEMS_FORMULA_NUM_SHARDS)
        )
        self.drop_remainder = bool(
            config.get(
                "gems_formula_drop_remainder",
                config.get("drop_remainder", False),
            )
        )

        self.metadata = _ensure_formula_processed(
            self.output_dir,
            raw_csv_path=self.raw_csv_path,
            gcs_uri=self.gcs_uri,
            gcp_key_path=self.gcp_key_path,
            formula_column_name=self.formula_column_name,
            adduct_column_name=self.adduct_column_name,
            split_seed=self.split_seed,
            num_shards=self.num_shards,
        )
        self.formula_train_files = [
            str(self.output_dir / "train" / fn)
            for fn in self.metadata["gems_formula_train_files"]
        ]
        self.formula_val_files = [
            str(self.output_dir / "validation" / fn)
            for fn in self.metadata["gems_formula_val_files"]
        ]
        self.formula_test_files = [
            str(self.output_dir / "test" / fn)
            for fn in self.metadata["gems_formula_test_files"]
        ]

        self.info = _compute_formula_info(self.metadata, output_dir=self.output_dir)
        self.steps = {
            "gems_formula_train": _steps_from_size(
                int(self.info["gems_formula_train_size"]),
                self.batch_size,
                self.drop_remainder,
            ),
            "gems_formula_val": _steps_from_size(
                int(self.info["gems_formula_val_size"]),
                self.batch_size,
                False,
            ),
            "gems_formula_test": _steps_from_size(
                int(self.info["gems_formula_test_size"]),
                self.batch_size,
                False,
            ),
        }
        self.train_steps = self.steps["gems_formula_train"]
        self.val_steps = self.steps["gems_formula_val"]
        self.test_steps = self.steps["gems_formula_test"]

        default_pin = torch.cuda.is_available()
        self.pin_memory = bool(config.get("dataloader_pin_memory", default_pin))
        self.dataloader_num_workers = int(config.get("dataloader_num_workers", 1))
        self.dataloader_prefetch_factor = int(
            config.get("dataloader_prefetch_factor", 2)
        )
        self.dataloader_persistent_workers = bool(
            config.get(
                "dataloader_persistent_workers",
                self.dataloader_num_workers > 0,
            )
        )

    def state_dict(self) -> dict[str, Any]:
        return {"seed": self.seed}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.seed = int(state_dict["seed"])

    def _build_formula_split_dataset(
        self,
        files: list[str],
        *,
        seed: int,
        shuffle: bool,
        drop_remainder: bool,
    ) -> tf.data.Dataset:
        shuffle_buffer = self.shuffle_buffer if shuffle else 0
        return _build_formula_dataset(
            files,
            self.batch_size,
            shuffle_buffer,
            seed,
            drop_remainder,
            tfrecord_buffer_size=self.tfrecord_buffer_size,
        )

    def _build_formula_train_dataset(self, seed: int) -> tf.data.Dataset:
        return self._build_formula_split_dataset(
            self.formula_train_files,
            seed=seed,
            shuffle=True,
            drop_remainder=self.drop_remainder,
        )

    def _build_formula_val_dataset(self, seed: int) -> tf.data.Dataset:
        return self._build_formula_split_dataset(
            self.formula_val_files,
            seed=seed,
            shuffle=False,
            drop_remainder=False,
        )

    def _build_formula_test_dataset(self, seed: int) -> tf.data.Dataset:
        return self._build_formula_split_dataset(
            self.formula_test_files,
            seed=seed,
            shuffle=False,
            drop_remainder=False,
        )

    def _make_loader(
        self,
        *,
        dataset: tf.data.Dataset,
        steps: int,
    ) -> DataLoader:
        dataset = _TfIterableDataset(
            dataset=dataset,
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

    def train_dataloader(self) -> DataLoader:
        dataset = self._build_formula_train_dataset(self.seed)
        return self._make_loader(
            dataset=dataset,
            steps=self.steps["gems_formula_train"],
        )

    def val_dataloader(self) -> DataLoader:
        seed = self.seed + 1_000_000
        dataset = self._build_formula_val_dataset(seed)
        return self._make_loader(
            dataset=dataset,
            steps=self.steps["gems_formula_val"],
        )

    def test_dataloader(self) -> DataLoader:
        seed = self.seed + 2_000_000
        dataset = self._build_formula_test_dataset(seed)
        return self._make_loader(
            dataset=dataset,
            steps=self.steps["gems_formula_test"],
        )


def create_lightning_dataloaders(
    config: config_dict.ConfigDict, seed: int
) -> tuple[DataLoader, Any, dict[str, Any]]:
    module = TfLightningDataModule(config, seed)
    return module.train_dataloader(), module.val_dataloader(), module.info


def create_gems_formula_lightning_dataloaders(
    config: config_dict.ConfigDict,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, Any]]:
    module = GeMSFormulaLightningDataModule(config, seed)
    return (
        module.train_dataloader(),
        module.val_dataloader(),
        module.test_dataloader(),
        module.info,
    )


# -----------------------------------------------------------------------------
# Legacy-style helpers
# -----------------------------------------------------------------------------


def create_gems_set_datasets(
    config: config_dict.ConfigDict,
    seed: Optional[int] = None,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    seed_value = int(config.seed if seed is None else seed)
    datamodule = TfLightningDataModule(config, seed=seed_value)

    train_ds = datamodule._build_gems_train_dataset(seed_value)

    val_iters: dict[str, Any] = {
        "gems_val": datamodule._build_gems_val_dataset(seed_value).as_numpy_iterator(),
        "massspec_val": datamodule._build_massspec_val_dataset(seed_value).as_numpy_iterator(),
        "massspec_test": datamodule._build_massspec_test_dataset(seed_value).as_numpy_iterator(),
    }

    return train_ds.as_numpy_iterator(), val_iters, datamodule.info


def create_gems_formula_set_datasets(
    config: config_dict.ConfigDict,
    seed: Optional[int] = None,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    seed_value = int(config.seed if seed is None else seed)
    datamodule = GeMSFormulaLightningDataModule(config, seed=seed_value)

    train_ds = datamodule._build_formula_train_dataset(seed_value)
    val_iters: dict[str, Any] = {
        "gems_formula_val": datamodule._build_formula_val_dataset(seed_value).as_numpy_iterator(),
        "gems_formula_test": datamodule._build_formula_test_dataset(seed_value).as_numpy_iterator(),
    }
    return train_ds.as_numpy_iterator(), val_iters, datamodule.info


def create_datasets(
    config: config_dict.ConfigDict, seed: int
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    info: dict[str, Any] = {}

    if config.dataset == "gems_a":
        train_dataset, eval_dataset, gems_info = create_gems_set_datasets(config, seed)
        info.update(gems_info)
        config.dataset_info = dict(info)
        config.vocab_size = info["vocab_size"]
        config.max_length = info["pair_sequence_length"]
        config.pad_token_id = info["special_tokens"]["[PAD]"]
        config.cls_token_id = info["special_tokens"]["[CLS]"]
        config.sep_token_id = info["special_tokens"]["[SEP]"]
        config.mask_token_id = info["special_tokens"]["[MASK]"]
        config.precursor_bins = info["precursor_bins"]
        config.precursor_offset = info["precursor_offset"]
        return train_dataset, eval_dataset, info
    if config.dataset == "gems_formula":
        train_dataset, eval_dataset, formula_info = create_gems_formula_set_datasets(
            config, seed
        )
        info.update(formula_info)
        config.dataset_info = dict(info)
        return train_dataset, eval_dataset, info

    raise NotImplementedError("Only gems_a and gems_formula datasets are supported.")


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
    if cfg.dataset == "gems_formula":
        train_iter, val_iters, info = create_gems_formula_set_datasets(
            cfg, seed=int(cfg.seed)
        )
    else:
        train_iter, val_iters, info = create_gems_set_datasets(cfg, seed=int(cfg.seed))

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

    if "token_ids" in batch:
        print("\nFirst sample token_ids:", batch["token_ids"][0][:20])
        print("First sample segment_ids:", batch["segment_ids"][0][:20])
    if "formula" in batch:
        print("\nFirst sample formula:", batch["formula"][0])
