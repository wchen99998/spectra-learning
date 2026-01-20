"""Input pipeline for GeMS_A mass spectrometry dataset using peak list representation.

This pipeline stores spectra as fixed-length 128-peak lists with (m/z, intensity) pairs,
along with retention time and precursor m/z. Unlike the splatted channel representation,
this preserves the discrete peak structure.

Configuration keys:
- ``tfrecord_dir``: Directory for TFRecord files (default ``data/gems_peaklist_tfrecord``).
- ``validation_fraction``: Fraction reserved for validation (default ``0.05``).
- ``batch_size``: Batch size (default ``512``).
- ``shuffle_buffer``: Shuffle buffer size for training (default ``10_000``).
- ``split_seed``: Seed for train/validation split (default ``42``).
- ``apply_scaling``: Add ``rt_scaled`` and ``precursor_scaled`` fields (default ``False``).
- ``apply_log_intensity``: Apply ``log1p`` to intensity values (default ``False``).
- ``apply_sqrt_tic_intensity``: Apply per-spectrum ``sqrt`` then TIC normalization to intensities
  (default ``False``). This matches the intensity handling used in the splatted GeMS pipeline.
- ``pre_keep_top_k``: Keep only the top-K peaks by intensity before any scaling or filtering,
  truncating the output to length K with peaks sorted by m/z and padding at the end
  (default ``0`` which keeps all 128 peaks).
- ``noise_quantile``: Per-spectrum noise estimate from the given quantile of non-zero intensities
  (default ``0.1``). Used with ``min_snr``.
- ``min_snr``: Zero-out peaks with intensity below ``min_snr * noise`` within each spectrum
  (default ``0.0``; no filtering). Applied before any intensity transform.
- ``keep_top_k``: Keep only the top-K peaks by intensity (others are zeroed; default ``0`` which keeps all).
  Applied after ``min_snr``.
- ``massspec_mix_ratio``: Fraction of MassSpecGym batches mixed into the training iterator.
  Defaults to the MassSpecGym train proportion in the metadata.

MassSpecGym preprocessing also computes top-16 Morgan fingerprint bits (1024-bit, radius 2)
from distinct SMILES and stores them in TFRecords under ``massspec_morgan_top16``. The bit
indices are saved in metadata as ``massspec_morgan_top_bits`` for downstream probing.
The stored target vector is binary (0/1) in ``massspec_morgan_top16``. MassSpecGym entries
are split into train/test using the HDF5 ``fold`` field.
Mixed training can add a per-example label mask in ``massspec_label_mask`` to indicate
which batches carry Morgan labels.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from huggingface_hub import hf_hub_download
from ml_collections import config_dict
from tqdm import tqdm

logger = logging.getLogger(__name__)

_DEFAULT_BATCH_SIZE = 512
_DEFAULT_SHUFFLE_BUFFER = 10_000
_DEFAULT_VALIDATION_FRACTION = 0.05
_DEFAULT_TFRECORD_DIR = Path("data/gems_peaklist_tfrecord")
_DEFAULT_SPLIT_SEED = 42
_DEFAULT_NUM_SHARDS = 4
_NUM_PEAKS = 128

_MASS_SPEC_MORGAN_BITS = 1024
_MASS_SPEC_MORGAN_RADIUS = 2
_MASS_SPEC_MORGAN_TOP_K = 16
_MASS_SPEC_MORGAN_FEATURE = "massspec_morgan_top16"
_MASS_SPEC_LABEL_MASK_FEATURE = "massspec_label_mask"
_MASS_SPEC_MORGAN_DEFAULT = [0] * _MASS_SPEC_MORGAN_TOP_K

_PRECURSOR_MIN = 50.0
_PRECURSOR_MAX = 1500.0
_METADATA_FILENAME = "metadata.json"

_GEMS_HF_REPO = "roman-bushuiev/GeMS"
_GEMS_HDF5_PATH = "data/GeMS_A/GeMS_A.hdf5"
_MASSSPEC_HDF5_PATH = "data/auxiliary/MassSpecGym_MurckoHist_split.hdf5"


def _download_hdf5(repo_id: str, filename: str, local_dir: Path) -> Path:
    """Download HDF5 file from HuggingFace."""
    local_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=str(local_dir),
    )
    return Path(path)


def _load_gems_arrays(hdf5_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load GeMS HDF5 data."""
    import h5py

    with h5py.File(hdf5_path, "r") as f:
        spectra = f["spectrum"][:]
        retention = np.asarray(f["RT"], dtype=np.float32)
        precursor = np.asarray(f["precursor_mz"], dtype=np.float32)
    return spectra, retention, precursor


def _load_massspec_arrays(
    hdf5_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MassSpecGym HDF5 data."""
    import h5py

    with h5py.File(hdf5_path, "r") as f:
        spectra = f["spectrum"][:]
        precursor = np.asarray(f["precursor_mz"], dtype=np.float32)
        smiles = f["smiles"].astype("T")[:]
        fold = f["fold"].astype("T")[:]

    # MassSpecGym has no RT - use placeholder
    retention = np.full(len(spectra), 392.3146, dtype=np.float32)
    return spectra, retention, precursor, smiles, fold


def _compute_massspec_top_bits(smiles: np.ndarray) -> np.ndarray:
    """Compute top-K Morgan fingerprint bit positions over distinct SMILES."""
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    unique_smiles = np.unique(smiles)
    counts = np.zeros(_MASS_SPEC_MORGAN_BITS, dtype=np.int32)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=_MASS_SPEC_MORGAN_RADIUS,
        fpSize=_MASS_SPEC_MORGAN_BITS,
    )

    for smi in unique_smiles:
        mol = Chem.MolFromSmiles(smi)
        fp = generator.GetFingerprint(mol)
        for bit in fp.GetOnBits():
            counts[bit] += 1

    top_bits = np.argsort(-counts)[:_MASS_SPEC_MORGAN_TOP_K].astype(np.int32)
    return top_bits


def _compute_massspec_top_bit_vectors(smiles: np.ndarray, top_bits: np.ndarray) -> np.ndarray:
    """Build per-spectrum top-bit vectors from SMILES."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    unique_smiles, inverse = np.unique(smiles, return_inverse=True)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=_MASS_SPEC_MORGAN_RADIUS,
        fpSize=_MASS_SPEC_MORGAN_BITS,
    )

    unique_vectors = np.zeros((len(unique_smiles), len(top_bits)), dtype=np.int8)
    for idx, smi in enumerate(unique_smiles):
        mol = Chem.MolFromSmiles(smi)
        fp = generator.GetFingerprint(mol)
        arr = np.zeros(_MASS_SPEC_MORGAN_BITS, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        unique_vectors[idx] = arr[top_bits]

    return unique_vectors[inverse]


def _write_tfrecords(
    spectra: np.ndarray,
    retention: np.ndarray,
    precursor: np.ndarray,
    output_path: Path,
    num_shards: int,
    desc: str = "Writing TFRecords",
    extra_float_features: Optional[dict[str, np.ndarray]] = None,
    extra_int_features: Optional[dict[str, np.ndarray]] = None,
) -> tuple[list[str], list[int]]:
    """Write peak list data to TFRecords."""
    n = len(spectra)
    num_shards = max(1, min(num_shards, n))
    shard_size = math.ceil(n / num_shards)

    output_path.mkdir(parents=True, exist_ok=True)
    files, lengths = [], []

    for shard_id in range(num_shards):
        start = shard_id * shard_size
        end = min(start + shard_size, n)
        if start >= end:
            break

        shard_file = output_path / f"shard-{shard_id:05d}-of-{num_shards:05d}.tfrecord"
        options = tf.io.TFRecordOptions(compression_type="GZIP")

        with tf.io.TFRecordWriter(str(shard_file), options=options) as writer:
            for i in tqdm(range(start, end), desc=f"{desc} [{shard_id+1}/{num_shards}]"):
                mz = spectra[i, 0].astype(np.float32)
                intensity = spectra[i, 1].astype(np.float32)

                features = {
                    "mz": tf.train.Feature(float_list=tf.train.FloatList(value=mz)),
                    "intensity": tf.train.Feature(float_list=tf.train.FloatList(value=intensity)),
                    "rt": tf.train.Feature(float_list=tf.train.FloatList(value=[retention[i]])),
                    "precursor_mz": tf.train.Feature(float_list=tf.train.FloatList(value=[precursor[i]])),
                }
                if extra_float_features is not None:
                    for name, values in extra_float_features.items():
                        features[name] = tf.train.Feature(
                            float_list=tf.train.FloatList(value=values[i].astype(np.float32))
                        )
                if extra_int_features is not None:
                    for name, values in extra_int_features.items():
                        features[name] = tf.train.Feature(
                            int64_list=tf.train.Int64List(value=values[i].astype(np.int64))
                        )

                example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(example.SerializeToString())

        files.append(shard_file.name)
        lengths.append(end - start)

    return files, lengths


def _process_gems(
    output_dir: Path,
    validation_fraction: float,
    split_seed: int,
    num_shards: int,
) -> dict:
    """Download and process GeMS dataset."""
    logger.info("Downloading GeMS HDF5...")
    hdf5_path = _download_hdf5(_GEMS_HF_REPO, _GEMS_HDF5_PATH, output_dir.parent)

    logger.info("Loading GeMS data...")
    spectra, retention, precursor = _load_gems_arrays(hdf5_path)

    # Filter valid RT
    mask = np.isfinite(retention) & (retention > 0.0)
    spectra = spectra[mask]
    retention = retention[mask]
    precursor = precursor[mask]

    n = len(spectra)
    logger.info("Valid GeMS spectra: %d", n)

    # Train/val split
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n)
    train_size = int(n * (1.0 - validation_fraction))

    train_idx = perm[:train_size]
    val_idx = perm[train_size:]

    # Compute RT scaling from train set
    rt_p1, rt_p95 = np.percentile(retention[train_idx], [1, 95])

    # Write train
    train_files, train_lengths = _write_tfrecords(
        spectra[train_idx], retention[train_idx], precursor[train_idx],
        output_dir / "train", num_shards, desc="Train"
    )

    # Write validation
    val_files, val_lengths = _write_tfrecords(
        spectra[val_idx], retention[val_idx], precursor[val_idx],
        output_dir / "validation", max(1, num_shards // 4), desc="Validation"
    )

    return {
        "train_files": train_files,
        "train_lengths": train_lengths,
        "validation_files": val_files,
        "validation_lengths": val_lengths,
        "train_size": int(train_size),
        "validation_size": int(n - train_size),
        "rt_percentile_1": float(rt_p1),
        "rt_percentile_95": float(rt_p95),
    }


def _process_massspec(output_dir: Path, num_shards: int) -> dict:
    """Download and process MassSpecGym dataset."""
    logger.info("Downloading MassSpecGym HDF5...")
    hdf5_path = _download_hdf5(_GEMS_HF_REPO, _MASSSPEC_HDF5_PATH, output_dir.parent)

    logger.info("Loading MassSpecGym data...")
    spectra, retention, precursor, smiles, fold = _load_massspec_arrays(hdf5_path)

    train_mask = fold == "train"
    test_mask = fold != "train"

    train_size = int(np.count_nonzero(train_mask))
    test_size = int(np.count_nonzero(test_mask))
    logger.info("MassSpecGym spectra: %d (train=%d, test=%d)", len(spectra), train_size, test_size)

    logger.info("Computing MassSpecGym Morgan fingerprint targets...")
    top_bits = _compute_massspec_top_bits(smiles)
    top_vectors = _compute_massspec_top_bit_vectors(smiles, top_bits)

    # Write train split
    train_files, train_lengths = _write_tfrecords(
        spectra[train_mask],
        retention[train_mask],
        precursor[train_mask],
        output_dir / "massspec_train",
        num_shards,
        desc="MassSpec Train",
        extra_int_features={_MASS_SPEC_MORGAN_FEATURE: top_vectors[train_mask]},
    )

    # Write test split
    test_files, test_lengths = _write_tfrecords(
        spectra[test_mask],
        retention[test_mask],
        precursor[test_mask],
        output_dir / "massspec_test",
        max(1, num_shards // 4),
        desc="MassSpec Test",
        extra_int_features={_MASS_SPEC_MORGAN_FEATURE: top_vectors[test_mask]},
    )

    return {
        "massspec_train_files": train_files,
        "massspec_train_lengths": train_lengths,
        "massspec_train_size": train_size,
        "massspec_test_files": test_files,
        "massspec_test_lengths": test_lengths,
        "massspec_test_size": test_size,
        "massspec_morgan_top_bits": top_bits.astype(int).tolist(),
        "massspec_morgan_radius": _MASS_SPEC_MORGAN_RADIUS,
        "massspec_morgan_bits": _MASS_SPEC_MORGAN_BITS,
        "massspec_morgan_top_k": _MASS_SPEC_MORGAN_TOP_K,
        "massspec_morgan_feature_dtype": "int64",
    }


def _ensure_processed(
    output_dir: Path,
    validation_fraction: float,
    split_seed: int,
    num_shards: int,
) -> dict:
    """Ensure TFRecords exist, creating them if needed."""
    metadata_path = output_dir / _METADATA_FILENAME

    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Verify files exist
        train_ok = all((output_dir / "train" / fn).exists() for fn in metadata.get("train_files", []))
        val_ok = all((output_dir / "validation" / fn).exists() for fn in metadata.get("validation_files", []))
        massspec_train_files = metadata.get("massspec_train_files", [])
        massspec_test_files = metadata.get("massspec_test_files", [])
        massspec_train_ok = all(
            (output_dir / "massspec_train" / fn).exists() for fn in massspec_train_files
        )
        massspec_test_ok = all(
            (output_dir / "massspec_test" / fn).exists() for fn in massspec_test_files
        )
        has_morgan = metadata.get("massspec_morgan_feature_dtype") == "int64"

        if (
            train_ok
            and val_ok
            and massspec_train_files
            and massspec_test_files
            and massspec_train_ok
            and massspec_test_ok
            and has_morgan
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

    # Process GeMS
    metadata = _process_gems(output_dir, validation_fraction, split_seed, num_shards)

    # Process MassSpecGym
    massspec_info = _process_massspec(output_dir, num_shards)
    metadata.update(massspec_info)

    # Save metadata
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved metadata to %s", metadata_path)
    return metadata


def _parse_example(serialized: tf.Tensor) -> dict[str, tf.Tensor]:
    """Parse a TFRecord example."""
    features = {
        "mz": tf.io.FixedLenFeature([_NUM_PEAKS], tf.float32),
        "intensity": tf.io.FixedLenFeature([_NUM_PEAKS], tf.float32),
        "rt": tf.io.FixedLenFeature([1], tf.float32),
        "precursor_mz": tf.io.FixedLenFeature([1], tf.float32),
        _MASS_SPEC_MORGAN_FEATURE: tf.io.FixedLenFeature(
            [_MASS_SPEC_MORGAN_TOP_K],
            tf.int64,
            default_value=_MASS_SPEC_MORGAN_DEFAULT,
        ),
    }
    parsed = tf.io.parse_single_example(serialized, features)
    return {
        "mz": parsed["mz"],
        "intensity": parsed["intensity"],
        "rt_raw": parsed["rt"][0],
        "precursor_mz": parsed["precursor_mz"],
        _MASS_SPEC_MORGAN_FEATURE: parsed[_MASS_SPEC_MORGAN_FEATURE],
    }


def _add_massspec_label_mask(value: float):
    mask = tf.constant([value], dtype=tf.float32)

    def add(example: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        example[_MASS_SPEC_LABEL_MASK_FEATURE] = mask
        return example

    return add


def _apply_scaling(rt_p1: float, rt_p95: float):
    """Create scaling function."""
    rt_lower = tf.constant(rt_p1, tf.float32)
    rt_upper = tf.constant(rt_p95, tf.float32)
    rt_denom = tf.maximum(rt_upper - rt_lower, 1e-6)

    prec_min = tf.constant(_PRECURSOR_MIN, tf.float32)
    prec_max = tf.constant(_PRECURSOR_MAX, tf.float32)
    prec_denom = prec_max - prec_min

    def scale(example: dict) -> dict:
        rt_scaled = tf.clip_by_value((example["rt_raw"] - rt_lower) / rt_denom, 0.0, 1.0)
        example["rt_scaled"] = rt_scaled
        prec_scaled = tf.clip_by_value((example["precursor_mz"] - prec_min) / prec_denom, 0.0, 1.0)
        example["precursor_scaled"] = prec_scaled
        return example

    return scale


def _apply_log_intensity(example: dict) -> dict:
    """Apply log1p transformation to intensity values."""
    example["intensity"] = tf.math.log1p(example["intensity"])
    return example


def _apply_sqrt_tic_intensity(example: dict) -> dict:
    """Apply sqrt(intensity) then normalize by per-spectrum TIC (sum of sqrt intensities)."""
    weights = tf.sqrt(example["intensity"])
    tic = tf.reduce_sum(weights)
    weights = weights / tf.maximum(tic, 1e-6)
    example["intensity"] = weights
    return example


def _apply_min_snr(noise_quantile: float, min_snr: float):
    q = tf.constant(noise_quantile, tf.float32)
    snr = tf.constant(min_snr, tf.float32)

    def apply(example: dict) -> dict:
        intensity = example["intensity"]
        non_zero = tf.boolean_mask(intensity, intensity > 0)
        sorted_vals = tf.sort(non_zero)
        n = tf.shape(sorted_vals)[0]
        idx = tf.cast(tf.floor(q * tf.cast(n - 1, tf.float32)), tf.int32)
        noise = sorted_vals[idx]
        keep = intensity >= (noise * snr)
        example["intensity"] = tf.where(keep, intensity, 0.0)
        example["mz"] = tf.where(keep, example["mz"], 0.0)
        return example

    return apply


def _apply_keep_top_k(keep_top_k: int):
    k = int(keep_top_k)

    def apply(example: dict) -> dict:
        intensity = example["intensity"]
        _, idx = tf.math.top_k(intensity, k=k, sorted=False)
        keep = tf.scatter_nd(idx[:, None], tf.ones((k,), tf.bool), (tf.shape(intensity)[0],))
        example["intensity"] = tf.where(keep, intensity, 0.0)
        example["mz"] = tf.where(keep, example["mz"], 0.0)
        return example

    return apply


def _apply_pre_keep_top_k(pre_keep_top_k: int):
    """Keep top-k peaks by intensity, truncate to length k, sort by m/z with padding at end."""
    k = int(pre_keep_top_k)

    def apply(example: dict) -> dict:
        mz = example["mz"]
        intensity = example["intensity"]

        # Get indices of top-k peaks by intensity
        _, top_k_idx = tf.math.top_k(intensity, k=k, sorted=False)

        # Gather the top-k peaks
        top_k_mz = tf.gather(mz, top_k_idx)
        top_k_intensity = tf.gather(intensity, top_k_idx)

        # Sort by m/z, pushing zero m/z values to the end
        sort_key = tf.where(top_k_mz > 0, top_k_mz, tf.constant(float("inf"), dtype=tf.float32))
        sorted_idx = tf.argsort(sort_key)

        example["mz"] = tf.gather(top_k_mz, sorted_idx)
        example["intensity"] = tf.gather(top_k_intensity, sorted_idx)

        return example

    return apply


def _build_dataset(
    filenames: list[str],
    rt_p1: float,
    rt_p95: float,
    batch_size: int,
    shuffle_buffer: int,
    seed: Optional[int],
    repeat: bool,
    drop_remainder: bool,
    label_mask_value: float,
    apply_scaling: bool = False,
    apply_log_intensity: bool = False,
    apply_sqrt_tic_intensity: bool = False,
    pre_keep_top_k: int = 0,
    noise_quantile: float = 0.1,
    min_snr: float = 0.0,
    keep_top_k: int = 0,
) -> tf.data.Dataset:
    """Build tf.data pipeline."""
    ds = tf.data.TFRecordDataset(filenames, compression_type="GZIP", num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(_add_massspec_label_mask(label_mask_value), num_parallel_calls=tf.data.AUTOTUNE)
    if min_snr > 0.0:
        ds = ds.map(_apply_min_snr(noise_quantile, min_snr), num_parallel_calls=tf.data.AUTOTUNE)
    if pre_keep_top_k > 0:
        ds = ds.map(_apply_pre_keep_top_k(pre_keep_top_k), num_parallel_calls=tf.data.AUTOTUNE)
    if apply_scaling:
        ds = ds.map(_apply_scaling(rt_p1, rt_p95), num_parallel_calls=tf.data.AUTOTUNE)
    if apply_sqrt_tic_intensity:
        ds = ds.map(_apply_sqrt_tic_intensity, num_parallel_calls=tf.data.AUTOTUNE)
    elif apply_log_intensity:
        ds = ds.map(_apply_log_intensity, num_parallel_calls=tf.data.AUTOTUNE)
    if keep_top_k > 0:
        ds = ds.map(_apply_keep_top_k(keep_top_k), num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle_buffer > 0:
        ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)

    if repeat:
        ds = ds.repeat()

    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def create_gems_set_datasets(
    config: config_dict.ConfigDict,
    seed: Optional[int] = None,
) -> tuple[tf.data.Iterator, dict[str, tf.data.Iterator], dict]:
    """Create GeMS peak list datasets.

    Returns:
        (train_iterator, {"validation": val_iter, "massspec_test": test_iter}, info_dict)
    """
    output_dir = Path(config.get("tfrecord_dir", str(_DEFAULT_TFRECORD_DIR))).expanduser().resolve()
    validation_fraction = float(config.get("validation_fraction", _DEFAULT_VALIDATION_FRACTION))
    batch_size = int(config.get("batch_size", _DEFAULT_BATCH_SIZE))
    shuffle_buffer = int(config.get("shuffle_buffer", _DEFAULT_SHUFFLE_BUFFER))
    split_seed = int(config.get("split_seed", _DEFAULT_SPLIT_SEED))
    num_shards = int(config.get("num_shards", _DEFAULT_NUM_SHARDS))
    drop_remainder = bool(config.get("drop_remainder", False))
    apply_scaling = bool(config.get("apply_scaling", False))
    apply_log_intensity = bool(config.get("apply_log_intensity", False))
    apply_sqrt_tic_intensity = bool(config.get("apply_sqrt_tic_intensity", False))
    pre_keep_top_k = int(config.get("pre_keep_top_k", 0))
    noise_quantile = float(config.get("noise_quantile", 0.1))
    min_snr = float(config.get("min_snr", 0.0))
    keep_top_k = int(config.get("keep_top_k", 0))
    massspec_mix_ratio = config.get("massspec_mix_ratio", None)

    metadata = _ensure_processed(output_dir, validation_fraction, split_seed, num_shards)

    rt_p1 = metadata["rt_percentile_1"]
    rt_p95 = metadata["rt_percentile_95"]

    train_files = [str(output_dir / "train" / fn) for fn in metadata["train_files"]]
    val_files = [str(output_dir / "validation" / fn) for fn in metadata["validation_files"]]
    massspec_train_files = [
        str(output_dir / "massspec_train" / fn) for fn in metadata.get("massspec_train_files", [])
    ]
    massspec_test_files = [
        str(output_dir / "massspec_test" / fn) for fn in metadata.get("massspec_test_files", [])
    ]

    train_ds = _build_dataset(
        train_files,
        rt_p1,
        rt_p95,
        batch_size,
        shuffle_buffer,
        seed,
        repeat=True,
        drop_remainder=drop_remainder,
        label_mask_value=0.0,
        apply_scaling=apply_scaling,
        apply_log_intensity=apply_log_intensity,
        apply_sqrt_tic_intensity=apply_sqrt_tic_intensity,
        pre_keep_top_k=pre_keep_top_k,
        noise_quantile=noise_quantile,
        min_snr=min_snr,
        keep_top_k=keep_top_k,
    )
    val_ds = _build_dataset(
        val_files,
        rt_p1,
        rt_p95,
        batch_size,
        0,
        seed,
        repeat=True,
        drop_remainder=False,
        label_mask_value=0.0,
        apply_scaling=apply_scaling,
        apply_log_intensity=apply_log_intensity,
        apply_sqrt_tic_intensity=apply_sqrt_tic_intensity,
        pre_keep_top_k=pre_keep_top_k,
        noise_quantile=noise_quantile,
        min_snr=min_snr,
        keep_top_k=keep_top_k,
    )

    val_iters = {"validation": val_ds.as_numpy_iterator()}

    massspec_train_ds = None
    if massspec_train_files:
        massspec_train_ds = _build_dataset(
            massspec_train_files,
            rt_p1,
            rt_p95,
            batch_size,
            shuffle_buffer,
            seed,
            repeat=True,
            drop_remainder=drop_remainder,
            label_mask_value=1.0,
            apply_scaling=apply_scaling,
            apply_log_intensity=apply_log_intensity,
            apply_sqrt_tic_intensity=apply_sqrt_tic_intensity,
            pre_keep_top_k=pre_keep_top_k,
            noise_quantile=noise_quantile,
            min_snr=min_snr,
            keep_top_k=keep_top_k,
        )

    if massspec_train_ds is not None:
        if massspec_mix_ratio is None:
            massspec_train_size = int(metadata.get("massspec_train_size", 0))
            total_size = int(metadata["train_size"]) + massspec_train_size
            massspec_mix_ratio = massspec_train_size / total_size
        massspec_mix_ratio = float(massspec_mix_ratio)
        if massspec_mix_ratio > 0.0:
            train_ds = tf.data.Dataset.sample_from_datasets(
                [train_ds, massspec_train_ds],
                weights=[1.0 - massspec_mix_ratio, massspec_mix_ratio],
                seed=seed,
            )

    if massspec_test_files:
        test_ds = _build_dataset(
            massspec_test_files,
            rt_p1,
            rt_p95,
            batch_size,
            0,
            seed,
            repeat=True,
            drop_remainder=True,
            label_mask_value=1.0,
            apply_scaling=apply_scaling,
            apply_log_intensity=apply_log_intensity,
            apply_sqrt_tic_intensity=apply_sqrt_tic_intensity,
            pre_keep_top_k=pre_keep_top_k,
            noise_quantile=noise_quantile,
            min_snr=min_snr,
            keep_top_k=keep_top_k,
        )
        val_iters["massspec_test"] = test_ds.as_numpy_iterator()

    info = {
        "tfrecord_dir": str(output_dir),
        "train_size": metadata["train_size"],
        "validation_size": metadata["validation_size"],
        "massspec_train_size": metadata.get("massspec_train_size", 0),
        "massspec_test_size": metadata.get("massspec_test_size", 0),
        "num_peaks": _NUM_PEAKS if pre_keep_top_k == 0 else pre_keep_top_k,
        "rt_percentile_1": rt_p1,
        "rt_percentile_95": rt_p95,
    }
    if massspec_mix_ratio is not None:
        info["massspec_mix_ratio"] = float(massspec_mix_ratio)
    if "massspec_morgan_top_bits" in metadata:
        info["massspec_morgan_top_bits"] = metadata["massspec_morgan_top_bits"]
        info["massspec_morgan_radius"] = metadata.get("massspec_morgan_radius")
        info["massspec_morgan_bits"] = metadata.get("massspec_morgan_bits")
        info["massspec_morgan_top_k"] = metadata.get("massspec_morgan_top_k")

    return train_ds.as_numpy_iterator(), val_iters, info


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    cfg = config_dict.ConfigDict()
    cfg.tfrecord_dir = "data/gems_peaklist_tfrecord"
    cfg.batch_size = 32
    cfg.validation_fraction = 0.05

    train_iter, val_iters, info = create_gems_set_datasets(cfg, seed=42)

    print("\nDataset info:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    print("\nSample batch:")
    batch = next(train_iter)
    for k, v in batch.items():
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

    print("\nFirst sample m/z (non-zero):", batch["mz"][0][batch["mz"][0] > 0][:10])
    print("First sample intensity (non-zero):", batch["intensity"][0][batch["intensity"][0] > 0][:10])
