"""Unified TF input pipeline and Lightning DataModule for GeMS_A peak lists."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
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
_DEFAULT_SPLIT_SEED = 42
_DEFAULT_NUM_SHARDS = 4
_NUM_PEAKS_INPUT = 128
_NUM_PEAKS_OUTPUT = 64
_DEFAULT_MAX_PRECURSOR_MZ = 1000.0
_PEAK_MZ_MIN = 20.0
_PEAK_MZ_MAX = 1000.0
_PRECURSOR_MZ_WINDOW = 2.5
_INTENSITY_BINS = 128
_INTENSITY_EPS = 1e-4
_SPECIAL_TOKENS = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3}
_NUM_SPECIAL_TOKENS = len(_SPECIAL_TOKENS)
_DEFAULT_PAIR_SEQUENCE_LENGTH = 128

_METADATA_FILENAME = "metadata.json"

_GEMS_HF_REPO = "roman-bushuiev/GeMS"
_GEMS_HDF5_PATH = "data/GeMS_A/GeMS_A.hdf5"
_MASSSPEC_HDF5_PATH = "data/auxiliary/MassSpecGym_MurckoHist_split.hdf5"


# -----------------------------------------------------------------------------
# TFRecord creation and dataset preparation
# -----------------------------------------------------------------------------


def _download_hdf5(repo_id: str, filename: str, local_dir: Path) -> Path:
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


def _load_massspec_arrays(
    hdf5_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import h5py

    with h5py.File(hdf5_path, "r") as f:
        spectra = f["spectrum"][:]
        precursor = np.asarray(f["precursor_mz"], dtype=np.float32)
        fold = f["fold"].astype("T")[:]

    retention = np.full(len(spectra), 392.3146, dtype=np.float32)
    return spectra, retention, precursor, fold


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


def _process_gems(
    output_dir: Path,
    validation_fraction: float,
    split_seed: int,
    num_shards: int,
) -> dict[str, Any]:
    logger.info("Downloading GeMS HDF5...")
    hdf5_path = _download_hdf5(_GEMS_HF_REPO, _GEMS_HDF5_PATH, output_dir.parent)

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
    logger.info("Downloading MassSpecGym HDF5...")
    hdf5_path = _download_hdf5(_GEMS_HF_REPO, _MASSSPEC_HDF5_PATH, output_dir.parent)

    logger.info("Loading MassSpecGym data...")
    spectra, retention, precursor, fold = _load_massspec_arrays(hdf5_path)

    test_mask = fold != "train"
    test_size = int(np.count_nonzero(test_mask))
    logger.info("MassSpecGym spectra: %d (test=%d)", len(spectra), test_size)

    test_files, test_lengths = _write_tfrecords(
        spectra[test_mask],
        retention[test_mask],
        precursor[test_mask],
        output_dir / "massspec_test",
        max(1, num_shards // 4),
        desc="MassSpec Test",
    )

    return {
        "massspec_test_files": test_files,
        "massspec_test_lengths": test_lengths,
        "massspec_test_size": test_size,
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
        massspec_test_files = metadata.get("massspec_test_files", [])
        massspec_test_ok = all(
            (output_dir / "massspec_test" / fn).exists()
            for fn in massspec_test_files
        )

        if train_ok and val_ok and massspec_test_files and massspec_test_ok:
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


def _strip_padding_and_tokenize(max_precursor_mz: float) -> Callable[[dict], dict]:
    eps = tf.constant(_INTENSITY_EPS, tf.float32)
    log_eps = tf.math.log(eps)
    denom = -log_eps
    bins = tf.constant(_INTENSITY_BINS - 1, tf.float32)
    mz_bins = int(_PEAK_MZ_MAX) + 1
    precursor_bins = int(max_precursor_mz) + 1
    mz_offset = tf.constant(_NUM_SPECIAL_TOKENS, tf.int32)
    precursor_offset = mz_offset
    intensity_offset = tf.constant(_NUM_SPECIAL_TOKENS + mz_bins, tf.int32)

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
        s = (tf.math.log(intensity) - log_eps) / denom
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
) -> dict[str, Any] | list[dict[str, Any]]:
    tokens = token_ids
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.detach().cpu().numpy()
    tokens = np.asarray(tokens, dtype=np.int32)

    if tokens.ndim == 2:
        return [
            detokenize_spectrum(row, max_precursor_mz=max_precursor_mz) for row in tokens
        ]

    pad_id = _SPECIAL_TOKENS["[PAD]"]
    pad_positions = np.where(tokens == pad_id)[0]
    end = int(pad_positions[0]) if pad_positions.size > 0 else int(tokens.shape[0])
    content = tokens[1:end]
    precursor_token = content[0]
    peaks = content[1:]
    mz_tokens = peaks[0::2]
    intensity_tokens = peaks[1::2]

    mz = (mz_tokens - _NUM_SPECIAL_TOKENS).astype(np.float32)
    precursor = float(
        np.clip(precursor_token - _NUM_SPECIAL_TOKENS, 0, max_precursor_mz)
    )

    bins = _INTENSITY_BINS - 1
    intensity_offset = _NUM_SPECIAL_TOKENS + int(_PEAK_MZ_MAX) + 1
    intensity_idx = intensity_tokens - intensity_offset
    s = intensity_idx.astype(np.float32) / float(bins)
    log_eps = math.log(_INTENSITY_EPS)
    denom = -log_eps
    intensity = np.exp(s * denom + log_eps).astype(np.float32)

    return {
        "precursor_mz": precursor,
        "mz": mz,
        "intensity": intensity,
    }


def _build_single_spectrum_input(max_len: int) -> Callable[[dict], dict]:
    cls_id = tf.constant(_SPECIAL_TOKENS["[CLS]"], tf.int32)
    pad_id = tf.constant(_SPECIAL_TOKENS["[PAD]"], tf.int32)
    max_peaks = tf.constant((max_len - 2) // 2, tf.int32)

    def interleave(mz: tf.Tensor, intensity: tf.Tensor) -> tf.Tensor:
        pair = tf.stack([mz, intensity], axis=1)
        return tf.reshape(pair, [-1])

    def build_sequence(mz: tf.Tensor, intensity: tf.Tensor, precursor: tf.Tensor) -> tf.Tensor:
        peaks = interleave(mz, intensity)
        return tf.concat([precursor, peaks], axis=0)

    def apply(example: dict) -> dict:
        mz = example["mz"][:max_peaks]
        intensity = example["intensity"][:max_peaks]
        precursor = tf.reshape(example["precursor_mz"], [1])
        seq = build_sequence(mz, intensity, precursor)

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
        seq_len = tf.shape(token_ids)[1]
        mask_count = tf.cast(mask_ratio_t * tf.cast(seq_len, tf.float32), tf.int32)
        scores = tf.random.uniform(tf.shape(token_ids), dtype=tf.float32)
        scores = tf.where(maskable, scores, tf.constant(-1.0, tf.float32))
        _, mask_idx = tf.math.top_k(scores, k=mask_count, sorted=False)
        mask = tf.reduce_any(
            tf.one_hot(mask_idx, seq_len, on_value=True, off_value=False, dtype=tf.bool),
            axis=1,
        )
        batch["masked_token_ids"] = tf.where(mask, mask_token, token_ids)
        batch["mlm_mask"] = mask
        return batch

    return apply


def _build_dataset(
    filenames: list[str],
    batch_size: int,
    shuffle_buffer: int,
    seed: Optional[int],
    repeat: bool,
    drop_remainder: bool,
    *,
    max_precursor_mz: float,
    pair_sequence_length: int,
    mask_ratio: float,
    mask_token_id: int,
) -> tf.data.Dataset:
    ds = tf.data.TFRecordDataset(
        filenames,
        compression_type="GZIP",
        num_parallel_reads=tf.data.AUTOTUNE,
    )
    ds = ds.map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.filter(_filter_max_precursor_mz(max_precursor_mz))
    ds = ds.map(
        _filter_peak_mz_range(_PEAK_MZ_MIN, _PEAK_MZ_MAX, _PRECURSOR_MZ_WINDOW),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.map(_topk_peaks(_NUM_PEAKS_OUTPUT), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(_compact_sort_peaks(), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(
        _strip_padding_and_tokenize(max_precursor_mz),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if repeat:
        ds = ds.repeat()
    if shuffle_buffer > 0:
        ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=False)

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


# -----------------------------------------------------------------------------
# Info and step resolution
# -----------------------------------------------------------------------------


def _compute_info(
    metadata: dict[str, Any],
    *,
    output_dir: Path,
    max_precursor_mz: float,
    pair_sequence_length: int,
) -> dict[str, Any]:
    mz_bins = int(_PEAK_MZ_MAX) + 1
    precursor_bins = int(max_precursor_mz) + 1
    mz_offset = _NUM_SPECIAL_TOKENS
    precursor_offset = mz_offset
    intensity_offset = mz_offset + mz_bins
    vocab_size = intensity_offset + _INTENSITY_BINS
    return {
        "tfrecord_dir": str(output_dir),
        "train_size": metadata["train_size"],
        "validation_size": metadata["validation_size"],
        "massspec_test_size": metadata.get("massspec_test_size", 0),
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
    }


def _steps_from_size(size: int, batch_size: int, drop_remainder: bool) -> int:
    if drop_remainder:
        return int(size // batch_size)
    return int(math.ceil(size / batch_size))


def _resolve_train_steps(config: config_dict.ConfigDict, info: dict[str, Any]) -> int:
    return int(config.num_train_steps)


def _resolve_eval_steps(
    config: config_dict.ConfigDict, info: dict[str, Any], split: str
) -> int:
    if config.get("num_eval_steps", 0) > 0:
        return int(config.num_eval_steps)
    size_key = "validation_size" if split == "validation" else "massspec_test_size"
    size = int(info[size_key])
    drop_remainder = split == "massspec_test"
    return _steps_from_size(size, int(config.batch_size), drop_remainder)


def get_num_train_steps(config: config_dict.ConfigDict) -> int:
    if config.num_train_steps > 0:
        return int(config.num_train_steps)
    raise NotImplementedError()


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
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return [[_to_torch(item) for item in row] for row in value.tolist()]
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


class _StatefulDataLoader(DataLoader):
    def state_dict(self) -> dict[str, Any]:
        return self.dataset.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
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
        self.split_seed = int(config.get("split_seed", _DEFAULT_SPLIT_SEED))
        self.num_shards = int(config.get("num_shards", _DEFAULT_NUM_SHARDS))
        self.drop_remainder = bool(config.get("drop_remainder", False))
        self.max_precursor_mz = float(
            config.get("max_precursor_mz", _DEFAULT_MAX_PRECURSOR_MZ)
        )
        self.pair_sequence_length = int(
            config.get("pair_sequence_length", _DEFAULT_PAIR_SEQUENCE_LENGTH)
        )
        self.mask_ratio = float(config.get("mask_ratio", 0.15))
        self.mask_token_id = int(config.get("mask_token_id", _SPECIAL_TOKENS["[MASK]"]))

        self.metadata = _ensure_processed(
            self.output_dir,
            self.validation_fraction,
            self.split_seed,
            self.num_shards,
        )

        self.train_files = [
            str(self.output_dir / "train" / fn) for fn in self.metadata["train_files"]
        ]
        self.val_files = [
            str(self.output_dir / "validation" / fn)
            for fn in self.metadata["validation_files"]
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
        )
        self.eval_splits = ["validation"]
        if self.info["massspec_test_size"] > 0:
            self.eval_splits.append("massspec_test")

        self.train_steps = _resolve_train_steps(config, self.info)
        self.eval_steps = {
            split: _resolve_eval_steps(config, self.info, split)
            for split in self.eval_splits
        }

        default_pin = torch.cuda.is_available()
        self.pin_memory = bool(config.get("dataloader_pin_memory", default_pin))

    def state_dict(self) -> dict[str, Any]:
        return {"seed": self.seed}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.seed = int(state_dict["seed"])

    def _build_train_dataset(self, seed: int) -> tf.data.Dataset:
        return _build_dataset(
            self.train_files,
            self.batch_size,
            self.shuffle_buffer,
            seed,
            repeat=True,
            drop_remainder=self.drop_remainder,
            max_precursor_mz=self.max_precursor_mz,
            pair_sequence_length=self.pair_sequence_length,
            mask_ratio=self.mask_ratio,
            mask_token_id=self.mask_token_id,
        )

    def _build_eval_dataset(self, split: str, seed: int) -> tf.data.Dataset:
        files = self.val_files if split == "validation" else self.massspec_test_files
        return _build_dataset(
            files,
            self.batch_size,
            shuffle_buffer=0,
            seed=seed,
            repeat=True,
            drop_remainder=(split == "massspec_test"),
            max_precursor_mz=self.max_precursor_mz,
            pair_sequence_length=self.pair_sequence_length,
            mask_ratio=self.mask_ratio,
            mask_token_id=self.mask_token_id,
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
            "pin_memory": self.pin_memory,
            "collate_fn": _identity_collate,
        }
        return _StatefulDataLoader(**loader_kwargs)

    def train_dataloader(self) -> DataLoader:
        base_seed = self.seed
        return self._make_loader(
            dataset=self._build_train_dataset(base_seed),
            steps=self.train_steps,
        )

    def val_dataloader(self):
        base_seed = self.seed + 1_000_000
        loaders = [
            self._make_loader(
                dataset=self._build_eval_dataset(split, base_seed + i * 10_000),
                steps=self.eval_steps[split],
            )
            for i, split in enumerate(self.eval_splits)
        ]
        if len(loaders) == 1:
            return loaders[0]
        return loaders


def create_lightning_dataloaders(
    config: config_dict.ConfigDict, seed: int
) -> tuple[DataLoader, Any, dict[str, Any]]:
    module = TfLightningDataModule(config, seed)
    return module.train_dataloader(), module.val_dataloader(), module.info


# -----------------------------------------------------------------------------
# Legacy-style helpers
# -----------------------------------------------------------------------------


def create_gems_set_datasets(
    config: config_dict.ConfigDict,
    seed: Optional[int] = None,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    seed_value = int(config.seed if seed is None else seed)
    datamodule = TfLightningDataModule(config, seed=seed_value)

    train_ds = datamodule._build_train_dataset(seed_value)
    val_ds = datamodule._build_eval_dataset("validation", seed_value)

    val_iters: dict[str, Any] = {"validation": val_ds.as_numpy_iterator()}
    if datamodule.massspec_test_files:
        test_ds = datamodule._build_eval_dataset("massspec_test", seed_value)
        val_iters["massspec_test"] = test_ds.as_numpy_iterator()

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

    raise NotImplementedError("Only gems_a (peak set) dataset is supported.")


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

    print("\nFirst sample token_ids:", batch["token_ids"][0][:20])
    print("First sample segment_ids:", batch["segment_ids"][0][:20])
