"""TF input pipeline and DataModule for GeMS peak lists."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Callable, Optional

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import torch
from huggingface_hub import snapshot_download
from ml_collections import config_dict
from torch.utils.data import DataLoader, IterableDataset
from utils.gems_tfrecords import load_gems_metadata, validate_gems_artifact

tf.config.set_visible_devices([], "GPU")

logger = logging.getLogger(__name__)

_DEFAULT_BATCH_SIZE = 512
_DEFAULT_SHUFFLE_BUFFER = 10_000
_DEFAULT_TFRECORD_DIR = Path("data/gems_peaklist_tfrecord")
_DEFAULT_TFRECORD_BUFFER_SIZE = 250_000
_NUM_PEAKS_INPUT = 128
_NUM_PEAKS_OUTPUT = 60
_DEFAULT_MAX_PRECURSOR_MZ = 1000.0
_PEAK_MZ_MIN = 20.0
_PEAK_MZ_MAX = 1000.0
_PRECURSOR_MZ_WINDOW = 2.5
_INTENSITY_EPS = 1e-4
_DEFAULT_MIN_PEAK_INTENSITY = _INTENSITY_EPS
_METADATA_FILENAME = "metadata.json"


def _ensure_gems_downloaded(
    *,
    output_dir: Path,
    repo_id: str,
    revision: str,
) -> dict[str, Any]:
    logger.info("Downloading GeMS TFRecords from %s@%s", repo_id, revision)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=output_dir,
        allow_patterns=[_METADATA_FILENAME, "train/*", "validation/*"],
    )
    metadata = load_gems_metadata(output_dir)
    validate_gems_artifact(output_dir, metadata)
    return metadata



# -----------------------------------------------------------------------------
# tf.data pipeline
# -----------------------------------------------------------------------------

@tf.function
def _ensure_nonempty_peakset_tf(
    peak_mz: tf.Tensor,
    peak_intensity: tf.Tensor,
    peak_valid_mask: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Ensure each sample has at least one valid peak (fallback to index 0)."""
    num_peaks = tf.shape(peak_valid_mask)[1]
    has_valid = tf.reduce_any(peak_valid_mask, axis=1)  # [B]
    needs_fallback = tf.logical_not(has_valid)
    positions = tf.range(num_peaks, dtype=tf.int32)[tf.newaxis, :]
    fallback = tf.logical_and(needs_fallback[:, tf.newaxis], positions == 0)
    safe_valid_mask = tf.logical_or(peak_valid_mask, fallback)
    return peak_mz, peak_intensity, safe_valid_mask


@tf.function
def _apply_peak_jitter_tf(
    peak_mz: tf.Tensor,
    peak_intensity: tf.Tensor,
    peak_valid_mask: tf.Tensor,
    *,
    mz_jitter_std: float,
    intensity_jitter_std: float,
) -> tuple[tf.Tensor, tf.Tensor]:
    mz_noise = tf.random.normal(tf.shape(peak_mz), stddev=mz_jitter_std, dtype=peak_mz.dtype)
    mz = tf.where(peak_valid_mask, peak_mz + mz_noise, tf.zeros_like(peak_mz))
    mz = tf.clip_by_value(mz, 0.0, 1.0)

    intensity_noise = tf.random.normal(
        tf.shape(peak_intensity),
        stddev=intensity_jitter_std,
        dtype=peak_intensity.dtype,
    )
    intensity = tf.where(
        peak_valid_mask,
        peak_intensity + intensity_noise,
        tf.zeros_like(peak_intensity),
    )
    intensity = tf.clip_by_value(intensity, 0.0, 1.0)
    intensity = tf.where(peak_valid_mask, intensity, tf.zeros_like(intensity))
    return mz, intensity


def _sample_block_masks_tf(
    peak_valid_mask: tf.Tensor,
    *,
    num_target_blocks: int,
    context_fraction: float,
    target_fraction: float,
    block_min_len: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    num_targets = int(num_target_blocks)
    num_blocks = 1 + num_targets
    min_len = int(block_min_len)
    num_peaks = peak_valid_mask.shape[1]
    context_fraction_value = float(context_fraction)
    target_fraction_value = float(target_fraction)

    def sample_one(row_valid: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        valid_count = tf.reduce_sum(tf.cast(row_valid, tf.int32))
        desired_context = tf.maximum(
            tf.cast(
                tf.round(tf.cast(valid_count, tf.float32) * context_fraction_value),
                tf.int32,
            ),
            min_len,
        )
        if num_targets > 0:
            reserve_for_targets = tf.minimum(
                valid_count,
                tf.constant(num_targets * min_len, dtype=tf.int32),
            )
            max_context_len = tf.maximum(valid_count - reserve_for_targets, 1)
            context_len = tf.minimum(desired_context, max_context_len)
            target_budget = tf.maximum(valid_count - context_len, 0)
            desired_target = tf.maximum(
                tf.cast(
                    tf.round(tf.cast(valid_count, tf.float32) * target_fraction_value),
                    tf.int32,
                ),
                min_len,
            )
            target_len = tf.minimum(desired_target, target_budget // num_targets)
            target_lengths = tf.fill([num_targets], target_len)
        else:
            context_len = tf.minimum(desired_context, valid_count)
            target_lengths = tf.zeros([0], dtype=tf.int32)
        lengths = tf.concat([tf.reshape(context_len, [1]), target_lengths], axis=0)
        perm = tf.random.shuffle(tf.range(num_blocks, dtype=tf.int32))
        shuffled_lengths = tf.gather(lengths, perm)
        total_len = tf.reduce_sum(shuffled_lengths)
        gap_budget = valid_count - total_len
        gap_choices = tf.random.uniform(
            [gap_budget],
            minval=0,
            maxval=num_blocks + 1,
            dtype=tf.int32,
        )
        gaps = tf.math.bincount(
            gap_choices,
            minlength=num_blocks + 1,
            maxlength=num_blocks + 1,
            dtype=tf.int32,
        )
        if num_blocks == 1:
            starts = gaps[:1]
        else:
            starts = tf.concat(
                [
                    gaps[:1],
                    gaps[:1] + tf.cumsum(shuffled_lengths[:-1] + gaps[1:num_blocks]),
                ],
                axis=0,
            )
        positions = tf.range(tf.shape(row_valid)[0], dtype=tf.int32)[tf.newaxis, :]
        block_masks = tf.logical_and(
            positions >= starts[:, tf.newaxis],
            positions < (starts + shuffled_lengths)[:, tf.newaxis],
        )
        block_masks = tf.logical_and(block_masks, row_valid[tf.newaxis, :])
        ordered_masks = tf.gather(block_masks, tf.argsort(perm))
        return ordered_masks[0], ordered_masks[1:]

    return tf.map_fn(
        sample_one,
        peak_valid_mask,
        fn_output_signature=(
            tf.TensorSpec(shape=(num_peaks,), dtype=tf.bool),
            tf.TensorSpec(shape=(num_targets, num_peaks), dtype=tf.bool),
        ),
    )


def _augment_block_jepa_batch_tf(
    *,
    num_target_blocks: int,
    context_fraction: float,
    target_fraction: float,
    block_min_len: int,
    mz_jitter_std: float,
    intensity_jitter_std: float,
) -> Callable[[dict], dict]:
    def apply(batch: dict) -> dict:
        peak_mz = batch["peak_mz"]
        peak_intensity = batch["peak_intensity"]
        peak_valid_mask = batch["peak_valid_mask"]
        peak_mz, peak_intensity, peak_valid_mask = _ensure_nonempty_peakset_tf(
            peak_mz,
            peak_intensity,
            peak_valid_mask,
        )
        peak_mz, peak_intensity = _apply_peak_jitter_tf(
            peak_mz,
            peak_intensity,
            peak_valid_mask,
            mz_jitter_std=mz_jitter_std,
            intensity_jitter_std=intensity_jitter_std,
        )
        context_mask, target_masks = _sample_block_masks_tf(
            peak_valid_mask,
            num_target_blocks=num_target_blocks,
            context_fraction=context_fraction,
            target_fraction=target_fraction,
            block_min_len=block_min_len,
        )
        out = dict(batch)
        out["peak_mz"] = peak_mz
        out["peak_intensity"] = peak_intensity
        out["peak_valid_mask"] = peak_valid_mask
        out["context_mask"] = context_mask
        out["target_masks"] = target_masks
        return out

    return apply



def _batched_parse_and_transform(
    *,
    max_precursor_mz: float,
    min_peak_intensity: float,
    num_peaks: int,
    peak_ordering: str,
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

        return {
            "peak_mz": mz / peak_mz_max_c,
            "peak_intensity": tf.where(valid_final, intensity, 0.0),
            "peak_valid_mask": valid_final,
            "precursor_mz": precursor_mz_norm,
            "rt": rt,
            "mz": mz,
            "intensity": intensity,
        }

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
    min_peak_intensity: float,
    augmentation_type: str = "block_jepa",
    jepa_num_target_blocks: int = 2,
    jepa_context_fraction: float = 0.5,
    jepa_target_fraction: float = 0.25,
    jepa_block_min_len: int = 1,
    mz_jitter_std: float = 0.0001,
    intensity_jitter_std: float = 0.001,
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
        peak_ordering=peak_ordering,
    )
    ds = ds.map(batched_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if augmentation_type == "block_jepa":
        ds = ds.map(
            _augment_block_jepa_batch_tf(
                num_target_blocks=jepa_num_target_blocks,
                context_fraction=jepa_context_fraction,
                target_fraction=jepa_target_fraction,
                block_min_len=jepa_block_min_len,
                mz_jitter_std=mz_jitter_std,
                intensity_jitter_std=intensity_jitter_std,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    elif augmentation_type != "none":
        raise ValueError(f"Unsupported augmentation_type: {augmentation_type}")
    options = tf.data.Options()
    options.deterministic = True
    ds = ds.with_options(options)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds



# -----------------------------------------------------------------------------
# Info and step resolution
# -----------------------------------------------------------------------------


def _compute_info(
    gems_metadata: dict[str, Any],
    *,
    output_dir: Path,
    max_precursor_mz: float,
) -> dict[str, Any]:
    return {
        "tfrecord_dir": str(output_dir),
        "train_size": gems_metadata["train_size"],
        "validation_size": gems_metadata["validation_size"],
        "num_peaks": _NUM_PEAKS_OUTPUT,
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
        self.gems_dir = self.output_dir / "gems"
        self.gems_tfrecord_repo_id = str(
            config.get("gems_tfrecord_repo_id", "")
        ).strip()
        if not self.gems_tfrecord_repo_id:
            raise ValueError("GeMS configs must set gems_tfrecord_repo_id")
        self.gems_tfrecord_revision = str(config.get("gems_tfrecord_revision", "main"))
        self.batch_size = int(config.get("batch_size", _DEFAULT_BATCH_SIZE))
        self.shuffle_buffer = int(config.get("shuffle_buffer", _DEFAULT_SHUFFLE_BUFFER))
        self.tfrecord_buffer_size = int(
            config.get("tfrecord_buffer_size", _DEFAULT_TFRECORD_BUFFER_SIZE)
        )
        self.drop_remainder = bool(config.get("drop_remainder", True))
        self.max_precursor_mz = float(
            config.get("max_precursor_mz", _DEFAULT_MAX_PRECURSOR_MZ)
        )
        self.min_peak_intensity = float(
            config.get("min_peak_intensity", _DEFAULT_MIN_PEAK_INTENSITY)
        )
        self.peak_ordering = str(config.get("peak_ordering", "intensity"))
        self.jepa_num_target_blocks = int(config.get("jepa_num_target_blocks", 2))
        self.jepa_context_fraction = float(config.get("jepa_context_fraction", 0.5))
        self.jepa_target_fraction = float(config.get("jepa_target_fraction", 0.25))
        self.jepa_block_min_len = int(config.get("jepa_block_min_len", 1))
        self.mz_jitter_std = float(config.get("sigreg_mz_jitter_std", 0.0001))
        self.intensity_jitter_std = float(
            config.get("sigreg_intensity_jitter_std", 0.001)
        )

        self.gems_metadata = _ensure_gems_downloaded(
            output_dir=self.gems_dir,
            repo_id=self.gems_tfrecord_repo_id,
            revision=self.gems_tfrecord_revision,
        )

        self.gems_train_files = [
            str(self.gems_dir / "train" / fn)
            for fn in self.gems_metadata["train_files"]
        ]
        self.gems_val_files = [
            str(self.gems_dir / "validation" / fn)
            for fn in self.gems_metadata["validation_files"]
        ]
        self.gems_test_files = list(self.gems_val_files)

        self.info = _compute_info(
            self.gems_metadata,
            output_dir=self.output_dir,
            max_precursor_mz=self.max_precursor_mz,
        )

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

    def _build_dataset_for_files(
        self,
        files: list[str],
        *,
        seed: int,
        shuffle: bool,
        drop_remainder: bool,
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
            min_peak_intensity=self.min_peak_intensity,
            augmentation_type="block_jepa",
            jepa_num_target_blocks=self.jepa_num_target_blocks,
            jepa_context_fraction=self.jepa_context_fraction,
            jepa_target_fraction=self.jepa_target_fraction,
            jepa_block_min_len=self.jepa_block_min_len,
            mz_jitter_std=self.mz_jitter_std,
            intensity_jitter_std=self.intensity_jitter_std,
            peak_ordering=self.peak_ordering,
        )

    def _build_gems_train_dataset(self, seed: int) -> tf.data.Dataset:
        return self._build_dataset_for_files(
            self.gems_train_files,
            seed=seed,
            shuffle=True,
            drop_remainder=self.drop_remainder,
        )

    def _build_gems_val_dataset(self, seed: int) -> tf.data.Dataset:
        return self._build_dataset_for_files(
            self.gems_val_files,
            seed=seed,
            shuffle=False,
            drop_remainder=True,
        )

    def _build_gems_test_dataset(self, seed: int) -> tf.data.Dataset:
        return self._build_dataset_for_files(
            self.gems_test_files,
            seed=seed,
            shuffle=False,
            drop_remainder=True,
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
