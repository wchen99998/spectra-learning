import logging
import math
import os
import shutil
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import torch
from huggingface_hub import snapshot_download
from ml_collections import config_dict
from torch.utils.data import DataLoader, Dataset, IterableDataset
from utils.gems_native import (
    load_gems_native_metadata,
    validate_gems_native_artifact,
)

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
_DEFAULT_MIN_PEAK_INTENSITY = 1e-4
_METADATA_FILENAME = "metadata.json"


def _sample_block_masks_tf(
    peak_valid_mask: tf.Tensor,
    *,
    num_target_blocks: int,
    context_fraction: float,
    target_fraction: float,
    block_min_len: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    num_targets = int(num_target_blocks)
    min_len = int(block_min_len)
    num_peaks = peak_valid_mask.shape[1]

    def sample_one(row_valid: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        valid_count = tf.reduce_sum(tf.cast(row_valid, tf.int32))
        desired_context = tf.maximum(
            tf.cast(
                tf.round(tf.cast(valid_count, tf.float32) * float(context_fraction)),
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
            available_for_targets = tf.maximum(valid_count - context_len, 0)
            desired_target = tf.maximum(
                tf.cast(
                    tf.round(tf.cast(valid_count, tf.float32) * float(target_fraction)),
                    tf.int32,
                ),
                min_len,
            )
            target_len = tf.minimum(desired_target, available_for_targets)
        else:
            context_len = tf.minimum(desired_context, valid_count)
            target_len = tf.constant(0, dtype=tf.int32)
        context_start = tf.random.uniform(
            [],
            minval=0,
            maxval=valid_count - context_len + 1,
            dtype=tf.int32,
        )
        positions = tf.range(tf.shape(row_valid)[0], dtype=tf.int32)
        context_mask = tf.logical_and(
            positions >= context_start,
            positions < context_start + context_len,
        )
        context_mask = tf.logical_and(context_mask, row_valid)
        if num_targets == 0:
            return context_mask, tf.zeros([0, tf.shape(row_valid)[0]], dtype=tf.bool)

        available_for_targets = valid_count - context_len
        target_starts = tf.random.uniform(
            [num_targets],
            minval=0,
            maxval=available_for_targets - target_len + 1,
            dtype=tf.int32,
        )
        compressed_positions = tf.where(
            positions < context_start,
            positions,
            positions - context_len,
        )
        valid_target_positions = tf.logical_and(row_valid, ~context_mask)
        target_masks = tf.logical_and(
            compressed_positions[tf.newaxis, :] >= target_starts[:, tf.newaxis],
            compressed_positions[tf.newaxis, :]
            < (target_starts + target_len)[:, tf.newaxis],
        )
        target_masks = tf.logical_and(
            target_masks, valid_target_positions[tf.newaxis, :]
        )
        return context_mask, target_masks

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
        num_peaks = tf.shape(peak_valid_mask)[1]
        no_valid = ~tf.reduce_any(peak_valid_mask, axis=1)
        positions = tf.range(num_peaks, dtype=tf.int32)[tf.newaxis, :]
        fallback = tf.logical_and(no_valid[:, tf.newaxis], positions == 0)
        peak_valid_mask = tf.logical_or(peak_valid_mask, fallback)
        mz_noise = tf.random.normal(
            tf.shape(peak_mz), stddev=mz_jitter_std, dtype=peak_mz.dtype
        )
        peak_mz = tf.clip_by_value(
            tf.where(peak_valid_mask, peak_mz + mz_noise, tf.zeros_like(peak_mz)),
            0.0,
            1.0,
        )
        int_noise = tf.random.normal(
            tf.shape(peak_intensity),
            stddev=intensity_jitter_std,
            dtype=peak_intensity.dtype,
        )
        peak_intensity = tf.clip_by_value(
            tf.where(
                peak_valid_mask,
                peak_intensity + int_noise,
                tf.zeros_like(peak_intensity),
            ),
            0.0,
            1.0,
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


def _prepend_precursor_token_tf(batch: dict) -> dict:
    peak_mz = batch["peak_mz"]
    peak_intensity = batch["peak_intensity"]
    peak_valid_mask = batch["peak_valid_mask"]
    precursor_mz = batch["precursor_mz"]
    B = tf.shape(peak_mz)[0]
    out = dict(batch)
    out["peak_mz"] = tf.concat([tf.expand_dims(precursor_mz, 1), peak_mz], axis=1)
    out["peak_intensity"] = tf.concat([tf.fill([B, 1], -1.0), peak_intensity], axis=1)
    out["peak_valid_mask"] = tf.concat(
        [tf.ones([B, 1], dtype=tf.bool), peak_valid_mask], axis=1
    )
    if "context_mask" in batch:
        out["context_mask"] = tf.concat(
            [tf.ones([B, 1], dtype=tf.bool), batch["context_mask"]],
            axis=1,
        )
    if "target_masks" in batch:
        K = tf.shape(batch["target_masks"])[1]
        out["target_masks"] = tf.concat(
            [tf.zeros([B, K, 1], dtype=tf.bool), batch["target_masks"]],
            axis=2,
        )
    del out["precursor_mz"]
    return out


def apply_peak_transforms_tf(
    mz: tf.Tensor,
    intensity: tf.Tensor,
    precursor_mz: tf.Tensor,
    *,
    peak_mz_min: tf.Tensor,
    peak_mz_max: tf.Tensor,
    min_int: tf.Tensor,
    max_prec: tf.Tensor,
    num_peaks: int,
    peak_ordering: str,
) -> dict[str, tf.Tensor]:
    """Shared peak filtering, top-k selection, normalisation and ordering.

    Called inside @tf.function bodies from both the GeMS training pipeline and
    the probe dataset pipeline so both produce identical peak representations.

    Args:
        mz: raw m/z values [B, N_in], zero-padded.
        intensity: raw intensities [B, N_in], max-normalised to [0,1], zero-padded.
        precursor_mz: precursor m/z [B].
        peak_mz_min / peak_mz_max: tf.constant scalars for m/z range filter.
        min_int: tf.constant scalar for minimum intensity filter.
        max_prec: tf.constant scalar for precursor m/z clipping.
        num_peaks: number of peaks to keep (top-k).
        peak_ordering: ``"mz"`` (ascending) or ``"intensity"`` (descending).

    Returns:
        Dict with ``peak_mz``, ``peak_intensity``, ``peak_valid_mask``,
        ``precursor_mz``, ``mz``, ``intensity``.
    """
    keep = (mz >= peak_mz_min) & (mz <= peak_mz_max) & (intensity >= min_int)
    mz = tf.where(keep, mz, 0.0)
    intensity = tf.where(keep, intensity, 0.0)
    intensity, indices = tf.math.top_k(intensity, k=num_peaks, sorted=True)
    mz = tf.gather(mz, indices, batch_dims=1)
    max_intensity = tf.reduce_max(intensity, axis=1, keepdims=True)
    max_intensity = tf.maximum(max_intensity, 1e-8)
    intensity = intensity / max_intensity
    valid = intensity > 0
    if peak_ordering == "mz":
        sort_key = tf.where(valid, mz, tf.fill(tf.shape(mz), float("inf")))
        direction = "ASCENDING"
    else:
        sort_key = tf.where(
            valid, intensity, tf.fill(tf.shape(intensity), float("-inf"))
        )
        direction = "DESCENDING"
    sorted_idx = tf.argsort(sort_key, axis=1, direction=direction, stable=True)
    mz = tf.gather(mz, sorted_idx, batch_dims=1)
    intensity = tf.gather(intensity, sorted_idx, batch_dims=1)
    valid = tf.gather(valid, sorted_idx, batch_dims=1)
    mz = tf.where(valid, mz, 0.0)
    intensity = tf.where(valid, intensity, 0.0)
    return {
        "peak_mz": mz / peak_mz_max,
        "peak_intensity": intensity,
        "peak_valid_mask": valid,
        "precursor_mz": tf.clip_by_value(precursor_mz, 0.0, max_prec) / max_prec,
        "mz": mz,
        "intensity": intensity,
    }


def _batched_parse_and_transform(
    *,
    max_precursor_mz: float,
    min_peak_intensity: float,
    num_peaks: int,
    peak_ordering: str,
) -> Callable[[tf.Tensor], dict[str, tf.Tensor]]:
    peak_mz_min = tf.constant(_PEAK_MZ_MIN, tf.float32)
    peak_mz_max = tf.constant(_PEAK_MZ_MAX, tf.float32)
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
        out = apply_peak_transforms_tf(
            parsed["mz"],
            parsed["intensity"],
            parsed["precursor_mz"][:, 0],
            peak_mz_min=peak_mz_min,
            peak_mz_max=peak_mz_max,
            min_int=min_int,
            max_prec=max_prec,
            num_peaks=num_peaks,
            peak_ordering=peak_ordering,
        )
        out["rt"] = parsed["rt"][:, 0]
        return out

    return transform


def _build_dataset(
    filenames: list[str],
    batch_size: int,
    shuffle_buffer: int,
    seed: int | None,
    drop_remainder: bool,
    *,
    tfrecord_buffer_size: int,
    max_precursor_mz: float,
    min_peak_intensity: float,
    augment: bool = True,
    jepa_num_target_blocks: int = 2,
    jepa_context_fraction: float = 0.5,
    jepa_target_fraction: float = 0.25,
    jepa_block_min_len: int = 1,
    mz_jitter_std: float = 0.0001,
    intensity_jitter_std: float = 0.001,
    peak_ordering: str = "intensity",
    num_parallel_reads: int = tf.data.AUTOTUNE,
    use_precursor_token: bool = False,
    num_peaks: int = _NUM_PEAKS_OUTPUT,
) -> tf.data.Dataset:
    ds = tf.data.TFRecordDataset(
        filenames,
        compression_type="GZIP",
        buffer_size=int(tfrecord_buffer_size),
        num_parallel_reads=num_parallel_reads,
    )
    if shuffle_buffer > 0:
        ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    batched_fn = _batched_parse_and_transform(
        max_precursor_mz=max_precursor_mz,
        min_peak_intensity=min_peak_intensity,
        num_peaks=num_peaks,
        peak_ordering=peak_ordering,
    )
    ds = ds.map(batched_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
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
    if use_precursor_token:
        ds = ds.map(_prepend_precursor_token_tf, num_parallel_calls=tf.data.AUTOTUNE)
    options = tf.data.Options()
    options.deterministic = True
    ds = ds.with_options(options)
    return ds.prefetch(tf.data.AUTOTUNE)


def _to_torch(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return _to_torch(value.tolist())
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

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self):
        if self._dataset is None:
            self._dataset = self._dataset_builder()
        dataset = self._dataset
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            dataset = dataset.shard(
                num_shards=worker_info.num_workers,
                index=worker_info.id,
            )
        for batch in dataset.as_numpy_iterator():
            yield numpy_batch_to_torch(batch)


def _validate_gems_native_metadata(
    metadata: dict[str, Any],
    *,
    num_peaks: int,
    min_peak_intensity: float,
    peak_ordering: str,
    max_precursor_mz: float,
) -> None:
    expected = {
        "num_peaks": int(num_peaks),
        "min_peak_intensity": float(min_peak_intensity),
        "peak_ordering": str(peak_ordering),
        "max_precursor_mz": float(max_precursor_mz),
    }
    actual = {
        "num_peaks": int(metadata["num_peaks"]),
        "min_peak_intensity": float(metadata["min_peak_intensity"]),
        "peak_ordering": str(metadata["peak_ordering"]),
        "max_precursor_mz": float(metadata["max_precursor_mz"]),
    }
    if actual != expected:
        raise ValueError(
            f"GeMS native artifact preprocessing mismatch: expected {expected}, got {actual}"
        )


class _GemsMemmapDataset(Dataset):
    def __init__(self, shard_entries: list[dict[str, Any]]) -> None:
        self._shard_entries = [
            {"dir": Path(entry["dir"]), "length": int(entry["length"])}
            for entry in shard_entries
        ]
        lengths = np.asarray([entry["length"] for entry in self._shard_entries], dtype=np.int64)
        self._starts = np.zeros(len(lengths) + 1, dtype=np.int64)
        np.cumsum(lengths, out=self._starts[1:])
        self._arrays: list[dict[str, np.ndarray]] | None = None

    def __len__(self) -> int:
        return int(self._starts[-1])

    def _ensure_arrays(self) -> None:
        if self._arrays is not None:
            return
        # This is a map-style dataset, so the DataLoader sampler handles
        # cross-worker index partitioning; each worker only needs local memmaps.
        self._arrays = []
        for entry in self._shard_entries:
            shard_dir = entry["dir"]
            self._arrays.append(
                {
                    "peak_mz": np.load(shard_dir / "peak_mz.npy", mmap_mode="r"),
                    "peak_intensity": np.load(
                        shard_dir / "peak_intensity.npy",
                        mmap_mode="r",
                    ),
                    "peak_valid_mask": np.load(
                        shard_dir / "peak_valid_mask.npy",
                        mmap_mode="r",
                    ),
                    "precursor_mz": np.load(
                        shard_dir / "precursor_mz.npy",
                        mmap_mode="r",
                    ),
                }
            )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        self._ensure_arrays()
        assert self._arrays is not None
        idx = int(idx)
        shard_idx = int(np.searchsorted(self._starts, idx, side="right") - 1)
        local_idx = idx - int(self._starts[shard_idx])
        arrays = self._arrays[shard_idx]
        return {
            "peak_mz": torch.from_numpy(arrays["peak_mz"][local_idx].copy()),
            "peak_intensity": torch.from_numpy(
                arrays["peak_intensity"][local_idx].copy()
            ),
            "peak_valid_mask": torch.from_numpy(
                arrays["peak_valid_mask"][local_idx].copy()
            ),
            "precursor_mz": torch.tensor(
                float(arrays["precursor_mz"][local_idx]),
                dtype=torch.float32,
            ),
        }


def _sample_block_masks_torch(
    peak_valid_mask: torch.Tensor,
    *,
    num_target_blocks: int,
    context_fraction: float,
    target_fraction: float,
    block_min_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_peaks = peak_valid_mask.shape
    context_mask = torch.zeros(batch_size, num_peaks, dtype=torch.bool)
    target_masks = torch.zeros(
        batch_size,
        int(num_target_blocks),
        num_peaks,
        dtype=torch.bool,
    )
    positions = torch.arange(num_peaks)

    for row_idx in range(batch_size):
        row_valid = peak_valid_mask[row_idx]
        valid_count = int(row_valid.sum().item())
        desired_context = max(
            int(round(valid_count * float(context_fraction))),
            int(block_min_len),
        )
        if num_target_blocks > 0:
            reserve_for_targets = min(valid_count, int(num_target_blocks) * int(block_min_len))
            max_context_len = max(valid_count - reserve_for_targets, 1)
            context_len = min(desired_context, max_context_len)
            available_for_targets = max(valid_count - context_len, 0)
            desired_target = max(
                int(round(valid_count * float(target_fraction))),
                int(block_min_len),
            )
            target_len = min(desired_target, available_for_targets)
        else:
            context_len = min(desired_context, valid_count)
            target_len = 0

        context_start = int(torch.randint(valid_count - context_len + 1, ()).item())
        row_context = (positions >= context_start) & (
            positions < context_start + context_len
        )
        row_context &= row_valid
        context_mask[row_idx] = row_context

        if num_target_blocks == 0 or target_len == 0:
            continue

        compressed_positions = torch.where(
            positions < context_start,
            positions,
            positions - context_len,
        )
        valid_target_positions = row_valid & ~row_context
        max_target_start = valid_count - context_len - target_len + 1
        starts = torch.randint(max_target_start, (int(num_target_blocks),))
        for block_idx, block_start in enumerate(starts.tolist()):
            row_target = (compressed_positions >= block_start) & (
                compressed_positions < block_start + target_len
            )
            target_masks[row_idx, block_idx] = row_target & valid_target_positions

    return context_mask, target_masks


def _prepend_precursor_token_torch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    batch_size = int(batch["peak_mz"].shape[0])
    precursor_mz = batch["precursor_mz"]
    result: dict[str, torch.Tensor] = {
        "peak_mz": torch.cat([precursor_mz.unsqueeze(1), batch["peak_mz"]], dim=1),
        "peak_intensity": torch.cat(
            [
                torch.full(
                    (batch_size, 1),
                    -1.0,
                    dtype=batch["peak_intensity"].dtype,
                ),
                batch["peak_intensity"],
            ],
            dim=1,
        ),
        "peak_valid_mask": torch.cat(
            [
                torch.ones((batch_size, 1), dtype=torch.bool),
                batch["peak_valid_mask"],
            ],
            dim=1,
        ),
    }
    if "context_mask" in batch:
        result["context_mask"] = torch.cat(
            [torch.ones((batch_size, 1), dtype=torch.bool), batch["context_mask"]],
            dim=1,
        )
    if "target_masks" in batch:
        num_targets = int(batch["target_masks"].shape[1])
        result["target_masks"] = torch.cat(
            [
                torch.zeros((batch_size, num_targets, 1), dtype=torch.bool),
                batch["target_masks"],
            ],
            dim=2,
        )
    return result


class _GemsBatchCollator:
    def __init__(
        self,
        *,
        augment: bool,
        num_target_blocks: int,
        context_fraction: float,
        target_fraction: float,
        block_min_len: int,
        mz_jitter_std: float,
        intensity_jitter_std: float,
        use_precursor_token: bool,
    ) -> None:
        self.augment = bool(augment)
        self.num_target_blocks = int(num_target_blocks)
        self.context_fraction = float(context_fraction)
        self.target_fraction = float(target_fraction)
        self.block_min_len = int(block_min_len)
        self.mz_jitter_std = float(mz_jitter_std)
        self.intensity_jitter_std = float(intensity_jitter_std)
        self.use_precursor_token = bool(use_precursor_token)

    def __call__(self, samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        peak_mz = torch.stack([sample["peak_mz"] for sample in samples], dim=0)
        peak_intensity = torch.stack(
            [sample["peak_intensity"] for sample in samples],
            dim=0,
        )
        peak_valid_mask = torch.stack(
            [sample["peak_valid_mask"] for sample in samples],
            dim=0,
        )
        precursor_mz = torch.stack(
            [sample["precursor_mz"] for sample in samples],
            dim=0,
        )

        no_valid = ~peak_valid_mask.any(dim=1)
        if bool(no_valid.any()):
            peak_valid_mask = peak_valid_mask.clone()
            peak_valid_mask[no_valid, 0] = True

        batch: dict[str, torch.Tensor] = {
            "peak_mz": peak_mz,
            "peak_intensity": peak_intensity,
            "peak_valid_mask": peak_valid_mask,
            "precursor_mz": precursor_mz,
        }

        if self.augment:
            if self.mz_jitter_std > 0:
                batch["peak_mz"] = torch.where(
                    batch["peak_valid_mask"],
                    (batch["peak_mz"] + torch.randn_like(batch["peak_mz"]) * self.mz_jitter_std).clamp(0.0, 1.0),
                    torch.zeros_like(batch["peak_mz"]),
                )
            if self.intensity_jitter_std > 0:
                batch["peak_intensity"] = torch.where(
                    batch["peak_valid_mask"],
                    (
                        batch["peak_intensity"]
                        + torch.randn_like(batch["peak_intensity"]) * self.intensity_jitter_std
                    ).clamp(0.0, 1.0),
                    torch.zeros_like(batch["peak_intensity"]),
                )
            context_mask, target_masks = _sample_block_masks_torch(
                batch["peak_valid_mask"],
                num_target_blocks=self.num_target_blocks,
                context_fraction=self.context_fraction,
                target_fraction=self.target_fraction,
                block_min_len=self.block_min_len,
            )
            batch["context_mask"] = context_mask
            batch["target_masks"] = target_masks

        if self.use_precursor_token:
            batch = _prepend_precursor_token_torch(batch)

        return batch


class TfLightningDataModule:
    def __init__(self, config: config_dict.ConfigDict, seed: int) -> None:
        self.config = config
        self.seed = int(seed)
        self.output_dir = (
            Path(config.get("tfrecord_dir", str(_DEFAULT_TFRECORD_DIR)))
            .expanduser()
            .resolve()
        )
        self.gems_dir = self.output_dir / "gems"
        self.gems_native_repo_id = str(config.get("gems_native_repo_id", "")).strip()
        if not self.gems_native_repo_id:
            raise ValueError("GeMS configs must set gems_native_repo_id")
        self.gems_native_revision = str(config.get("gems_native_revision", "main"))
        self.batch_size = int(config.get("batch_size", _DEFAULT_BATCH_SIZE))
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
        self.mz_jitter_std = float(
            config.get(
                "augmentation_mz_jitter_std",
                config.get("sigreg_mz_jitter_std", 0.0001),
            )
        )
        self.intensity_jitter_std = float(
            config.get(
                "augmentation_intensity_jitter_std",
                config.get("sigreg_intensity_jitter_std", 0.001),
            )
        )
        self.use_precursor_token = bool(config.get("use_precursor_token", False))
        self.num_peaks_output = int(config.get("num_peaks", _NUM_PEAKS_OUTPUT))

        metadata_path = self.gems_dir / _METADATA_FILENAME
        should_download = True
        if metadata_path.exists():
            existing_metadata = load_gems_native_metadata(self.gems_dir)
            should_download = "gems_native_metadata_version" not in existing_metadata
            if should_download:
                shutil.rmtree(self.gems_dir)
        if should_download:
            logger.info(
                "Downloading GeMS native artifact from %s@%s",
                self.gems_native_repo_id,
                self.gems_native_revision,
            )
            snapshot_download(
                repo_id=self.gems_native_repo_id,
                repo_type="dataset",
                revision=self.gems_native_revision,
                local_dir=self.gems_dir,
                allow_patterns=[_METADATA_FILENAME, "train/*", "validation/*"],
            )
        self.gems_metadata = load_gems_native_metadata(self.gems_dir)
        validate_gems_native_artifact(self.gems_dir, self.gems_metadata)
        _validate_gems_native_metadata(
            self.gems_metadata,
            num_peaks=self.num_peaks_output,
            min_peak_intensity=self.min_peak_intensity,
            peak_ordering=self.peak_ordering,
            max_precursor_mz=self.max_precursor_mz,
        )
        self.gems_train_shards = [
            str(self.gems_dir / "train" / name)
            for name in self.gems_metadata["train_shards"]
        ]
        self.gems_validation_shards = [
            str(self.gems_dir / "validation" / name)
            for name in self.gems_metadata["validation_shards"]
        ]
        self.gems_train_files = list(self.gems_train_shards)
        self.gems_validation_files = list(self.gems_validation_shards)
        self._train_entries = [
            {"dir": path, "length": int(length)}
            for path, length in zip(
                self.gems_train_shards,
                self.gems_metadata["train_lengths"],
                strict=True,
            )
        ]
        self._val_entries = [
            {"dir": path, "length": int(length)}
            for path, length in zip(
                self.gems_validation_shards,
                self.gems_metadata["validation_lengths"],
                strict=True,
            )
        ]
        self.info = {
            "tfrecord_dir": str(self.output_dir),
            "gems_dir": str(self.gems_dir),
            "train_size": self.gems_metadata["train_size"],
            "validation_size": self.gems_metadata["validation_size"],
            "num_peaks": self.num_peaks_output + (1 if self.use_precursor_token else 0),
            "max_precursor_mz": self.max_precursor_mz,
            "peak_mz_min": _PEAK_MZ_MIN,
            "peak_mz_max": _PEAK_MZ_MAX,
        }
        train_size = int(self.info["train_size"])
        if self.drop_remainder:
            self.train_steps = train_size // self.batch_size
        else:
            self.train_steps = math.ceil(train_size / self.batch_size)
        default_pin = torch.cuda.is_available()
        self.pin_memory = bool(config.get("dataloader_pin_memory", default_pin))
        self.dataloader_num_workers = int(config.get("dataloader_num_workers", 1))
        self.dataloader_prefetch_factor = int(
            config.get("dataloader_prefetch_factor", 2)
        )
        self.dataloader_persistent_workers = bool(
            config.get("dataloader_persistent_workers", self.dataloader_num_workers > 0)
        )
        self._train_dataset: _GemsMemmapDataset | None = None
        self._val_dataset: _GemsMemmapDataset | None = None
        self._train_loader: DataLoader | None = None
        self._val_loader: DataLoader | None = None

    def _get_dataset(self, split: str) -> _GemsMemmapDataset:
        if split == "train":
            if self._train_dataset is None:
                self._train_dataset = _GemsMemmapDataset(self._train_entries)
            return self._train_dataset
        if self._val_dataset is None:
            self._val_dataset = _GemsMemmapDataset(self._val_entries)
        return self._val_dataset

    def _make_loader(
        self,
        *,
        dataset: _GemsMemmapDataset,
        augment: bool,
        shuffle: bool,
        seed: int,
        drop_last: bool,
    ) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        loader_kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_size": self.batch_size,
            "num_workers": self.dataloader_num_workers,
            "pin_memory": self.pin_memory,
            "drop_last": bool(drop_last),
            "collate_fn": _GemsBatchCollator(
                augment=augment,
                num_target_blocks=self.jepa_num_target_blocks,
                context_fraction=self.jepa_context_fraction,
                target_fraction=self.jepa_target_fraction,
                block_min_len=self.jepa_block_min_len,
                mz_jitter_std=self.mz_jitter_std,
                intensity_jitter_std=self.intensity_jitter_std,
                use_precursor_token=self.use_precursor_token,
            ),
            "generator": generator,
        }
        loader_kwargs["shuffle"] = bool(shuffle)
        if self.dataloader_num_workers > 0:
            loader_kwargs["persistent_workers"] = self.dataloader_persistent_workers
            loader_kwargs["prefetch_factor"] = self.dataloader_prefetch_factor
        return DataLoader(**loader_kwargs)

    @property
    def train_loader(self) -> DataLoader:
        if self._train_loader is None:
            self._train_loader = self.train_loader_for_epoch(0)
        return self._train_loader

    @property
    def val_loader(self) -> DataLoader:
        if self._val_loader is None:
            self._val_loader = self._make_loader(
                dataset=self._get_dataset("validation"),
                augment=False,
                shuffle=False,
                seed=self.seed,
                drop_last=False,
            )
        return self._val_loader

    def train_loader_for_epoch(self, epoch: int) -> DataLoader:
        return self._make_loader(
            dataset=self._get_dataset("train"),
            augment=True,
            shuffle=True,
            seed=self.seed + int(epoch),
            drop_last=self.drop_remainder,
        )
