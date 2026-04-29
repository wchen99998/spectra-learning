import logging
import math
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlretrieve

import numpy as np
import torch
from huggingface_hub import snapshot_download
from ml_collections import config_dict
from torch.utils.data import DataLoader, Dataset

from utils.gems_native import (
    build_gems_native_artifact,
    load_gems_native_metadata,
    validate_gems_native_artifact,
)
from utils.spectra_preprocessing import (
    DEFAULT_MAX_PRECURSOR_MZ,
    DEFAULT_MIN_PEAK_INTENSITY,
    DEFAULT_PRECURSOR_PEAK_EXCLUSION_WINDOW_DA,
    NUM_PEAKS_INPUT,
    PEAK_MZ_MAX,
    PEAK_MZ_MIN,
    PRECURSOR_TOKEN_INTENSITY,
    preprocess_peak_batch_torch,
)

logger = logging.getLogger(__name__)

_DEFAULT_BATCH_SIZE = 512
_DEFAULT_ARTIFACT_DIR = Path("data/gems_artifacts")
_NUM_PEAKS_OUTPUT = 60
_METADATA_FILENAME = "metadata.json"
_DEFAULT_JEPA_MASK_STRATEGY = "contiguous"
_DEFAULT_JEPA_MASK_LENGTHS = (1, 2, 4, 8, 16)
_JEPA_MASK_STRATEGIES = ("contiguous", "ragged")


def numpy_batch_to_torch(batch: dict[str, Any]) -> dict[str, Any]:
    def _convert(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            if value.dtype == object:
                return [_convert(item) for item in value.tolist()]
            if value.dtype.kind in {"U", "S"}:
                return value.tolist()
            if not value.flags.c_contiguous or not value.flags.writeable:
                value = value.copy()
            return torch.from_numpy(value)
        if isinstance(value, list):
            return [_convert(item) for item in value]
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return value

    return {key: _convert(value) for key, value in batch.items()}


def _validate_gems_native_metadata(
    metadata: dict[str, Any],
    *,
    max_precursor_mz: float,
) -> None:
    expected = {
        "num_peaks_input": int(NUM_PEAKS_INPUT),
        "max_precursor_mz": float(max_precursor_mz),
        "artifact_format": "raw_peaklist_v1",
    }
    actual = {
        "num_peaks_input": int(metadata["num_peaks_input"]),
        "max_precursor_mz": float(metadata["max_precursor_mz"]),
        "artifact_format": str(metadata.get("artifact_format", "")),
    }
    if actual != expected:
        raise ValueError(
            f"GeMS native artifact preprocessing mismatch: expected {expected}, got {actual}"
        )


def _gems_native_artifact_dir_name(
    *,
    max_precursor_mz: float,
) -> str:
    def _fmt(value: object) -> str:
        return str(value).replace(".", "p").replace("-", "m")

    return "_".join(["gems_native_raw", f"pmax{_fmt(max_precursor_mz)}"])


def _download_gems_source_hdf5(source_url: str, output_dir: Path) -> Path:
    filename = Path(urlparse(source_url).path).name or "source.hdf5"
    source_dir = output_dir / "gems_source"
    source_dir.mkdir(parents=True, exist_ok=True)
    download_path = source_dir / filename
    if not download_path.exists():
        logger.info("Downloading GeMS source HDF5 from %s", source_url)
        urlretrieve(source_url, download_path)
    return download_path


class _GemsMemmapDataset(Dataset):
    def __init__(self, shard_entries: list[dict[str, Any]]) -> None:
        self._shard_entries = [
            {"dir": Path(entry["dir"]), "length": int(entry["length"])}
            for entry in shard_entries
        ]
        lengths = np.asarray(
            [entry["length"] for entry in self._shard_entries], dtype=np.int64
        )
        self._starts = np.zeros(len(lengths) + 1, dtype=np.int64)
        np.cumsum(lengths, out=self._starts[1:])
        self._arrays: list[dict[str, np.ndarray]] | None = None

    def __len__(self) -> int:
        return int(self._starts[-1])

    def _ensure_arrays(self) -> None:
        if self._arrays is not None:
            return
        self._arrays = []
        for entry in self._shard_entries:
            shard_dir = entry["dir"]
            self._arrays.append(
                {
                    "spectra": np.load(
                        shard_dir / "spectra.npy",
                        mmap_mode="r",
                    ),
                    "precursor_mz_raw": np.load(
                        shard_dir / "precursor_mz_raw.npy",
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
            "spectra": torch.from_numpy(arrays["spectra"][local_idx].copy()),
            "precursor_mz_raw": torch.tensor(
                float(arrays["precursor_mz_raw"][local_idx]),
                dtype=torch.float32,
            ),
        }


def _sample_ragged_block_mask_1d_torch(
    active_positions: torch.Tensor,
    *,
    masked_fraction: float,
    lengths: tuple[int, ...],
    round_from: int,
) -> torch.Tensor:
    active_count = int(active_positions.sum().item())
    if active_count == 0:
        return torch.zeros_like(active_positions)
    compressed_positions = torch.cumsum(active_positions.to(torch.int64), dim=0)
    compressed_positions = compressed_positions - active_positions.to(torch.int64)
    bs = torch.rand(len(lengths), device=active_positions.device)
    bs = bs / bs.sum()
    masks_by_length: list[torch.Tensor] = []
    for length_idx, length in enumerate(lengths):
        block_len = int(length)
        max_elem = int(
            math.ceil(float(masked_fraction) * float(active_count) / float(block_len))
        )
        coeff_float = float(bs[length_idx].item()) * float(max_elem)
        if length_idx < round_from:
            coeff = int(math.ceil(coeff_float))
        else:
            coeff = int(round(coeff_float))
        if coeff == 0:
            masks_by_length.append(torch.zeros_like(active_positions))
            continue
        effective_len = min(block_len, active_count)
        max_start = active_count - effective_len
        starts = torch.randint(
            0,
            max_start + 1,
            (coeff,),
            device=active_positions.device,
        )
        block_mask = (compressed_positions.unsqueeze(0) >= starts.unsqueeze(1)) & (
            compressed_positions.unsqueeze(0)
            < (starts + effective_len).unsqueeze(1)
        )
        block_mask &= active_positions.unsqueeze(0)
        masks_by_length.append(block_mask.any(dim=0))
    return torch.stack(masks_by_length, dim=0).any(dim=0)


def _sample_contiguous_mask_1d_torch(
    active_positions: torch.Tensor,
    *,
    mask_count: int,
) -> torch.Tensor:
    active_count = int(active_positions.sum().item())
    if active_count == 0:
        return torch.zeros_like(active_positions)
    count = min(int(mask_count), active_count)
    start = int(
        torch.randint(
            active_count - count + 1,
            (),
            device=active_positions.device,
        ).item()
    )
    compressed_positions = torch.cumsum(active_positions.to(torch.int64), dim=0)
    compressed_positions = compressed_positions - active_positions.to(torch.int64)
    mask = (compressed_positions >= start) & (compressed_positions < start + count)
    return mask & active_positions


def _sample_mask_strategy_torch(
    mask_strategy: str,
    *,
    device: torch.device,
) -> str:
    strategy = str(mask_strategy).lower()
    if strategy == "ragged_blocks":
        strategy = "ragged"
    if strategy == "all":
        strategy = _JEPA_MASK_STRATEGIES[
            int(torch.randint(len(_JEPA_MASK_STRATEGIES), (), device=device).item())
        ]
    return strategy


def _sample_block_masks_torch(
    peak_valid_mask: torch.Tensor,
    *,
    num_target_blocks: int,
    context_fraction: float,
    target_fraction: float,
    block_min_len: int,
    mask_strategy: str = _DEFAULT_JEPA_MASK_STRATEGY,
    mask_lengths: tuple[int, ...] = _DEFAULT_JEPA_MASK_LENGTHS,
    mask_round_from: int = len(_DEFAULT_JEPA_MASK_LENGTHS),
) -> tuple[torch.Tensor, torch.Tensor]:
    strategy = str(mask_strategy).lower()
    if strategy == "ragged_blocks":
        strategy = "ragged"
    if strategy not in {*_JEPA_MASK_STRATEGIES, "all"}:
        raise ValueError(f"Unsupported JEPA mask strategy: {mask_strategy!r}")
    lengths = tuple(int(length) for length in mask_lengths)
    round_from = int(mask_round_from)
    device = peak_valid_mask.device
    batch_size, num_peaks = peak_valid_mask.shape
    context_mask = torch.zeros(batch_size, num_peaks, dtype=torch.bool, device=device)
    target_masks = torch.zeros(
        batch_size,
        int(num_target_blocks),
        num_peaks,
        dtype=torch.bool,
        device=device,
    )
    for row_idx in range(batch_size):
        row_strategy = _sample_mask_strategy_torch(strategy, device=device)
        row_valid = peak_valid_mask[row_idx]
        valid_count = int(row_valid.sum().item())
        if valid_count == 0:
            continue
        row_context_fraction = float(context_fraction)
        row_target_fraction = float(target_fraction)
        desired_context = max(
            int(round(valid_count * row_context_fraction)),
            int(block_min_len),
        )
        if num_target_blocks > 0:
            reserve_for_targets = min(
                valid_count,
                int(num_target_blocks) * int(block_min_len),
            )
            max_context_len = max(valid_count - reserve_for_targets, 1)
            context_len = min(desired_context, max_context_len)
            available_for_targets = max(valid_count - context_len, 0)
            desired_target = max(
                int(round(valid_count * row_target_fraction)),
                int(block_min_len),
            )
            target_len = min(desired_target, available_for_targets)
        else:
            context_len = min(desired_context, valid_count)
            target_len = 0
        if row_strategy == "contiguous":
            row_context = _sample_contiguous_mask_1d_torch(
                row_valid,
                mask_count=context_len,
            )
        elif row_strategy == "ragged":
            row_context = _sample_ragged_block_mask_1d_torch(
                row_valid,
                masked_fraction=float(context_len) / float(valid_count),
                lengths=lengths,
                round_from=round_from,
            )
        context_mask[row_idx] = row_context
        if num_target_blocks == 0 or target_len == 0:
            continue
        valid_target_positions = row_valid & ~row_context
        if row_strategy == "contiguous":
            for block_idx in range(int(num_target_blocks)):
                target_masks[row_idx, block_idx] = _sample_contiguous_mask_1d_torch(
                    valid_target_positions,
                    mask_count=target_len,
                )
            continue
        available_for_targets = int(valid_target_positions.sum().item())
        if available_for_targets == 0:
            continue
        target_fraction_on_available = float(target_len) / float(available_for_targets)
        for block_idx in range(int(num_target_blocks)):
            target_masks[row_idx, block_idx] = _sample_ragged_block_mask_1d_torch(
                valid_target_positions,
                masked_fraction=target_fraction_on_available,
                lengths=lengths,
                round_from=round_from,
            )
    return context_mask, target_masks


def _prepend_precursor_token_torch(
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    batch_size = int(batch["peak_mz"].shape[0])
    device = batch["peak_intensity"].device
    result: dict[str, torch.Tensor] = {
        "peak_mz": torch.cat(
            [batch["precursor_mz"].unsqueeze(1), batch["peak_mz"]],
            dim=1,
        ),
        "peak_intensity": torch.cat(
            [
                torch.full(
                    (batch_size, 1),
                    PRECURSOR_TOKEN_INTENSITY,
                    dtype=batch["peak_intensity"].dtype,
                    device=device,
                ),
                batch["peak_intensity"],
            ],
            dim=1,
        ),
        "peak_valid_mask": torch.cat(
            [
                torch.ones((batch_size, 1), dtype=torch.bool, device=device),
                batch["peak_valid_mask"],
            ],
            dim=1,
        ),
    }
    for key in batch:
        if key not in result and key != "precursor_mz":
            result[key] = batch[key]
    if "context_mask" in batch:
        result["context_mask"] = torch.cat(
            [
                torch.ones((batch_size, 1), dtype=torch.bool, device=device),
                batch["context_mask"],
            ],
            dim=1,
        )
    if "target_masks" in batch:
        num_targets = int(batch["target_masks"].shape[1])
        result["target_masks"] = torch.cat(
            [
                torch.zeros(
                    (batch_size, num_targets, 1), dtype=torch.bool, device=device
                ),
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
        mask_strategy: str = _DEFAULT_JEPA_MASK_STRATEGY,
        mask_lengths: tuple[int, ...] = _DEFAULT_JEPA_MASK_LENGTHS,
        mask_round_from: int = len(_DEFAULT_JEPA_MASK_LENGTHS),
        use_precursor_token: bool,
        num_peaks: int,
        max_precursor_mz: float,
        min_peak_intensity: float,
        peak_drop_min_intensity: float,
        peak_ordering: str,
        precursor_peak_exclusion_window_da: float,
    ) -> None:
        self.augment = bool(augment)
        self.num_target_blocks = int(num_target_blocks)
        self.context_fraction = float(context_fraction)
        self.target_fraction = float(target_fraction)
        self.block_min_len = int(block_min_len)
        self.mask_strategy = str(mask_strategy)
        self.mask_lengths = tuple(int(length) for length in mask_lengths)
        self.mask_round_from = int(mask_round_from)
        self.use_precursor_token = bool(use_precursor_token)
        self.num_peaks = int(num_peaks)
        self.max_precursor_mz = float(max_precursor_mz)
        self.min_peak_intensity = float(min_peak_intensity)
        self.peak_drop_min_intensity = float(peak_drop_min_intensity)
        self.peak_ordering = str(peak_ordering)
        self.precursor_peak_exclusion_window_da = float(
            precursor_peak_exclusion_window_da
        )

    def __call__(self, samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        spectra = torch.stack([sample["spectra"] for sample in samples], dim=0)
        precursor_mz_raw = torch.stack(
            [sample["precursor_mz_raw"] for sample in samples],
            dim=0,
        )
        batch = preprocess_peak_batch_torch(
            spectra[:, 0, :],
            spectra[:, 1, :],
            precursor_mz_raw,
            num_peaks=self.num_peaks,
            peak_drop_min_intensity=self.peak_drop_min_intensity,
            peak_ordering=self.peak_ordering,
            max_precursor_mz=self.max_precursor_mz,
            precursor_peak_exclusion_window_da=self.precursor_peak_exclusion_window_da,
            min_peak_intensity=self.min_peak_intensity,
        )
        no_valid = ~batch["peak_valid_mask"].any(dim=1)
        if bool(no_valid.any()) and not self.use_precursor_token:
            batch["peak_valid_mask"] = batch["peak_valid_mask"].clone()
            batch["peak_valid_mask"][no_valid, 0] = True
        if self.augment:
            context_mask, target_masks = _sample_block_masks_torch(
                batch["peak_valid_mask"],
                num_target_blocks=self.num_target_blocks,
                context_fraction=self.context_fraction,
                target_fraction=self.target_fraction,
                block_min_len=self.block_min_len,
                mask_strategy=self.mask_strategy,
                mask_lengths=self.mask_lengths,
                mask_round_from=self.mask_round_from,
            )
            batch["context_mask"] = context_mask
            batch["target_masks"] = target_masks
        if self.use_precursor_token:
            batch = _prepend_precursor_token_torch(batch)
        return batch


class GemsNativeDataModule:
    def __init__(self, config: config_dict.ConfigDict, seed: int) -> None:
        self.config = config
        self.seed = int(seed)
        self.output_dir = (
            Path(config.get("artifact_dir", str(_DEFAULT_ARTIFACT_DIR)))
            .expanduser()
            .resolve()
        )
        self.gems_base_dir = self.output_dir / "gems"
        self.gems_native_repo_id = str(config.get("gems_native_repo_id", "")).strip()
        if not self.gems_native_repo_id:
            raise ValueError("GeMS configs must set gems_native_repo_id")
        self.gems_native_revision = str(config.get("gems_native_revision", "main"))
        self.batch_size = int(config.get("batch_size", _DEFAULT_BATCH_SIZE))
        self.drop_remainder = bool(config.get("drop_remainder", True))
        self.max_precursor_mz = float(
            config.get("max_precursor_mz", DEFAULT_MAX_PRECURSOR_MZ)
        )
        self.min_peak_intensity = float(
            config.get("min_peak_intensity", DEFAULT_MIN_PEAK_INTENSITY)
        )
        self.peak_drop_min_intensity = float(
            config.get("peak_drop_min_intensity", self.min_peak_intensity)
        )
        self.peak_ordering = str(config.get("peak_ordering", "mz"))
        self.precursor_peak_exclusion_window_da = float(
            config.get(
                "precursor_peak_exclusion_window_da",
                DEFAULT_PRECURSOR_PEAK_EXCLUSION_WINDOW_DA,
            )
        )
        self.gems_native_source_hdf5_path = str(
            config.get("gems_native_source_hdf5_path", "")
        ).strip()
        self.gems_native_source_url = str(
            config.get("gems_native_source_url", "")
        ).strip()
        self.jepa_num_target_blocks = int(config.get("jepa_num_target_blocks", 2))
        self.jepa_context_fraction = float(config.get("jepa_context_fraction", 0.5))
        self.jepa_target_fraction = float(config.get("jepa_target_fraction", 0.25))
        self.jepa_block_min_len = int(config.get("jepa_block_min_len", 1))
        self.jepa_mask_strategy = str(
            config.get("jepa_mask_strategy", _DEFAULT_JEPA_MASK_STRATEGY)
        )
        self.jepa_mask_lengths = tuple(
            int(length)
            for length in config.get("jepa_mask_lengths", _DEFAULT_JEPA_MASK_LENGTHS)
        )
        self.jepa_mask_round_from = int(
            config.get("jepa_mask_round_from", len(self.jepa_mask_lengths))
        )
        self.use_precursor_token = bool(config.get("use_precursor_token", False))
        self.num_peaks_output = int(config.get("num_peaks", _NUM_PEAKS_OUTPUT))
        self.gems_dir, self.gems_metadata = self._resolve_gems_artifact()
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
            "artifact_dir": str(self.output_dir),
            "gems_dir": str(self.gems_dir),
            "train_size": int(self.gems_metadata["train_size"]),
            "validation_size": int(self.gems_metadata["validation_size"]),
            "num_peaks": self.num_peaks_output,
            "max_precursor_mz": self.max_precursor_mz,
            "peak_mz_min": PEAK_MZ_MIN,
            "peak_mz_max": PEAK_MZ_MAX,
        }
        train_size = int(self.info["train_size"])
        self.train_steps = (
            train_size // self.batch_size
            if self.drop_remainder
            else math.ceil(train_size / self.batch_size)
        )
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

    def _ensure_base_gems_artifact(self) -> dict[str, Any]:
        metadata_path = self.gems_base_dir / _METADATA_FILENAME
        should_download = True
        if metadata_path.exists():
            existing_metadata = load_gems_native_metadata(self.gems_base_dir)
            should_download = "gems_native_metadata_version" not in existing_metadata
            if should_download:
                shutil.rmtree(self.gems_base_dir)
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
                local_dir=self.gems_base_dir,
                allow_patterns=[_METADATA_FILENAME, "train/*", "validation/*"],
            )
        metadata = load_gems_native_metadata(self.gems_base_dir)
        validate_gems_native_artifact(self.gems_base_dir, metadata)
        return metadata

    def _ensure_custom_gems_artifact(self, base_metadata: dict[str, Any]) -> Path:
        variant_dir = self.output_dir / "gems_variants" / _gems_native_artifact_dir_name(
            max_precursor_mz=self.max_precursor_mz,
        )
        metadata_path = variant_dir / _METADATA_FILENAME
        if metadata_path.exists():
            metadata = load_gems_native_metadata(variant_dir)
            validate_gems_native_artifact(variant_dir, metadata)
            _validate_gems_native_metadata(
                metadata,
                max_precursor_mz=self.max_precursor_mz,
            )
            return variant_dir
        source_path = self.gems_native_source_hdf5_path or str(
            base_metadata.get("source_hdf5_path", "")
        ).strip()
        source_url = self.gems_native_source_url or str(
            base_metadata.get("source_url", "")
        ).strip()
        if source_path:
            hdf5_path = Path(source_path).expanduser().resolve()
        elif source_url:
            hdf5_path = _download_gems_source_hdf5(source_url, self.output_dir)
        else:
            raise FileNotFoundError(
                "Need source HDF5 or source URL to build a custom GeMS native artifact."
            )
        build_gems_native_artifact(
            hdf5_path=hdf5_path,
            output_dir=variant_dir,
            max_precursor_mz=self.max_precursor_mz,
            num_workers=1,
            source_path=str(hdf5_path),
            source_url=source_url or None,
        )
        return variant_dir

    def _resolve_gems_artifact(self) -> tuple[Path, dict[str, Any]]:
        base_metadata = self._ensure_base_gems_artifact()
        try:
            _validate_gems_native_metadata(
                base_metadata,
                max_precursor_mz=self.max_precursor_mz,
            )
            return self.gems_base_dir, base_metadata
        except ValueError:
            variant_dir = self._ensure_custom_gems_artifact(base_metadata)
            metadata = load_gems_native_metadata(variant_dir)
            validate_gems_native_artifact(variant_dir, metadata)
            _validate_gems_native_metadata(
                metadata,
                max_precursor_mz=self.max_precursor_mz,
            )
            return variant_dir, metadata

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
            "shuffle": bool(shuffle),
            "num_workers": self.dataloader_num_workers,
            "pin_memory": self.pin_memory,
            "drop_last": bool(drop_last),
            "collate_fn": _GemsBatchCollator(
                augment=augment,
                num_target_blocks=self.jepa_num_target_blocks,
                context_fraction=self.jepa_context_fraction,
                target_fraction=self.jepa_target_fraction,
                block_min_len=self.jepa_block_min_len,
                mask_strategy=self.jepa_mask_strategy,
                mask_lengths=self.jepa_mask_lengths,
                mask_round_from=self.jepa_mask_round_from,
                use_precursor_token=self.use_precursor_token,
                num_peaks=self.num_peaks_output,
                max_precursor_mz=self.max_precursor_mz,
                min_peak_intensity=self.min_peak_intensity,
                peak_drop_min_intensity=self.peak_drop_min_intensity,
                peak_ordering=self.peak_ordering,
                precursor_peak_exclusion_window_da=self.precursor_peak_exclusion_window_da,
            ),
            "generator": generator,
        }
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


def _mask_block_ranges(mask: torch.Tensor) -> list[tuple[int, int]]:
    positions = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    if positions.numel() == 0:
        return []
    ranges: list[tuple[int, int]] = []
    start = int(positions[0].item())
    prev = start
    for value in positions[1:].tolist():
        current = int(value)
        if current != prev + 1:
            ranges.append((start, prev))
            start = current
        prev = current
    ranges.append((start, prev))
    return ranges


def _mask_block_ranges_in_active_order(
    mask: torch.Tensor,
    active_positions: torch.Tensor,
) -> list[tuple[int, int]]:
    active_indices = torch.nonzero(active_positions, as_tuple=False).squeeze(-1)
    if active_indices.numel() == 0:
        return []
    return _mask_block_ranges(mask[active_indices])


def _format_block_ranges(ranges: list[tuple[int, int]]) -> str:
    if not ranges:
        return "[]"
    return "[" + ", ".join(f"({start}, {end})" for start, end in ranges) + "]"


def _make_visualization_collator_kwargs(
    datamodule: GemsNativeDataModule,
) -> dict[str, Any]:
    return {
        "num_target_blocks": datamodule.jepa_num_target_blocks,
        "context_fraction": datamodule.jepa_context_fraction,
        "target_fraction": datamodule.jepa_target_fraction,
        "block_min_len": datamodule.jepa_block_min_len,
        "mask_strategy": datamodule.jepa_mask_strategy,
        "mask_lengths": datamodule.jepa_mask_lengths,
        "mask_round_from": datamodule.jepa_mask_round_from,
        "use_precursor_token": datamodule.use_precursor_token,
        "num_peaks": datamodule.num_peaks_output,
        "max_precursor_mz": datamodule.max_precursor_mz,
        "min_peak_intensity": datamodule.min_peak_intensity,
        "peak_drop_min_intensity": datamodule.peak_drop_min_intensity,
        "peak_ordering": datamodule.peak_ordering,
        "precursor_peak_exclusion_window_da": datamodule.precursor_peak_exclusion_window_da,
    }


def _normalize_mask_strategy_name(mask_strategy: str) -> str:
    strategy = str(mask_strategy).lower()
    if strategy == "ragged_blocks":
        strategy = "ragged"
    return strategy


def _resolve_visualization_strategies(
    config_mask_strategy: str,
    strategies: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    if strategies is not None:
        resolved = [_normalize_mask_strategy_name(strategy) for strategy in strategies]
        return tuple(dict.fromkeys(resolved))
    default_strategies = list(_JEPA_MASK_STRATEGIES)
    config_strategy = _normalize_mask_strategy_name(config_mask_strategy)
    if config_strategy not in default_strategies:
        default_strategies.append(config_strategy)
    return tuple(default_strategies)


def _load_real_mask_visualization_batches(
    *,
    config_path: str | Path,
    split: str,
    start_index: int,
    num_samples: int,
    seed: int,
    strategies: tuple[str, ...] | None = None,
) -> tuple[
    config_dict.ConfigDict,
    dict[str, torch.Tensor],
    dict[str, dict[str, torch.Tensor]],
    list[int],
]:
    from utils.training import load_config

    config = load_config(Path(config_path).expanduser().resolve())
    datamodule = GemsNativeDataModule(config, seed=seed)
    dataset = datamodule._get_dataset(split)
    sample_indices = [int(start_index) + offset for offset in range(int(num_samples))]
    samples = [dataset[index] for index in sample_indices]
    collator_kwargs = _make_visualization_collator_kwargs(datamodule)
    resolved_strategies = _resolve_visualization_strategies(
        datamodule.jepa_mask_strategy,
        strategies,
    )
    raw_batch = _GemsBatchCollator(
        augment=False,
        **collator_kwargs,
    )(samples)
    strategy_batches: dict[str, dict[str, torch.Tensor]] = {}
    for strategy in resolved_strategies:
        torch.manual_seed(int(seed))
        strategy_batches[strategy] = _GemsBatchCollator(
            augment=True,
            **(collator_kwargs | {"mask_strategy": strategy}),
        )(samples)
    return config, raw_batch, strategy_batches, sample_indices


def _mask_rows_for_plot(
    peak_valid_mask: torch.Tensor,
    context_mask: torch.Tensor,
    target_masks: torch.Tensor,
) -> tuple[np.ndarray, list[str]]:
    rows = [
        peak_valid_mask,
        context_mask,
        *[target_masks[target_idx] for target_idx in range(int(target_masks.shape[0]))],
    ]
    labels = [
        f"valid ({int(peak_valid_mask.sum().item())})",
        f"context ({int(context_mask.sum().item())})",
        *[
            f"target {target_idx} ({int(target_masks[target_idx].sum().item())})"
            for target_idx in range(int(target_masks.shape[0]))
        ],
    ]
    matrix = torch.stack(rows, dim=0).to(torch.float32).cpu().numpy()
    return matrix, labels


def _set_slot_ticks(
    ax: Any,
    *,
    num_slots: int,
    use_precursor_token: bool,
) -> None:
    step = max(int(math.ceil(float(num_slots) / 8.0)), 1)
    ticks = list(range(0, int(num_slots), step))
    if ticks[-1] != int(num_slots) - 1:
        ticks.append(int(num_slots) - 1)
    labels = [str(tick) for tick in ticks]
    if bool(use_precursor_token) and ticks:
        labels[0] = "P"
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)


def _plot_mask_strategy_panel(
    *,
    ax_slots: Any,
    ax_masks: Any,
    peak_intensity: torch.Tensor,
    peak_valid_mask: torch.Tensor,
    context_mask: torch.Tensor,
    target_masks: torch.Tensor,
    title: str,
    use_precursor_token: bool,
) -> None:
    x = np.arange(int(peak_intensity.shape[0]))
    valid = peak_valid_mask.cpu().numpy().astype(bool)
    context = context_mask.cpu().numpy().astype(bool)
    any_target = target_masks.any(dim=0).cpu().numpy().astype(bool)
    free_valid = valid & (~context) & (~any_target)
    padded = ~valid
    heights = peak_intensity.cpu().numpy()

    if np.any(free_valid):
        ax_slots.bar(
            x[free_valid],
            heights[free_valid],
            width=0.82,
            color="#cbd5e1",
            edgecolor="none",
            label="Valid unused",
        )
    if np.any(context):
        ax_slots.bar(
            x[context],
            heights[context],
            width=0.82,
            color="#2563eb",
            edgecolor="none",
            label="Context",
        )
    if np.any(any_target):
        ax_slots.bar(
            x[any_target],
            heights[any_target],
            width=0.82,
            color="#f97316",
            edgecolor="none",
            label="Target (any block)",
        )
    if np.any(padded):
        ax_slots.scatter(
            x[padded],
            np.zeros(int(padded.sum())),
            color="#94a3b8",
            marker="x",
            s=18,
            linewidths=1.0,
            label="Padding",
            zorder=5,
        )

    ax_slots.set_xlim(-0.5, len(x) - 0.5)
    ax_slots.set_title(title, fontsize=11, fontweight="bold")
    ax_slots.set_ylabel("Intensity")
    ax_slots.grid(axis="y", alpha=0.2)
    if bool(use_precursor_token):
        ax_slots.axvline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.35)
        ax_slots.text(
            0.01,
            0.95,
            "slot P = precursor",
            transform=ax_slots.transAxes,
            va="top",
            ha="left",
            fontsize=8,
        )
    ax_slots.legend(fontsize=7, loc="upper right")
    _set_slot_ticks(
        ax_slots,
        num_slots=int(peak_intensity.shape[0]),
        use_precursor_token=use_precursor_token,
    )

    mask_matrix, row_labels = _mask_rows_for_plot(
        peak_valid_mask=peak_valid_mask,
        context_mask=context_mask,
        target_masks=target_masks,
    )
    ax_masks.imshow(
        mask_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
    )
    ax_masks.set_yticks(np.arange(len(row_labels)))
    ax_masks.set_yticklabels(row_labels, fontsize=8)
    ax_masks.set_xlabel("Model input slot (P = precursor)" if use_precursor_token else "Peak slot")
    _set_slot_ticks(
        ax_masks,
        num_slots=int(peak_intensity.shape[0]),
        use_precursor_token=use_precursor_token,
    )


def _print_mask_strategy_summary(
    *,
    strategy: str,
    batch: dict[str, torch.Tensor],
    sample_index: int,
    dataset_index: int,
    use_precursor_token: bool,
) -> None:
    full_valid = batch["peak_valid_mask"][sample_index]
    full_context = batch["context_mask"][sample_index]
    full_targets = batch["target_masks"][sample_index]
    peak_valid = full_valid[1:] if bool(use_precursor_token) else full_valid
    peak_context = full_context[1:] if bool(use_precursor_token) else full_context
    peak_targets = full_targets[:, 1:] if bool(use_precursor_token) else full_targets
    valid_target_positions = peak_valid & (~peak_context)

    print(
        f"{strategy} | dataset_index={dataset_index} | "
        f"valid={int(full_valid.sum().item())} | "
        f"context={int(full_context.sum().item())} | "
        f"target_counts={[int(mask.sum().item()) for mask in full_targets]}"
    )
    if bool(use_precursor_token):
        print("  model slot P is the precursor token; active-order blocks ignore it")
    print(
        "  context: "
        f"model-slot={_format_block_ranges(_mask_block_ranges(full_context))} | "
        f"active-order={_format_block_ranges(_mask_block_ranges_in_active_order(peak_context, peak_valid))}"
    )
    for target_idx in range(int(full_targets.shape[0])):
        print(
            f"  target {target_idx}: "
            f"model-slot={_format_block_ranges(_mask_block_ranges(full_targets[target_idx]))} | "
            f"active-order={_format_block_ranges(_mask_block_ranges_in_active_order(peak_targets[target_idx], valid_target_positions))}"
        )


def visualize_real_mask_strategies(
    *,
    config_path: str | Path,
    split: str,
    start_index: int,
    num_samples: int,
    seed: int,
    output_path: str | Path,
    strategies: tuple[str, ...] | None = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config, raw_batch, strategy_batches, sample_indices = _load_real_mask_visualization_batches(
        config_path=config_path,
        split=split,
        start_index=start_index,
        num_samples=num_samples,
        seed=seed,
        strategies=strategies,
    )
    resolved_strategies = tuple(strategy_batches.keys())
    use_precursor_token = bool(config.get("use_precursor_token", False))
    fig, axes = plt.subplots(
        int(num_samples) * 2,
        len(resolved_strategies),
        figsize=(5.8 * len(resolved_strategies), 4.2 * int(num_samples)),
        height_ratios=[ratio for _ in range(int(num_samples)) for ratio in (3.0, 1.2)],
        squeeze=False,
    )
    fig.suptitle(
        "Real-data JEPA masking on "
        f"{split} split | samples {sample_indices[0]}-{sample_indices[-1]} | "
        f"seed={seed} | strategies={','.join(resolved_strategies)}",
        fontsize=14,
        fontweight="bold",
    )
    for row_offset, dataset_index in enumerate(sample_indices):
        peak_intensity = raw_batch["peak_intensity"][row_offset]
        peak_valid_mask = raw_batch["peak_valid_mask"][row_offset]
        for col_idx, strategy in enumerate(resolved_strategies):
            batch = strategy_batches[strategy]
            context_mask = batch["context_mask"][row_offset]
            target_masks = batch["target_masks"][row_offset]
            title = (
                f"{strategy.title()} | dataset[{dataset_index}] | "
                f"context={int(context_mask.sum().item())} | "
                f"targets={[int(mask.sum().item()) for mask in target_masks]}"
            )
            _plot_mask_strategy_panel(
                ax_slots=axes[row_offset * 2, col_idx],
                ax_masks=axes[row_offset * 2 + 1, col_idx],
                peak_intensity=peak_intensity,
                peak_valid_mask=peak_valid_mask,
                context_mask=context_mask,
                target_masks=target_masks,
                title=title,
                use_precursor_token=use_precursor_token,
            )
            if row_offset == int(num_samples) - 1:
                axes[row_offset * 2, col_idx].set_xlabel(
                    "Model input slot (P = precursor)" if use_precursor_token else "Peak slot"
                )
            _print_mask_strategy_summary(
                strategy=strategy,
                batch=batch,
                sample_index=row_offset,
                dataset_index=dataset_index,
                use_precursor_token=use_precursor_token,
            )
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved real-data mask visualization to {output_path}")
    return output_path


def _parse_args() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    visualize_parser = subparsers.add_parser(
        "visualize-masks",
        help="Visualize JEPA mask modes on real GeMS samples.",
    )
    visualize_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/gems_small.py"),
        help="Path to an experiment config.",
    )
    visualize_parser.add_argument(
        "--split",
        choices=("train", "validation"),
        default="validation",
        help="Dataset split to sample from.",
    )
    visualize_parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Dataset index for the first sample in the figure.",
    )
    visualize_parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of consecutive real samples to visualize.",
    )
    visualize_parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Torch RNG seed used when sampling context and target masks.",
    )
    visualize_parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/input_pipeline_real_masks.png"),
        help="Output image path.",
    )
    visualize_parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help=(
            "Mask strategies to render. Defaults to all concrete modes plus the config "
            "mode if it is additional."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "visualize-masks":
        visualize_real_mask_strategies(
            config_path=args.config,
            split=args.split,
            start_index=args.start_index,
            num_samples=args.num_samples,
            seed=args.seed,
            output_path=args.output,
            strategies=None if args.strategies is None else tuple(args.strategies),
        )


if __name__ == "__main__":
    main()
