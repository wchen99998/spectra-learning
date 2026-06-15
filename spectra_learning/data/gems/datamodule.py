import math
from collections.abc import Iterator, Sized
from pathlib import Path
from typing import Any, cast

import torch
from ml_collections import config_dict
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

from spectra_learning.data.gems.artifacts import resolve_gems_artifact
from spectra_learning.data.gems.collate import GemsBatchCollator
from spectra_learning.data.gems.dataset import GemsMemmapDataset
from spectra_learning.data.gems.settings import GemsDataConfig
from spectra_learning.data.spectra import PEAK_MZ_MAX, PEAK_MZ_MIN


class _OffsetSampler(Sampler[int]):
    def __init__(
        self,
        sampler: Sampler[int],
        *,
        start_index: int,
    ) -> None:
        self.sampler = sampler
        self.start_index = start_index

    def __iter__(self) -> Iterator[int]:
        for position, idx in enumerate(self.sampler):
            if position >= self.start_index:
                yield idx

    def __len__(self) -> int:
        return len(cast(Sized, self.sampler)) - self.start_index


class _ShuffledSampler(Sampler[int]):
    def __init__(
        self,
        dataset: GemsMemmapDataset,
        *,
        shuffle: bool,
        generator: torch.Generator,
    ) -> None:
        self.dataset = dataset
        self.shuffle = shuffle
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            order = torch.randperm(len(self.dataset), generator=self.generator)
            for idx in order:
                yield int(idx)
            return
        yield from range(len(self.dataset))

    def __len__(self) -> int:
        return len(self.dataset)


class GemsNativeDataModule:
    config: GemsDataConfig
    seed: int
    output_dir: Path
    gems_base_dir: Path
    distributed_world_size: int
    distributed_rank: int
    gems_dir: Path
    gems_metadata: dict[str, Any]
    info: dict[str, Any]
    train_steps: int
    global_batch_size: int
    batch_size: int
    gradient_accumulation_steps: int
    drop_remainder: bool
    max_precursor_mz: float
    min_peak_intensity: float
    peak_drop_min_intensity: float
    peak_filtering: str
    grouped_peak_shoulder_da: float
    grouped_peak_isotope_charges: tuple[int, ...]
    peak_ordering: str
    precursor_peak_exclusion_window_da: float
    jepa_num_target_blocks: int
    jepa_context_fraction: float
    jepa_target_fraction: float
    jepa_block_min_len: int
    jepa_mask_strategy: str | tuple[str, ...]
    jepa_mask_lengths: tuple[int, ...]
    jepa_mask_round_from: int
    jepa_intensity_aware_mask_config: dict[str, float]
    jepa_allow_target_overlap: bool
    num_peaks_output: int
    dataloader_pin_memory: bool
    dataloader_num_workers: int
    dataloader_prefetch_factor: int
    dataloader_persistent_workers: bool
    dataloader_multiprocessing_context: str
    dataloader_output_format: str
    gems_train_shards: list[str]
    gems_validation_shards: list[str]
    gems_train_files: list[str]
    gems_validation_files: list[str]
    _train_entries: list[dict[str, Any]]
    _val_entries: list[dict[str, Any]]

    def __init__(
        self,
        config: config_dict.ConfigDict,
        seed: int,
        *,
        distributed_world_size: int = 1,
        distributed_rank: int = 0,
    ) -> None:
        self.config = GemsDataConfig.from_config(config)
        self.seed = seed
        self.output_dir = self.config.artifact_dir
        self.gems_base_dir = self.output_dir / "gems"
        self.distributed_world_size = distributed_world_size
        self.distributed_rank = distributed_rank
        if not self.config.gems_native_repo_id:
            raise ValueError("GeMS configs must set gems_native_repo_id")
        self.gems_dir, self.gems_metadata = resolve_gems_artifact(
            output_dir=self.output_dir,
            gems_base_dir=self.gems_base_dir,
            repo_id=self.config.gems_native_repo_id,
            revision=self.config.gems_native_revision,
            max_precursor_mz=self.config.max_precursor_mz,
            source_hdf5_path=self.config.gems_native_source_hdf5_path,
            source_url=self.config.gems_native_source_url,
            repo_subdir=self.config.gems_native_hf_subdir,
            distributed_world_size=distributed_world_size,
            distributed_rank=distributed_rank,
        )
        self._set_public_config_attrs()
        self._set_distributed_batch_attrs()
        self._set_shard_entries()
        self.info = self._info()
        self.train_steps = self._train_steps()
        self._train_dataset: GemsMemmapDataset | None = None
        self._val_dataset: GemsMemmapDataset | None = None
        self._train_loader: DataLoader | None = None
        self._val_loader: DataLoader | None = None

    def _set_public_config_attrs(self) -> None:
        for key, value in self.config.__dict__.items():
            if key == "num_peaks":
                self.num_peaks_output = value
            else:
                setattr(self, key, value)

    def _set_distributed_batch_attrs(self) -> None:
        self.gradient_accumulation_steps = self.config.gradient_accumulation_steps
        self.global_batch_size = self.batch_size
        denominator = self.distributed_world_size * self.gradient_accumulation_steps
        assert self.global_batch_size % denominator == 0
        self.batch_size = self.global_batch_size // denominator
        if self.distributed_world_size > 1 and self.dataloader_num_workers > 0:
            self.dataloader_num_workers = max(
                1,
                self.dataloader_num_workers // self.distributed_world_size,
            )

    def _set_shard_entries(self) -> None:
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
        self._train_entries = self._entries("train", self.gems_train_shards)
        self._val_entries = self._entries("validation", self.gems_validation_shards)

    def _entries(self, split: str, shard_paths: list[str]) -> list[dict[str, Any]]:
        return [
            {"dir": path, "length": int(length)}
            for path, length in zip(
                shard_paths,
                self.gems_metadata[f"{split}_lengths"],
                strict=True,
            )
        ]

    def _info(self) -> dict[str, Any]:
        return {
            "artifact_dir": str(self.output_dir),
            "gems_dir": str(self.gems_dir),
            "train_size": int(self.gems_metadata["train_size"]),
            "validation_size": int(self.gems_metadata["validation_size"]),
            "num_peaks": self.num_peaks_output,
            "max_precursor_mz": self.max_precursor_mz,
            "peak_mz_min": PEAK_MZ_MIN,
            "peak_mz_max": PEAK_MZ_MAX,
            "peak_filtering": self.peak_filtering,
            "grouped_peak_shoulder_da": self.grouped_peak_shoulder_da,
            "grouped_peak_isotope_charges": list(self.grouped_peak_isotope_charges),
        }

    def _train_steps(self) -> int:
        train_size = int(self.info["train_size"])
        if self.drop_remainder:
            return train_size // self.global_batch_size
        return math.ceil(train_size / self.global_batch_size)

    def _get_dataset(self, split: str) -> GemsMemmapDataset:
        if split == "train":
            if self._train_dataset is None:
                self._train_dataset = GemsMemmapDataset(
                    self._train_entries,
                    return_numpy=True,
                )
            return self._train_dataset
        if self._val_dataset is None:
            self._val_dataset = GemsMemmapDataset(
                self._val_entries,
                return_numpy=True,
            )
        return self._val_dataset

    def _make_loader(
        self,
        *,
        dataset: GemsMemmapDataset,
        augment: bool,
        shuffle: bool,
        seed: int,
        drop_last: bool,
        start_batch: int = 0,
        epoch: int = 0,
    ) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(seed)
        start_index = (
            start_batch * self.batch_size * self.gradient_accumulation_steps
        )
        loader_kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_size": self.batch_size,
            "num_workers": self.dataloader_num_workers,
            "pin_memory": self.dataloader_pin_memory,
            "drop_last": drop_last,
            "collate_fn": self._collator(augment=augment),
            "generator": generator,
        }
        if self.distributed_world_size > 1:
            sampler = DistributedSampler(
                dataset,
                shuffle=shuffle,
                num_replicas=self.distributed_world_size,
                rank=self.distributed_rank,
                seed=self.seed,
                drop_last=drop_last,
            )
            sampler.set_epoch(epoch)
            if start_index:
                sampler = _OffsetSampler(sampler, start_index=start_index)
            loader_kwargs["sampler"] = sampler
        elif start_index:
            loader_kwargs["sampler"] = _OffsetSampler(
                _ShuffledSampler(dataset, shuffle=shuffle, generator=generator),
                start_index=start_index,
            )
        else:
            loader_kwargs["shuffle"] = shuffle
        if self.dataloader_num_workers > 0:
            loader_kwargs["persistent_workers"] = self.dataloader_persistent_workers
            loader_kwargs["prefetch_factor"] = self.dataloader_prefetch_factor
            if self.dataloader_multiprocessing_context:
                loader_kwargs["multiprocessing_context"] = (
                    self.dataloader_multiprocessing_context
                )
        return DataLoader(**loader_kwargs)

    def _collator(self, *, augment: bool) -> GemsBatchCollator:
        return GemsBatchCollator(
            augment=augment,
            num_target_blocks=self.jepa_num_target_blocks,
            context_fraction=self.jepa_context_fraction,
            target_fraction=self.jepa_target_fraction,
            block_min_len=self.jepa_block_min_len,
            mask_strategy=self.jepa_mask_strategy,
            mask_lengths=self.jepa_mask_lengths,
            mask_round_from=self.jepa_mask_round_from,
            intensity_aware_mask_config=self.jepa_intensity_aware_mask_config,
            allow_target_overlap=self.jepa_allow_target_overlap,
            num_peaks=self.num_peaks_output,
            max_precursor_mz=self.max_precursor_mz,
            min_peak_intensity=self.min_peak_intensity,
            peak_drop_min_intensity=self.peak_drop_min_intensity,
            peak_ordering=self.peak_ordering,
            precursor_peak_exclusion_window_da=self.precursor_peak_exclusion_window_da,
            peak_filtering=self.peak_filtering,
            grouped_peak_shoulder_da=self.grouped_peak_shoulder_da,
            grouped_peak_isotope_charges=self.grouped_peak_isotope_charges,
            output_format=self.dataloader_output_format,
        )

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

    def train_loader_for_epoch(self, epoch: int, start_batch: int = 0) -> DataLoader:
        return self._make_loader(
            dataset=self._get_dataset("train"),
            augment=True,
            shuffle=True,
            seed=self.seed + epoch,
            drop_last=self.drop_remainder,
            start_batch=start_batch,
            epoch=epoch,
        )
