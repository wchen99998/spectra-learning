from pathlib import Path
from typing import Any

from ml_collections import config_dict
from torch.utils.data import DataLoader, Sampler

from spectra_learning.data.gems.artifacts import resolve_gems_hdf5_manifest
from spectra_learning.data.gems.collate import GemsBatchCollator
from spectra_learning.data.gems.hdf5 import GemsHdf5ShardDataset
from spectra_learning.data.gems.sampling import (
    ChunkedDistributedBatchSampler,
    LimitBatchSampler,
    OffsetBatchSampler,
)
from spectra_learning.data.gems.settings import GemsDataConfig
from spectra_learning.data.spectra import PEAK_MZ_MAX, PEAK_MZ_MIN


class GemsDataModule:
    config: GemsDataConfig
    seed: int
    output_dir: Path
    gems_base_dir: Path
    distributed_world_size: int
    distributed_rank: int
    distributed_local_rank: int
    gems_dir: Path
    gems_manifest: Path
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

    def __init__(
        self,
        config: config_dict.ConfigDict,
        seed: int,
        *,
        distributed_world_size: int = 1,
        distributed_rank: int = 0,
        distributed_local_rank: int | None = None,
    ) -> None:
        distributed_local_rank = (
            distributed_rank if distributed_local_rank is None else distributed_local_rank
        )
        self.config = GemsDataConfig.from_config(config)
        self.seed = seed
        self.output_dir = self.config.artifact_dir
        self.gems_base_dir = self.output_dir / "gems"
        self.distributed_world_size = distributed_world_size
        self.distributed_rank = distributed_rank
        self.distributed_local_rank = distributed_local_rank
        self.gems_manifest = resolve_gems_hdf5_manifest(
            gems_base_dir=self.gems_base_dir,
            repo_id=self.config.gems_hdf5_repo_id,
            revision=self.config.gems_hdf5_revision,
            manifest_filename=self.config.gems_hdf5_manifest,
            distributed_world_size=distributed_world_size,
            distributed_rank=distributed_rank,
            distributed_local_rank=distributed_local_rank,
        )
        self.gems_dir = self.gems_manifest.parent
        self._set_public_config_attrs()
        self._set_distributed_batch_attrs()
        self._dataset = self._build_dataset()
        self.gems_train_shards = list(self._dataset.paths)
        self.gems_validation_shards = list(self._dataset.paths)
        self.gems_train_files = list(self.gems_train_shards)
        self.gems_validation_files = list(self.gems_validation_shards)
        self.info = self._info()
        self.train_steps = self._train_steps()
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

    def _build_dataset(self) -> GemsHdf5ShardDataset:
        return GemsHdf5ShardDataset(
            self.gems_manifest,
            spectrum_dataset=self.config.gems_hdf5_spectrum_dataset,
            precursor_dataset=self.config.gems_hdf5_precursor_dataset,
        )

    def _info(self) -> dict[str, Any]:
        return {
            "artifact_dir": str(self.output_dir),
            "gems_dir": str(self.gems_dir),
            "gems_manifest": str(self.gems_manifest),
            "train_size": len(self._dataset),
            "validation_size": len(self._dataset),
            "num_peaks": self.num_peaks_output,
            "max_precursor_mz": self.max_precursor_mz,
            "peak_mz_min": PEAK_MZ_MIN,
            "peak_mz_max": PEAK_MZ_MAX,
            "peak_filtering": self.peak_filtering,
            "grouped_peak_shoulder_da": self.grouped_peak_shoulder_da,
            "grouped_peak_isotope_charges": list(self.grouped_peak_isotope_charges),
        }

    def _train_steps(self) -> int:
        micro_batches = min(
            len(
                self._make_batch_sampler(
                    shuffle=True,
                    seed=self.seed,
                    drop_last=self.drop_remainder,
                    epoch=0,
                    rank=rank,
                )
            )
            for rank in range(self.distributed_world_size)
        )
        return micro_batches // self.gradient_accumulation_steps

    def _get_dataset(self, split: str) -> GemsHdf5ShardDataset:
        return self._dataset

    def _dataset_segments(self) -> list[tuple[int, int, int]]:
        return [
            (
                int(start),
                int(info["length"]),
                int(info["spectrum_chunk"][0] or 1),
            )
            for start, info in zip(self._dataset.starts, self._dataset.infos, strict=True)
        ]

    def _make_batch_sampler(
        self,
        *,
        shuffle: bool,
        seed: int,
        drop_last: bool,
        epoch: int,
        rank: int | None = None,
    ) -> Sampler[list[int]]:
        rank = self.distributed_rank if rank is None else rank
        sampler = ChunkedDistributedBatchSampler(
            self._dataset_segments(),
            batch_size=self.batch_size,
            rows_per_block=self.config.gems_hdf5_rows_per_block or None,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
            world_size=self.distributed_world_size,
            rank=rank,
        )
        sampler.set_epoch(epoch)
        return sampler

    def _make_loader(
        self,
        *,
        augment: bool,
        shuffle: bool,
        seed: int,
        drop_last: bool,
        start_batch: int = 0,
        epoch: int = 0,
        num_workers: int | None = None,
        max_batches: int | None = None,
    ) -> DataLoader:
        resolved_num_workers = (
            self.dataloader_num_workers if num_workers is None else num_workers
        )
        batch_sampler = self._make_batch_sampler(
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
            epoch=epoch,
        )
        start_index = start_batch * self.batch_size * self.gradient_accumulation_steps
        if start_index:
            batch_sampler = OffsetBatchSampler(
                batch_sampler,
                start_index=start_index,
                batch_size=self.batch_size,
                drop_last=drop_last,
            )
        if max_batches is not None:
            batch_sampler = LimitBatchSampler(
                batch_sampler,
                max_batches=max_batches,
            )
        if resolved_num_workers > 0:
            self._dataset.close()
        loader_kwargs: dict[str, Any] = {
            "dataset": self._dataset,
            "batch_sampler": batch_sampler,
            "num_workers": resolved_num_workers,
            "pin_memory": self.dataloader_pin_memory,
            "collate_fn": self._collator(augment=augment),
        }
        if resolved_num_workers > 0:
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
            self._val_loader = self.val_loader_for_eval(augment=False)
        return self._val_loader

    def val_loader_for_eval(self, *, augment: bool) -> DataLoader:
        return self._make_loader(
            augment=augment,
            shuffle=False,
            seed=self.seed,
            drop_last=False,
        )

    def train_loader_for_epoch(self, epoch: int, start_batch: int = 0) -> DataLoader:
        max_batches = max(
            0,
            (self.train_steps - start_batch) * self.gradient_accumulation_steps,
        )
        return self._make_loader(
            augment=True,
            shuffle=True,
            seed=self.seed,
            drop_last=self.drop_remainder,
            start_batch=start_batch,
            epoch=epoch,
            max_batches=max_batches,
        )
