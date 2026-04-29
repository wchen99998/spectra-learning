import math
from pathlib import Path
from typing import Any

import torch
from ml_collections import config_dict
from torch.utils.data import DataLoader

from spectra_learning.data.gems.artifacts import resolve_gems_artifact
from spectra_learning.data.gems.collate import GemsBatchCollator
from spectra_learning.data.gems.dataset import GemsMemmapDataset
from spectra_learning.data.gems.settings import GemsDataConfig
from utils.spectra_preprocessing import PEAK_MZ_MAX, PEAK_MZ_MIN


class GemsNativeDataModule:
    def __init__(self, config: config_dict.ConfigDict, seed: int) -> None:
        self.config = GemsDataConfig.from_config(config)
        self.seed = int(seed)
        self.output_dir = self.config.artifact_dir
        self.gems_base_dir = self.output_dir / "gems"
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
        )
        self._set_public_config_attrs()
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
        }

    def _train_steps(self) -> int:
        train_size = int(self.info["train_size"])
        if self.drop_remainder:
            return train_size // self.batch_size
        return math.ceil(train_size / self.batch_size)

    def _get_dataset(self, split: str) -> GemsMemmapDataset:
        if split == "train":
            if self._train_dataset is None:
                self._train_dataset = GemsMemmapDataset(self._train_entries)
            return self._train_dataset
        if self._val_dataset is None:
            self._val_dataset = GemsMemmapDataset(self._val_entries)
        return self._val_dataset

    def _make_loader(
        self,
        *,
        dataset: GemsMemmapDataset,
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
            "pin_memory": self.dataloader_pin_memory,
            "drop_last": bool(drop_last),
            "collate_fn": self._collator(augment=augment),
            "generator": generator,
        }
        if self.dataloader_num_workers > 0:
            loader_kwargs["persistent_workers"] = self.dataloader_persistent_workers
            loader_kwargs["prefetch_factor"] = self.dataloader_prefetch_factor
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
            use_precursor_token=self.use_precursor_token,
            num_peaks=self.num_peaks_output,
            max_precursor_mz=self.max_precursor_mz,
            min_peak_intensity=self.min_peak_intensity,
            peak_drop_min_intensity=self.peak_drop_min_intensity,
            peak_ordering=self.peak_ordering,
            precursor_peak_exclusion_window_da=self.precursor_peak_exclusion_window_da,
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

    def train_loader_for_epoch(self, epoch: int) -> DataLoader:
        return self._make_loader(
            dataset=self._get_dataset("train"),
            augment=True,
            shuffle=True,
            seed=self.seed + int(epoch),
            drop_last=self.drop_remainder,
        )
