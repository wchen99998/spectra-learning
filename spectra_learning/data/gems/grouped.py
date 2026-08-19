from __future__ import annotations

from collections.abc import Iterator, Sized
from typing import Any, cast

import jax
import numpy as np
import torch
from jax.experimental import multihost_utils
from ml_collections import config_dict
from torch.utils.data import DataLoader, Sampler

from spectra_learning.data.gems.collate import GemsBatchCollator
from spectra_learning.data.gems.datamodule import GemsDataModule
from spectra_learning.data.gems.hdf5 import MassiveV2Hdf5ShardDataset


class GroupedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        group_ranges: tuple[tuple[np.ndarray, np.ndarray], ...],
        *,
        groups_per_batch: int,
        spectra_per_group: int,
        shuffle: bool,
        seed: int,
        drop_last: bool,
        epoch: int,
        start_batch: int = 0,
        max_batches: int | None = None,
        partition_count: int = 1,
        partition_index: int = 0,
    ) -> None:
        self.group_ranges = group_ranges
        self.groups_per_batch = groups_per_batch
        self.spectra_per_group = spectra_per_group
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = epoch
        self.start_batch = start_batch
        self.max_batches = max_batches
        self.partition_count = partition_count
        self.partition_index = partition_index

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        shard_order = list(range(len(self.group_ranges)))
        if self.shuffle:
            shard_order = torch.randperm(
                len(shard_order),
                generator=generator,
            ).tolist()
        pending: list[int] = []
        pending_groups = 0
        batch_index = 0
        yielded = 0
        for shard_id in shard_order:
            starts, counts = self.group_ranges[shard_id]
            group_order = torch.arange(len(starts))
            if self.shuffle:
                group_order = torch.randperm(len(starts), generator=generator)
            group_order = group_order[
                self.partition_index :: self.partition_count
            ].tolist()
            for group_index in group_order:
                count = int(counts[group_index])
                offsets = torch.randperm(count, generator=generator)[
                    : self.spectra_per_group
                ].tolist()
                start = int(starts[group_index])
                pending.extend(start + offset for offset in offsets)
                pending_groups += 1
                if pending_groups != self.groups_per_batch:
                    continue
                if batch_index >= self.start_batch:
                    yield pending
                    yielded += 1
                    if self.max_batches is not None and yielded >= self.max_batches:
                        return
                batch_index += 1
                pending = []
                pending_groups = 0
        if pending and not self.drop_last and batch_index >= self.start_batch:
            yield pending

    def __len__(self) -> int:
        groups = sum(
            (len(starts) + self.partition_count - 1 - self.partition_index)
            // self.partition_count
            for starts, _counts in self.group_ranges
        )
        batches = (
            groups // self.groups_per_batch
            if self.drop_last
            else (groups + self.groups_per_batch - 1) // self.groups_per_batch
        )
        batches = max(0, batches - self.start_batch)
        if self.max_batches is not None:
            batches = min(batches, self.max_batches)
        return batches


class GroupedGemsBatchCollator:
    def __init__(self, base: GemsBatchCollator, spectra_per_group: int) -> None:
        self.base = base
        self.spectra_per_group = spectra_per_group

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        batch = cast(dict[str, np.ndarray], self.base(samples))
        groups = len(samples) // self.spectra_per_group
        return {
            key: value.reshape(groups, self.spectra_per_group, *value.shape[1:])
            for key, value in batch.items()
        }


class GroupedGemsDataModule(GemsDataModule):
    def __init__(
        self,
        config: config_dict.ConfigDict,
        seed: int,
        *,
        distributed_world_size: int = 1,
        distributed_rank: int = 0,
        distributed_local_rank: int | None = None,
    ) -> None:
        self.spectra_per_group = int(config.group_jepa_spectra_per_group)
        self.teacher_spectra_per_group = int(
            config.group_jepa_teacher_spectra_per_group
        )
        self.groups_per_batch = int(config.group_jepa_groups_per_batch)
        assert self.teacher_spectra_per_group < self.spectra_per_group
        assert self.groups_per_batch % (
            distributed_world_size * int(config.gradient_accumulation_steps)
        ) == 0

        data_config = config_dict.ConfigDict(config.to_dict())
        data_config.batch_size = self.groups_per_batch * self.spectra_per_group
        data_config.training_max_steps = None
        super().__init__(
            data_config,
            seed,
            distributed_world_size=distributed_world_size,
            distributed_rank=distributed_rank,
            distributed_local_rank=distributed_local_rank,
        )
        assert isinstance(self._datasets["train"], MassiveV2Hdf5ShardDataset)
        assert isinstance(self._datasets["validation"], MassiveV2Hdf5ShardDataset)
        self.global_spectra_batch_size = self.global_batch_size
        self.global_batch_size = self.groups_per_batch
        self.batch_size = self.groups_per_batch // (
            distributed_world_size * self.gradient_accumulation_steps
        )
        self._group_ranges = {
            split: self._scan_group_ranges(split)
            for split in ("train", "validation")
        }
        local_micro_batches = len(
            GroupedBatchSampler(
                self._group_ranges["train"],
                groups_per_batch=self.batch_size,
                spectra_per_group=self.spectra_per_group,
                shuffle=False,
                seed=self.seed,
                drop_last=True,
                epoch=0,
            )
        )
        local_steps = local_micro_batches // self.gradient_accumulation_steps
        if jax.process_count() > 1:
            process_steps = multihost_utils.process_allgather(
                np.asarray(local_steps, dtype=np.int64)
            )
            local_steps = int(np.asarray(process_steps).min())
        self.train_steps = local_steps
        self.info.update(
            {
                "train_groups": sum(
                    len(starts) for starts, _counts in self._group_ranges["train"]
                ),
                "validation_groups": sum(
                    len(starts)
                    for starts, _counts in self._group_ranges["validation"]
                ),
                "spectra_per_group": self.spectra_per_group,
                "teacher_spectra_per_group": self.teacher_spectra_per_group,
            }
        )
        self._val_loader = None

    def _scan_group_ranges(
        self,
        split: str,
    ) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        dataset = self._datasets[split]
        assert isinstance(dataset, MassiveV2Hdf5ShardDataset)
        dataset.wait_for_prefetch()
        ranges = tuple(
            dataset.grouped_logical_ranges(
                shard_id,
                minimum_size=self.spectra_per_group,
            )
            for shard_id in range(len(dataset.states))
        )
        dataset.close()
        return ranges

    def _grouped_collator(self) -> GroupedGemsBatchCollator:
        return GroupedGemsBatchCollator(
            super()._collator(augment=False),
            self.spectra_per_group,
        )

    def _grouped_loader(
        self,
        *,
        split: str,
        shuffle: bool,
        seed: int,
        drop_last: bool,
        epoch: int,
        start_batch: int = 0,
        max_batches: int | None = None,
    ) -> DataLoader:
        partition_count = 1
        partition_index = 0
        if split == "validation":
            partition_count, partition_index = (
                self.artifact.validation_sampler_partition(self.distributed_rank)
            )
        sampler = GroupedBatchSampler(
            self._group_ranges[split],
            groups_per_batch=self.batch_size,
            spectra_per_group=self.spectra_per_group,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
            epoch=epoch,
            start_batch=start_batch,
            max_batches=max_batches,
            partition_count=partition_count,
            partition_index=partition_index,
        )
        dataset = self._datasets[split]
        loader_kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_sampler": sampler,
            "num_workers": self.dataloader_num_workers,
            "pin_memory": False,
            "collate_fn": self._grouped_collator(),
        }
        if self.dataloader_num_workers > 0:
            loader_kwargs["persistent_workers"] = False
            loader_kwargs["prefetch_factor"] = self.dataloader_prefetch_factor
            if self.dataloader_multiprocessing_context:
                loader_kwargs["multiprocessing_context"] = (
                    self.dataloader_multiprocessing_context
                )
        return DataLoader(**loader_kwargs)

    def train_loader_for_epoch(self, epoch: int, start_batch: int = 0) -> DataLoader:
        remaining_steps = max(0, self.train_steps - start_batch)
        return self._grouped_loader(
            split="train",
            shuffle=True,
            seed=self.seed,
            drop_last=True,
            epoch=epoch,
            start_batch=start_batch * self.gradient_accumulation_steps,
            max_batches=remaining_steps * self.gradient_accumulation_steps,
        )

    def val_loader_for_eval(self, *, augment: bool) -> DataLoader:
        del augment
        return self._grouped_loader(
            split="validation",
            shuffle=False,
            seed=42,
            drop_last=True,
            epoch=0,
        )
