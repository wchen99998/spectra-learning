"""Lightning-friendly wrapper around the TF input pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import lightning as L
from ml_collections import config_dict
from torch.utils.data import DataLoader, IterableDataset

import input_pipeline


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
        return torch.as_tensor(value, dtype=_torch_dtype(value))
    return value


def numpy_batch_to_torch(batch: dict[str, Any]) -> dict[str, Any]:
    return {key: _to_torch(value) for key, value in batch.items()}


def _identity_collate(batch: dict[str, Any]) -> dict[str, Any]:
    return batch


def _resolve_train_steps(config: config_dict.ConfigDict, info: dict[str, Any]) -> int:
    if config.get("steps_per_epoch", 0) > 0:
        return int(config.steps_per_epoch)
    if config.get("num_epochs", 0) > 0:
        return int(config.num_train_steps // config.num_epochs)
    if config.get("num_train_steps", 0) > 0:
        return int(config.num_train_steps)
    return int(info["train_size"] // config.batch_size)


def _resolve_eval_steps(
    config: config_dict.ConfigDict, info: dict[str, Any], split: str
) -> int:
    if config.get("num_eval_steps", 0) > 0:
        return int(config.num_eval_steps)
    size_key = "validation_size" if split == "validation" else "massspec_test_size"
    return int(info[size_key] // config.batch_size)


class _TfIteratorDataset(IterableDataset):
    """Rebuilds TF iterators per epoch and yields torch batches."""

    def __init__(
        self,
        config: config_dict.ConfigDict,
        *,
        seed: int,
        split: str,
        steps_per_epoch: int,
    ):
        super().__init__()
        self.config = config
        self.seed = int(seed)
        self.split = split
        self.steps_per_epoch = int(steps_per_epoch)
        self._epoch = 0

    def __len__(self) -> int:
        return self.steps_per_epoch

    def _make_iterator(self, epoch: int):
        epoch_seed = self.seed + epoch
        train_iter, eval_iters, _ = input_pipeline.create_datasets(
            self.config, epoch_seed
        )
        if self.split == "train":
            return train_iter
        return eval_iters[self.split]

    def __iter__(self):
        epoch = self._epoch
        self._epoch += 1
        iterator = self._make_iterator(epoch)
        for _ in range(self.steps_per_epoch):
            yield numpy_batch_to_torch(next(iterator))


def _init_info(
    config: config_dict.ConfigDict, seed: int
) -> tuple[dict[str, Any], list[str]]:
    _, eval_iters, info = input_pipeline.create_datasets(config, seed)
    return info, list(eval_iters.keys())


class TfLightningDataModule(L.LightningDataModule):
    """Minimal DataModule-style wrapper that returns Lightning-ready dataloaders."""

    def __init__(self, config: config_dict.ConfigDict, seed: int):
        super().__init__()
        self.config = config
        self.seed = int(seed)
        self.info, self.eval_splits = _init_info(config, self.seed)
        self.train_steps = _resolve_train_steps(config, self.info)
        self.eval_steps = {
            split: _resolve_eval_steps(config, self.info, split)
            for split in self.eval_splits
        }

    def _make_loader(self, split: str, steps: int) -> DataLoader:
        dataset = _TfIteratorDataset(
            self.config,
            seed=self.seed,
            split=split,
            steps_per_epoch=steps,
        )
        return DataLoader(
            dataset,
            batch_size=None,
            num_workers=0,
            collate_fn=_identity_collate,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader("train", self.train_steps)

    def val_dataloader(self):
        loaders = [
            self._make_loader(split, self.eval_steps[split])
            for split in self.eval_splits
        ]
        if len(loaders) == 1:
            return loaders[0]
        return loaders


def create_lightning_dataloaders(
    config: config_dict.ConfigDict, seed: int
) -> tuple[DataLoader, Any, dict[str, Any]]:
    """Convenience helper mirroring input_pipeline.create_datasets."""
    module = TfLightningDataModule(config, seed)
    return module.train_dataloader(), module.val_dataloader(), module.info
