"""Lightning-friendly wrapper around the TF input pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import lightning as L
from ml_collections import config_dict
from torch.utils.data import DataLoader, IterableDataset

import input_pipeline_gems_set


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
    """Wraps a persistent TF numpy iterator and yields torch batches."""

    def __init__(self, iterator, steps_per_epoch: int):
        super().__init__()
        self._iterator = iterator
        self.steps_per_epoch = int(steps_per_epoch)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            yield numpy_batch_to_torch(next(self._iterator))


class TfLightningDataModule(L.LightningDataModule):
    """Minimal DataModule wrapping the TF gems_set pipeline with persistent iterators."""

    def __init__(self, config: config_dict.ConfigDict, seed: int):
        super().__init__()
        self.config = config
        self.seed = int(seed)
        
        # Create TF datasets once - they already use repeat() for infinite iteration
        train_iter, eval_iters, info = input_pipeline_gems_set.create_gems_set_datasets(
            config, seed
        )
        self._train_iter = train_iter
        self._eval_iters = eval_iters
        self.info = info
        self.eval_splits = list(eval_iters.keys())
        self.train_steps = _resolve_train_steps(config, info)
        self.eval_steps = {
            split: _resolve_eval_steps(config, info, split)
            for split in self.eval_splits
        }

    def _make_loader(self, iterator, steps: int) -> DataLoader:
        dataset = _TfIteratorDataset(iterator, steps)
        return DataLoader(
            dataset,
            batch_size=None,
            num_workers=0,
            collate_fn=_identity_collate,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self._train_iter, self.train_steps)

    def val_dataloader(self):
        loaders = [
            self._make_loader(self._eval_iters[split], self.eval_steps[split])
            for split in self.eval_splits
        ]
        if len(loaders) == 1:
            return loaders[0]
        return loaders


def create_lightning_dataloaders(
    config: config_dict.ConfigDict, seed: int
) -> tuple[DataLoader, Any, dict[str, Any]]:
    """Convenience helper mirroring input_pipeline_gems_set.create_gems_set_datasets."""
    module = TfLightningDataModule(config, seed)
    return module.train_dataloader(), module.val_dataloader(), module.info
