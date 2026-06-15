from __future__ import annotations

import torch
from torch.utils.data import Dataset, Subset
from torch.utils.data.distributed import DistributedSampler


def subset_for_max_samples(
    dataset: Dataset,
    *,
    max_samples: int | None,
    shuffle: bool,
    seed: int,
) -> tuple[Dataset, bool]:
    if max_samples is None:
        return dataset, shuffle
    n = min(len(dataset), max_samples)
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=generator)[:n].tolist()
    else:
        indices = list(range(n))
    if hasattr(dataset, "subset"):
        return dataset.subset([int(idx) for idx in indices]), False
    return Subset(dataset, [int(idx) for idx in indices]), False


def loader_sampler(
    dataset: Dataset,
    *,
    shuffle: bool,
    seed: int,
    drop_last: bool,
    distributed_world_size: int,
    distributed_rank: int,
) -> tuple[DistributedSampler | None, bool]:
    if distributed_world_size <= 1:
        return None, shuffle
    return (
        DistributedSampler(
            dataset,
            num_replicas=distributed_world_size,
            rank=distributed_rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        ),
        False,
    )


def local_batch_size(global_batch_size: int, distributed_world_size: int) -> int:
    if distributed_world_size <= 1:
        return global_batch_size
    assert global_batch_size % distributed_world_size == 0
    return global_batch_size // distributed_world_size
