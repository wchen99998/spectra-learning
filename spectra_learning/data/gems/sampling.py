import math
from collections.abc import Iterator, Sized
from typing import cast

import numpy as np
from torch.utils.data import Sampler

from spectra_learning.data.gems.randomness import data_seed


class ChunkedDistributedBatchSampler(Sampler[list[int]]):
    """Shuffle all rows in bounded windows of randomly ordered storage blocks.

    Block boundaries and permutations never depend on output batch size. Only
    the final epoch tail can be dropped; every window is drained completely.
    """

    def __init__(
        self,
        segments: list[tuple[int, int, int]],
        *,
        batch_size: int,
        rows_per_block: int | None,
        shuffle: bool,
        seed: int,
        drop_last: bool,
        world_size: int,
        rank: int,
        partition_batches: bool = False,
        shuffle_window_blocks: int = 2048,
    ) -> None:
        if batch_size < 1 or shuffle_window_blocks < 1:
            raise ValueError("batch_size and shuffle_window_blocks must be positive")
        if not 0 <= rank < world_size:
            raise ValueError("rank must be between zero and world_size - 1")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.world_size = world_size
        self.rank = rank
        self.partition_batches = partition_batches
        self.shuffle_window_blocks = shuffle_window_blocks
        self.epoch = 0
        blocks = []
        for start, length, chunk_rows in segments:
            block_rows = max(chunk_rows, rows_per_block or chunk_rows)
            stop = start + length
            blocks.extend(
                (row, min(row + block_rows, stop))
                for row in range(start, stop, block_rows)
            )
        self.blocks = np.asarray(
            blocks if partition_batches else blocks[rank::world_size],
            dtype=np.int64,
        ).reshape(-1, 2)
        self.row_count = int(np.sum(self.blocks[:, 1] - self.blocks[:, 0]))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(
            data_seed(
                self.seed,
                self.epoch,
                "rows",
                0 if self.partition_batches else self.rank,
            )
        )
        order = (
            rng.permutation(len(self.blocks))
            if self.shuffle
            else np.arange(len(self.blocks))
        )
        pending = np.empty(0, dtype=np.int64)
        batch_index = 0
        for offset in range(0, len(order), self.shuffle_window_blocks):
            window = order[offset : offset + self.shuffle_window_blocks]
            rows = np.concatenate(
                [np.arange(*self.blocks[i], dtype=np.int64) for i in window]
            )
            if self.shuffle:
                rng.shuffle(rows)
            if len(pending):
                rows = np.concatenate((pending, rows))
            full_stop = len(rows) // self.batch_size * self.batch_size
            for start in range(0, full_stop, self.batch_size):
                if (
                    not self.partition_batches
                    or batch_index % self.world_size == self.rank
                ):
                    yield rows[start : start + self.batch_size].tolist()
                batch_index += 1
            pending = rows[full_stop:]
        if len(pending) and not self.drop_last:
            if not self.partition_batches or batch_index % self.world_size == self.rank:
                yield pending.tolist()

    def __len__(self) -> int:
        batches = (
            self.row_count // self.batch_size
            if self.drop_last
            else math.ceil(self.row_count / self.batch_size)
        )
        if self.partition_batches:
            return max(
                0, (batches + self.world_size - 1 - self.rank) // self.world_size
            )
        return batches


class OffsetBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        sampler: Sampler[list[int]],
        *,
        start_index: int,
        batch_size: int,
        drop_last: bool,
    ) -> None:
        self.sampler = sampler
        self.start_index = start_index
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[list[int]]:
        remaining = self.start_index
        pending: list[int] = []
        for batch in self.sampler:
            if remaining >= len(batch):
                remaining -= len(batch)
                continue
            if remaining:
                batch = batch[remaining:]
                remaining = 0
            for index in batch:
                pending.append(index)
                if len(pending) == self.batch_size:
                    yield pending
                    pending = []
        if pending and not self.drop_last:
            yield pending

    def __len__(self) -> int:
        remaining = max(
            0,
            sum(len(batch) for batch in self.sampler) - self.start_index,
        )
        if self.drop_last:
            return remaining // self.batch_size
        return math.ceil(remaining / self.batch_size)


class LimitBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        sampler: Sampler[list[int]],
        *,
        max_batches: int,
    ) -> None:
        self.sampler = sampler
        self.max_batches = max_batches

    def __iter__(self) -> Iterator[list[int]]:
        for position, batch in enumerate(self.sampler):
            if position >= self.max_batches:
                break
            yield batch

    def __len__(self) -> int:
        return min(len(cast(Sized, self.sampler)), self.max_batches)
