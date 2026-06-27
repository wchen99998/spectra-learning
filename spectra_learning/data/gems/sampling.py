import math
from collections.abc import Iterator, Sized
from typing import cast

import torch
from torch.utils.data import Sampler


class ChunkedDistributedBatchSampler(Sampler[list[int]]):
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
    ) -> None:
        self.batch_size = batch_size
        self.rows_per_block = rows_per_block
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.world_size = world_size
        self.rank = rank
        self.blocks = self._build_blocks(segments)
        self.epoch = 0

    def _build_blocks(self, segments: list[tuple[int, int, int]]) -> list[tuple[int, int]]:
        blocks: list[tuple[int, int]] = []
        for start, length, chunk_rows in segments:
            block_rows = self.rows_per_block
            if block_rows is None:
                block_rows = math.ceil(self.batch_size / chunk_rows) * chunk_rows
            block_rows = max(chunk_rows, block_rows)
            stop = start + length
            for block_start in range(start, stop, block_rows):
                blocks.append((block_start, min(block_start + block_rows, stop)))
        return blocks

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        blocks = list(self.blocks)
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        if self.shuffle:
            order = torch.randperm(len(blocks), generator=generator).tolist()
            blocks = [blocks[index] for index in order]
        blocks = blocks[self.rank :: self.world_size]

        for block_start, block_stop in blocks:
            rows = list(range(block_start, block_stop))
            if self.shuffle:
                order = torch.randperm(len(rows), generator=generator).tolist()
                rows = [rows[index] for index in order]
            for offset in range(0, len(rows), self.batch_size):
                batch = rows[offset : offset + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch

    def __len__(self) -> int:
        batches = 0
        for block_start, block_stop in self.blocks[self.rank :: self.world_size]:
            block_len = block_stop - block_start
            if self.drop_last:
                batches += block_len // self.batch_size
            else:
                batches += math.ceil(block_len / self.batch_size)
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
