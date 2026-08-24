import math
from collections import deque
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
        shuffle_segments: bool = False,
        partition_batches: bool = False,
        active_segments: int = 1,
        mix_blocks_per_batch: int = 1,
    ) -> None:
        self.batch_size = batch_size
        self.rows_per_block = rows_per_block
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.world_size = world_size
        self.rank = rank
        self.shuffle_segments = shuffle_segments
        self.partition_batches = partition_batches
        self.active_segments = active_segments
        self.mix_blocks_per_batch = mix_blocks_per_batch
        assert self.active_segments >= 1
        assert self.mix_blocks_per_batch >= 1
        assert self.batch_size % self.mix_blocks_per_batch == 0
        self.segment_blocks = [
            self._build_blocks([segment]) for segment in segments
        ]
        self.blocks = [
            block for blocks in self.segment_blocks for block in blocks
        ]
        self.epoch = 0

    def _build_blocks(
        self,
        segments: list[tuple[int, int, int]],
    ) -> list[tuple[int, int]]:
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

    def segment_order(self) -> list[int]:
        if self.shuffle and self.shuffle_segments:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            return torch.randperm(
                len(self.segment_blocks),
                generator=generator,
            ).tolist()
        return list(range(len(self.segment_blocks)))

    def __iter__(self) -> Iterator[list[int]]:
        if self.shuffle and self.mix_blocks_per_batch > 1:
            for batch_index, batch in enumerate(self._mixed_batches()):
                if (
                    not self.partition_batches
                    or batch_index % self.world_size == self.rank
                ):
                    yield batch
            return

        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        if self.shuffle and self.shuffle_segments:
            segment_order = self.segment_order()
            torch.randperm(len(self.segment_blocks), generator=generator)
            blocks = []
            for segment_index in segment_order:
                segment_blocks = list(self.segment_blocks[segment_index])
                block_order = torch.randperm(
                    len(segment_blocks),
                    generator=generator,
                ).tolist()
                blocks.extend(
                    segment_blocks[index] for index in block_order
                )
        else:
            blocks = list(
                self.blocks
                if self.partition_batches
                else self.blocks[self.rank :: self.world_size]
            )
        if self.shuffle and not self.shuffle_segments:
            order = torch.randperm(len(blocks), generator=generator).tolist()
            blocks = [blocks[index] for index in order]

        partial_batches: list[list[int]] = []
        batch_index = 0
        for block_start, block_stop in blocks:
            rows = list(range(block_start, block_stop))
            if self.shuffle:
                order = torch.randperm(len(rows), generator=generator).tolist()
                rows = [rows[index] for index in order]
            for offset in range(0, len(rows), self.batch_size):
                batch = rows[offset : offset + self.batch_size]
                if len(batch) == self.batch_size:
                    if (
                        not self.partition_batches
                        or batch_index % self.world_size == self.rank
                    ):
                        yield batch
                    batch_index += 1
                elif not self.drop_last:
                    partial_batches.append(batch)
        for batch in partial_batches:
            if (
                not self.partition_batches
                or batch_index % self.world_size == self.rank
            ):
                yield batch
            batch_index += 1

    def _mixed_batches(self) -> Iterator[list[int]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        segment_order = self.segment_order()
        torch.randperm(len(self.segment_blocks), generator=generator)
        partial_batches: list[list[int]] = []
        for window_start in range(0, len(segment_order), self.active_segments):
            window = segment_order[
                window_start : window_start + self.active_segments
            ]
            queues = []
            for segment_index in window:
                blocks = self.segment_blocks[segment_index]
                block_order = torch.randperm(
                    len(blocks),
                    generator=generator,
                ).tolist()
                units: list[tuple[int, int]] = []
                for block_index in block_order:
                    block_start, block_stop = blocks[block_index]
                    full_stop = block_start + (
                        (block_stop - block_start) // self.batch_size
                    ) * self.batch_size
                    units.extend(
                        (start, start + self.batch_size)
                        for start in range(block_start, full_stop, self.batch_size)
                    )
                    if not self.drop_last and full_stop < block_stop:
                        partial_batches.append(list(range(full_stop, block_stop)))
                queues.append(deque(units))

            while any(queues):
                group: list[tuple[int, int]] = []
                queue_order = torch.randperm(
                    len(queues),
                    generator=generator,
                ).tolist()
                while len(group) < self.mix_blocks_per_batch and any(queues):
                    for queue_index in queue_order:
                        if queues[queue_index]:
                            group.append(queues[queue_index].popleft())
                            if len(group) == self.mix_blocks_per_batch:
                                break
                if len(group) == self.mix_blocks_per_batch:
                    yield from self._striped_batches(group, generator)
                else:
                    for start, stop in group:
                        rows = list(range(start, stop))
                        order = torch.randperm(
                            len(rows),
                            generator=generator,
                        ).tolist()
                        yield [rows[index] for index in order]

        for batch in partial_batches:
            order = torch.randperm(len(batch), generator=generator).tolist()
            yield [batch[index] for index in order]

    def _striped_batches(
        self,
        group: list[tuple[int, int]],
        generator: torch.Generator,
    ) -> Iterator[list[int]]:
        lane_rows = self.batch_size // self.mix_blocks_per_batch
        lane_orders = [
            torch.randperm(
                self.mix_blocks_per_batch,
                generator=generator,
            ).tolist()
            for _ in group
        ]
        for output_index in range(self.mix_blocks_per_batch):
            batch = []
            for (start, _stop), lanes in zip(group, lane_orders, strict=True):
                lane_start = start + lanes[output_index] * lane_rows
                batch.extend(range(lane_start, lane_start + lane_rows))
            order = torch.randperm(len(batch), generator=generator).tolist()
            yield [batch[index] for index in order]

    def _partitioned_batch_count(self, total: int) -> int:
        return max(
            0,
            (total + self.world_size - 1 - self.rank) // self.world_size,
        )

    @property
    def full_batch_count(self) -> int:
        total = sum(
            (block_stop - block_start) // self.batch_size
            for block_start, block_stop in (
                self.blocks
                if self.partition_batches
                else self.blocks[self.rank :: self.world_size]
            )
        )
        if self.partition_batches:
            return self._partitioned_batch_count(total)
        return total

    def __len__(self) -> int:
        blocks = (
            self.blocks
            if self.partition_batches
            else self.blocks[self.rank :: self.world_size]
        )
        batches = 0
        for block_start, block_stop in blocks:
            block_len = block_stop - block_start
            if self.drop_last:
                batches += block_len // self.batch_size
            else:
                batches += math.ceil(block_len / self.batch_size)
        if self.partition_batches:
            return self._partitioned_batch_count(batches)
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
