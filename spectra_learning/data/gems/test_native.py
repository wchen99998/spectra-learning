import unittest
from itertools import combinations, islice

import torch

import spectra_learning.data.massspec_probe as massspec_probe_data
from spectra_learning.data.gems.sampling import (
    ChunkedDistributedBatchSampler,
    OffsetBatchSampler,
)


class GemsSamplingTests(unittest.TestCase):
    def test_chunked_sampler_partitions_shard_blocks_by_rank(self):
        segments = [(0, 5, 2), (5, 4, 2)]
        rank0 = list(
            ChunkedDistributedBatchSampler(
                segments,
                batch_size=2,
                rows_per_block=4,
                shuffle=False,
                seed=123,
                drop_last=False,
                world_size=2,
                rank=0,
            )
        )
        rank1 = list(
            ChunkedDistributedBatchSampler(
                segments,
                batch_size=2,
                rows_per_block=4,
                shuffle=False,
                seed=123,
                drop_last=False,
                world_size=2,
                rank=1,
            )
        )

        self.assertEqual(rank0, [[0, 1], [2, 3], [5, 6], [7, 8]])
        self.assertEqual(rank1, [[4]])
        combined = [index for batch in rank0 + rank1 for index in batch]
        self.assertEqual(sorted(combined), list(range(9)))

    def test_shuffled_sampler_length_matches_every_rank_and_epoch(self):
        segments = [(0, 1025, 256)]
        for epoch in range(4):
            for rank in range(2):
                sampler = ChunkedDistributedBatchSampler(
                    segments,
                    batch_size=512,
                    rows_per_block=512,
                    shuffle=True,
                    seed=42,
                    drop_last=True,
                    world_size=2,
                    rank=rank,
                )
                sampler.set_epoch(epoch)
                self.assertEqual(len(list(sampler)), len(sampler))

    def test_shuffled_segments_are_consumed_one_shard_at_a_time(self):
        sampler = ChunkedDistributedBatchSampler(
            [(0, 8, 2), (8, 8, 2), (16, 8, 2)],
            batch_size=2,
            rows_per_block=4,
            shuffle=True,
            seed=42,
            drop_last=True,
            world_size=1,
            rank=0,
            shuffle_segments=True,
        )
        shard_order = [batch[0] // 8 for batch in sampler]
        transitions = [
            shard
            for position, shard in enumerate(shard_order)
            if position == 0 or shard != shard_order[position - 1]
        ]
        self.assertEqual(len(transitions), 3)
        self.assertEqual(set(transitions), {0, 1, 2})

    def test_mixed_sampler_stripes_batches_across_active_shards(self):
        segments = [(index * 4096, 4096, 256) for index in range(4)]
        sampler = ChunkedDistributedBatchSampler(
            segments,
            batch_size=256,
            rows_per_block=256,
            shuffle=True,
            seed=66,
            drop_last=True,
            world_size=1,
            rank=0,
            shuffle_segments=True,
            active_segments=2,
            mix_blocks_per_batch=8,
        )
        batches = list(sampler)
        repeated = list(sampler)

        rows = [row for batch in batches for row in batch]
        self.assertEqual(len(batches), len(sampler))
        self.assertEqual(sorted(rows), list(range(4 * 4096)))
        self.assertEqual(len(rows), len(set(rows)))
        self.assertEqual(batches, repeated)
        for batch in batches:
            self.assertEqual(len({row // 256 for row in batch}), 8)
            self.assertEqual(len({row // 4096 for row in batch}), 2)

    def test_mixed_sampler_simulates_disjoint_eight_host_optimizer_batch(self):
        host_batches = []
        host_stride = 1_048_576
        for host in range(8):
            base = host * host_stride
            sampler = ChunkedDistributedBatchSampler(
                [(base, 4096, 256), (base + 4096, 4096, 256)],
                batch_size=256,
                rows_per_block=256,
                shuffle=True,
                seed=66,
                drop_last=True,
                world_size=1,
                rank=0,
                shuffle_segments=True,
                active_segments=2,
                mix_blocks_per_batch=8,
            )
            host_batches.append(list(sampler))

        optimizer_batch = [
            row for batches in host_batches for batch in batches[:2] for row in batch
        ]
        self.assertEqual(len(optimizer_batch), 4096)
        self.assertEqual(len(set(optimizer_batch)), 4096)
        self.assertEqual(
            len({row // host_stride for row in optimizer_batch}),
            8,
        )
        self.assertEqual(len({row // 256 for row in optimizer_batch}), 64)

    def test_rolling_mixer_uses_unique_rows_and_changes_chunk_partners(self):
        sampler = ChunkedDistributedBatchSampler(
            [(index * 1024, 1024, 16) for index in range(4)],
            batch_size=16,
            rows_per_block=16,
            shuffle=True,
            seed=66,
            drop_last=True,
            world_size=1,
            rank=0,
            shuffle_segments=True,
            active_segments=4,
            mix_blocks_per_batch=4,
            mix_pool_blocks=16,
            mix_lanes_per_residency=2,
        )
        batches = list(sampler)
        repeated = list(sampler)

        rows = [row for batch in batches for row in batch]
        self.assertEqual(len(batches), len(sampler))
        self.assertEqual(len(rows), len(set(rows)))
        self.assertEqual(batches, repeated)
        for batch in batches:
            self.assertEqual(len(batch), 16)
            self.assertEqual(len({row // 16 for row in batch}), 4)
        for left, right in zip(batches[::2], batches[1::2], strict=True):
            left_blocks = {row // 16 for row in left}
            right_blocks = {row // 16 for row in right}
            self.assertTrue(left_blocks.isdisjoint(right_blocks))

        first_pair = {row // 16 for batch in batches[:2] for row in batch}
        second_pair = {row // 16 for batch in batches[2:4] for row in batch}
        self.assertNotEqual(first_pair, second_pair)

    def test_rolling_mixer_simulates_eight_host_optimizer_batch(self):
        host_batches = []
        host_stride = 16_777_216
        for host in range(8):
            base = host * host_stride
            sampler = ChunkedDistributedBatchSampler(
                [(base + shard * 262_144, 262_144, 256) for shard in range(4)],
                batch_size=256,
                rows_per_block=256,
                shuffle=True,
                seed=67 + host,
                drop_last=True,
                world_size=1,
                rank=0,
                shuffle_segments=True,
                active_segments=4,
                mix_blocks_per_batch=64,
                mix_pool_blocks=2_048,
                mix_lanes_per_residency=8,
            )
            host_batches.append(list(islice(sampler, 2)))

        optimizer_batch = [
            row for batches in host_batches for batch in batches for row in batch
        ]
        self.assertEqual(len(optimizer_batch), 4_096)
        self.assertEqual(len(set(optimizer_batch)), 4_096)
        self.assertEqual(
            len({row // host_stride for row in optimizer_batch}),
            8,
        )
        self.assertEqual(
            len({row // 256 for row in optimizer_batch}),
            1_024,
        )

    def test_mixed_sampler_offset_matches_full_sequence_suffix(self):
        kwargs = dict(
            segments=[(0, 4096, 256), (4096, 4096, 256)],
            batch_size=256,
            rows_per_block=256,
            shuffle=True,
            seed=66,
            drop_last=True,
            world_size=1,
            rank=0,
            shuffle_segments=True,
            active_segments=2,
            mix_blocks_per_batch=8,
        )
        full = list(ChunkedDistributedBatchSampler(**kwargs))
        offset = list(
            OffsetBatchSampler(
                ChunkedDistributedBatchSampler(**kwargs),
                start_index=13 * 256,
                batch_size=256,
                drop_last=True,
            )
        )
        self.assertEqual(offset, full[13:])

    def test_batch_partition_has_no_overlap_between_shard_peers(self):
        batch_size = 4
        required_batches = 5
        for group_size in (1, 2, 5, 6, 22):
            rows = group_size * required_batches * batch_size + 17
            rank_rows = []
            for rank in range(group_size):
                sampler = ChunkedDistributedBatchSampler(
                    [(0, rows, 2)],
                    batch_size=batch_size,
                    rows_per_block=16,
                    shuffle=True,
                    seed=42,
                    drop_last=True,
                    world_size=group_size,
                    rank=rank,
                    shuffle_segments=True,
                    partition_batches=True,
                )
                batches = list(islice(sampler, required_batches))
                self.assertGreaterEqual(
                    sampler.full_batch_count,
                    required_batches,
                )
                self.assertEqual(len(batches), required_batches)
                rank_rows.append({row for batch in batches for row in batch})

            for left, right in combinations(rank_rows, 2):
                self.assertTrue(left.isdisjoint(right))
            self.assertEqual(
                len(set().union(*rank_rows)),
                group_size * required_batches * batch_size,
            )

    def test_batch_partition_covers_every_row_once(self):
        rank_batches = []
        for rank in range(3):
            sampler = ChunkedDistributedBatchSampler(
                [(0, 5, 2), (5, 4, 2)],
                batch_size=2,
                rows_per_block=4,
                shuffle=True,
                seed=42,
                drop_last=False,
                world_size=3,
                rank=rank,
                shuffle_segments=True,
                partition_batches=True,
            )
            batches = list(sampler)
            self.assertEqual(len(batches), len(sampler))
            rank_batches.append(batches)

        rows = [row for batches in rank_batches for batch in batches for row in batch]
        self.assertEqual(sorted(rows), list(range(9)))


class MassSpecPreprocessTests(unittest.TestCase):
    def test_probe_collator_applies_peak_and_precursor_window_filters(self):
        collator = massspec_probe_data._ProbeBatchCollator(
            num_peaks=8,
            max_precursor_mz=1000.0,
            min_peak_intensity=1e-4,
            peak_drop_min_intensity=0.01,
            peak_ordering="mz",
            precursor_peak_exclusion_window_da=5.0,
        )
        spectra = torch.zeros((2, 128), dtype=torch.float32)
        spectra[0, :5] = torch.tensor([90.0, 95.0, 97.0, 98.5, 150.0])
        spectra[1, :4] = torch.tensor([50.0, 60.0, 194.0, 210.0])
        intensity = torch.zeros((2, 128), dtype=torch.float32)
        intensity[0, :5] = torch.tensor([1.0, 0.009, 0.8, 0.7, 0.5])
        intensity[1, :4] = torch.tensor([0.02, 0.5, 0.8, 0.9])
        batch = collator(
            [
                {
                    "spectra": torch.stack([spectra[0], intensity[0]], dim=0),
                    "precursor_mz_raw": 100.0,
                    "fingerprint": torch.zeros(1024, dtype=torch.int32),
                    "smiles": "CCO",
                    "adduct_id": 0,
                    "instrument_type_id": 0,
                    "collision_energy": 0.0,
                    "collision_energy_present": 0,
                    "charge": 1.0,
                    "probe_valid_mol": True,
                    "probe_maccs": torch.zeros(166, dtype=torch.int32),
                    "probe_morgan": torch.zeros(4096, dtype=torch.int32),
                    "probe_mol_weight": 0.0,
                    "probe_logp": 0.0,
                    "probe_num_heavy_atoms": 0.0,
                    "probe_num_rings": 0.0,
                },
                {
                    "spectra": torch.stack([spectra[1], intensity[1]], dim=0),
                    "precursor_mz_raw": 200.0,
                    "fingerprint": torch.zeros(1024, dtype=torch.int32),
                    "smiles": "CCN",
                    "adduct_id": 0,
                    "instrument_type_id": 0,
                    "collision_energy": 0.0,
                    "collision_energy_present": 0,
                    "charge": 1.0,
                    "probe_valid_mol": True,
                    "probe_maccs": torch.zeros(166, dtype=torch.int32),
                    "probe_morgan": torch.zeros(4096, dtype=torch.int32),
                    "probe_mol_weight": 0.0,
                    "probe_logp": 0.0,
                    "probe_num_heavy_atoms": 0.0,
                    "probe_num_rings": 0.0,
                },
            ]
        )

        peak_mz = batch["peak_mz"] * 1000.0
        valid = batch["peak_valid_mask"]
        precursor = batch["precursor_mz"] * 1000.0

        self.assertEqual(int(valid[0].sum().item()), 1)
        self.assertTrue(torch.allclose(peak_mz[0, :1], torch.tensor([90.0])))
        self.assertEqual(int(valid[1].sum().item()), 3)
        self.assertTrue(
            torch.allclose(peak_mz[1, :3], torch.tensor([50.0, 60.0, 194.0]))
        )
        self.assertEqual(
            int((valid & (peak_mz > (precursor.unsqueeze(1) - 5.0))).sum().item()),
            0,
        )
        self.assertGreaterEqual(
            float(batch["peak_intensity"][valid].min().item()),
            0.01 - 1e-6,
        )

    def test_probe_collator_keeps_num_peaks(self):
        collator = massspec_probe_data._ProbeBatchCollator(
            num_peaks=4,
            max_precursor_mz=1000.0,
            min_peak_intensity=1e-4,
            peak_drop_min_intensity=1e-4,
            peak_ordering="mz",
            precursor_peak_exclusion_window_da=0.0,
        )
        spectra = torch.zeros((2, 128), dtype=torch.float32)
        spectra[0, :6] = torch.tensor([90.0, 100.0, 110.0, 120.0, 130.0, 140.0])
        spectra[1, :6] = torch.tensor([190.0, 200.0, 210.0, 220.0, 230.0, 240.0])
        intensity = torch.zeros((2, 128), dtype=torch.float32)
        intensity[:, :6] = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
        samples = []
        for idx, precursor_mz in enumerate((150.0, 250.0)):
            samples.append(
                {
                    "spectra": torch.stack([spectra[idx], intensity[idx]], dim=0),
                    "precursor_mz_raw": precursor_mz,
                    "fingerprint": torch.zeros(1024, dtype=torch.int32),
                    "smiles": "CCO",
                    "adduct_id": 0,
                    "instrument_type_id": 0,
                    "collision_energy": 0.0,
                    "collision_energy_present": 0,
                    "charge": 1.0,
                    "probe_valid_mol": True,
                    "probe_maccs": torch.zeros(166, dtype=torch.int32),
                    "probe_morgan": torch.zeros(4096, dtype=torch.int32),
                    "probe_mol_weight": 0.0,
                    "probe_logp": 0.0,
                    "probe_num_heavy_atoms": 0.0,
                    "probe_num_rings": 0.0,
                }
            )

        batch = collator(samples)

        self.assertEqual(tuple(batch["peak_mz"].shape), (2, 4))
        self.assertEqual(batch["peak_valid_mask"].sum().item(), 8)


if __name__ == "__main__":
    unittest.main()
