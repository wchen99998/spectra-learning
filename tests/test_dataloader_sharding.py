"""Verify _TfIterableDataset correctly shards across DataLoader workers."""

import unittest

import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import DataLoader

from input_pipeline import _TfIterableDataset, numpy_batch_to_torch, _identity_collate


def _make_counting_dataset(n: int) -> tf.data.Dataset:
    """Return an unbatched tf.data.Dataset of integers 0..n-1, then batched by 1."""
    ds = tf.data.Dataset.range(n)
    ds = ds.map(lambda x: {"idx": x})
    ds = ds.batch(1)
    return ds


class DataLoaderShardingTests(unittest.TestCase):
    def _collect_indices(self, num_workers: int, n: int = 20) -> list[int]:
        ds = _make_counting_dataset(n)
        iterable = _TfIterableDataset(dataset=ds, steps_per_epoch=n)
        loader = DataLoader(
            iterable,
            batch_size=None,
            num_workers=num_workers,
            collate_fn=_identity_collate,
        )
        indices = []
        for batch in loader:
            indices.append(int(batch["idx"].item()))
        return sorted(indices)

    def test_single_worker_yields_all(self):
        indices = self._collect_indices(num_workers=0, n=20)
        self.assertEqual(indices, list(range(20)))

    def test_multi_worker_no_duplicates(self):
        indices = self._collect_indices(num_workers=2, n=20)
        self.assertEqual(len(indices), len(set(indices)), "Found duplicate indices")

    def test_multi_worker_covers_all(self):
        indices = self._collect_indices(num_workers=2, n=20)
        self.assertEqual(indices, list(range(20)))

    def test_four_workers_no_duplicates(self):
        indices = self._collect_indices(num_workers=4, n=40)
        self.assertEqual(len(indices), len(set(indices)), "Found duplicate indices")
        self.assertEqual(indices, list(range(40)))


if __name__ == "__main__":
    unittest.main()
