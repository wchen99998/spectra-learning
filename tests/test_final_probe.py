import unittest

import numpy as np
import torch
import torch.nn.functional as F

from utils.probing import (
    FinalAttentiveProbe,
    iter_massspec_probe,
    precursor_mz_to_bins,
    probe_steps_per_epoch,
)


class PrecursorBinningTests(unittest.TestCase):
    def test_precursor_mz_bins_are_clamped_to_valid_range(self):
        precursor_norm = torch.tensor([0.0, 0.0002, 0.9999, 1.0, 1.2, -0.1], dtype=torch.float32)
        bins = precursor_mz_to_bins(
            precursor_norm,
            num_bins=1000,
            max_mz=1000.0,
        )
        expected = torch.tensor([0, 0, 999, 999, 999, 0], dtype=torch.long)
        self.assertTrue(torch.equal(bins, expected))

    def test_precursor_mz_bins_respect_num_bins(self):
        precursor_norm = torch.tensor([0.0, 0.25, 0.5, 0.999, 1.0], dtype=torch.float32)
        bins = precursor_mz_to_bins(
            precursor_norm,
            num_bins=10,
            max_mz=1000.0,
        )
        expected = torch.tensor([0, 2, 5, 9, 9], dtype=torch.long)
        self.assertTrue(torch.equal(bins, expected))


class FinalAttentiveProbeTests(unittest.TestCase):
    def test_output_shapes_match_target_spaces(self):
        probe = FinalAttentiveProbe(
            input_dim=32,
            hidden_dim=64,
            num_attention_heads=4,
            head_dims={"adduct": 13, "precursor_bin": 1000, "instrument": 9},
        )
        features = torch.randn(7, 60, 32)
        feature_mask = torch.ones(7, 60, dtype=torch.bool)
        logits = probe(features, feature_mask)

        self.assertEqual(logits["adduct"].shape, (7, 13))
        self.assertEqual(logits["precursor_bin"].shape, (7, 1000))
        self.assertEqual(logits["instrument"].shape, (7, 9))

    def test_multitask_losses_are_finite(self):
        g = torch.Generator().manual_seed(11)
        probe = FinalAttentiveProbe(
            input_dim=24,
            hidden_dim=48,
            num_attention_heads=4,
            head_dims={"adduct": 5, "precursor_bin": 1000, "instrument": 4},
        )
        features = torch.randn(10, 1, 24, generator=g)
        feature_mask = torch.ones(10, 1, dtype=torch.bool)
        logits = probe(features, feature_mask)
        adduct_targets = torch.randint(0, 5, (10,), generator=g)
        precursor_targets = torch.randint(0, 1000, (10,), generator=g)
        instrument_targets = torch.randint(0, 4, (10,), generator=g)
        total_loss = (
            F.cross_entropy(logits["adduct"], adduct_targets)
            + F.cross_entropy(logits["precursor_bin"], precursor_targets)
            + F.cross_entropy(logits["instrument"], instrument_targets)
        )
        self.assertTrue(torch.isfinite(total_loss).item())


class _DummyDataset:
    def __init__(self, batches):
        self._batches = batches

    def as_numpy_iterator(self):
        return iter(self._batches)


class _DummyDataModule:
    def __init__(self, batches, info, batch_size):
        self._dataset = _DummyDataset(batches)
        self.info = info
        self.batch_size = batch_size
        self.calls = []

    def build_massspec_probe_dataset(
        self,
        split: str,
        seed: int,
        *,
        peak_ordering: str | None = None,
        shuffle: bool = False,
        drop_remainder: bool = True,
    ):
        self.calls.append({
            "split": split,
            "seed": seed,
            "peak_ordering": peak_ordering,
            "shuffle": shuffle,
            "drop_remainder": drop_remainder,
        })
        return self._dataset


class ProbeIterationTests(unittest.TestCase):
    def test_train_probe_uses_shuffle_and_includes_remainder(self):
        batches = [
            {"peak_mz": np.zeros((4, 60), dtype=np.float32)},
            {"peak_mz": np.ones((4, 60), dtype=np.float32)},
        ]
        dm = _DummyDataModule(
            batches=batches,
            info={
                "massspec_train_size": 6,
                "massspec_val_size": 0,
                "massspec_test_size": 0,
            },
            batch_size=4,
        )
        result = list(
            iter_massspec_probe(
                dm,
                "massspec_train",
                seed=123,
                peak_ordering="mz",
                drop_remainder=False,
            )
        )
        self.assertEqual(dm.calls[0]["shuffle"], True)
        self.assertEqual(dm.calls[0]["drop_remainder"], False)
        self.assertEqual(result[0]["peak_mz"].shape[0], 4)
        self.assertEqual(result[1]["peak_mz"].shape[0], 2)

    def test_eval_probe_does_not_shuffle(self):
        batches = [{"peak_mz": np.zeros((2, 60), dtype=np.float32)}]
        dm = _DummyDataModule(
            batches=batches,
            info={
                "massspec_train_size": 0,
                "massspec_val_size": 0,
                "massspec_test_size": 2,
            },
            batch_size=4,
        )
        _ = list(
            iter_massspec_probe(
                dm,
                "massspec_test",
                seed=321,
                peak_ordering="intensity",
                drop_remainder=False,
            )
        )
        self.assertEqual(dm.calls[0]["shuffle"], False)


class ProbeStepCountTests(unittest.TestCase):
    def testprobe_steps_per_epoch_matches_drop_remainder_policy(self):
        dm = _DummyDataModule(
            batches=[],
            info={
                "massspec_train_size": 10,
                "massspec_val_size": 0,
                "massspec_test_size": 0,
            },
            batch_size=4,
        )
        self.assertEqual(
            probe_steps_per_epoch(dm, split="massspec_train", drop_remainder=False),
            3,
        )
        self.assertEqual(
            probe_steps_per_epoch(dm, split="massspec_train", drop_remainder=True),
            2,
        )


if __name__ == "__main__":
    unittest.main()
