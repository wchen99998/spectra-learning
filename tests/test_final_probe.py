import unittest

import torch
import torch.nn.functional as F

from train import _FinalAttentiveProbe, _precursor_mz_to_bins


class PrecursorBinningTests(unittest.TestCase):
    def test_precursor_mz_bins_are_clamped_to_valid_range(self):
        precursor_norm = torch.tensor([0.0, 0.0002, 0.9999, 1.0, 1.2, -0.1], dtype=torch.float32)
        bins = _precursor_mz_to_bins(
            precursor_norm,
            num_bins=1000,
            max_mz=1000.0,
        )
        expected = torch.tensor([0, 0, 999, 999, 999, 0], dtype=torch.long)
        self.assertTrue(torch.equal(bins, expected))

    def test_precursor_mz_bins_respect_num_bins(self):
        precursor_norm = torch.tensor([0.0, 0.25, 0.5, 0.999, 1.0], dtype=torch.float32)
        bins = _precursor_mz_to_bins(
            precursor_norm,
            num_bins=10,
            max_mz=1000.0,
        )
        expected = torch.tensor([0, 2, 5, 9, 9], dtype=torch.long)
        self.assertTrue(torch.equal(bins, expected))


class FinalAttentiveProbeTests(unittest.TestCase):
    def test_output_shapes_match_target_spaces(self):
        probe = _FinalAttentiveProbe(
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
        probe = _FinalAttentiveProbe(
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


if __name__ == "__main__":
    unittest.main()
