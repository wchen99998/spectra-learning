import unittest

import tensorflow as tf
import torch

from input_pipeline import (
    _NUM_PEAKS_OUTPUT,
    _compact_sort_peaks,
    _convert_to_neutral_loss,
)
from models.losses import (
    BCSLoss,
    _epps_pulley,
    squared_prediction_loss,
)
from models.model import PeakMaskSampler, PeakSetJEPA


def _make_batch(batch_size: int = 4, num_peaks: int = 60) -> dict[str, torch.Tensor]:
    """Create a synthetic batch matching the JEPA data contract."""
    peak_mz = torch.rand(batch_size, num_peaks)
    peak_intensity = torch.rand(batch_size, num_peaks)
    valid_count = num_peaks - 10
    peak_valid_mask = torch.zeros(batch_size, num_peaks, dtype=torch.bool)
    peak_valid_mask[:, :valid_count] = True
    peak_mz[:, valid_count:] = 0.0
    peak_intensity[:, valid_count:] = 0.0
    precursor_mz = torch.rand(batch_size)
    return {
        "peak_mz": peak_mz,
        "peak_intensity": peak_intensity,
        "peak_valid_mask": peak_valid_mask,
        "precursor_mz": precursor_mz,
    }


class DataPipelineContractTests(unittest.TestCase):
    def test_batch_has_required_keys(self):
        batch = _make_batch()
        for key in ("peak_mz", "peak_intensity", "peak_valid_mask", "precursor_mz"):
            self.assertIn(key, batch)

    def test_batch_shapes(self):
        B, N = 4, 60
        batch = _make_batch(batch_size=B, num_peaks=N)
        self.assertEqual(batch["peak_mz"].shape, (B, N))
        self.assertEqual(batch["peak_intensity"].shape, (B, N))
        self.assertEqual(batch["peak_valid_mask"].shape, (B, N))
        self.assertEqual(batch["precursor_mz"].shape, (B,))

    def test_batch_does_not_contain_token_ids(self):
        batch = _make_batch()
        self.assertNotIn("token_ids", batch)
        self.assertNotIn("segment_ids", batch)

    def test_peak_ordering_supports_mz_and_intensity(self):
        example = {
            "mz": tf.constant([50.0, 10.0, 30.0, 0.0], dtype=tf.float32),
            "intensity": tf.constant([0.2, 0.5, 0.9, 0.0], dtype=tf.float32),
        }
        by_mz = _compact_sort_peaks("mz")(dict(example))
        by_intensity = _compact_sort_peaks("intensity")(dict(example))
        self.assertEqual(by_mz["mz"].shape[0], _NUM_PEAKS_OUTPUT)
        self.assertEqual(by_intensity["mz"].shape[0], _NUM_PEAKS_OUTPUT)

    def test_neutral_loss_conversion(self):
        original = {
            "mz": tf.constant([10.0, 70.0, 20.0, 0.0], dtype=tf.float32),
            "intensity": tf.constant([0.2, 0.9, 0.5, 0.0], dtype=tf.float32),
            "precursor_mz": tf.constant(100.0, dtype=tf.float32),
        }
        converted = _convert_to_neutral_loss()(dict(original))
        self.assertAlmostEqual(float(converted["mz"][0]), 90.0, places=1)


class MaskSamplerTests(unittest.TestCase):
    def test_output_shapes(self):
        N, N_ctx, N_tgt = 60, 36, 24
        sampler = PeakMaskSampler(N, N_ctx, N_tgt)
        mask = torch.ones(4, N, dtype=torch.bool)
        ctx_idx, tgt_idx, ctx_valid, tgt_valid = sampler(mask)
        self.assertEqual(ctx_idx.shape, (4, N_ctx))
        self.assertEqual(tgt_idx.shape, (4, N_tgt))
        self.assertEqual(ctx_valid.shape, (4, N_ctx))
        self.assertEqual(tgt_valid.shape, (4, N_tgt))

    def test_no_overlap(self):
        N, N_ctx, N_tgt = 60, 36, 24
        sampler = PeakMaskSampler(N, N_ctx, N_tgt)
        mask = torch.ones(2, N, dtype=torch.bool)
        ctx_idx, tgt_idx, _, _ = sampler(mask)
        for b in range(2):
            ctx_set = set(ctx_idx[b].tolist())
            tgt_set = set(tgt_idx[b].tolist())
            self.assertEqual(len(ctx_set & tgt_set), 0)

    def test_valid_peaks_preferred(self):
        N, N_ctx, N_tgt = 60, 36, 24
        sampler = PeakMaskSampler(N, N_ctx, N_tgt)
        mask = torch.zeros(1, N, dtype=torch.bool)
        mask[0, :40] = True
        ctx_idx, tgt_idx, ctx_valid, tgt_valid = sampler(mask)
        self.assertTrue(ctx_valid.all().item())
        self.assertTrue(tgt_valid[:, :4].all().item())


class JEPAForwardTests(unittest.TestCase):
    def _build_model(self) -> PeakSetJEPA:
        return PeakSetJEPA(
            num_peaks=60,
            model_dim=32,
            encoder_num_layers=1,
            encoder_num_heads=4,
            encoder_num_kv_heads=4,
            predictor_num_layers=1,
            predictor_num_heads=4,
            predictor_num_kv_heads=4,
            attention_mlp_multiple=2.0,
            feature_mlp_hidden_dim=16,
            target_ratio=0.4,
            pred_weight=1.0,
            bcs_num_slices=32,
            bcs_lambda=10.0,
        )

    def test_forward_loss_is_finite(self):
        model = self._build_model()
        batch = _make_batch()
        metrics = model(batch, train=True)
        self.assertTrue(torch.isfinite(metrics["loss"]).item())

    def test_forward_contains_expected_keys(self):
        model = self._build_model()
        batch = _make_batch()
        metrics = model(batch, train=True)
        for key in ("loss", "pred_loss", "bcs_loss", "target_valid_fraction"):
            self.assertIn(key, metrics, f"Missing key: {key}")

    def test_encode_output_shape(self):
        model = self._build_model()
        batch = _make_batch(batch_size=3)
        pooled = model.encode(batch, train=False)
        self.assertEqual(pooled.shape, (3, 32))

    def test_compute_loss_returns_loss_and_metrics(self):
        model = self._build_model()
        batch = _make_batch()
        loss, metrics = model.compute_loss(batch, train=True)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertIn("pred_loss", metrics)


class JEPALossTests(unittest.TestCase):
    def test_epps_pulley_finite(self):
        x = torch.randn(32, 16)
        t = _epps_pulley(x)
        self.assertEqual(t.shape, (16,))
        self.assertTrue(torch.isfinite(t).all().item())

    def test_bcs_loss_finite_and_nonnegative(self):
        bcs = BCSLoss(num_slices=32, lmbd=10.0)
        x = torch.randn(32, 16)
        loss = bcs(x)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreaterEqual(float(loss), 0.0)

    def test_bcs_loss_step_increments(self):
        bcs = BCSLoss(num_slices=32, lmbd=10.0)
        x = torch.randn(32, 16)
        self.assertEqual(bcs.step, 0)
        bcs(x)
        self.assertEqual(bcs.step, 1)
        bcs(x)
        self.assertEqual(bcs.step, 2)

    def test_squared_prediction_loss_with_mask(self):
        pred = torch.randn(2, 10, 8)
        target = torch.randn(2, 10, 8)
        mask = torch.ones(2, 10, dtype=torch.bool)
        mask[:, 8:] = False
        loss = squared_prediction_loss(pred, target, mask)
        self.assertTrue(torch.isfinite(loss).item())

    def test_squared_prediction_loss_without_mask(self):
        pred = torch.randn(2, 10, 8)
        target = torch.randn(2, 10, 8)
        loss = squared_prediction_loss(pred, target)
        self.assertTrue(torch.isfinite(loss).item())


if __name__ == "__main__":
    unittest.main()
