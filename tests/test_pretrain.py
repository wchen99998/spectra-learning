import unittest
import tempfile

import tensorflow as tf
import torch

from utils.training import load_pretrained_weights
from input_pipeline import (
    _NUM_PEAKS_OUTPUT,
    _compact_sort_peaks,
    _convert_to_neutral_loss,
    _filter_min_peak_intensity,
    _normalize_for_jepa,
    _topk_peaks,
)
from models.losses import (
    BCSLoss,
    _epps_pulley,
)
from models.model import PeakSetSIGReg


def _make_batch(batch_size: int = 4, num_peaks: int = 60) -> dict[str, torch.Tensor]:
    """Create a synthetic batch matching the SIGReg peak-set contract."""
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

    def test_compact_sort_orders_by_mz_and_preserves_pairs(self):
        example = {
            "mz": tf.constant([50.0, 10.0, 30.0, 0.0], dtype=tf.float32),
            "intensity": tf.constant([0.2, 0.5, 0.9, 0.0], dtype=tf.float32),
        }
        out = _compact_sort_peaks("mz")(dict(example))
        self.assertTrue(
            bool(tf.reduce_all(tf.abs(out["mz"][:3] - tf.constant([10.0, 30.0, 50.0], dtype=tf.float32)) < 1e-6).numpy()),
        )
        self.assertTrue(
            bool(tf.reduce_all(tf.abs(out["intensity"][:3] - tf.constant([0.5, 0.9, 0.2], dtype=tf.float32)) < 1e-6).numpy()),
        )

    def test_compact_sort_orders_by_intensity_and_preserves_pairs(self):
        example = {
            "mz": tf.constant([50.0, 10.0, 30.0, 0.0], dtype=tf.float32),
            "intensity": tf.constant([0.2, 0.5, 0.9, 0.0], dtype=tf.float32),
        }
        out = _compact_sort_peaks("intensity")(dict(example))
        self.assertTrue(
            bool(tf.reduce_all(tf.abs(out["intensity"][:3] - tf.constant([0.9, 0.5, 0.2], dtype=tf.float32)) < 1e-6).numpy()),
        )
        self.assertTrue(
            bool(tf.reduce_all(tf.abs(out["mz"][:3] - tf.constant([30.0, 10.0, 50.0], dtype=tf.float32)) < 1e-6).numpy()),
        )

    def test_topk_then_mz_sort_keeps_same_peak_pairs(self):
        example = {
            "mz": tf.constant([8.0, 1.0, 5.0, 2.0], dtype=tf.float32),
            "intensity": tf.constant([0.1, 0.9, 0.8, 0.2], dtype=tf.float32),
        }
        topk = _topk_peaks(3)(dict(example))
        out = _compact_sort_peaks("mz")(topk)
        self.assertTrue(
            bool(tf.reduce_all(tf.abs(out["mz"][:3] - tf.constant([1.0, 2.0, 5.0], dtype=tf.float32)) < 1e-6).numpy()),
        )
        self.assertTrue(
            bool(tf.reduce_all(tf.abs(out["intensity"][:3] - tf.constant([0.9, 0.2, 0.8], dtype=tf.float32)) < 1e-6).numpy()),
        )

    def test_neutral_loss_conversion(self):
        original = {
            "mz": tf.constant([10.0, 70.0, 20.0, 0.0], dtype=tf.float32),
            "intensity": tf.constant([0.2, 0.9, 0.5, 0.0], dtype=tf.float32),
            "precursor_mz": tf.constant(100.0, dtype=tf.float32),
        }
        converted = _convert_to_neutral_loss()(dict(original))
        self.assertAlmostEqual(float(converted["mz"][0]), 90.0, places=1)

    def test_neutral_loss_padding_is_not_marked_valid(self):
        example = {
            "mz": tf.constant([10.0, 0.0, 0.0], dtype=tf.float32),
            "intensity": tf.constant([0.7, 0.0, 0.0], dtype=tf.float32),
            "precursor_mz": tf.constant(100.0, dtype=tf.float32),
        }
        converted = _convert_to_neutral_loss()(dict(example))
        compacted = _compact_sort_peaks("mz")(converted)
        normalized = _normalize_for_jepa(1000.0)(compacted)
        mask = normalized["peak_valid_mask"].numpy()
        self.assertTrue(mask[0])
        self.assertFalse(mask[1])
        self.assertFalse(mask[2])

    def test_filter_min_peak_intensity_zeros_peaks_below_threshold(self):
        example = {
            "mz": tf.constant([50.0, 75.0, 100.0], dtype=tf.float32),
            "intensity": tf.constant([0.2, 0.01, 0.5], dtype=tf.float32),
        }
        out = _filter_min_peak_intensity(0.1)(dict(example))
        self.assertTrue(
            bool(
                tf.reduce_all(
                    tf.abs(out["mz"] - tf.constant([50.0, 0.0, 100.0], dtype=tf.float32)) < 1e-6,
                ).numpy(),
            ),
        )
        self.assertTrue(
            bool(
                tf.reduce_all(
                    tf.abs(out["intensity"] - tf.constant([0.2, 0.0, 0.5], dtype=tf.float32)) < 1e-6,
                ).numpy(),
            ),
        )

    def test_normalize_for_jepa_keeps_intensity_and_uses_fixed_precursor_scale(self):
        example = {
            "mz": tf.constant([100.0, 50.0, 0.0], dtype=tf.float32),
            "intensity": tf.constant([0.6, 0.02, 0.0], dtype=tf.float32),
            "precursor_mz": tf.constant(1200.0, dtype=tf.float32),
        }
        out = _normalize_for_jepa(1000.0)(dict(example))
        self.assertTrue(
            bool(
                tf.reduce_all(
                    tf.abs(out["peak_intensity"] - tf.constant([0.6, 0.02, 0.0], dtype=tf.float32)) < 1e-6,
                ).numpy(),
            ),
        )
        self.assertAlmostEqual(float(out["precursor_mz"].numpy()), 1.0, places=6)


class SIGRegForwardTests(unittest.TestCase):
    def _build_model(self, *, encoder_use_rope: bool = False) -> PeakSetSIGReg:
        return PeakSetSIGReg(
            num_peaks=60,
            model_dim=32,
            encoder_num_layers=1,
            encoder_num_heads=4,
            encoder_num_kv_heads=4,
            attention_mlp_multiple=2.0,
            feature_mlp_hidden_dim=16,
            sigreg_use_projector=True,
            sigreg_proj_hidden_dim=64,
            sigreg_proj_output_dim=16,
            bcs_num_slices=32,
            sigreg_lambda=10.0,
            sigreg_contiguous_mask_fraction=0.2,
            sigreg_contiguous_mask_min_len=1,
            sigreg_mz_jitter_std=0.005,
            sigreg_intensity_jitter_std=0.05,
            encoder_use_rope=encoder_use_rope,
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
        for key in (
            "loss",
            "bcs_loss",
            "invariance_loss",
            "valid_fraction",
            "masked_fraction",
            "density_interval_fraction",
            "representation_variance",
        ):
            self.assertIn(key, metrics, f"Missing key: {key}")

    def test_encode_output_shape(self):
        model = self._build_model()
        batch = _make_batch(batch_size=3)
        pooled = model.encode(batch, train=False)
        self.assertEqual(pooled.shape, (3, model.sigreg_dim))

    def test_compute_loss_returns_loss_and_metrics(self):
        model = self._build_model()
        batch = _make_batch()
        loss, metrics = model.compute_loss(batch, train=True)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertIn("invariance_loss", metrics)

    def test_backward_populates_encoder_gradients(self):
        model = self._build_model()
        batch = _make_batch()
        loss = model(batch, train=True)["loss"]
        loss.backward()
        grads = [p.grad for p in model.encoder.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None for g in grads))

    def test_encoder_rope_toggle_changes_output(self):
        batch = _make_batch(batch_size=2)

        torch.manual_seed(1234)
        model_no_rope = self._build_model(encoder_use_rope=False)
        torch.manual_seed(1234)
        model_with_rope = self._build_model(encoder_use_rope=True)

        with torch.no_grad():
            emb_no_rope = model_no_rope.encoder(
                batch["peak_mz"],
                batch["peak_intensity"],
                batch["precursor_mz"],
                valid_mask=batch["peak_valid_mask"],
            )
            emb_with_rope = model_with_rope.encoder(
                batch["peak_mz"],
                batch["peak_intensity"],
                batch["precursor_mz"],
                valid_mask=batch["peak_valid_mask"],
            )

        self.assertFalse(torch.allclose(emb_no_rope, emb_with_rope))


class AugmentationTests(unittest.TestCase):
    def _build_model(self) -> PeakSetSIGReg:
        return PeakSetSIGReg(
            num_peaks=60,
            model_dim=32,
            encoder_num_layers=1,
            encoder_num_heads=4,
            encoder_num_kv_heads=4,
            attention_mlp_multiple=2.0,
            feature_mlp_hidden_dim=16,
            sigreg_use_projector=False,
            bcs_num_slices=32,
            sigreg_lambda=10.0,
            sigreg_contiguous_mask_fraction=0.2,
            sigreg_contiguous_mask_min_len=1,
            sigreg_mz_jitter_std=0.005,
            sigreg_intensity_jitter_std=0.05,
        )

    def test_augment_view_shapes(self):
        model = self._build_model()
        batch = _make_batch()
        mz, intensity, valid, masked, masked_fraction, density_interval_fraction = model._augment_view(
            batch["peak_mz"],
            batch["peak_intensity"],
            batch["peak_valid_mask"],
        )
        self.assertEqual(mz.shape, batch["peak_mz"].shape)
        self.assertEqual(intensity.shape, batch["peak_intensity"].shape)
        self.assertEqual(valid.shape, batch["peak_valid_mask"].shape)
        self.assertEqual(masked.shape, batch["peak_valid_mask"].shape)
        self.assertEqual(masked_fraction.ndim, 0)
        self.assertEqual(density_interval_fraction.ndim, 0)

    def test_augment_view_value_bounds(self):
        model = self._build_model()
        batch = _make_batch()
        mz, intensity, _, _, _, _ = model._augment_view(
            batch["peak_mz"],
            batch["peak_intensity"],
            batch["peak_valid_mask"],
        )
        self.assertTrue((mz >= 0.0).all().item())
        self.assertTrue((mz <= 1.0).all().item())
        self.assertTrue((intensity >= 0.0).all().item())
        self.assertTrue((intensity <= 1.0).all().item())

    def test_invalid_positions_are_zeroed(self):
        model = self._build_model()
        batch = _make_batch()
        mz, intensity, _, _, _, _ = model._augment_view(
            batch["peak_mz"],
            batch["peak_intensity"],
            batch["peak_valid_mask"],
        )
        invalid = ~batch["peak_valid_mask"]
        self.assertTrue((mz[invalid] == 0.0).all().item())
        self.assertTrue((intensity[invalid] == 0.0).all().item())

    def test_masked_positions_remain_valid(self):
        model = self._build_model()
        batch = _make_batch(batch_size=3)
        _, _, valid, masked, _, _ = model._augment_view(
            batch["peak_mz"],
            batch["peak_intensity"],
            batch["peak_valid_mask"],
        )
        self.assertTrue((valid[masked]).all().item())
        self.assertTrue((~batch["peak_valid_mask"][masked]).sum().item() == 0)

    def test_masked_positions_are_zeroed_for_mask_token(self):
        model = self._build_model()
        batch = _make_batch(batch_size=3)
        mz, intensity, _, masked, _, _ = model._augment_view(
            batch["peak_mz"],
            batch["peak_intensity"],
            batch["peak_valid_mask"],
        )
        self.assertTrue((mz[masked] == 0.0).all().item())
        self.assertTrue((intensity[masked] == 0.0).all().item())

    def test_mask_is_contiguous_in_mz_sorted_order(self):
        model = self._build_model()
        batch = _make_batch(batch_size=3)
        torch.manual_seed(7)
        _, _, _, masked, _, _ = model._augment_view(
            batch["peak_mz"],
            batch["peak_intensity"],
            batch["peak_valid_mask"],
        )
        for row in range(masked.shape[0]):
            valid_idx = torch.nonzero(batch["peak_valid_mask"][row], as_tuple=False).squeeze(-1)
            sorted_valid_idx = valid_idx[torch.argsort(batch["peak_mz"][row, valid_idx])]
            masked_sorted_positions = torch.nonzero(
                masked[row, sorted_valid_idx], as_tuple=False,
            ).squeeze(-1)
            self.assertGreater(int(masked_sorted_positions.numel()), 0)
            if masked_sorted_positions.numel() > 1:
                deltas = masked_sorted_positions[1:] - masked_sorted_positions[:-1]
                self.assertTrue((deltas == 1).all().item())

    def test_encoder_uses_learnable_mask_token(self):
        model = PeakSetSIGReg(
            num_peaks=8,
            model_dim=16,
            encoder_num_layers=0,
            encoder_num_heads=4,
            encoder_num_kv_heads=4,
            attention_mlp_multiple=2.0,
            feature_mlp_hidden_dim=16,
            sigreg_use_projector=False,
            bcs_num_slices=32,
            sigreg_lambda=10.0,
            sigreg_contiguous_mask_fraction=0.2,
            sigreg_contiguous_mask_min_len=1,
            sigreg_mz_jitter_std=0.005,
            sigreg_intensity_jitter_std=0.05,
        )
        batch = _make_batch(batch_size=1, num_peaks=8)
        masked = torch.zeros_like(batch["peak_valid_mask"])
        masked[:, 0] = True
        masked[:, 1] = True
        embeddings = model.encoder(
            batch["peak_mz"],
            batch["peak_intensity"],
            batch["precursor_mz"],
            valid_mask=batch["peak_valid_mask"],
            masked_positions=masked,
            mask_token=model.mask_token,
        )
        self.assertTrue(torch.allclose(embeddings[:, 0, :], embeddings[:, 1, :], atol=1e-6, rtol=1e-5))

    def test_unmasked_view_has_no_masked_positions(self):
        model = self._build_model()
        batch = _make_batch(batch_size=3)
        _, _, valid, masked = model._augment_unmasked_view(
            batch["peak_mz"],
            batch["peak_intensity"],
            batch["peak_valid_mask"],
        )
        self.assertTrue(torch.equal(valid, batch["peak_valid_mask"]))
        self.assertEqual(int(masked.sum().item()), 0)

    def test_unmasked_view_invalid_positions_are_zeroed(self):
        model = self._build_model()
        batch = _make_batch(batch_size=3)
        mz, intensity, _, _ = model._augment_unmasked_view(
            batch["peak_mz"],
            batch["peak_intensity"],
            batch["peak_valid_mask"],
        )
        invalid = ~batch["peak_valid_mask"]
        self.assertTrue((mz[invalid] == 0.0).all().item())
        self.assertTrue((intensity[invalid] == 0.0).all().item())

    def test_forward_masked_fraction_matches_masked_view(self):
        model = self._build_model()
        batch = _make_batch(batch_size=3)
        torch.manual_seed(23)
        metrics = model(batch, train=True)
        torch.manual_seed(23)
        _, _, _, _, masked_fraction, density_interval_fraction = model._augment_view(
            batch["peak_mz"],
            batch["peak_intensity"],
            batch["peak_valid_mask"],
        )
        self.assertTrue(torch.allclose(metrics["masked_fraction"], masked_fraction))
        self.assertTrue(torch.allclose(metrics["density_interval_fraction"], density_interval_fraction))


class PMAPoolingTests(unittest.TestCase):
    def _build_model(self) -> PeakSetSIGReg:
        return PeakSetSIGReg(
            num_peaks=60,
            model_dim=32,
            encoder_num_layers=1,
            encoder_num_heads=4,
            encoder_num_kv_heads=4,
            attention_mlp_multiple=2.0,
            feature_mlp_hidden_dim=16,
            sigreg_use_projector=False,
            bcs_num_slices=32,
            sigreg_lambda=10.0,
            pooling_type="pma",
            pma_num_heads=4,
            pma_num_seeds=2,
        )

    def test_pool_ignores_invalid_positions(self):
        model = self._build_model()
        embeddings = torch.randn(3, 8, 32)
        valid = torch.zeros(3, 8, dtype=torch.bool)
        valid[:, :5] = True

        pooled_a = model.pool(embeddings, valid)
        embeddings[:, 5:] = torch.randn(3, 3, 32)
        pooled_b = model.pool(embeddings, valid)
        self.assertTrue(torch.allclose(pooled_a, pooled_b, atol=1e-6, rtol=1e-5))

    def test_pool_backward_populates_pma_gradients(self):
        model = self._build_model()
        batch = _make_batch()
        loss = model(batch, train=True)["loss"]
        loss.backward()

        self.assertIsNotNone(model.pool_query.grad)
        self.assertGreater(float(model.pool_query.grad.abs().sum()), 0.0)

        grads = [p.grad for p in model.pool_mha.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None for g in grads))
        self.assertGreater(sum(float(g.abs().sum()) for g in grads if g is not None), 0.0)


class SIGRegLossTests(unittest.TestCase):
    def test_epps_pulley_finite(self):
        x = torch.randn(32, 16)
        t = _epps_pulley(x)
        self.assertEqual(t.shape, (16,))
        self.assertTrue(torch.isfinite(t).all().item())

    def test_bcs_loss_output_structure(self):
        loss_fn = BCSLoss(num_slices=32, lmbd=10.0)
        z1 = torch.randn(32, 16)
        z2 = torch.randn(32, 16)
        out = loss_fn(z1, z2)
        for key in ("loss", "bcs_loss", "invariance_loss"):
            self.assertIn(key, out)
        self.assertTrue(torch.isfinite(out["loss"]).item())
        self.assertGreaterEqual(float(out["bcs_loss"]), 0.0)

    def test_bcs_projection_seed_is_reproducible(self):
        loss_fn = BCSLoss(num_slices=32, lmbd=10.0)
        proj1 = loss_fn.sample_projection(16, device=torch.device("cpu"), seed=7)
        proj2 = loss_fn.sample_projection(16, device=torch.device("cpu"), seed=7)
        proj3 = loss_fn.sample_projection(16, device=torch.device("cpu"), seed=8)
        self.assertTrue(torch.allclose(proj1, proj2))
        self.assertFalse(torch.allclose(proj1, proj3))

    def test_loss_backpropagates_to_both_branches(self):
        loss_fn = BCSLoss(num_slices=32, lmbd=10.0)
        z1 = torch.randn(32, 16, requires_grad=True)
        z2 = torch.randn(32, 16, requires_grad=True)
        out = loss_fn(z1, z2)
        out["loss"].backward()
        self.assertIsNotNone(z1.grad)
        self.assertIsNotNone(z2.grad)
        self.assertGreater(float(z1.grad.abs().sum()), 0.0)
        self.assertGreater(float(z2.grad.abs().sum()), 0.0)


class CheckpointCompatibilityTests(unittest.TestCase):
    def test_load_pretrained_requires_pma_weights(self):
        model = PeakSetSIGReg(
            num_peaks=60,
            model_dim=32,
            encoder_num_layers=1,
            encoder_num_heads=4,
            encoder_num_kv_heads=4,
            attention_mlp_multiple=2.0,
            feature_mlp_hidden_dim=16,
            sigreg_use_projector=False,
            pooling_type="pma",
            pma_num_heads=4,
            pma_num_seeds=1,
        )

        full_state = model.state_dict()
        without_pma = {
            key: value
            for key, value in full_state.items()
            if key != "pool_query" and not key.startswith("pool_mha.")
        }
        ckpt = {"state_dict": {f"model.{key}": value for key, value in without_pma.items()}}

        with tempfile.NamedTemporaryFile(suffix=".ckpt") as tmp:
            torch.save(ckpt, tmp.name)
            with self.assertRaises(RuntimeError):
                load_pretrained_weights(model, tmp.name)


if __name__ == "__main__":
    unittest.main()
