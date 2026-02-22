import unittest
import tempfile

import torch

from utils.training import load_pretrained_weights
from models.losses import SIGReg
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
            sigreg_num_slices=32,
            sigreg_lambda=0.1,
            multicrop_num_global_views=2,
            multicrop_num_local_views=2,
            multicrop_global_keep_fraction=0.80,
            multicrop_local_keep_fraction=0.25,
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
            "sigreg_loss",
            "invariance_loss",
            "valid_fraction",
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
                valid_mask=batch["peak_valid_mask"],
                precursor_mz=batch["precursor_mz"],
            )
            emb_with_rope = model_with_rope.encoder(
                batch["peak_mz"],
                batch["peak_intensity"],
                valid_mask=batch["peak_valid_mask"],
                precursor_mz=batch["precursor_mz"],
            )

        self.assertFalse(torch.allclose(emb_no_rope, emb_with_rope))


class MulticropAugmentationTests(unittest.TestCase):
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
            sigreg_num_slices=32,
            sigreg_lambda=0.1,
            multicrop_num_global_views=2,
            multicrop_num_local_views=3,
            multicrop_global_keep_fraction=0.80,
            multicrop_local_keep_fraction=0.25,
            sigreg_mz_jitter_std=0.005,
            sigreg_intensity_jitter_std=0.05,
        )

    def test_augment_batch_shapes(self):
        model = self._build_model()
        batch = _make_batch(batch_size=4, num_peaks=60)
        aug = model.augment_batch(batch)
        V = model.num_views  # 2 + 3 = 5
        B = 4
        N = 60
        self.assertEqual(aug["fused_mz"].shape, (V * B, N))
        self.assertEqual(aug["fused_intensity"].shape, (V * B, N))
        self.assertEqual(aug["fused_valid_mask"].shape, (V * B, N))
        self.assertEqual(aug["fused_precursor_mz"].shape, (V * B,))

    def test_augment_batch_value_bounds(self):
        model = self._build_model()
        batch = _make_batch()
        aug = model.augment_batch(batch)
        self.assertTrue((aug["fused_mz"] >= 0.0).all().item())
        self.assertTrue((aug["fused_mz"] <= 1.0).all().item())
        self.assertTrue((aug["fused_intensity"] >= 0.0).all().item())
        self.assertTrue((aug["fused_intensity"] <= 1.0).all().item())

    def test_augment_batch_no_fused_masked_positions(self):
        """Multicrop augmentation should not produce fused_masked_positions."""
        model = self._build_model()
        batch = _make_batch()
        aug = model.augment_batch(batch)
        self.assertNotIn("fused_masked_positions", aug)
        self.assertNotIn("view1_masked_fraction", aug)

    def test_global_views_keep_more_peaks_than_local(self):
        model = self._build_model()
        batch = _make_batch(batch_size=4, num_peaks=60)
        aug = model.augment_batch(batch)
        B = 4
        num_global = model.multicrop_num_global_views
        num_local = model.multicrop_num_local_views

        global_valid = aug["fused_valid_mask"][:num_global * B]
        local_valid = aug["fused_valid_mask"][num_global * B:]

        global_avg = global_valid.float().sum(dim=1).mean()
        local_avg = local_valid.float().sum(dim=1).mean()
        self.assertGreater(float(global_avg), float(local_avg))

    def test_forward_augmented_num_views(self):
        """Verify forward_augmented works with the multicrop V-view structure."""
        model = self._build_model()
        batch = _make_batch(batch_size=4)
        metrics = model(batch, train=True)
        self.assertTrue(torch.isfinite(metrics["loss"]).item())
        self.assertIn("sigreg_loss", metrics)
        self.assertIn("invariance_loss", metrics)


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
            sigreg_num_slices=32,
            sigreg_lambda=0.1,
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
    def test_sigreg_output_is_scalar(self):
        sigreg = SIGReg(num_slices=32)
        proj = torch.randn(4, 8, 16)  # [V, B, D]
        result = sigreg(proj)
        self.assertEqual(result.ndim, 0)
        self.assertTrue(torch.isfinite(result).item())

    def test_sigreg_backpropagates(self):
        sigreg = SIGReg(num_slices=32)
        proj = torch.randn(4, 8, 16, requires_grad=True)
        result = sigreg(proj)
        result.backward()
        self.assertIsNotNone(proj.grad)
        self.assertGreater(float(proj.grad.abs().sum()), 0.0)

    def test_sigreg_gaussian_input_has_low_statistic(self):
        sigreg = SIGReg(num_slices=64)
        proj = torch.randn(4, 256, 32)  # large B for statistical accuracy
        result = sigreg(proj)
        # Gaussian input should produce a small statistic
        self.assertLess(float(result), 100.0)


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
