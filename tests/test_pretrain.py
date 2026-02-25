import unittest
import tempfile
import math

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


def _make_fused_batch(
    batch_size: int = 4, num_peaks: int = 60, num_views: int = 4,
) -> dict[str, torch.Tensor]:
    """Create a synthetic fused (pre-augmented) batch for forward_augmented."""
    VB = num_views * batch_size
    valid_count = num_peaks - 10
    mz = torch.rand(VB, num_peaks)
    intensity = torch.rand(VB, num_peaks)
    valid = torch.zeros(VB, num_peaks, dtype=torch.bool)
    valid[:, :valid_count] = True
    mz[:, valid_count:] = 0.0
    intensity[:, valid_count:] = 0.0
    return {
        "fused_mz": mz,
        "fused_intensity": intensity,
        "fused_valid_mask": valid,
        "fused_precursor_mz": torch.rand(VB),
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
            multicrop_num_local_views=3,
            multicrop_local_keep_fraction=0.25,
            sigreg_mz_jitter_std=0.005,
            sigreg_intensity_jitter_std=0.05,
            encoder_use_rope=encoder_use_rope,
        )

    def test_forward_loss_is_finite(self):
        model = self._build_model()
        batch = _make_fused_batch(num_views=model.num_views)
        metrics = model.forward_augmented(batch)
        self.assertTrue(torch.isfinite(metrics["loss"]).item())

    def test_forward_contains_expected_keys(self):
        model = self._build_model()
        batch = _make_fused_batch(num_views=model.num_views)
        metrics = model.forward_augmented(batch)
        for key in (
            "loss",
            "sigreg_loss",
            "token_sigreg_loss",
            "local_global_l1_loss",
            "valid_fraction",
            "representation_variance",
        ):
            self.assertIn(key, metrics, f"Missing key: {key}")

    def test_encode_output_shape(self):
        model = self._build_model()
        batch = _make_batch(batch_size=3)
        pooled = model.encode(batch, train=False)
        self.assertEqual(pooled.shape, (3, model.sigreg_dim))

    def test_backward_populates_encoder_gradients(self):
        model = self._build_model()
        batch = _make_fused_batch(num_views=model.num_views)
        loss = model.forward_augmented(batch)["loss"]
        loss.backward()
        grads = [p.grad for p in model.encoder.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None for g in grads))

    def test_local_global_loss_stops_grad_on_global_target_branch(self):
        model = PeakSetSIGReg(
            num_peaks=6,
            model_dim=32,
            encoder_num_layers=2,
            encoder_num_heads=4,
            feature_mlp_hidden_dim=32,
            encoder_use_rope=True,
            sigreg_use_projector=False,
            pooling_type="mean",
            sigreg_lambda=0.0,
            multicrop_num_local_views=1,
            use_masked_token_input=True,
            masked_token_position_mode="index",
            masked_token_loss_weight=1.0,
        )
        fused_mz = torch.tensor(
            [
                [0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
                [0.12, 0.22, 0.32, 0.42, 0.52, 0.62],
                [0.11, 0.21, 0.31, 0.41, 0.51, 0.61],
                [0.13, 0.23, 0.33, 0.43, 0.53, 0.63],
            ],
            dtype=torch.float32,
        )
        fused_intensity = torch.tensor(
            [
                [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
                [0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
                [0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
                [0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            ],
            dtype=torch.float32,
            requires_grad=True,
        )
        fused_valid_mask = torch.ones_like(fused_mz, dtype=torch.bool)
        fused_masked_positions = torch.tensor(
            [
                [False, False, False, False, False, False],
                [False, False, False, False, False, False],
                [False, True, False, False, False, False],
                [False, False, False, True, False, False],
            ]
        )
        fused_precursor_mz = torch.tensor([0.95, 0.95, 0.95, 0.95], dtype=torch.float32)

        loss = model.forward_augmented(
            {
                "fused_mz": fused_mz,
                "fused_intensity": fused_intensity,
                "fused_valid_mask": fused_valid_mask,
                "fused_masked_positions": fused_masked_positions,
                "fused_precursor_mz": fused_precursor_mz,
            }
        )["loss"]
        loss.backward()
        global_branch_grad = fused_intensity.grad[:2].abs().sum()
        self.assertEqual(float(global_branch_grad), 0.0)

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


class MassAwareRoPETests(unittest.TestCase):
    def _build_encoder_model(
        self,
        *,
        rope_mz_precision: float = 0.1,
        rope_complement_heads: int | None = None,
        encoder_num_kv_heads: int = 4,
    ) -> PeakSetSIGReg:
        return PeakSetSIGReg(
            num_peaks=60,
            model_dim=32,
            encoder_num_layers=1,
            encoder_num_heads=4,
            encoder_num_kv_heads=encoder_num_kv_heads,
            attention_mlp_multiple=2.0,
            feature_mlp_hidden_dim=16,
            encoder_use_rope=True,
            rope_mz_max=1000.0,
            rope_mz_precision=rope_mz_precision,
            rope_complement_heads=rope_complement_heads,
            rope_modulo_2pi=True,
            sigreg_use_projector=False,
            pooling_type="mean",
        )

    def test_mass_rope_precision_0p1_omega_range(self):
        model = self._build_encoder_model(rope_mz_precision=0.1)
        omega = model.encoder.rope_omega
        expected_max = (2.0 * math.pi) / 0.2
        expected_min = (2.0 * math.pi) / 1000.0
        self.assertTrue(torch.isclose(omega[0], torch.tensor(expected_max), rtol=1e-5, atol=1e-6))
        self.assertTrue(torch.isclose(omega[-1], torch.tensor(expected_min), rtol=1e-5, atol=1e-6))

    def test_mass_rope_precision_0p01_omega_range(self):
        model = self._build_encoder_model(rope_mz_precision=0.01)
        omega = model.encoder.rope_omega
        expected_max = (2.0 * math.pi) / 0.02
        expected_min = (2.0 * math.pi) / 1000.0
        self.assertTrue(torch.isclose(omega[0], torch.tensor(expected_max), rtol=1e-5, atol=1e-6))
        self.assertTrue(torch.isclose(omega[-1], torch.tensor(expected_min), rtol=1e-5, atol=1e-6))

    def test_complement_head_split_changes_output(self):
        batch = _make_batch(batch_size=2)

        torch.manual_seed(2026)
        model_all_mass = self._build_encoder_model(rope_complement_heads=0)
        torch.manual_seed(2026)
        model_split = self._build_encoder_model(rope_complement_heads=2)

        with torch.no_grad():
            emb_all_mass = model_all_mass.encoder(
                batch["peak_mz"],
                batch["peak_intensity"],
                valid_mask=batch["peak_valid_mask"],
                precursor_mz=batch["precursor_mz"],
            )
            emb_split = model_split.encoder(
                batch["peak_mz"],
                batch["peak_intensity"],
                valid_mask=batch["peak_valid_mask"],
                precursor_mz=batch["precursor_mz"],
            )

        self.assertFalse(torch.allclose(emb_all_mass, emb_split))

    def test_rope_with_gqa_and_complement_heads_is_finite(self):
        model = self._build_encoder_model(
            rope_mz_precision=0.1,
            rope_complement_heads=2,
            encoder_num_kv_heads=2,
        )
        batch = _make_batch(batch_size=3)
        with torch.no_grad():
            emb = model.encoder(
                batch["peak_mz"],
                batch["peak_intensity"],
                valid_mask=batch["peak_valid_mask"],
                precursor_mz=batch["precursor_mz"],
            )
        self.assertTrue(torch.isfinite(emb).all())


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
            masked_token_loss_weight=1.0,
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

    def test_pool_parameters_are_not_in_loss_path(self):
        model = self._build_model()
        batch = _make_fused_batch(num_views=model.num_views)
        loss = model.forward_augmented(batch)["loss"]
        loss.backward()

        self.assertIsNone(model.pool_query.grad)
        mha_grads = [p.grad for p in model.pool_mha.parameters() if p.requires_grad]
        self.assertTrue(all(g is None for g in mha_grads))

        predictor_grads = [p.grad for p in model.masked_latent_predictor.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None for g in predictor_grads))
        self.assertGreater(sum(float(g.abs().sum()) for g in predictor_grads if g is not None), 0.0)


class SIGRegLossTests(unittest.TestCase):
    def test_sigreg_output_is_scalar(self):
        sigreg = SIGReg(num_slices=32)
        proj = torch.randn(4, 8, 16)  # [..., D]
        result = sigreg(proj)
        self.assertEqual(result.ndim, 0)
        self.assertTrue(torch.isfinite(result).item())

    def test_sigreg_accepts_any_leading_shape(self):
        sigreg = SIGReg(num_slices=32)
        proj = torch.randn(2, 3, 5, 16)
        result = sigreg(proj)
        self.assertEqual(result.ndim, 0)
        self.assertTrue(torch.isfinite(result).item())

    def test_sigreg_matches_flattened_input(self):
        sigreg = SIGReg(num_slices=32)
        proj = torch.randn(2, 3, 5, 16)
        torch.manual_seed(2026)
        result_shaped = sigreg(proj)
        torch.manual_seed(2026)
        result_flat = sigreg(proj.reshape(-1, proj.size(-1)))
        self.assertTrue(torch.allclose(result_shaped, result_flat))

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
