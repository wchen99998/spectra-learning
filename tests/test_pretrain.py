import tempfile
import unittest

import torch

from models.model import PeakSetSIGReg
from utils.training import load_pretrained_weights


def _make_batch(batch_size: int = 4, num_peaks: int = 6, num_targets: int = 2, include_precursor: bool = False) -> dict[str, torch.Tensor]:
    peak_mz = torch.rand(batch_size, num_peaks)
    peak_intensity = torch.rand(batch_size, num_peaks)
    peak_valid_mask = torch.ones(batch_size, num_peaks, dtype=torch.bool)
    context_mask = torch.zeros(batch_size, num_peaks, dtype=torch.bool)
    context_mask[:, :2] = True
    target_masks = torch.zeros(batch_size, num_targets, num_peaks, dtype=torch.bool)
    for target_idx in range(num_targets):
        target_masks[:, target_idx, 2 + target_idx] = True
    batch = {
        "peak_mz": peak_mz,
        "peak_intensity": peak_intensity,
        "peak_valid_mask": peak_valid_mask,
        "context_mask": context_mask,
        "target_masks": target_masks,
    }
    if include_precursor:
        batch["precursor_mz"] = torch.rand(batch_size) * 500 + 100
    return batch


class DataPipelineContractTests(unittest.TestCase):
    def test_batch_has_required_keys(self):
        batch = _make_batch()
        for key in ("peak_mz", "peak_intensity", "peak_valid_mask", "context_mask", "target_masks"):
            self.assertIn(key, batch)

    def test_batch_shapes(self):
        batch = _make_batch(batch_size=3, num_peaks=8, num_targets=3)
        self.assertEqual(batch["peak_mz"].shape, (3, 8))
        self.assertEqual(batch["peak_intensity"].shape, (3, 8))
        self.assertEqual(batch["peak_valid_mask"].shape, (3, 8))
        self.assertEqual(batch["context_mask"].shape, (3, 8))
        self.assertEqual(batch["target_masks"].shape, (3, 3, 8))


class BlockJEPATests(unittest.TestCase):
    def _build_model(self, **kwargs) -> PeakSetSIGReg:
        model_kwargs = {
            "num_peaks": 6,
            "model_dim": 32,
            "encoder_num_layers": 1,
            "encoder_num_heads": 4,
            "encoder_num_kv_heads": 4,
            "attention_mlp_multiple": 2.0,
            "feature_mlp_hidden_dim": 16,
            "sigreg_num_slices": 32,
            "sigreg_lambda": 0.1,
            "jepa_num_target_blocks": 2,
            "jepa_context_fraction": 0.25,
            "jepa_target_fraction": 0.25,
            "jepa_block_min_len": 1,
        }
        model_kwargs.update(kwargs)
        return PeakSetSIGReg(**model_kwargs)

    def test_forward_loss_is_finite(self):
        model = self._build_model()
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)
        metrics = model.forward_augmented(batch)
        self.assertTrue(torch.isfinite(metrics["loss"]).item())

    def test_forward_contains_expected_keys(self):
        model = self._build_model()
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)
        metrics = model.forward_augmented(batch)
        for key in (
            "loss",
            "sigreg_loss",
            "token_sigreg_loss",
            "local_global_l1_loss",
            "context_fraction",
            "masked_fraction",
            "sigreg_lambda_current",
            "global_emb_var_floor",
            "local_emb_var_floor",
            "global_emb_cov_offdiag_abs_mean",
            "local_emb_cov_offdiag_abs_mean",
            "global_emb_corr_offdiag_abs_mean",
            "local_emb_corr_offdiag_abs_mean",
        ):
            self.assertIn(key, metrics, f"Missing key: {key}")

    def test_sigreg_regularizer_uses_all_visible_branches(self):
        model = self._build_model(representation_regularizer="sigreg", sigreg_lambda=0.1)

        class CaptureSIGReg(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = None
                self.valid_mask = None

            def forward(self, proj, valid_mask=None):
                self.proj = proj.detach().clone()
                self.valid_mask = None if valid_mask is None else valid_mask.detach().clone()
                return proj.new_tensor(0.5)

        capture = CaptureSIGReg()
        model.sigreg = capture
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)
        B = batch["peak_mz"].shape[0]
        K = batch["target_masks"].shape[1]

        model.forward_augmented(batch)

        expected_visible = torch.cat(
            [
                batch["context_mask"].unsqueeze(0),
                batch["target_masks"].permute(1, 0, 2),
            ],
            dim=0,
        ).reshape((1 + K) * B, -1)
        self.assertIsNotNone(capture.proj)
        self.assertIsNotNone(capture.valid_mask)
        self.assertTrue(torch.equal(capture.valid_mask, expected_visible))

    def test_forward_without_regularizer(self):
        model = self._build_model(
            representation_regularizer=None,
            masked_token_loss_weight=1.0,
            sigreg_lambda=0.1,
        )
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)
        metrics = model.forward_augmented(batch)
        self.assertAlmostEqual(float(metrics["sigreg_loss"]), 0.0, places=7)
        self.assertAlmostEqual(float(metrics["regularizer_term"]), 0.0, places=7)
        self.assertTrue(torch.allclose(metrics["loss"], metrics["jepa_term"]))

    def test_encode_output_shape(self):
        model = self._build_model()
        batch = {
            "peak_mz": torch.rand(3, 6),
            "peak_intensity": torch.rand(3, 6),
            "peak_valid_mask": torch.ones(3, 6, dtype=torch.bool),
        }
        pooled = model.encode(batch, train=False)
        self.assertEqual(pooled.shape, (3, model.model_dim))

    def test_backward_populates_encoder_gradients(self):
        model = self._build_model()
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)
        loss = model.forward_augmented(batch)["loss"]
        loss.backward()
        grads = [p.grad for p in model.encoder.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None for g in grads))

    def test_vicreg_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "only supports"):
            self._build_model(representation_regularizer="vicreg")

    def test_load_pretrained_weights_roundtrip(self):
        model = self._build_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/ckpt.pt"
            torch.save({"state_dict": {f"model.{k}": v for k, v in model.state_dict().items()}}, path)
            loaded = self._build_model()
            load_pretrained_weights(loaded, path)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, loaded.state_dict()[key]), key)


class PrecursorTokenTests(unittest.TestCase):
    def _build_model(self, **kwargs) -> PeakSetSIGReg:
        model_kwargs = {
            "num_peaks": 6,
            "model_dim": 32,
            "encoder_num_layers": 1,
            "encoder_num_heads": 4,
            "encoder_num_kv_heads": 4,
            "attention_mlp_multiple": 2.0,
            "feature_mlp_hidden_dim": 16,
            "sigreg_num_slices": 32,
            "sigreg_lambda": 0.1,
            "jepa_num_target_blocks": 2,
            "jepa_context_fraction": 0.25,
            "jepa_target_fraction": 0.25,
            "jepa_block_min_len": 1,
            "use_precursor_token": True,
        }
        model_kwargs.update(kwargs)
        return PeakSetSIGReg(**model_kwargs)

    def test_forward_loss_is_finite_with_precursor(self):
        model = self._build_model()
        batch = _make_batch(num_targets=model.jepa_num_target_blocks, include_precursor=True)
        metrics = model.forward_augmented(batch)
        self.assertTrue(torch.isfinite(metrics["loss"]).item())

    def test_encode_output_shape_with_precursor(self):
        model = self._build_model()
        batch = {
            "peak_mz": torch.rand(3, 6),
            "peak_intensity": torch.rand(3, 6),
            "peak_valid_mask": torch.ones(3, 6, dtype=torch.bool),
            "precursor_mz": torch.rand(3) * 500 + 100,
        }
        pooled = model.encode(batch, train=False)
        self.assertEqual(pooled.shape, (3, model.model_dim))

    def test_no_nan_from_sentinel_intensity(self):
        """intensity=-1 must not produce NaN via log1p clamp."""
        model = self._build_model()
        batch = _make_batch(num_targets=model.jepa_num_target_blocks, include_precursor=True)
        metrics = model.forward_augmented(batch)
        self.assertFalse(torch.isnan(metrics["loss"]).item())

    def test_gradients_through_precursor_token(self):
        model = self._build_model()
        batch = _make_batch(num_targets=model.jepa_num_target_blocks, include_precursor=True)
        loss = model.forward_augmented(batch)["loss"]
        loss.backward()
        grads = [p.grad for p in model.encoder.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None for g in grads))

    def test_prepend_precursor_token_shapes(self):
        B, N, K = 4, 6, 2
        peak_mz = torch.rand(B, N)
        peak_intensity = torch.rand(B, N)
        peak_valid_mask = torch.ones(B, N, dtype=torch.bool)
        precursor_mz = torch.rand(B) * 500
        context_mask = torch.ones(B, N, dtype=torch.bool)
        target_masks = torch.zeros(B, K, N, dtype=torch.bool)

        result = PeakSetSIGReg.prepend_precursor_token(
            peak_mz, peak_intensity, peak_valid_mask, precursor_mz,
            context_mask=context_mask, target_masks=target_masks,
        )
        self.assertEqual(result["peak_mz"].shape, (B, N + 1))
        self.assertEqual(result["peak_intensity"].shape, (B, N + 1))
        self.assertEqual(result["peak_valid_mask"].shape, (B, N + 1))
        self.assertEqual(result["context_mask"].shape, (B, N + 1))
        self.assertEqual(result["target_masks"].shape, (B, K, N + 1))
        # Precursor token: valid=True, context=True, target=False
        self.assertTrue(result["peak_valid_mask"][:, 0].all())
        self.assertTrue(result["context_mask"][:, 0].all())
        self.assertFalse(result["target_masks"][:, :, 0].any())
        # Precursor intensity sentinel
        self.assertTrue((result["peak_intensity"][:, 0] == -1.0).all())


if __name__ == "__main__":
    unittest.main()
