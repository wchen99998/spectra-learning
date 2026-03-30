import tempfile
import unittest

import torch

from models.model import PeakSetSIGReg
from train import _is_weight_decay_target
from utils.training import load_pretrained_weights


def _make_batch(
    batch_size: int = 4,
    num_peaks: int = 6,
    num_targets: int = 2,
    include_precursor: bool = False,
) -> dict[str, torch.Tensor]:
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


def _make_pipeline_prepended_batch(
    batch_size: int = 4,
    num_peaks: int = 6,
    num_targets: int = 2,
) -> dict[str, torch.Tensor]:
    """Create a batch that mimics what the TF pipeline produces when use_precursor_token=True.

    The precursor token is already at position 0, and `precursor_mz` is absent.
    Total sequence length is num_peaks + 1.
    """
    N = num_peaks + 1  # includes prepended precursor token
    peak_mz = torch.rand(batch_size, N)
    peak_intensity = torch.rand(batch_size, N)
    peak_intensity[:, 0] = -1.0  # sentinel
    peak_valid_mask = torch.ones(batch_size, N, dtype=torch.bool)
    context_mask = torch.zeros(batch_size, N, dtype=torch.bool)
    context_mask[:, 0] = True  # precursor always in context
    context_mask[:, 1:3] = True
    target_masks = torch.zeros(batch_size, num_targets, N, dtype=torch.bool)
    for target_idx in range(num_targets):
        target_masks[:, target_idx, 3 + target_idx] = True
    return {
        "peak_mz": peak_mz,
        "peak_intensity": peak_intensity,
        "peak_valid_mask": peak_valid_mask,
        "context_mask": context_mask,
        "target_masks": target_masks,
    }


class DataPipelineContractTests(unittest.TestCase):
    def test_batch_has_required_keys(self):
        batch = _make_batch()
        for key in (
            "peak_mz",
            "peak_intensity",
            "peak_valid_mask",
            "context_mask",
            "target_masks",
        ):
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
            "model_dim": 32,
            "encoder_num_layers": 1,
            "encoder_num_heads": 4,
            "encoder_num_kv_heads": 4,
            "attention_mlp_multiple": 2.0,
            "feature_mlp_hidden_dim": 16,
            "sigreg_num_slices": 32,
            "sigreg_lambda": 0.1,
            "jepa_num_target_blocks": 2,
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
            "token_sigreg_loss",
            "local_global_loss",
            "peak_recon_loss",
            "peak_recon_term",
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
        model = self._build_model(sigreg_lambda=0.1)

        class CaptureSIGReg(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = None
                self.valid_mask = None

            def forward(self, proj, valid_mask=None):
                self.proj = proj.detach().clone()
                self.valid_mask = (
                    None if valid_mask is None else valid_mask.detach().clone()
                )
                return proj.new_tensor(0.5)

        capture = CaptureSIGReg()
        model.sigreg = capture
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)
        B = batch["peak_mz"].shape[0]
        K = batch["target_masks"].shape[1]

        model.forward_augmented(batch)

        expected_visible = torch.cat(
            [
                batch["context_mask"].unsqueeze(1),
                batch["target_masks"],
            ],
            dim=1,
        ).reshape((1 + K) * B, -1)
        self.assertIsNotNone(capture.proj)
        self.assertIsNotNone(capture.valid_mask)
        self.assertTrue(torch.equal(capture.valid_mask, expected_visible))

    def test_forward_without_regularizer(self):
        model = self._build_model(
            masked_token_loss_weight=1.0,
            mae_loss_weight=0.0,
            sigreg_lambda=0.0,
        )
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)
        metrics = model.forward_augmented(batch)
        self.assertAlmostEqual(float(metrics["token_sigreg_loss"]), 0.0, places=7)
        self.assertAlmostEqual(float(metrics["sigreg_term"]), 0.0, places=7)
        self.assertTrue(torch.allclose(metrics["loss"], metrics["jepa_term"]))

    def test_encode_output_shape(self):
        model = self._build_model()
        batch = {
            "peak_mz": torch.rand(3, 6),
            "peak_intensity": torch.rand(3, 6),
            "peak_valid_mask": torch.ones(3, 6, dtype=torch.bool),
        }
        pooled = model.encode(batch)
        self.assertEqual(pooled.shape, (3, model.model_dim))

    def test_encode_returns_cls_token_state(self):
        model = self._build_model(encoder_num_register_tokens=2)
        batch = {
            "peak_mz": torch.rand(3, 6),
            "peak_intensity": torch.rand(3, 6),
            "peak_valid_mask": torch.ones(3, 6, dtype=torch.bool),
        }
        peak_emb, cls_emb = model.encoder(
            batch["peak_mz"],
            batch["peak_intensity"],
            valid_mask=batch["peak_valid_mask"],
            visible_mask=batch["peak_valid_mask"],
            return_cls_token=True,
        )
        pooled = model.encode(batch)
        self.assertEqual(peak_emb.shape, (3, 7, model.model_dim))
        self.assertTrue(torch.allclose(pooled, cls_emb))

    def test_backward_populates_encoder_gradients(self):
        model = self._build_model()
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)
        loss = model.forward_augmented(batch)["loss"]
        loss.backward()
        grads = [p.grad for p in model.encoder.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None for g in grads))

    def test_load_pretrained_weights_roundtrip(self):
        model = self._build_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/ckpt.pt"
            torch.save(
                {
                    "state_dict": {
                        f"model.{k}": v for k, v in model.state_dict().items()
                    }
                },
                path,
            )
            loaded = self._build_model()
            load_pretrained_weights(loaded, path)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, loaded.state_dict()[key]), key)

    def test_load_pretrained_weights_allows_missing_position_embeddings(self):
        model = self._build_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/ckpt.pt"
            old_state = {
                f"model.{k}": v
                for k, v in model.state_dict().items()
                if not k.endswith(
                    (
                        "position_embedding.weight",
                        "predictor_position_embedding.weight",
                        "cls_token",
                        "register_tokens",
                        "predictor_register_tokens",
                    )
                )
            }
            torch.save({"state_dict": old_state}, path)
            loaded = self._build_model()
            load_pretrained_weights(loaded, path)

    def test_load_pretrained_weights_allows_missing_masked_latent_readout(self):
        model = self._build_model(jepa_target_layers=[1])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/ckpt.pt"
            old_state = {
                f"model.{k}": v
                for k, v in model.state_dict().items()
                if not k.startswith("masked_latent_readout.")
            }
            torch.save({"state_dict": old_state}, path)
            loaded = self._build_model(jepa_target_layers=[1])
            load_pretrained_weights(loaded, path)

    def test_load_pretrained_weights_allows_missing_masked_peak_readout(self):
        model = self._build_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/ckpt.pt"
            old_state = {
                f"model.{k}": v
                for k, v in model.state_dict().items()
                if not k.startswith("masked_peak_readout.")
            }
            torch.save({"state_dict": old_state}, path)
            loaded = self._build_model()
            load_pretrained_weights(loaded, path)

    def test_online_targets_require_grad(self):
        model = self._build_model()
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)
        targets = model._compute_jepa_online_targets(
            batch["peak_mz"],
            batch["peak_intensity"],
            batch["peak_valid_mask"],
        )
        self.assertTrue(targets.requires_grad)

    def test_weight_decay_targets_all_2d_weights(self):
        model = self._build_model()
        self.assertTrue(
            _is_weight_decay_target(
                "encoder.embedder.output_proj.weight",
                model.encoder.embedder.output_proj.weight,
            )
        )
        self.assertTrue(
            _is_weight_decay_target(
                "encoder.embedder.fourier_ffn.0.weight",
                model.encoder.embedder.fourier_ffn[0].weight,
            )
        )
        self.assertFalse(
            _is_weight_decay_target(
                "encoder.embedder.mz_fourier.b",
                model.encoder.embedder.mz_fourier.b,
            )
        )


class PrecursorTokenTests(unittest.TestCase):
    def _build_model(self, **kwargs) -> PeakSetSIGReg:
        model_kwargs = {
            "model_dim": 32,
            "encoder_num_layers": 1,
            "encoder_num_heads": 4,
            "encoder_num_kv_heads": 4,
            "attention_mlp_multiple": 2.0,
            "feature_mlp_hidden_dim": 16,
            "sigreg_num_slices": 32,
            "sigreg_lambda": 0.1,
            "jepa_num_target_blocks": 2,
            "use_precursor_token": True,
            "num_peaks": 6,
        }
        model_kwargs.update(kwargs)
        return PeakSetSIGReg(**model_kwargs)

    def test_forward_with_pipeline_prepended_batch(self):
        """forward_augmented works with pipeline-prepended batch (N+1 tensors, no precursor_mz key)."""
        model = self._build_model()
        batch = _make_pipeline_prepended_batch(
            num_peaks=6,
            num_targets=model.jepa_num_target_blocks,
        )
        self.assertNotIn("precursor_mz", batch)
        metrics = model.forward_augmented(batch)
        self.assertTrue(torch.isfinite(metrics["loss"]).item())

    def test_encode_with_prepended_batch(self):
        """encode() works with a batch where precursor is already prepended."""
        model = self._build_model()
        N = 7  # 6 peaks + 1 precursor
        batch = {
            "peak_mz": torch.rand(3, N),
            "peak_intensity": torch.rand(3, N),
            "peak_valid_mask": torch.ones(3, N, dtype=torch.bool),
        }
        batch["peak_intensity"][:, 0] = -1.0
        pooled = model.encode(batch)
        self.assertEqual(pooled.shape, (3, model.model_dim))

    def test_no_nan_from_sentinel_intensity(self):
        """intensity=-1 must not produce NaN via log1p clamp."""
        model = self._build_model()
        batch = _make_pipeline_prepended_batch(
            num_peaks=6,
            num_targets=model.jepa_num_target_blocks,
        )
        metrics = model.forward_augmented(batch)
        self.assertFalse(torch.isnan(metrics["loss"]).item())

    def test_gradients_through_precursor_token(self):
        model = self._build_model()
        batch = _make_pipeline_prepended_batch(
            num_peaks=6,
            num_targets=model.jepa_num_target_blocks,
        )
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
            peak_mz,
            peak_intensity,
            peak_valid_mask,
            precursor_mz,
            context_mask=context_mask,
            target_masks=target_masks,
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


class TfPrependPrecursorTokenTests(unittest.TestCase):
    """Test the TF-side _prepend_precursor_token_tf function."""

    def test_shapes_and_values(self):
        import tensorflow as tf
        from input_pipeline import _prepend_precursor_token_tf

        B, N, K = 4, 8, 2
        batch = {
            "peak_mz": tf.random.uniform([B, N]),
            "peak_intensity": tf.random.uniform([B, N]),
            "peak_valid_mask": tf.ones([B, N], dtype=tf.bool),
            "precursor_mz": tf.random.uniform([B]),
            "context_mask": tf.ones([B, N], dtype=tf.bool),
            "target_masks": tf.zeros([B, K, N], dtype=tf.bool),
            "rt": tf.zeros([B]),
        }
        out = _prepend_precursor_token_tf(batch)

        # precursor_mz should be removed
        self.assertNotIn("precursor_mz", out)
        # rt should be preserved
        self.assertIn("rt", out)

        # Shapes: N+1 in sequence dim
        self.assertEqual(out["peak_mz"].shape, (B, N + 1))
        self.assertEqual(out["peak_intensity"].shape, (B, N + 1))
        self.assertEqual(out["peak_valid_mask"].shape, (B, N + 1))
        self.assertEqual(out["context_mask"].shape, (B, N + 1))
        self.assertEqual(out["target_masks"].shape, (B, K, N + 1))

        # Sentinel intensity at position 0
        self.assertTrue((out["peak_intensity"][:, 0].numpy() == -1.0).all())
        # Valid at position 0
        self.assertTrue(out["peak_valid_mask"][:, 0].numpy().all())
        # Context at position 0
        self.assertTrue(out["context_mask"][:, 0].numpy().all())
        # Not a target at position 0
        self.assertFalse(out["target_masks"][:, :, 0].numpy().any())
        # Precursor mz at position 0
        self.assertTrue(
            (out["peak_mz"][:, 0].numpy() == batch["precursor_mz"].numpy()).all()
        )

    def test_without_masks(self):
        """Works on raw (pre-augmentation) batches without context_mask/target_masks."""
        import tensorflow as tf
        from input_pipeline import _prepend_precursor_token_tf

        B, N = 3, 5
        batch = {
            "peak_mz": tf.random.uniform([B, N]),
            "peak_intensity": tf.random.uniform([B, N]),
            "peak_valid_mask": tf.ones([B, N], dtype=tf.bool),
            "precursor_mz": tf.random.uniform([B]),
        }
        out = _prepend_precursor_token_tf(batch)

        self.assertNotIn("precursor_mz", out)
        self.assertEqual(out["peak_mz"].shape, (B, N + 1))
        self.assertNotIn("context_mask", out)
        self.assertNotIn("target_masks", out)


if __name__ == "__main__":
    unittest.main()
