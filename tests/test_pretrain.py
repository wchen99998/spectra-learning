import tempfile
import unittest
import math
from unittest import mock

import numpy as np
import torch

from models.losses import SlotwiseSIGReg
from models.model import PeakSetSIGReg
from models.peak_features import FourierFeatures, PeakFeatureEmbedder
from train import _is_weight_decay_target
from utils.spectra_preprocessing import PRECURSOR_TOKEN_INTENSITY
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
    *,
    precursor_in_context: bool = False,
    precursor_in_targets: bool = False,
) -> dict[str, torch.Tensor]:
    """Create a batch that mimics what the TF pipeline produces when use_precursor_token=True.

    The precursor token is already at position 0, and `precursor_mz` is absent.
    Total sequence length is num_peaks + 1.
    """
    N = num_peaks + 1  # includes prepended precursor token
    peak_mz = torch.rand(batch_size, N)
    peak_intensity = torch.rand(batch_size, N)
    peak_intensity[:, 0] = PRECURSOR_TOKEN_INTENSITY
    peak_valid_mask = torch.ones(batch_size, N, dtype=torch.bool)
    context_mask = torch.zeros(batch_size, N, dtype=torch.bool)
    context_mask[:, 0] = precursor_in_context
    context_mask[:, 1:3] = True
    target_masks = torch.zeros(batch_size, num_targets, N, dtype=torch.bool)
    target_masks[:, :, 0] = precursor_in_targets
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


class FourierFeatureTests(unittest.TestCase):
    def test_log_spaced_both_funcs_preserves_feature_width(self):
        fourier = FourierFeatures(
            strategy="log_spaced",
            x_min=3e-3,
            x_max=1000.0,
            funcs="both",
            num_freqs=256,
        )
        self.assertEqual(fourier.num_features(), 512)

    def test_peak_embedder_fourier_branch_uses_configured_input_scale(self):
        embedder = PeakFeatureEmbedder(
            model_dim=32,
            hidden_dim=16,
            fourier_strategy="log_spaced",
            fourier_x_min=3e-3,
            fourier_x_max=1000.0,
            fourier_num_freqs=8,
            fourier_input_scale=750.0,
        )
        normalized_peak_mz = torch.tensor([[0.5]], dtype=torch.float32)
        prepared = embedder._prepare_fourier_mz(normalized_peak_mz)
        self.assertAlmostEqual(
            float(prepared.item()),
            375.0,
            places=6,
        )

    def test_peak_embedder_raw_branch_uses_normal_log_intensity(self):
        embedder = PeakFeatureEmbedder(
            model_dim=32,
            hidden_dim=16,
            fourier_strategy="log_spaced",
            fourier_x_min=3e-3,
            fourier_x_max=1000.0,
            fourier_num_freqs=8,
        )
        captured: dict[str, torch.Tensor] = {}

        def capture_raw_input(_module, args):
            captured["raw_input"] = args[0].detach().clone()

        handle = embedder.raw_ffn.register_forward_pre_hook(capture_raw_input)
        peak_mz = torch.tensor([[0.25]], dtype=torch.float32)
        peak_intensity = torch.tensor(
            [[PRECURSOR_TOKEN_INTENSITY]],
            dtype=torch.float32,
        )
        embedder(peak_mz, peak_intensity)
        handle.remove()

        expected = torch.tensor(
            [[[0.25, PRECURSOR_TOKEN_INTENSITY, math.log1p(PRECURSOR_TOKEN_INTENSITY)]]],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(captured["raw_input"], expected))


class BlockJEPATests(unittest.TestCase):
    def _build_model(self, **kwargs) -> PeakSetSIGReg:
        model_kwargs = {
            "model_dim": 32,
            "encoder_num_layers": 1,
            "encoder_num_heads": 4,
            "encoder_num_kv_heads": 4,
            "attention_mlp_multiple": 2.0,
            "feature_mlp_hidden_dim": 16,
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
            "local_global_loss",
            "cls_embedding_loss",
            "cls_embedding_term",
            "cls_visible_fraction",
            "context_fraction",
            "masked_fraction",
            "global_emb_var_floor",
            "encoder_emb_var_floor",
            "global_emb_cov_offdiag_abs_mean",
            "encoder_emb_cov_offdiag_abs_mean",
            "global_emb_corr_offdiag_abs_mean",
            "encoder_emb_corr_offdiag_abs_mean",
        ):
            self.assertIn(key, metrics, f"Missing key: {key}")

    def test_sigreg_on_encoder_output_contributes_to_loss(self):
        for regularizer in ("sigreg-enc", "slot-sigreg-enc"):
            with self.subTest(regularizer=regularizer):
                model = self._build_model(
                    masked_token_loss_weight=1.0,
                    representation_regularizer=regularizer,
                    sigreg_lambda=0.02,
                )
                if regularizer.startswith("slot-"):
                    self.assertIsInstance(model.sigreg, SlotwiseSIGReg)
                batch = _make_batch(num_targets=model.jepa_num_target_blocks)
                metrics = model.forward_augmented(batch)
                self.assertIn("sigreg_term", metrics)
                self.assertIn("token_sigreg_loss", metrics)
                self.assertGreater(float(metrics["token_sigreg_loss"].detach()), 0.0)
                self.assertGreater(float(metrics["sigreg_term"].detach()), 0.0)
                self.assertTrue(
                    torch.allclose(
                        metrics["loss"],
                        metrics["jepa_term"]
                        + metrics["cls_embedding_term"]
                        + metrics["sigreg_term"],
                    )
                )

    def test_sigreg_on_predictor_outputs_uses_masked_predictions(self):
        for regularizer in ("sigreg-pred", "slot-sigreg-pred", "slog-sigreg-pred"):
            with self.subTest(regularizer=regularizer):
                model = self._build_model(
                    representation_regularizer=regularizer,
                    sigreg_lambda=0.02,
                    predictor_dim=24,
                )
                if regularizer.startswith("slot-") or regularizer.startswith("slog-"):
                    self.assertIsInstance(model.sigreg, SlotwiseSIGReg)
                batch = _make_batch(num_targets=model.jepa_num_target_blocks)
                captured: dict[str, torch.Tensor] = {}

                def fake_sigreg_forward(
                    proj: torch.Tensor,
                    valid_mask: torch.Tensor | None = None,
                ) -> torch.Tensor:
                    captured["proj"] = proj.detach().clone()
                    captured["valid_mask"] = valid_mask.detach().clone()
                    return proj.new_zeros(())

                with mock.patch.object(
                    model.sigreg,
                    "forward",
                    side_effect=fake_sigreg_forward,
                ):
                    model.forward_augmented(batch)

                self.assertIn("proj", captured)
                self.assertEqual(
                    captured["proj"].shape,
                    (*batch["target_masks"].shape, model.jepa_target_dim),
                )
                self.assertNotEqual(captured["proj"].shape[-1], model.predictor_dim)
                torch.testing.assert_close(
                    captured["valid_mask"],
                    batch["target_masks"].float(),
                )

    def test_vicreg_on_visible_context_contributes_to_loss(self):
        model = self._build_model(
            masked_token_loss_weight=1.0,
            representation_regularizer="vicreg",
            vicreg_lambda=0.02,
        )
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)
        metrics = model.forward_augmented(batch)
        for key in (
            "vicreg_term",
            "token_vicreg_loss",
            "vicreg_var_loss",
            "vicreg_cov_loss",
        ):
            self.assertIn(key, metrics)
        self.assertEqual(float(metrics["vicreg_inv_loss"].detach()), 0.0)
        self.assertGreater(float(metrics["vicreg_var_loss"].detach()), 0.0)
        self.assertGreaterEqual(float(metrics["vicreg_cov_loss"].detach()), 0.0)
        self.assertGreater(float(metrics["token_vicreg_loss"].detach()), 0.0)
        self.assertGreater(float(metrics["vicreg_term"].detach()), 0.0)
        self.assertTrue(
            torch.allclose(metrics["regularizer_loss"], metrics["token_vicreg_loss"])
        )
        self.assertTrue(
            torch.allclose(metrics["regularizer_term"], metrics["vicreg_term"])
        )
        self.assertTrue(
            torch.allclose(
                metrics["loss"],
                metrics["jepa_term"]
                + metrics["cls_embedding_term"]
                + metrics["vicreg_term"],
            )
        )

    def test_teacher_targets_require_grad(self):
        model = self._build_model(masked_token_loss_weight=1.0)
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)
        teacher_targets = model._compute_jepa_teacher_targets(
            batch["peak_mz"],
            batch["peak_intensity"],
            batch["peak_valid_mask"],
        )
        self.assertTrue(teacher_targets.requires_grad)

    def test_compute_teacher_targets_expands_per_target_block(self):
        model = self._build_model(
            masked_token_loss_weight=1.0,
            jepa_target_layers=[1],
        )
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)
        expected = model._compute_jepa_teacher_targets(
            batch["peak_mz"],
            batch["peak_intensity"],
            batch["peak_valid_mask"],
        ).unsqueeze(1).expand(
            -1,
            model.jepa_num_target_blocks,
            -1,
            -1,
        )
        actual = model.compute_teacher_targets(batch)
        self.assertTrue(actual.requires_grad)
        self.assertTrue(torch.allclose(actual, expected))

    def test_pooled_teacher_peak_targets_require_grad(self):
        model = self._build_model()
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)
        cls_targets = model._compute_pooled_teacher_peak_targets(
            batch["peak_mz"],
            batch["peak_intensity"],
            batch["peak_valid_mask"],
        )
        self.assertTrue(cls_targets.requires_grad)

    def test_forward_augmented_uses_full_spectrum_teacher_targets(self):
        model = self._build_model(
            masked_token_loss_weight=1.0,
            jepa_target_layers=[1],
        )
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)
        teacher_targets = model._compute_jepa_teacher_targets(
            batch["peak_mz"],
            batch["peak_intensity"],
            batch["peak_valid_mask"],
        )
        teacher_targets = teacher_targets.unsqueeze(1).expand(
            -1,
            model.jepa_num_target_blocks,
            -1,
            -1,
        )

        expected = model.forward_augmented(batch, teacher_targets=teacher_targets)
        actual = model.forward_augmented(batch)

        for key in ("loss", "local_global_loss", "jepa_term"):
            self.assertTrue(
                torch.allclose(actual[key], expected[key], atol=1e-6, rtol=1e-6),
                key,
            )

    def test_cls_embedding_term_is_disabled(self):
        model = self._build_model(
            masked_token_loss_weight=1.0,
        )
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)
        metrics = model.forward_augmented(batch)
        self.assertEqual(float(metrics["cls_embedding_loss"].detach()), 0.0)
        self.assertEqual(float(metrics["cls_embedding_term"].detach()), 0.0)
        self.assertTrue(
            torch.allclose(
                metrics["loss"],
                metrics["jepa_term"]
                + metrics["cls_embedding_term"]
                + metrics["sigreg_term"],
            )
        )

    def test_cls_embedding_is_independent_of_token_target_normalization(self):
        torch.manual_seed(0)
        model_none = self._build_model(
            masked_token_loss_weight=0.0,
            jepa_target_normalization="none",
        )
        torch.manual_seed(0)
        model_zscore = self._build_model(
            masked_token_loss_weight=0.0,
            jepa_target_normalization="zscore",
        )
        batch = _make_batch(num_targets=model_none.jepa_num_target_blocks)

        metrics_none = model_none.forward_augmented(batch)
        metrics_zscore = model_zscore.forward_augmented(batch)

        self.assertEqual(float(metrics_none["cls_embedding_loss"]), 0.0)
        self.assertEqual(float(metrics_zscore["cls_embedding_loss"]), 0.0)

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
        batch["peak_intensity"][:, 0] = PRECURSOR_TOKEN_INTENSITY
        pooled = model.encode(batch)
        self.assertEqual(pooled.shape, (3, model.model_dim))

    def test_no_nan_from_sentinel_intensity(self):
        """The precursor sentinel intensity keeps log1p finite."""
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

    def test_sigreg_precursor_scale_weights_precursor_slot(self):
        for regularizer in ("sigreg-enc", "slot-sigreg-enc"):
            with self.subTest(regularizer=regularizer):
                model = self._build_model(
                    representation_regularizer=regularizer,
                    sigreg_lambda=0.02,
                    sigreg_precursor_scale=4.0,
                )
                batch = _make_pipeline_prepended_batch(
                    num_peaks=6,
                    num_targets=model.jepa_num_target_blocks,
                    precursor_in_context=True,
                )
                captured: dict[str, torch.Tensor] = {}

                def fake_sigreg_forward(
                    proj: torch.Tensor,
                    valid_mask: torch.Tensor | None = None,
                ) -> torch.Tensor:
                    captured["valid_mask"] = valid_mask.detach().clone()
                    return proj.new_zeros(())

                with mock.patch.object(
                    model.sigreg,
                    "forward",
                    side_effect=fake_sigreg_forward,
                ):
                    model.forward_augmented(batch)

                self.assertIn("valid_mask", captured)
                torch.testing.assert_close(
                    captured["valid_mask"][:, 0],
                    torch.full_like(captured["valid_mask"][:, 0], 4.0),
                )
                torch.testing.assert_close(
                    captured["valid_mask"][:, 1:],
                    batch["context_mask"][:, 1:].float(),
                )

    def test_sigreg_pred_respects_precursor_target_mask(self):
        for regularizer in ("sigreg-pred", "slot-sigreg-pred", "slog-sigreg-pred"):
            with self.subTest(regularizer=regularizer):
                model = self._build_model(
                    representation_regularizer=regularizer,
                    sigreg_lambda=0.02,
                )
                batch = _make_pipeline_prepended_batch(
                    num_peaks=6,
                    num_targets=model.jepa_num_target_blocks,
                )
                captured: dict[str, torch.Tensor] = {}

                def fake_sigreg_forward(
                    proj: torch.Tensor,
                    valid_mask: torch.Tensor | None = None,
                ) -> torch.Tensor:
                    captured["valid_mask"] = valid_mask.detach().clone()
                    return proj.new_zeros(())

                with mock.patch.object(
                    model.sigreg,
                    "forward",
                    side_effect=fake_sigreg_forward,
                ):
                    model.forward_augmented(batch)

                self.assertIn("valid_mask", captured)
                torch.testing.assert_close(
                    captured["valid_mask"],
                    batch["target_masks"].float(),
                )

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
        # Precursor token: valid=True, masks preserved with an unselected new slot.
        self.assertTrue(result["peak_valid_mask"][:, 0].all())
        self.assertFalse(result["context_mask"][:, 0].any())
        self.assertFalse(result["target_masks"][:, :, 0].any())
        # Precursor intensity sentinel
        torch.testing.assert_close(
            result["peak_intensity"][:, 0],
            torch.full((B,), PRECURSOR_TOKEN_INTENSITY),
        )


class PrependPrecursorTokenTests(unittest.TestCase):
    """Test the torch-side _prepend_precursor_token_torch function."""

    def test_shapes_and_values(self):
        from input_pipeline import _prepend_precursor_token_torch

        B, N, K = 4, 8, 2
        batch = {
            "peak_mz": torch.rand(B, N),
            "peak_intensity": torch.rand(B, N),
            "peak_valid_mask": torch.ones(B, N, dtype=torch.bool),
            "precursor_mz": torch.rand(B),
            "context_mask": torch.ones(B, N, dtype=torch.bool),
            "target_masks": torch.zeros(B, K, N, dtype=torch.bool),
            "rt": torch.zeros(B),
        }
        out = _prepend_precursor_token_torch(batch)

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
        np.testing.assert_allclose(
            out["peak_intensity"][:, 0].numpy(),
            np.full(B, PRECURSOR_TOKEN_INTENSITY, dtype=np.float32),
        )
        # Valid at position 0
        self.assertTrue(out["peak_valid_mask"][:, 0].numpy().all())
        # Existing masks are preserved; prepending does not force-select the precursor.
        self.assertFalse(out["context_mask"][:, 0].numpy().any())
        self.assertFalse(out["target_masks"][:, :, 0].numpy().any())
        # Precursor mz at position 0
        self.assertTrue(
            (out["peak_mz"][:, 0].numpy() == batch["precursor_mz"].numpy()).all()
        )

    def test_without_masks(self):
        """Works on raw (pre-augmentation) batches without context_mask/target_masks."""
        from input_pipeline import _prepend_precursor_token_torch

        B, N = 3, 5
        batch = {
            "peak_mz": torch.rand(B, N),
            "peak_intensity": torch.rand(B, N),
            "peak_valid_mask": torch.ones(B, N, dtype=torch.bool),
            "precursor_mz": torch.rand(B),
        }
        out = _prepend_precursor_token_torch(batch)

        self.assertNotIn("precursor_mz", out)
        self.assertEqual(out["peak_mz"].shape, (B, N + 1))
        self.assertNotIn("context_mask", out)
        self.assertNotIn("target_masks", out)


if __name__ == "__main__":
    unittest.main()
