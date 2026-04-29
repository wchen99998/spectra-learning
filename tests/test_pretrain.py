import tempfile
import unittest
import math
from unittest import mock

import numpy as np
import torch

from models.losses import SlotwiseSIGReg
from models.model import PeakSetSIGReg
from models.peak_features import FourierFeatures, PeakFeatureEmbedder
from train import _is_weight_decay_target, _train_step_impl
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

    def test_peak_embedder_fourier_branch_supports_deeper_mlp(self):
        embedder = PeakFeatureEmbedder(
            model_dim=32,
            hidden_dim=16,
            fourier_mlp_hidden_dim=64,
            fourier_mlp_num_layers=4,
            fourier_strategy="log_spaced",
            fourier_x_min=3e-3,
            fourier_x_max=1000.0,
            fourier_num_freqs=8,
        )
        linear_layers = [
            layer for layer in embedder.fourier_ffn if isinstance(layer, torch.nn.Linear)
        ]
        self.assertEqual(len(linear_layers), 4)
        self.assertEqual(linear_layers[0].in_features, 16)
        self.assertEqual(linear_layers[0].out_features, 64)
        self.assertEqual(linear_layers[-1].out_features, 16)

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

    def test_target_projector_maps_multilayer_targets_to_model_dim(self):
        model = self._build_model(
            encoder_num_layers=2,
            jepa_target_layers=[1, 2],
        )
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)

        metrics, collapse_data = model.forward_augmented(
            batch,
            return_collapse_data=True,
        )

        self.assertTrue(torch.isfinite(metrics["loss"]).item())
        self.assertIsInstance(model.target_projector, torch.nn.Sequential)
        self.assertEqual(model.target_projector_dim, model.model_dim)
        self.assertEqual(
            collapse_data["teacher_targets"].shape[-1],
            model.model_dim,
        )
        self.assertEqual(
            collapse_data["predictor_output"].shape[-1],
            model.model_dim,
        )

    def test_target_projector_can_be_disabled(self):
        model = self._build_model(
            encoder_num_layers=2,
            jepa_target_layers=[1, 2],
            use_target_projector=False,
        )
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)

        metrics, collapse_data = model.forward_augmented(
            batch,
            return_collapse_data=True,
        )

        self.assertTrue(torch.isfinite(metrics["loss"]).item())
        self.assertIsInstance(model.target_projector, torch.nn.Identity)
        self.assertEqual(model.target_projector_dim, model.jepa_target_dim)
        self.assertEqual(
            collapse_data["teacher_targets"].shape[-1],
            model.jepa_target_dim,
        )
        self.assertEqual(
            collapse_data["predictor_output"].shape[-1],
            model.jepa_target_dim,
        )
        torch.testing.assert_close(
            collapse_data["teacher_targets"],
            collapse_data["teacher_target_features_normalized"],
        )

    def test_forward_contains_expected_keys(self):
        model = self._build_model()
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)
        metrics = model.forward_augmented(batch)
        for key in (
            "loss",
            "masked_prediction_loss",
            "masked_prediction_term",
            "context_fraction",
            "target_fraction",
        ):
            self.assertIn(key, metrics, f"Missing key: {key}")

    def test_forward_can_return_raw_collapse_data(self):
        model = self._build_model()
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)
        metrics, collapse_data = model.forward_augmented(batch, return_collapse_data=True)
        self.assertIn("loss", metrics)
        for key in (
            "teacher_peak_emb",
            "teacher_cls_emb",
            "context_emb",
            "context_mask",
            "peak_valid_mask",
            "target_masks",
            "teacher_target_features",
            "teacher_target_features_normalized",
            "teacher_targets",
            "predictor_output_features",
            "predictor_output",
            "pooled_mean",
        ):
            self.assertIn(key, collapse_data, f"Missing key: {key}")
            self.assertFalse(collapse_data[key].requires_grad)

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
                self.assertIn("sigreg_loss", metrics)
                self.assertGreater(float(metrics["sigreg_loss"].detach()), 0.0)
                self.assertGreater(float(metrics["sigreg_term"].detach()), 0.0)
                self.assertTrue(
                    torch.allclose(
                        metrics["loss"],
                        metrics["masked_prediction_term"] + metrics["sigreg_term"],
                    )
                )

    def test_sigreg_on_predictor_outputs_uses_masked_predictions(self):
        for regularizer in ("sigreg-pred", "slot-sigreg-pred", "slog-sigreg-pred"):
            with self.subTest(regularizer=regularizer):
                model = self._build_model(
                    encoder_num_layers=2,
                    jepa_target_layers=[1, 2],
                    representation_regularizer=regularizer,
                    sigreg_lambda=0.02,
                    predictor_dim=16,
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
                self.assertEqual(captured["proj"].shape[-1], model.jepa_target_dim)
                self.assertNotEqual(
                    captured["proj"].shape[-1], model.target_projector_dim
                )

    def test_sigreg_on_encoder_and_predictor_outputs_combines_losses(self):
        for regularizer in ("sigreg-enc-pred", "slot-sigreg-enc-pred"):
            with self.subTest(regularizer=regularizer):
                model = self._build_model(
                    encoder_num_layers=2,
                    jepa_target_layers=[1, 2],
                    representation_regularizer=regularizer,
                    sigreg_lambda=0.02,
                    predictor_dim=16,
                )
                if regularizer.startswith("slot-"):
                    self.assertIsInstance(model.sigreg, SlotwiseSIGReg)
                batch = _make_batch(num_targets=model.jepa_num_target_blocks)
                captured: list[tuple[torch.Tensor, torch.Tensor]] = []

                def fake_sigreg_forward(
                    proj: torch.Tensor,
                    valid_mask: torch.Tensor | None = None,
                ) -> torch.Tensor:
                    captured.append(
                        (proj.detach().clone(), valid_mask.detach().clone())
                    )
                    return proj.new_tensor(float(len(captured)))

                with mock.patch.object(
                    model.sigreg,
                    "forward",
                    side_effect=fake_sigreg_forward,
                ):
                    metrics = model.forward_augmented(batch)

                self.assertEqual(len(captured), 2)
                encoder_proj, encoder_mask = captured[0]
                predictor_proj, predictor_mask = captured[1]
                self.assertEqual(
                    encoder_proj.shape,
                    (*batch["context_mask"].shape, model.model_dim),
                )
                torch.testing.assert_close(
                    encoder_mask,
                    batch["context_mask"].float(),
                )
                self.assertEqual(
                    predictor_proj.shape,
                    (*batch["target_masks"].shape, model.jepa_target_dim),
                )
                torch.testing.assert_close(
                    predictor_mask,
                    batch["target_masks"].float(),
                )
                torch.testing.assert_close(
                    metrics["sigreg_encoder_loss"],
                    metrics["sigreg_encoder_loss"].new_tensor(1.0),
                )
                torch.testing.assert_close(
                    metrics["sigreg_predictor_loss"],
                    metrics["sigreg_predictor_loss"].new_tensor(2.0),
                )
                torch.testing.assert_close(
                    metrics["sigreg_loss"],
                    metrics["sigreg_loss"].new_tensor(3.0),
                )

    def test_sigreg_on_projected_outputs_uses_projected_student_targets(self):
        for regularizer in ("sigreg-proj", "slot-sigreg-proj", "slog-sigreg-proj"):
            with self.subTest(regularizer=regularizer):
                model = self._build_model(
                    encoder_num_layers=2,
                    representation_regularizer=regularizer,
                    sigreg_lambda=0.02,
                    jepa_target_layers=[1, 2],
                )
                if regularizer.startswith("slot-") or regularizer.startswith("slog-"):
                    self.assertIsInstance(model.sigreg, SlotwiseSIGReg)
                batch = _make_batch(num_targets=model.jepa_num_target_blocks)
                captured: list[tuple[torch.Tensor, torch.Tensor]] = []

                def fake_sigreg_forward(
                    proj: torch.Tensor,
                    valid_mask: torch.Tensor | None = None,
                ) -> torch.Tensor:
                    captured.append(
                        (proj.detach().clone(), valid_mask.detach().clone())
                    )
                    return proj.new_tensor(float(len(captured)))

                with mock.patch.object(
                    model.sigreg,
                    "forward",
                    side_effect=fake_sigreg_forward,
                ):
                    metrics = model.forward_augmented(batch)

                self.assertEqual(len(captured), 1)
                student_proj, student_mask = captured[0]
                self.assertEqual(
                    student_proj.shape,
                    (*batch["target_masks"].shape, model.target_projector_dim),
                )
                torch.testing.assert_close(
                    student_mask,
                    batch["target_masks"].float(),
                )
                torch.testing.assert_close(
                    metrics["sigreg_loss"],
                    metrics["sigreg_loss"].new_tensor(1.0),
                )

    def test_sigreg_on_encoder_outputs_uses_visible_context_only(self):
        for regularizer in ("sigreg-enc", "slot-sigreg-enc"):
            with self.subTest(regularizer=regularizer):
                model = self._build_model(
                    encoder_num_layers=2,
                    representation_regularizer=regularizer,
                    sigreg_lambda=0.02,
                    jepa_target_layers=[1, 2],
                )
                batch = _make_batch(num_targets=model.jepa_num_target_blocks)
                captured: list[tuple[torch.Tensor, torch.Tensor]] = []

                def fake_sigreg_forward(
                    proj: torch.Tensor,
                    valid_mask: torch.Tensor | None = None,
                ) -> torch.Tensor:
                    captured.append(
                        (proj.detach().clone(), valid_mask.detach().clone())
                    )
                    return proj.new_tensor(float(len(captured)))

                with mock.patch.object(
                    model.sigreg,
                    "forward",
                    side_effect=fake_sigreg_forward,
                ):
                    metrics = model.forward_augmented(batch)

                self.assertEqual(len(captured), 1)
                context_proj, context_mask = captured[0]
                self.assertEqual(
                    context_proj.shape,
                    (*batch["context_mask"].shape, model.model_dim),
                )
                torch.testing.assert_close(
                    context_mask,
                    batch["context_mask"].float(),
                )
                torch.testing.assert_close(metrics["sigreg_loss"], metrics["sigreg_loss"].new_tensor(1.0))

    def test_covariance_pooling_does_not_add_sigreg_loss(self):
        model = self._build_model(
            masked_token_loss_weight=1.0,
            train_covariance_pooling=True,
            covariance_pooling_dim=4,
            sigreg_lambda=0.03,
        )
        self.assertTrue(hasattr(model, "covariance_pooler"))
        self.assertFalse(hasattr(model, "covariance_sigreg"))
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)
        metrics = model.forward_augmented(batch)

        self.assertNotIn("covariance_sigreg_loss", metrics)
        self.assertNotIn("covariance_sigreg_term", metrics)
        torch.testing.assert_close(metrics["loss"], metrics["masked_prediction_term"])

    def test_teacher_targets_are_detached(self):
        model = self._build_model(masked_token_loss_weight=1.0)
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)
        teacher_targets = model._compute_jepa_teacher_targets(
            batch["peak_mz"],
            batch["peak_intensity"],
            batch["peak_valid_mask"],
        )
        self.assertFalse(teacher_targets.requires_grad)

    def test_compute_teacher_targets_returns_full_spectrum_once(self):
        model = self._build_model(
            masked_token_loss_weight=1.0,
            jepa_target_layers=[1],
        )
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)
        expected = model._compute_jepa_teacher_targets(
            batch["peak_mz"],
            batch["peak_intensity"],
            batch["peak_valid_mask"],
        )
        actual = model.compute_teacher_targets(batch)
        self.assertFalse(actual.requires_grad)
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
        peak_mz = batch["peak_mz"]
        peak_intensity = batch["peak_intensity"]
        peak_valid_mask = batch["peak_valid_mask"]
        context_mask = batch["context_mask"] & peak_valid_mask
        target_masks = batch["target_masks"] & peak_valid_mask.unsqueeze(1)
        B, K, N = target_masks.shape
        teacher_targets = model._compute_jepa_teacher_targets(
            peak_mz,
            peak_intensity,
            peak_valid_mask,
        )
        context_encoded = model.encoder(
            peak_mz,
            peak_intensity,
            valid_mask=peak_valid_mask,
            visible_mask=context_mask,
        )
        context_emb, _ = model.encoder.split_peak_and_cls(context_encoded)
        predictor_input = context_emb.unsqueeze(1).expand(-1, K, -1, -1)
        predictor_input = predictor_input * context_mask.unsqueeze(1).unsqueeze(-1)
        predictor_input = torch.where(
            target_masks.unsqueeze(-1),
            model.latent_mask_token.view(1, 1, 1, -1).to(context_emb),
            predictor_input,
        )
        predictor_output = model.predict_masked_targets(
            predictor_input.reshape(B * K, N, -1),
            (context_mask.unsqueeze(1) | target_masks).reshape(B * K, N),
        ).reshape(B, K, N, -1)
        expected_masked_prediction_loss = (
            model._embedding_loss(predictor_output, teacher_targets.unsqueeze(1))
            * target_masks.float()
        ).sum() / target_masks.float().sum().clamp_min(1.0)
        expected_masked_prediction_term = model.masked_token_loss_weight * expected_masked_prediction_loss

        actual = model.forward_augmented(batch)

        self.assertTrue(
            torch.allclose(
                actual["masked_prediction_loss"],
                expected_masked_prediction_loss,
                atol=1e-6,
                rtol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(actual["masked_prediction_term"], expected_masked_prediction_term, atol=1e-6, rtol=1e-6)
        )
        self.assertTrue(
            torch.allclose(actual["loss"], expected_masked_prediction_term, atol=1e-6, rtol=1e-6)
        )

    def test_jepa_mae_targets_use_configured_bins(self):
        model = self._build_model(
            jepa_mae_loss_weight=1.0,
            jepa_mae_mz_bin_size=2.5,
            jepa_mae_intensity_bin_size=0.1,
        )
        peak_mz = torch.tensor([[0.0, 0.0024, 0.0025, 0.999, 1.1]])
        peak_intensity = torch.tensor([[0.0, 0.099, 0.1, 0.999, 1.1]])

        mz_target, intensity_target = model._jepa_mae_targets(
            peak_mz,
            peak_intensity,
        )

        torch.testing.assert_close(
            mz_target,
            torch.tensor([[0, 0, 1, 399, 399]]),
        )
        torch.testing.assert_close(
            intensity_target,
            torch.tensor([[0, 0, 1, 9, 9]]),
        )

    def test_jepa_mae_value_prediction_contributes_to_loss(self):
        model = self._build_model(
            masked_token_loss_weight=1.0,
            jepa_mae_loss_weight=0.25,
        )
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)

        metrics = model.forward_augmented(batch)

        self.assertGreater(float(metrics["jepa_mae_loss"].detach()), 0.0)
        torch.testing.assert_close(
            metrics["jepa_mae_term"],
            metrics["jepa_mae_loss"] * 0.25,
        )
        torch.testing.assert_close(
            metrics["loss"],
            metrics["masked_prediction_term"]
            + metrics["jepa_mae_term"],
        )

    def test_forward_augmented_uses_single_encoder_pass(self):
        model = self._build_model(masked_token_loss_weight=1.0)
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)

        with mock.patch.object(
            model.encoder,
            "forward_with_block_outputs",
            wraps=model.encoder.forward_with_block_outputs,
        ) as encoder_forward:
            metrics = model.forward_augmented(batch)

        self.assertEqual(encoder_forward.call_count, 1)
        self.assertTrue(torch.isfinite(metrics["loss"]))

    def test_ema_teacher_uses_separate_stop_gradient_encoder(self):
        model = self._build_model(
            masked_token_loss_weight=1.0,
            use_ema_teacher=True,
        )
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)

        with (
            mock.patch.object(
                model.teacher_encoder,
                "forward_with_block_outputs",
                wraps=model.teacher_encoder.forward_with_block_outputs,
            ) as teacher_forward,
            mock.patch.object(
                model.encoder,
                "forward",
                wraps=model.encoder.forward,
            ) as student_forward,
        ):
            loss = model.forward_augmented(batch)["loss"]
            loss.backward()

        self.assertEqual(teacher_forward.call_count, 1)
        self.assertEqual(student_forward.call_count, 1)
        teacher_grads = [p.grad for p in model.teacher_encoder.parameters()]
        self.assertTrue(all(grad is None for grad in teacher_grads))
        student_grads = [
            p.grad for p in model.encoder.parameters() if p.requires_grad
        ]
        self.assertTrue(any(grad is not None for grad in student_grads))

    def test_ema_teacher_uses_separate_stop_gradient_target_projector(self):
        model = self._build_model(
            masked_token_loss_weight=1.0,
            use_ema_teacher=True,
        )
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)

        with (
            mock.patch.object(
                model.teacher_target_projector,
                "forward",
                wraps=model.teacher_target_projector.forward,
            ) as teacher_projector_forward,
            mock.patch.object(
                model.target_projector,
                "forward",
                wraps=model.target_projector.forward,
            ) as student_projector_forward,
        ):
            loss = model.forward_augmented(batch)["loss"]
            loss.backward()

        self.assertEqual(teacher_projector_forward.call_count, 1)
        self.assertEqual(student_projector_forward.call_count, 1)
        teacher_projector_grads = [
            p.grad for p in model.teacher_target_projector.parameters()
        ]
        self.assertTrue(all(grad is None for grad in teacher_projector_grads))
        student_projector_grads = [
            p.grad for p in model.target_projector.parameters() if p.requires_grad
        ]
        self.assertTrue(any(grad is not None for grad in student_projector_grads))

    def test_train_step_updates_ema_teacher_with_schedule(self):
        model = self._build_model(
            masked_token_loss_weight=1.0,
            use_ema_teacher=True,
            ema_teacher_momentum=0.5,
            ema_teacher_momentum_final=0.9,
            ema_teacher_schedule="linear",
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)
        before_student = next(model.encoder.parameters()).detach().clone()
        before_teacher = next(model.teacher_encoder.parameters()).detach().clone()
        before_student_projector = next(model.target_projector.parameters()).detach().clone()
        before_teacher_projector = (
            next(model.teacher_target_projector.parameters()).detach().clone()
        )

        metrics = _train_step_impl(
            model,
            batch,
            [optimizer],
            [scheduler],
            autocast_dtype=None,
            grad_clip_norm=None,
            global_step=1,
            total_steps=4,
        )

        after_student = next(model.encoder.parameters()).detach()
        after_teacher = next(model.teacher_encoder.parameters()).detach()
        after_student_projector = next(model.target_projector.parameters()).detach()
        after_teacher_projector = (
            next(model.teacher_target_projector.parameters()).detach()
        )
        expected_momentum = 0.7
        self.assertIn("ema_teacher_momentum", metrics)
        self.assertAlmostEqual(
            float(metrics["ema_teacher_momentum"]),
            expected_momentum,
            places=6,
        )
        self.assertFalse(torch.equal(before_student, after_student))
        self.assertTrue(
            torch.allclose(
                after_teacher,
                before_teacher * expected_momentum
                + after_student * (1.0 - expected_momentum),
                atol=1e-6,
                rtol=1e-6,
            )
        )
        self.assertFalse(torch.equal(before_student_projector, after_student_projector))
        self.assertTrue(
            torch.allclose(
                after_teacher_projector,
                before_teacher_projector * expected_momentum
                + after_student_projector * (1.0 - expected_momentum),
                atol=1e-6,
                rtol=1e-6,
            )
        )

    def test_slow_fast_slow_ema_schedule_uses_total_steps(self):
        model = self._build_model(
            use_ema_teacher=True,
            ema_teacher_momentum=0.9995,
            ema_teacher_momentum_mid=0.99,
            ema_teacher_momentum_final=0.999,
            ema_teacher_schedule_peak_fraction=0.35,
            ema_teacher_schedule="slow-fast-slow",
        )

        self.assertAlmostEqual(model.ema_teacher_momentum_at(0, 100), 0.9995)
        self.assertAlmostEqual(model.ema_teacher_momentum_at(35, 100), 0.99)
        self.assertAlmostEqual(model.ema_teacher_momentum_at(100, 100), 0.999)
        self.assertLess(
            model.ema_teacher_momentum_at(20, 100),
            model.ema_teacher_momentum_at(5, 100),
        )
        self.assertGreater(
            model.ema_teacher_momentum_at(70, 100),
            model.ema_teacher_momentum_at(45, 100),
        )

    def test_forward_augmented_does_not_report_disabled_cls_metrics(self):
        model = self._build_model(
            masked_token_loss_weight=1.0,
        )
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)
        metrics = model.forward_augmented(batch)
        self.assertNotIn("cls_embedding_loss", metrics)
        self.assertNotIn("cls_embedding_term", metrics)
        self.assertTrue(
            torch.allclose(
                metrics["loss"],
                metrics["masked_prediction_term"],
            )
        )

    def test_disabled_cls_metrics_are_not_affected_by_target_normalization(self):
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

        self.assertNotIn("cls_embedding_loss", metrics_none)
        self.assertNotIn("cls_embedding_loss", metrics_zscore)

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

    def test_encoder_cls_token_can_be_disabled(self):
        model = self._build_model(
            encoder_use_cls_token=False,
            encoder_num_register_tokens=2,
        )
        batch = {
            "peak_mz": torch.rand(3, 6),
            "peak_intensity": torch.rand(3, 6),
            "peak_valid_mask": torch.ones(3, 6, dtype=torch.bool),
        }

        peak_emb, pooled_emb = model.encoder(
            batch["peak_mz"],
            batch["peak_intensity"],
            valid_mask=batch["peak_valid_mask"],
            visible_mask=batch["peak_valid_mask"],
            return_cls_token=True,
        )
        encoded = model.encode(batch)

        self.assertIsNone(model.encoder.cls_token)
        self.assertEqual(peak_emb.shape, (3, 6, model.model_dim))
        self.assertTrue(torch.allclose(encoded, pooled_emb))
        self.assertTrue(
            torch.allclose(
                encoded,
                model.pool(peak_emb, batch["peak_valid_mask"]),
            )
        )

    def test_register_tokens_can_be_disabled(self):
        model = self._build_model(
            encoder_num_register_tokens=0,
            predictor_num_register_tokens=0,
        )
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)

        metrics = model.forward_augmented(batch)

        self.assertTrue(torch.isfinite(metrics["loss"]).item())
        self.assertEqual(model.encoder.num_register_tokens, 0)
        self.assertIsNone(model.encoder.register_tokens)
        self.assertEqual(model.predictor_num_register_tokens, 0)
        self.assertIsNone(model.predictor_register_tokens)

    def test_all_special_tokens_can_be_disabled(self):
        model = self._build_model(
            encoder_use_cls_token=False,
            encoder_num_register_tokens=0,
            predictor_num_register_tokens=0,
        )
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)

        metrics, collapse_data = model.forward_augmented(
            batch,
            return_collapse_data=True,
        )

        self.assertTrue(torch.isfinite(metrics["loss"]).item())
        self.assertEqual(
            collapse_data["teacher_peak_emb"].shape[1],
            batch["peak_mz"].shape[1],
        )
        self.assertEqual(
            collapse_data["context_emb"].shape[1],
            batch["peak_mz"].shape[1],
        )
        self.assertNotIn("encoder.cls_token", model.state_dict())
        self.assertNotIn("encoder.register_tokens", model.state_dict())
        self.assertNotIn("predictor_register_tokens", model.state_dict())

    def test_backward_populates_encoder_gradients(self):
        model = self._build_model()
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)
        loss = model.forward_augmented(batch)["loss"]
        loss.backward()
        grads = [p.grad for p in model.encoder.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None for g in grads))

    def test_train_step_impl_uses_forward_augmented_entrypoint(self):
        model = self._build_model(masked_token_loss_weight=1.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)
        fake_loss = torch.tensor(1.0, requires_grad=True)

        with mock.patch.object(
            model,
            "forward_augmented",
            return_value={"loss": fake_loss},
        ) as forward_augmented_mock:
            metrics = _train_step_impl(
                model,
                batch,
                [optimizer],
                [scheduler],
                autocast_dtype=None,
                grad_clip_norm=None,
            )

        forward_augmented_mock.assert_called_once_with(batch)
        self.assertIn("loss", metrics)

    def test_load_pretrained_weights_roundtrip(self):
        model = self._build_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/ckpt.pt"
            torch.save({"state_dict": model.state_dict()}, path)
            loaded = self._build_model()
            load_pretrained_weights(loaded, path)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, loaded.state_dict()[key]), key)

    def test_load_pretrained_weights_rejects_missing_position_embeddings(self):
        model = self._build_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/ckpt.pt"
            old_state = {
                k: v
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
            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                load_pretrained_weights(loaded, path)

    def test_load_pretrained_weights_rejects_removed_special_tokens(self):
        model = self._build_model(
            encoder_num_register_tokens=2,
            predictor_num_register_tokens=2,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/ckpt.pt"
            torch.save({"state_dict": model.state_dict()}, path)
            loaded = self._build_model(
                encoder_use_cls_token=False,
                encoder_num_register_tokens=0,
                predictor_num_register_tokens=0,
            )
            with self.assertRaisesRegex(RuntimeError, "Unexpected key"):
                load_pretrained_weights(loaded, path)

    def test_load_pretrained_weights_rejects_missing_masked_latent_readout(self):
        model = self._build_model(jepa_target_layers=[1])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/ckpt.pt"
            old_state = {
                k: v
                for k, v in model.state_dict().items()
                if not k.startswith(("masked_latent_readout.", "target_projector."))
            }
            torch.save({"state_dict": old_state}, path)
            loaded = self._build_model(jepa_target_layers=[1])
            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                load_pretrained_weights(loaded, path)

    def test_load_pretrained_weights_rejects_missing_ema_teacher(self):
        model = self._build_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/ckpt.pt"
            torch.save({"model": model.state_dict()}, path)
            loaded = self._build_model(use_ema_teacher=True)
            with self.assertRaisesRegex(RuntimeError, "Missing key"):
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

    def test_model_sizes_position_tables_from_real_peaks_plus_precursor(self):
        model = self._build_model(num_peaks=6, use_precursor_token=True)

        self.assertEqual(model.encoder.position_embedding.num_embeddings, 7)
        self.assertEqual(model.predictor_position_embedding.num_embeddings, 7)

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

    def test_forward_augmented_forces_precursor_context_not_target(self):
        model = self._build_model()
        batch = _make_pipeline_prepended_batch(
            num_peaks=6,
            num_targets=model.jepa_num_target_blocks,
            precursor_in_context=False,
            precursor_in_targets=True,
        )

        _, collapse_data = model.forward_augmented(batch, return_collapse_data=True)

        self.assertTrue(collapse_data["context_mask"][:, 0].all())
        self.assertFalse(collapse_data["target_masks"][:, :, 0].any())

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
                captured_masks: list[torch.Tensor] = []

                def fake_sigreg_forward(
                    proj: torch.Tensor,
                    valid_mask: torch.Tensor | None = None,
                ) -> torch.Tensor:
                    captured_masks.append(valid_mask.detach().clone())
                    return proj.new_zeros(())

                with mock.patch.object(
                    model.sigreg,
                    "forward",
                    side_effect=fake_sigreg_forward,
                ):
                    model.forward_augmented(batch)

                self.assertEqual(len(captured_masks), 1)
                context_mask = captured_masks[0]
                torch.testing.assert_close(
                    context_mask[:, 0],
                    torch.full_like(context_mask[:, 0], 4.0),
                )
                torch.testing.assert_close(
                    context_mask[:, 1:],
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
        # Precursor token: valid=True, context-visible, never a target.
        self.assertTrue(result["peak_valid_mask"][:, 0].all())
        self.assertTrue(result["context_mask"][:, 0].all())
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
        # Existing masks are preserved after the always-visible precursor.
        self.assertTrue(out["context_mask"][:, 0].numpy().all())
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
