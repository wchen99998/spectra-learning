import tempfile
import unittest
import math
from typing import cast
from unittest import mock

import torch

from spectra_learning.models.model import PeakSetJEPA
from spectra_learning.models.pairformer import PairFeatureEmbedder
from spectra_learning.models.peak_features import FourierFeatures, PeakFeatureEmbedder
from spectra_learning.training.optimization import is_weight_decay_target
from spectra_learning.training.steps import train_step_impl
from spectra_learning.training.api import (
    load_frozen_teacher_weights,
    load_pretrained_weights,
)


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
    def test_log_spaced_fourier_preserves_feature_width_without_trainable_params(self):
        fourier = FourierFeatures(
            x_min=3e-3,
            x_max=1000.0,
            num_freqs=256,
        )
        self.assertEqual(fourier.num_features(), 512)
        self.assertEqual(list(fourier.parameters()), [])
        self.assertFalse(fourier.b.requires_grad)

    def test_peak_embedder_fourier_branch_uses_configured_input_scale(self):
        embedder = PeakFeatureEmbedder(
            model_dim=32,
            hidden_dim=16,
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

    def test_peak_embedder_without_fourier_uses_full_capacity_raw_mlp(self):
        embedder = PeakFeatureEmbedder(
            model_dim=32,
            hidden_dim=16,
            fourier_mlp_hidden_dim=64,
            fourier_mlp_num_layers=4,
            fourier_x_min=3e-3,
            fourier_x_max=1000.0,
            fourier_num_freqs=8,
            use_fourier_features=False,
        )

        self.assertFalse(hasattr(embedder, "mz_fourier"))
        self.assertFalse(hasattr(embedder, "fourier_ffn"))
        linear_layers = [
            layer for layer in embedder.raw_ffn if isinstance(layer, torch.nn.Linear)
        ]
        self.assertEqual(len(linear_layers), 4)
        self.assertEqual(linear_layers[0].in_features, 3)
        self.assertEqual(linear_layers[0].out_features, 64)
        self.assertEqual(linear_layers[-1].out_features, 32)

        peak_mz = torch.rand(2, 5)
        peak_intensity = torch.rand(2, 5)
        output = embedder(peak_mz, peak_intensity)
        self.assertEqual(output.shape, (2, 5, 32))

    def test_peak_embedder_raw_branch_uses_normal_log_intensity(self):
        embedder = PeakFeatureEmbedder(
            model_dim=32,
            hidden_dim=16,
            fourier_x_min=3e-3,
            fourier_x_max=1000.0,
            fourier_num_freqs=8,
        )
        captured: dict[str, torch.Tensor] = {}

        def capture_raw_input(_module, args):
            captured["raw_input"] = args[0].detach().clone()

        handle = embedder.raw_ffn.register_forward_pre_hook(capture_raw_input)
        peak_mz = torch.tensor([[0.25]], dtype=torch.float32)
        peak_intensity = torch.tensor([[0.5]], dtype=torch.float32)
        embedder(peak_mz, peak_intensity)
        handle.remove()

        expected = torch.tensor(
            [[[0.25, 0.5, math.log1p(0.5)]]],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(captured["raw_input"], expected))


class PairformerEncoderTests(unittest.TestCase):
    def _build_model(self, **kwargs) -> PeakSetJEPA:
        model_kwargs = {
            "model_dim": 32,
            "encoder_num_layers": 1,
            "encoder_num_heads": 4,
            "attention_mlp_multiple": 2.0,
            "feature_mlp_hidden_dim": 16,
            "num_peaks": 6,
            "jepa_num_target_blocks": 2,
            "masked_token_loss_weight": 1.0,
            "pairformer_pair_dim": 32,
            "pairformer_pair_num_heads": 4,
            "pairformer_pair_feature_hidden_dim": 16,
        }
        model_kwargs.update(kwargs)
        return PeakSetJEPA(**model_kwargs)

    def test_forward_uses_precursor_mass_features(self):
        model = self._build_model()
        batch = _make_batch(
            num_peaks=6,
            num_targets=model.jepa_num_target_blocks,
            include_precursor=True,
        )
        batch["precursor_mz"] = batch["precursor_mz"] / 1000.0

        metrics = model.forward_augmented(batch)

        self.assertEqual(model.num_peak_tokens, 6)
        self.assertTrue(torch.isfinite(metrics["loss"]).item())

    def test_pair_features_use_precursor_mz_without_adduct_shift(self):
        embedder = PairFeatureEmbedder(
            single_dim=4,
            pair_dim=8,
            hidden_dim=16,
            mz_scale=1000.0,
            precursor_mz_scale=2000.0,
        )
        peak_mz = torch.tensor([[0.10, 0.20, 0.30]])
        valid_mask = torch.tensor([[True, True, False]])
        precursor_mz = torch.tensor([0.40])

        reference_mass = embedder._reference_mass_da(
            peak_mz,
            valid_mask,
            precursor_mz,
        )
        fallback_mass = embedder._reference_mass_da(
            peak_mz,
            valid_mask,
            None,
        )

        torch.testing.assert_close(reference_mass, torch.tensor([800.0]))
        torch.testing.assert_close(fallback_mass, torch.tensor([200.0]))

    def test_pair_features_include_compact_fourier_encodings(self):
        embedder = PairFeatureEmbedder(
            single_dim=4,
            pair_dim=8,
            hidden_dim=16,
            mz_scale=1000.0,
            precursor_mz_scale=1000.0,
            fourier_num_freqs=4,
            fourier_x_min=1e-2,
            fourier_x_max=1000.0,
            relative_fourier_x_min=1e-3,
            relative_fourier_x_max=1.0,
        )
        captured: dict[str, torch.Tensor] = {}

        def capture_raw_input(_module, args):
            captured["raw_input"] = args[0].detach().clone()

        handle = embedder.raw_proj.register_forward_pre_hook(capture_raw_input)
        peak_mz = torch.tensor([[0.10, 0.25]], dtype=torch.float32)
        peak_intensity = torch.tensor([[1.0, 0.5]], dtype=torch.float32)
        single = torch.zeros(1, 2, 4)
        valid_mask = torch.ones(1, 2, dtype=torch.bool)
        precursor_mz = torch.tensor([0.50], dtype=torch.float32)
        embedder(
            peak_mz,
            peak_intensity,
            single,
            valid_mask,
            precursor_mz=precursor_mz,
        )
        handle.remove()

        raw = captured["raw_input"]
        scalar_dim = 14 + embedder.mass_differences.numel()
        pair_fourier_dim = 3 * embedder.pair_fourier.num_features()
        relative_fourier_dim = embedder.relative_pair_fourier.num_features()
        self.assertEqual(
            raw.shape[-1],
            scalar_dim + pair_fourier_dim + relative_fourier_dim,
        )

        raw_pair = raw[0, 0, 1]
        torch.testing.assert_close(
            raw_pair[:4],
            torch.tensor([0.15, 0.15, 0.30, -0.15]),
        )
        expected_da_fourier = embedder._fourier_values(
            embedder.pair_fourier,
            torch.tensor([[[[150.0, 150.0, -150.0]]]]),
        ).flatten()
        expected_relative_fourier = embedder._fourier_values(
            embedder.relative_pair_fourier,
            torch.tensor([[[[0.30]]]]),
        ).flatten()
        torch.testing.assert_close(
            raw_pair[scalar_dim : scalar_dim + pair_fourier_dim],
            expected_da_fourier,
        )
        torch.testing.assert_close(
            raw_pair[scalar_dim + pair_fourier_dim :],
            expected_relative_fourier,
        )

    def test_padding_values_do_not_leak_into_visible_peak_embeddings(self):
        model = self._build_model()
        model.eval()
        valid_mask = torch.tensor(
            [
                [True, True, True, True, False, False],
                [True, True, True, False, False, False],
            ]
        )
        peak_mz = torch.rand(2, 6)
        peak_intensity = torch.rand(2, 6)
        peak_mz[:, 4:] = 0.0
        peak_intensity[:, 4:] = 0.0
        randomized_mz = peak_mz.clone()
        randomized_intensity = peak_intensity.clone()
        randomized_mz[~valid_mask] = torch.rand_like(randomized_mz[~valid_mask])
        randomized_intensity[~valid_mask] = torch.rand_like(
            randomized_intensity[~valid_mask]
        )

        with torch.no_grad():
            base = model.encoder(
                peak_mz,
                peak_intensity,
                valid_mask=valid_mask,
                visible_mask=valid_mask,
            )
            randomized = model.encoder(
                randomized_mz,
                randomized_intensity,
                valid_mask=valid_mask,
                visible_mask=valid_mask,
            )

        torch.testing.assert_close(base[valid_mask], randomized[valid_mask])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for compile test")
    def test_cuda_pairformer_works_with_torch_compile(self):
        model = self._build_model(
            num_peaks=8,
            jepa_num_target_blocks=1,
            pairformer_use_cuequivariance=True,
        ).cuda()
        compiled_encoder = torch.compile(model.encoder, mode="reduce-overhead")
        peak_mz = torch.linspace(0.05, 0.6, 8, device="cuda").view(1, 8)
        peak_intensity = torch.rand(1, 8, device="cuda")
        valid_mask = torch.ones(1, 8, device="cuda", dtype=torch.bool)
        precursor_mz = torch.tensor([0.8], device="cuda")

        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = compiled_encoder(
                peak_mz,
                peak_intensity,
                valid_mask=valid_mask,
                visible_mask=valid_mask,
                precursor_mz=precursor_mz,
            )

        loss = output.float().square().mean()
        loss.backward()
        grads = [p.grad for p in model.encoder.parameters() if p.requires_grad]
        self.assertEqual(output.shape, (1, 8, model.model_dim))
        self.assertTrue(torch.isfinite(output.float()).all().item())
        self.assertTrue(any(g is not None for g in grads))


class BlockJEPATests(unittest.TestCase):
    def _build_model(self, **kwargs) -> PeakSetJEPA:
        model_kwargs = {
            "model_dim": 32,
            "encoder_num_layers": 1,
            "encoder_num_heads": 4,
            "attention_mlp_multiple": 2.0,
            "feature_mlp_hidden_dim": 16,
            "jepa_num_target_blocks": 2,
        }
        model_kwargs.update(kwargs)
        return PeakSetJEPA(**model_kwargs)

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
            target_projector_dim=-1,
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
            "target_fraction_per_view",
            "target_union_fraction",
            "target_entry_fraction",
            "target_overlap_entries",
        ):
            self.assertIn(key, metrics, f"Missing key: {key}")

    def test_forward_can_return_raw_collapse_data(self):
        model = self._build_model()
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)
        metrics, collapse_data = model.forward_augmented(batch, return_collapse_data=True)
        self.assertIn("loss", metrics)
        for key in (
            "teacher_peak_emb",
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
        pooled_targets = model._compute_pooled_teacher_peak_targets(
            batch["peak_mz"],
            batch["peak_intensity"],
            batch["peak_valid_mask"],
        )
        self.assertTrue(pooled_targets.requires_grad)

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
        context_encoded, _, context_pair = model.encoder.forward_with_block_outputs(
            peak_mz,
            peak_intensity,
            valid_mask=peak_valid_mask,
            visible_mask=context_mask,
        )
        context_emb = context_encoded
        _, predictor_output = model._predict_augmented_targets(
            context_emb,
            context_pair,
            context_mask,
            target_masks,
        )
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

    def test_distogram_targets_use_shared_mz_bin_size(self):
        model = self._build_model(
            distogram_loss_weight=1.0,
            jepa_mae_mz_bin_size=250.0,
        )
        peak_mz = torch.tensor([[0.0, 0.249, 0.25, 1.0, 1.5]])

        targets = model._distogram_targets(peak_mz)

        self.assertEqual(model.distogram_num_bins, 4)
        torch.testing.assert_close(
            targets[0, 0],
            torch.tensor([0, 0, 1, 3, 3]),
        )

    def test_distogram_pair_mask_uses_pairs_with_either_target_endpoint(self):
        model = self._build_model(distogram_loss_weight=1.0)
        target_masks = torch.tensor([[[False, True, False, False]]])
        predictor_visible_masks = torch.tensor([[[True, True, True, False]]])

        pair_mask = model._distogram_pair_mask(
            target_masks,
            predictor_visible_masks,
        )

        expected = torch.tensor(
            [
                [
                    [
                        [False, True, False, False],
                        [True, False, True, False],
                        [False, True, False, False],
                        [False, False, False, False],
                    ]
                ]
            ]
        )
        torch.testing.assert_close(pair_mask, expected)

    def test_distogram_logits_symmetrise_predictor_pairs(self):
        model = self._build_model(
            distogram_loss_weight=1.0,
            jepa_mae_mz_bin_size=250.0,
        )
        predictor_pair = torch.randn(2, 1, 4, 4, model.predictor_pair_dim)

        logits = model._distogram_logits(predictor_pair)

        self.assertEqual(logits.shape[-1], 4)
        torch.testing.assert_close(logits[:, :, 1, 2], logits[:, :, 2, 1])

    def test_distogram_loss_contributes_to_loss(self):
        model = self._build_model(
            masked_token_loss_weight=1.0,
            distogram_loss_weight=0.25,
        )
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)

        metrics = model.forward_augmented(batch)

        self.assertGreater(float(metrics["distogram_loss"].detach()), 0.0)
        torch.testing.assert_close(
            metrics["distogram_term"],
            metrics["distogram_loss"] * 0.25,
        )
        torch.testing.assert_close(
            metrics["loss"],
            metrics["masked_prediction_term"] + metrics["distogram_term"],
        )

    def test_mae_training_mode_uses_binned_value_prediction_only(self):
        model = self._build_model(
            training_mode="mae",
            masked_token_loss_weight=1.0,
            jepa_mae_loss_weight=1.0,
            mae_loss_weight=0.5,
        )
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)

        with mock.patch.object(
            model,
            "_encode_augmented_teacher_and_context",
            side_effect=AssertionError("teacher path should not run"),
        ):
            metrics = model.forward_augmented(batch)

        self.assertIn("mae_loss", metrics)
        self.assertIn("mae_mz_loss", metrics)
        self.assertIn("mae_intensity_loss", metrics)
        self.assertIn("mae_mz_accuracy", metrics)
        self.assertIn("mae_intensity_accuracy", metrics)
        self.assertNotIn("masked_prediction_loss", metrics)
        self.assertNotIn("jepa_mae_loss", metrics)
        torch.testing.assert_close(metrics["mae_term"], metrics["mae_loss"] * 0.5)
        torch.testing.assert_close(metrics["loss"], metrics["mae_term"])

    def test_mae_training_mode_disables_ema_teacher(self):
        model = self._build_model(
            training_mode="mae",
            use_ema_teacher=True,
            masked_token_loss_weight=1.0,
            jepa_mae_loss_weight=1.0,
        )
        self.assertFalse(model.use_ema_teacher)
        self.assertIsNone(model.teacher_encoder)
        self.assertIsNone(model.teacher_target_projector)
        self.assertEqual(model.masked_token_loss_weight, 0.0)
        self.assertEqual(model.jepa_mae_loss_weight, 0.0)

    def test_mae_teacher_jepa_mode_uses_frozen_teacher_without_ema(self):
        model = self._build_model(
            training_mode="mae_teacher_jepa",
            use_ema_teacher=True,
            masked_token_loss_weight=1.0,
            jepa_mae_loss_weight=1.0,
        )
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)

        metrics = model.forward_augmented(batch)
        teacher_encoder = model.teacher_encoder
        assert teacher_encoder is not None
        before_teacher = next(teacher_encoder.parameters()).detach().clone()
        momentum = model.update_ema_teacher(step=1, total_steps=10)
        after_teacher = next(teacher_encoder.parameters()).detach()

        self.assertFalse(model.use_ema_teacher)
        self.assertTrue(model.use_frozen_teacher)
        self.assertIsNotNone(model.teacher_encoder)
        self.assertIsNotNone(model.teacher_target_projector)
        self.assertIn("masked_prediction_loss", metrics)
        self.assertNotIn("mae_loss", metrics)
        self.assertEqual(model.jepa_mae_loss_weight, 1.0)
        self.assertIsNone(momentum)
        torch.testing.assert_close(after_teacher, before_teacher)
        self.assertTrue(
            all(not param.requires_grad for param in teacher_encoder.parameters())
        )

    def test_mae_teacher_jepa_uses_separate_frozen_teacher_config_last_layer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            teacher_config_path = f"{tmpdir}/teacher_config.py"
            with open(teacher_config_path, "w") as f:
                f.write(
                    "from ml_collections import config_dict\n\n"
                    "def get_config():\n"
                    "    cfg = config_dict.ConfigDict()\n"
                    "    cfg.training_mode = 'mae'\n"
                    "    cfg.model_dim = 24\n"
                    "    cfg.encoder_num_layers = 2\n"
                    "    cfg.encoder_num_heads = 4\n"
                    "    cfg.feature_mlp_hidden_dim = 16\n"
                    "    cfg.encoder_fourier_num_freqs = 8\n"
                    "    return cfg\n"
                )
            model = self._build_model(
                training_mode="mae_teacher_jepa",
                frozen_teacher_config_path=teacher_config_path,
                encoder_num_layers=1,
                jepa_target_layers=[1, 2],
                jepa_target_normalization="zscore",
                target_projector_dim=-1,
            )

        self.assertEqual(model.model_dim, 32)
        self.assertEqual(model.teacher_model_dim, 24)
        self.assertEqual(model.teacher_encoder_num_layers, 2)
        self.assertEqual(model.jepa_target_layers, [2])
        self.assertEqual(model.jepa_target_dim, 24)
        self.assertEqual(model.jepa_target_group_dim, 24)
        self.assertEqual(model.masked_latent_readout.out_features, 24)
        self.assertIsNotNone(model.teacher_encoder)

        batch = _make_batch(num_targets=model.jepa_num_target_blocks)
        metrics = model.forward_augmented(batch)

        self.assertIn("masked_prediction_loss", metrics)
        self.assertTrue(torch.isfinite(metrics["loss"]).item())

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
        teacher_encoder = model.teacher_encoder
        assert teacher_encoder is not None

        with (
            mock.patch.object(
                teacher_encoder,
                "forward_with_block_outputs",
                wraps=teacher_encoder.forward_with_block_outputs,
            ) as teacher_forward,
            mock.patch.object(
                model.encoder,
                "forward_with_block_outputs",
                wraps=model.encoder.forward_with_block_outputs,
            ) as student_forward,
        ):
            loss = model.forward_augmented(batch)["loss"]
            loss.backward()

        self.assertEqual(teacher_forward.call_count, 1)
        self.assertEqual(student_forward.call_count, 1)
        teacher_grads = [p.grad for p in teacher_encoder.parameters()]
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
        teacher_target_projector = model.teacher_target_projector
        assert teacher_target_projector is not None

        with (
            mock.patch.object(
                teacher_target_projector,
                "forward",
                wraps=teacher_target_projector.forward,
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
            p.grad for p in teacher_target_projector.parameters()
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
            ema_teacher_momentum_start=0.5,
            ema_teacher_momentum_final=0.9,
            ema_teacher_schedule="linear",
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)
        before_student = next(model.encoder.parameters()).detach().clone()
        teacher_encoder = model.teacher_encoder
        teacher_target_projector = model.teacher_target_projector
        assert teacher_encoder is not None
        assert teacher_target_projector is not None
        before_teacher = next(teacher_encoder.parameters()).detach().clone()
        before_student_projector = next(model.target_projector.parameters()).detach().clone()
        before_teacher_projector = (
            next(teacher_target_projector.parameters()).detach().clone()
        )

        metrics = train_step_impl(
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
        after_teacher = next(teacher_encoder.parameters()).detach()
        after_student_projector = next(model.target_projector.parameters()).detach()
        after_teacher_projector = (
            next(teacher_target_projector.parameters()).detach()
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

    def test_train_step_updates_ema_teacher_with_disabled_target_projector(self):
        model = self._build_model(
            encoder_num_layers=2,
            jepa_target_layers=[1, 2],
            masked_token_loss_weight=1.0,
            target_projector_dim=-1,
            use_ema_teacher=True,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)

        metrics = train_step_impl(
            model,
            batch,
            [optimizer],
            [scheduler],
            autocast_dtype=None,
            grad_clip_norm=None,
            global_step=1,
            total_steps=4,
        )

        self.assertIsInstance(model.target_projector, torch.nn.Identity)
        self.assertIn("ema_teacher_momentum", metrics)
        self.assertTrue(torch.isfinite(metrics["loss"]).item())

    def test_slow_fast_slow_ema_schedule_uses_total_steps(self):
        model = self._build_model(
            use_ema_teacher=True,
            ema_teacher_momentum_start=0.9995,
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

    def test_encode_output_shape(self):
        model = self._build_model()
        batch = {
            "peak_mz": torch.rand(3, 6),
            "peak_intensity": torch.rand(3, 6),
            "peak_valid_mask": torch.ones(3, 6, dtype=torch.bool),
        }
        pooled = model.encode(batch)
        self.assertEqual(pooled.shape, (3, model.model_dim))

    def test_encode_returns_masked_mean_pool(self):
        model = self._build_model()
        batch = {
            "peak_mz": torch.rand(3, 6),
            "peak_intensity": torch.rand(3, 6),
            "peak_valid_mask": torch.ones(3, 6, dtype=torch.bool),
        }

        peak_emb = model.encoder(
            batch["peak_mz"],
            batch["peak_intensity"],
            valid_mask=batch["peak_valid_mask"],
            visible_mask=batch["peak_valid_mask"],
        )
        encoded = model.encode(batch)

        self.assertEqual(peak_emb.shape, (3, 6, model.model_dim))
        self.assertTrue(
            torch.allclose(
                encoded,
                model.pool(peak_emb, batch["peak_valid_mask"]),
            )
        )

    def test_model_omits_special_token_parameters(self):
        model = self._build_model()
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
            metrics = train_step_impl(
                model,
                batch,
                [optimizer],
                [scheduler],
                autocast_dtype=None,
                grad_clip_norm=None,
            )

        forward_augmented_mock.assert_called_once_with(batch)
        self.assertIn("loss", metrics)

    def test_train_step_impl_supports_grad_scaler(self):
        model = self._build_model(masked_token_loss_weight=1.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        grad_scaler = torch.amp.GradScaler(
            device="cpu",
            init_scale=16.0,
            growth_interval=1,
            enabled=True,
        )
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)
        before = next(model.encoder.parameters()).detach().clone()

        metrics = train_step_impl(
            model,
            batch,
            [optimizer],
            [scheduler],
            autocast_dtype=None,
            grad_clip_norm=1.0,
            grad_scaler=grad_scaler,
        )

        after = next(model.encoder.parameters()).detach()
        self.assertIn("grad_scale", metrics)
        self.assertEqual(float(metrics["optimizer_step_skipped"]), 0.0)
        self.assertGreaterEqual(float(metrics["grad_scale"]), 16.0)
        self.assertFalse(torch.equal(before, after))
        self.assertEqual(scheduler.last_epoch, 1)
        self.assertTrue(all(param.grad is None for param in model.parameters()))

    def test_load_pretrained_weights_roundtrip(self):
        model = self._build_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/ckpt.pt"
            torch.save({"state_dict": model.state_dict()}, path)
            loaded = self._build_model()
            load_pretrained_weights(loaded, path)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, loaded.state_dict()[key]), key)

    def test_load_frozen_teacher_weights_uses_mae_encoder_only(self):
        source = self._build_model(training_mode="mae")
        with torch.no_grad():
            for param in source.encoder.parameters():
                param.fill_(0.123)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/mae.pt"
            torch.save({"model": source.state_dict()}, path)
            loaded = self._build_model(training_mode="mae_teacher_jepa")
            before_student = next(loaded.encoder.parameters()).detach().clone()

            load_frozen_teacher_weights(loaded, path)

            source_encoder_param = next(source.encoder.parameters()).detach()
            loaded_teacher_encoder = loaded.teacher_encoder
            assert loaded_teacher_encoder is not None
            loaded_teacher_param = next(loaded_teacher_encoder.parameters()).detach()
            loaded_student_param = next(loaded.encoder.parameters()).detach()
            torch.testing.assert_close(loaded_teacher_param, source_encoder_param)
            torch.testing.assert_close(loaded_student_param, before_student)
            self.assertTrue(
                all(
                    not param.requires_grad
                    for param in loaded_teacher_encoder.parameters()
                )
            )

    def test_load_pretrained_weights_rejects_missing_mask_and_position_embeddings(self):
        model = self._build_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/ckpt.pt"
            old_state = {
                k: v
                for k, v in model.state_dict().items()
                if not k.endswith(
                    (
                        "position_embedding.weight",
                        "latent_mask_token",
                    )
                )
            }
            torch.save({"state_dict": old_state}, path)
            loaded = self._build_model()
            with self.assertRaisesRegex(RuntimeError, "Missing key"):
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
            is_weight_decay_target(
                "encoder.embedder.output_proj.weight",
                model.encoder.embedder.output_proj.weight,
            )
        )
        self.assertTrue(
            is_weight_decay_target(
                "encoder.embedder.fourier_ffn.0.weight",
                cast(torch.nn.Linear, model.encoder.embedder.fourier_ffn[0]).weight,
            )
        )
        self.assertFalse(
            is_weight_decay_target(
                "encoder.embedder.mz_fourier.b",
                model.encoder.embedder.mz_fourier.b,
            )
        )


if __name__ == "__main__":
    unittest.main()
