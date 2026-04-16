import tempfile
import unittest
import math

import torch

from models.model import PeakSetSIGReg
from models.peak_features import FourierFeatures, PeakFeatureEmbedder
from train import _is_weight_decay_target
from utils.spectra_preprocessing import PEAK_MZ_MAX
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


class FourierFeatureTests(unittest.TestCase):
    def test_lin_float_int_respects_num_freqs(self):
        fourier = FourierFeatures(
            strategy="lin_float_int",
            x_min=1e-4,
            x_max=1000.0,
            num_freqs=512,
        )
        self.assertEqual(fourier.num_features(), 512)

    def test_peak_embedder_fourier_branch_recovers_raw_mz_scale(self):
        embedder = PeakFeatureEmbedder(
            model_dim=32,
            hidden_dim=16,
            fourier_strategy="lin_float_int",
            fourier_x_min=1e-4,
            fourier_x_max=1000.0,
            fourier_num_freqs=8,
        )
        normalized_peak_mz = torch.tensor([[0.5]], dtype=torch.float32)
        prepared = embedder._prepare_fourier_mz(normalized_peak_mz)
        self.assertAlmostEqual(
            float(prepared.item()),
            0.5 * PEAK_MZ_MAX,
            places=6,
        )


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
        model = self._build_model(
            masked_token_loss_weight=1.0,
            representation_regularizer="sigreg",
            sigreg_lambda=0.02,
        )
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

    def test_teacher_targets_require_grad_without_ema(self):
        model = self._build_model(masked_token_loss_weight=1.0)
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)
        teacher_targets = model._compute_jepa_teacher_targets(
            batch["peak_mz"],
            batch["peak_intensity"],
            batch["peak_valid_mask"],
        )
        self.assertTrue(teacher_targets.requires_grad)

    def test_teacher_targets_are_detached_with_ema(self):
        model = self._build_model(
            masked_token_loss_weight=1.0,
            use_ema_teacher_target=True,
        )
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)
        teacher_targets = model._compute_jepa_teacher_targets(
            batch["peak_mz"],
            batch["peak_intensity"],
            batch["peak_valid_mask"],
        )
        self.assertFalse(teacher_targets.requires_grad)

    def test_pooled_teacher_peak_targets_require_grad_without_ema(self):
        model = self._build_model(cls_embedding_loss_weight=1.0)
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)
        cls_targets = model._compute_pooled_teacher_peak_targets(
            batch["peak_mz"],
            batch["peak_intensity"],
            batch["peak_valid_mask"],
        )
        self.assertTrue(cls_targets.requires_grad)

    def test_pooled_teacher_peak_targets_are_detached_with_ema(self):
        model = self._build_model(
            cls_embedding_loss_weight=1.0,
            use_ema_teacher_target=True,
        )
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)
        cls_targets = model._compute_pooled_teacher_peak_targets(
            batch["peak_mz"],
            batch["peak_intensity"],
            batch["peak_valid_mask"],
        )
        self.assertFalse(cls_targets.requires_grad)

    def test_pooled_teacher_peak_targets_per_block_match_looped_teacher_forwards(self):
        model = self._build_model(
            cls_embedding_loss_weight=1.0,
            use_ema_teacher_target=True,
        )
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)

        batched_targets = model._compute_pooled_teacher_peak_targets_per_block(
            batch["peak_mz"],
            batch["peak_intensity"],
            batch["peak_valid_mask"],
            batch["context_mask"],
            batch["target_masks"],
        )

        expected_targets = []
        for target_idx in range(model.jepa_num_target_blocks):
            teacher_visible = batch["context_mask"] | batch["target_masks"][:, target_idx]
            pooled_target = model._compute_pooled_teacher_peak_targets(
                batch["peak_mz"],
                batch["peak_intensity"],
                batch["peak_valid_mask"],
                visible_mask=teacher_visible,
                pack_n=model._predictor_pack_n,
                prefix_pack=False,
            )
            expected_targets.append(pooled_target)
        expected_targets = torch.stack(expected_targets, dim=1)

        torch.testing.assert_close(batched_targets, expected_targets)

    def test_per_block_teacher_targets_match_looped_teacher_forwards(self):
        model = self._build_model(
            masked_token_loss_weight=1.0,
            use_ema_teacher_target=True,
            jepa_teacher_targets_per_block=True,
            jepa_target_layers=[1],
        )
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)

        batched_targets = model._compute_jepa_teacher_targets_per_block(
            batch["peak_mz"],
            batch["peak_intensity"],
            batch["peak_valid_mask"],
            batch["context_mask"],
            batch["target_masks"],
        )

        teacher = model._teacher_encoder_module()
        expected_targets = []
        for target_idx in range(model.jepa_num_target_blocks):
            teacher_visible = batch["context_mask"] | batch["target_masks"][:, target_idx]
            target_layers = teacher.forward_peak_block_outputs(
                batch["peak_mz"],
                batch["peak_intensity"],
                valid_mask=batch["peak_valid_mask"],
                visible_mask=teacher_visible,
                pack_n=model._predictor_pack_n,
                prefix_pack=False,
                block_indices=model.jepa_target_layers,
            )
            expected_targets.append(torch.cat(target_layers, dim=-1))
        expected_targets = torch.stack(expected_targets, dim=1)

        torch.testing.assert_close(batched_targets, expected_targets)

    def test_forward_augmented_uses_per_block_teacher_targets_when_enabled(self):
        model = self._build_model(
            masked_token_loss_weight=1.0,
            use_ema_teacher_target=True,
            jepa_teacher_targets_per_block=True,
        )
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)
        teacher_targets = model._compute_jepa_teacher_targets_per_block(
            batch["peak_mz"],
            batch["peak_intensity"],
            batch["peak_valid_mask"],
            batch["context_mask"],
            batch["target_masks"],
        )

        expected = model.forward_augmented(batch, teacher_targets=teacher_targets)
        actual = model.forward_augmented(batch)

        for key in ("loss", "local_global_loss", "jepa_term"):
            self.assertTrue(
                torch.allclose(actual[key], expected[key], atol=1e-6, rtol=1e-6),
                key,
            )

    def test_cls_embedding_term_contributes_to_loss(self):
        model = self._build_model(
            masked_token_loss_weight=1.0,
            cls_embedding_loss_weight=0.5,
        )
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)
        metrics = model.forward_augmented(batch)
        self.assertGreater(float(metrics["cls_embedding_loss"].detach()), 0.0)
        self.assertGreater(float(metrics["cls_embedding_term"].detach()), 0.0)
        self.assertTrue(
            torch.allclose(
                metrics["loss"],
                metrics["jepa_term"]
                + metrics["cls_embedding_term"]
                + metrics["sigreg_term"],
            )
        )

    def test_cls_embedding_uses_context_plus_target_visibility_in_block_mode(self):
        model = self._build_model(
            masked_token_loss_weight=0.0,
            cls_embedding_loss_weight=1.0,
            use_ema_teacher_target=True,
            jepa_teacher_targets_per_block=True,
        )
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)

        context_encoded = model.encoder(
            batch["peak_mz"],
            batch["peak_intensity"],
            valid_mask=batch["peak_valid_mask"],
            visible_mask=batch["context_mask"],
            pack_n=model._context_pack_n,
        )
        _, context_cls = model.encoder.split_peak_and_cls(context_encoded)
        cls_target = model._compute_pooled_teacher_peak_targets_per_block(
            batch["peak_mz"],
            batch["peak_intensity"],
            batch["peak_valid_mask"],
            batch["context_mask"],
            batch["target_masks"],
        )

        metrics = model.forward_augmented(batch)

        torch.testing.assert_close(
            metrics["cls_embedding_loss"],
            model._embedding_loss(
                context_cls.unsqueeze(1).expand(-1, model.jepa_num_target_blocks, -1),
                cls_target,
            ).mean(),
        )
        expected_visible_fraction = (
            ((batch["context_mask"].unsqueeze(1) | batch["target_masks"]).float().sum())
            / batch["peak_valid_mask"].float().sum()
        )
        torch.testing.assert_close(
            metrics["cls_visible_fraction"],
            expected_visible_fraction,
        )

    def test_cls_embedding_uses_full_visibility_in_full_mode(self):
        model = self._build_model(
            masked_token_loss_weight=0.0,
            cls_embedding_loss_weight=1.0,
            use_ema_teacher_target=True,
            jepa_teacher_targets_per_block=False,
        )
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)

        full_encoded = model.encoder(
            batch["peak_mz"],
            batch["peak_intensity"],
            valid_mask=batch["peak_valid_mask"],
            visible_mask=batch["peak_valid_mask"],
            pack_n=model._full_pack_n,
            prefix_pack=True,
        )
        _, full_cls = model.encoder.split_peak_and_cls(full_encoded)
        cls_target = model._compute_pooled_teacher_peak_targets(
            batch["peak_mz"],
            batch["peak_intensity"],
            batch["peak_valid_mask"],
        )

        metrics = model.forward_augmented(batch)

        torch.testing.assert_close(
            metrics["cls_embedding_loss"],
            model._embedding_loss(full_cls, cls_target).mean(),
        )
        self.assertEqual(float(metrics["cls_visible_fraction"]), 1.0)

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
            pack_n=model._full_pack_n,
            prefix_pack=True,
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

    def test_teacher_ema_warmup_uses_cosine_schedule(self):
        model = self._build_model(
            use_ema_teacher_target=True,
            teacher_ema_decay_start=0.9,
            teacher_ema_decay=0.99,
            teacher_ema_decay_warmup_steps=4,
        )
        expected = []
        for step in range(5):
            ratio = min(step / 4.0, 1.0)
            cosine_ratio = 0.5 * (1.0 - math.cos(math.pi * ratio))
            expected.append(0.9 + 0.09 * cosine_ratio)

        actual = []
        for _ in range(5):
            model.advance_teacher_ema_decay_schedule()
            actual.append(float(model.teacher_ema_decay_current))

        for got, want in zip(actual, expected, strict=True):
            self.assertAlmostEqual(got, want, places=6)

    def test_teacher_ema_zero_warmup_stays_at_target_decay(self):
        model = self._build_model(
            use_ema_teacher_target=True,
            teacher_ema_decay_start=0.9,
            teacher_ema_decay=0.99,
            teacher_ema_decay_warmup_steps=0,
        )
        self.assertAlmostEqual(float(model.teacher_ema_decay_current), 0.99, places=6)
        model.update_teacher()
        self.assertAlmostEqual(float(model.teacher_ema_decay_current), 0.99, places=6)

    def test_teacher_ema_schedule_advances_per_train_step_not_update_cadence(self):
        shared_kwargs = dict(
            use_ema_teacher_target=True,
            teacher_ema_decay_start=0.9,
            teacher_ema_decay=0.99,
            teacher_ema_decay_warmup_steps=4,
        )
        model_u1 = self._build_model(
            **shared_kwargs,
            teacher_ema_update_every=1,
        )
        model_u2 = self._build_model(
            **shared_kwargs,
            teacher_ema_update_every=2,
        )

        for _ in range(5):
            model_u1.update_teacher()
            model_u2.update_teacher()
            self.assertAlmostEqual(
                float(model_u1.teacher_ema_decay_current),
                float(model_u2.teacher_ema_decay_current),
                places=6,
            )

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
