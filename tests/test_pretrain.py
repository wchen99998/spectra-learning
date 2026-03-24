import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from models.model import PeakSetSIGReg
from utils.training import build_model_from_config
from utils.training import collect_runtime_norm_metrics
from utils.training import load_pretrained_weights
from ml_collections import config_dict


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
            "jepa_projector_num_layers": 0,
            "jepa_projector_dim": None,
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
            sigreg_lambda=0.0,
        )
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)
        metrics = model.forward_augmented(batch)
        self.assertAlmostEqual(float(metrics["token_sigreg_loss"]), 0.0, places=7)
        self.assertAlmostEqual(float(metrics["sigreg_term"]), 0.0, places=7)
        self.assertTrue(torch.allclose(metrics["loss"], metrics["jepa_term"]))
        for key in (
            "context_encoder_output_norm",
            "teacher_encoder_output_norm",
            "predictor_output_norm",
        ):
            self.assertIn(key, metrics)
            self.assertTrue(torch.isfinite(metrics[key]).item())

    def test_encode_output_shape(self):
        model = self._build_model()
        batch = {
            "peak_mz": torch.rand(3, 6),
            "peak_intensity": torch.rand(3, 6),
            "peak_valid_mask": torch.ones(3, 6, dtype=torch.bool),
        }
        pooled = model.encode(batch)
        self.assertEqual(pooled.shape, (3, model.model_dim))

    def test_backward_populates_encoder_gradients(self):
        model = self._build_model()
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)
        loss = model.forward_augmented(batch)["loss"]
        loss.backward()
        grads = [p.grad for p in model.encoder.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None for g in grads))

    def test_projector_teacher_path_runs_under_no_grad(self):
        model = self._build_model(jepa_projector_num_layers=1)

        class CaptureProjector(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.grad_enabled: list[bool] = []

            def forward(self, x):
                self.grad_enabled.append(torch.is_grad_enabled())
                return x

        capture = CaptureProjector()
        model.jepa_projector = capture
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)
        model.forward_augmented(batch)
        self.assertEqual(capture.grad_enabled, [True, False])

    def test_projector_receives_gradients(self):
        model = self._build_model(jepa_projector_num_layers=2, jepa_projector_dim=24)
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)
        loss = model.forward_augmented(batch)["loss"]
        loss.backward()
        grads = [p.grad for p in model.jepa_projector.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None for g in grads))

    def test_projector_supports_custom_output_dim(self):
        model = self._build_model(jepa_projector_num_layers=2, jepa_projector_dim=24)
        last_linear = [m for m in model.jepa_projector.modules() if isinstance(m, torch.nn.Linear)][-1]
        self.assertEqual(last_linear.out_features, 24)
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)
        metrics = model.forward_augmented(batch)
        self.assertTrue(torch.isfinite(metrics["loss"]).item())

    def test_projector_uses_packed_masked_tokens(self):
        model = self._build_model(jepa_projector_num_layers=1, num_peaks=64)

        class CaptureProjector(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.shapes: list[tuple[int, ...]] = []

            def forward(self, x):
                self.shapes.append(tuple(x.shape))
                return x

        capture = CaptureProjector()
        model.jepa_projector = capture
        batch = _make_batch(num_peaks=64, num_targets=model.jepa_num_target_blocks)

        model.forward_augmented(batch)

        expected_shape = (
            batch["peak_mz"].shape[0] * model.jepa_num_target_blocks,
            min(model._target_pack_n, batch["peak_mz"].shape[1]),
            model.model_dim,
        )
        self.assertEqual(capture.shapes, [expected_shape, expected_shape])

    @torch.no_grad()
    def test_packed_projector_matches_dense_loss(self):
        model = self._build_model(
            num_peaks=12,
            jepa_projector_num_layers=2,
            masked_token_loss_type="l2",
            normalize_jepa_targets=True,
            sigreg_lambda=0.0,
        )
        batch = _make_batch(num_peaks=12, num_targets=model.jepa_num_target_blocks)
        metrics = model.forward_augmented(batch)

        peak_mz = batch["peak_mz"]
        peak_intensity = batch["peak_intensity"]
        peak_valid_mask = batch["peak_valid_mask"]
        context_mask = batch["context_mask"] & peak_valid_mask
        target_masks = batch["target_masks"] & peak_valid_mask.unsqueeze(1)
        B, K, N = target_masks.shape
        target_mask_flat = target_masks.reshape(B * K, N)

        context_emb = model._encoder_forward(
            peak_mz,
            peak_intensity,
            valid_mask=peak_valid_mask,
            visible_mask=context_mask,
            pack_n=model._context_pack_n,
            prefix_pack=False,
            pad_to=model._context_pad_to,
        )
        teacher = model._teacher_encoder_forward or model._encoder_forward
        teacher_full = teacher(
            peak_mz,
            peak_intensity,
            valid_mask=peak_valid_mask,
            visible_mask=peak_valid_mask,
            pack_n=model._teacher_pack_n,
            prefix_pack=True,
            pad_to=model._teacher_pad_to,
        ).detach()
        target_token_target = teacher_full.unsqueeze(1).expand(-1, K, -1, -1)

        ctx_mask_v = context_mask.unsqueeze(1)
        predictor_queries = torch.zeros_like(context_emb.unsqueeze(1).expand(-1, K, -1, -1))
        predictor_queries = torch.where(
            target_masks.unsqueeze(-1),
            model.latent_mask_token.view(1, 1, 1, -1).to(context_emb),
            predictor_queries,
        )
        predictor_output = model.predict_masked_latents(
            predictor_queries.reshape(B * K, N, -1),
            context_emb.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, N, -1),
            target_mask=target_masks.reshape(B * K, N),
            context_mask=ctx_mask_v.expand(-1, K, -1).reshape(B * K, N),
            pack_n=model._predictor_pack_n,
        )

        dense_pred = model.jepa_projector(predictor_output)
        dense_target = model.jepa_projector(target_token_target.reshape(B * K, N, -1))
        dense_pred = F.layer_norm(dense_pred, (dense_pred.shape[-1],))
        dense_target = F.layer_norm(dense_target, (dense_target.shape[-1],))
        dense_pred = F.normalize(dense_pred, dim=-1)
        dense_target = F.normalize(dense_target, dim=-1)
        per_token_reg = (dense_pred - dense_target).square().mean(dim=-1)
        manual_loss = (
            per_token_reg * target_mask_flat.float()
        ).sum() / target_mask_flat.float().sum().clamp_min(1.0)

        self.assertTrue(torch.allclose(metrics["local_global_loss"], manual_loss, atol=1e-6, rtol=1e-6))

    @torch.no_grad()
    def test_target_pack_capacity_does_not_truncate_two_token_targets(self):
        model = self._build_model(
            num_peaks=6,
            jepa_projector_num_layers=2,
            masked_token_loss_type="l2",
            normalize_jepa_targets=True,
            sigreg_lambda=0.0,
            jepa_projector_dim=24,
        )
        batch = _make_batch(num_peaks=6, num_targets=model.jepa_num_target_blocks)
        batch["target_masks"] = torch.tensor(
            [
                [
                    [False, False, True, True, False, False],
                    [False, False, False, False, True, True],
                ],
                [
                    [False, True, True, False, False, False],
                    [False, False, False, True, True, False],
                ],
                [
                    [False, False, True, True, False, False],
                    [False, False, False, False, True, True],
                ],
                [
                    [False, True, True, False, False, False],
                    [False, False, False, True, True, False],
                ],
            ],
            dtype=torch.bool,
        )

        metrics = model.forward_augmented(batch)
        self.assertEqual(model._target_pack_n, 2)
        peak_mz = batch["peak_mz"]
        peak_intensity = batch["peak_intensity"]
        peak_valid_mask = batch["peak_valid_mask"]
        context_mask = batch["context_mask"] & peak_valid_mask
        target_masks = batch["target_masks"] & peak_valid_mask.unsqueeze(1)
        B, K, N = target_masks.shape
        target_mask_flat = target_masks.reshape(B * K, N)

        context_emb = model._encoder_forward(
            peak_mz,
            peak_intensity,
            valid_mask=peak_valid_mask,
            visible_mask=context_mask,
            pack_n=model._context_pack_n,
            prefix_pack=False,
            pad_to=model._context_pad_to,
        )
        teacher = model._teacher_encoder_forward or model._encoder_forward
        teacher_full = teacher(
            peak_mz,
            peak_intensity,
            valid_mask=peak_valid_mask,
            visible_mask=peak_valid_mask,
            pack_n=model._teacher_pack_n,
            prefix_pack=True,
            pad_to=model._teacher_pad_to,
        ).detach()
        target_token_target = teacher_full.unsqueeze(1).expand(-1, K, -1, -1)

        ctx_mask_v = context_mask.unsqueeze(1)
        predictor_queries = torch.zeros_like(context_emb.unsqueeze(1).expand(-1, K, -1, -1))
        predictor_queries = torch.where(
            target_masks.unsqueeze(-1),
            model.latent_mask_token.view(1, 1, 1, -1).to(context_emb),
            predictor_queries,
        )
        predictor_output = model.predict_masked_latents(
            predictor_queries.reshape(B * K, N, -1),
            context_emb.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, N, -1),
            target_mask=target_mask_flat,
            context_mask=ctx_mask_v.expand(-1, K, -1).reshape(B * K, N),
            pack_n=model._predictor_pack_n,
        )

        dense_pred = model.jepa_projector(predictor_output)
        dense_target = model.jepa_projector(target_token_target.reshape(B * K, N, -1))
        dense_pred = F.layer_norm(dense_pred, (dense_pred.shape[-1],))
        dense_target = F.layer_norm(dense_target, (dense_target.shape[-1],))
        dense_pred = F.normalize(dense_pred, dim=-1)
        dense_target = F.normalize(dense_target, dim=-1)
        per_token_reg = (dense_pred - dense_target).square().mean(dim=-1)
        manual_loss = (
            per_token_reg * target_mask_flat.float()
        ).sum() / target_mask_flat.float().sum().clamp_min(1.0)

        self.assertTrue(torch.allclose(metrics["local_global_loss"], manual_loss, atol=1e-6, rtol=1e-6))

    def test_projector_space_outputs_are_layer_normed(self):
        model = self._build_model(
            jepa_projector_num_layers=2,
            jepa_projector_dim=24,
            masked_token_loss_type="l2",
            normalize_jepa_targets=False,
            sigreg_lambda=0.0,
        )
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)
        metrics = model.forward_augmented(batch)
        expected_norm = 24**0.5
        self.assertAlmostEqual(float(metrics["projected_pred_norm"]), expected_norm, places=3)
        self.assertAlmostEqual(float(metrics["projected_target_norm"]), expected_norm, places=3)

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

    def test_build_model_from_config_wires_projector_dim(self):
        cfg = config_dict.ConfigDict(
            {
                "model_dim": 32,
                "num_layers": 1,
                "num_heads": 4,
                "num_kv_heads": 4,
                "attention_mlp_multiple": 2.0,
                "feature_mlp_hidden_dim": 16,
                "sigreg_num_slices": 32,
                "sigreg_lambda": 0.1,
                "jepa_num_target_blocks": 2,
                "jepa_projector_num_layers": 2,
                "jepa_projector_dim": 20,
            }
        )
        model = build_model_from_config(cfg)
        self.assertEqual(model.jepa_projector_dim, 20)

    def test_collect_runtime_norm_metrics_reports_encoder_and_projector(self):
        model = self._build_model(jepa_projector_num_layers=2, jepa_projector_dim=24)
        batch = _make_batch(num_targets=model.jepa_num_target_blocks)
        loss = model.forward_augmented(batch)["loss"]
        loss.backward()
        metrics = collect_runtime_norm_metrics(model)
        for key in (
            "param_norm/total",
            "param_norm/encoder",
            "param_norm/jepa_projector",
            "grad_norm/total",
            "grad_norm/encoder",
            "grad_norm/jepa_projector",
        ):
            self.assertIn(key, metrics)
            self.assertTrue(np.isfinite(metrics[key]))


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
