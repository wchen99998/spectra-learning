import unittest
import tempfile

import torch

from utils.training import load_pretrained_weights
from models.losses import SIGReg, VICRegLoss
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
        "fused_masked_positions": torch.zeros(VB, num_peaks, dtype=torch.bool),
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
    def _build_model(self, *, encoder_use_rope: bool = False, **kwargs) -> PeakSetSIGReg:
        model_kwargs = {
            "num_peaks": 60,
            "model_dim": 32,
            "encoder_num_layers": 1,
            "encoder_num_heads": 4,
            "encoder_num_kv_heads": 4,
            "attention_mlp_multiple": 2.0,
            "feature_mlp_hidden_dim": 16,
            "sigreg_num_slices": 32,
            "sigreg_lambda": 0.1,
            "multicrop_num_local_views": 3,
            "multicrop_local_keep_fraction": 0.25,
            "sigreg_mz_jitter_std": 0.005,
            "sigreg_intensity_jitter_std": 0.05,
            "encoder_use_rope": encoder_use_rope,
        }
        model_kwargs.update(kwargs)
        return PeakSetSIGReg(**model_kwargs)

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
            "masked_fraction",
            "sigreg_lambda_current",
        ):
            self.assertIn(key, metrics, f"Missing key: {key}")

    def test_forward_vicreg_contains_expected_keys(self):
        model = self._build_model(
            representation_regularizer="vicreg",
            sigreg_lambda=0.0,
            vicreg_beta=1e-3,
            vicreg_sim_coeff=0.0,
            vicreg_std_coeff=25.0,
            vicreg_cov_coeff=1.0,
            masked_token_loss_weight=1.0,
            multicrop_num_local_views=3,
        )
        batch = _make_fused_batch(num_views=model.num_views)
        metrics = model.forward_augmented(batch)
        for key in (
            "loss",
            "vicreg_loss",
            "vicreg_term",
            "regularizer_loss",
            "regularizer_term",
            "jepa_term",
            "target_regularizer_term_over_jepa_term",
            "target_vicreg_term_over_jepa_term",
        ):
            self.assertIn(key, metrics, f"Missing key: {key}")
        self.assertAlmostEqual(float(metrics["sigreg_loss"]), 0.0, places=7)
        self.assertAlmostEqual(float(metrics["token_sigreg_loss"]), 0.0, places=7)
        expected = metrics["jepa_term"] + metrics["vicreg_term"]
        self.assertTrue(torch.allclose(metrics["loss"], expected))

    def test_forward_gco_sigreg_contains_expected_keys(self):
        model = self._build_model(
            representation_regularizer="gco-sigreg",
            sigreg_lambda=0.0,
            masked_token_loss_weight=1.0,
            gco_std_target=10.0,
            gco_alpha=0.0,
            gco_eta=1e-2,
            multicrop_num_local_views=3,
        )
        batch = _make_fused_batch(num_views=model.num_views)
        metrics = model.forward_augmented(batch)
        for key in (
            "gco_lambda",
            "gco_log_lambda",
            "gco_c_ema",
            "gco_constraint",
            "gco_std_penalty",
        ):
            self.assertIn(key, metrics, f"Missing key: {key}")
        self.assertGreater(float(metrics["gco_std_penalty"].detach()), 0.0)
        self.assertTrue(
            torch.allclose(
                metrics["sigreg_term"],
                metrics["gco_lambda"] * metrics["token_sigreg_loss"],
            )
        )

    def test_forward_gco_vicreg_contains_expected_keys(self):
        model = self._build_model(
            representation_regularizer="gco-vicreg",
            sigreg_lambda=0.0,
            masked_token_loss_weight=1.0,
            gco_std_target=10.0,
            gco_alpha=0.0,
            gco_eta=1e-2,
            multicrop_num_local_views=3,
        )
        batch = _make_fused_batch(num_views=model.num_views)
        metrics = model.forward_augmented(batch)
        for key in (
            "gco_lambda",
            "gco_log_lambda",
            "gco_c_ema",
            "gco_constraint",
            "gco_std_penalty",
            "vicreg_loss",
            "vicreg_term",
        ):
            self.assertIn(key, metrics, f"Missing key: {key}")
        self.assertGreater(float(metrics["gco_std_penalty"].detach()), 0.0)
        self.assertTrue(
            torch.allclose(
                metrics["vicreg_term"],
                metrics["gco_lambda"] * metrics["vicreg_loss"],
            )
        )
        self.assertAlmostEqual(float(metrics["sigreg_loss"]), 0.0, places=7)
        self.assertAlmostEqual(float(metrics["token_sigreg_loss"]), 0.0, places=7)

    def test_gco_lambda_updates_only_in_train_mode(self):
        for regularizer in ("gco-sigreg", "gco-vicreg"):
            with self.subTest(representation_regularizer=regularizer):
                model = self._build_model(
                    representation_regularizer=regularizer,
                    sigreg_lambda=0.0,
                    masked_token_loss_weight=1.0,
                    gco_std_target=10.0,
                    gco_alpha=0.0,
                    gco_eta=1e-2,
                    multicrop_num_local_views=3,
                )
                batch = _make_fused_batch(num_views=model.num_views)

                model.train()
                before_train = float(model.gco_log_lambda)
                model.forward_augmented(batch)
                after_train = float(model.gco_log_lambda)
                self.assertGreater(after_train, before_train)

                model.eval()
                before_eval = float(model.gco_log_lambda)
                model.forward_augmented(batch)
                after_eval = float(model.gco_log_lambda)
                self.assertAlmostEqual(after_eval, before_eval, places=7)

    def test_sigreg_regularizer_uses_encoder_outputs_and_padding_mask(self):
        model = self._build_model(
            representation_regularizer="sigreg",
            sigreg_lambda=0.1,
            masked_token_loss_weight=0.0,
            multicrop_num_local_views=3,
        )

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

        batch = _make_fused_batch(num_views=model.num_views)
        V = model.num_views
        batch_size = batch["fused_mz"].shape[0] // V
        model.forward_augmented(batch)

        # Regularizer should receive local views' encoder output with their valid masks
        L = V - 1
        expected_valid = (
            batch["fused_valid_mask"]
            .reshape(V, batch_size, -1)[1:]
            .reshape(L * batch_size, -1)
        )
        self.assertIsNotNone(capture.proj)
        self.assertIsNotNone(capture.valid_mask)
        self.assertTrue(torch.equal(capture.valid_mask, expected_valid))
        # Shape should be [L*B, N, D] — local views only
        self.assertEqual(capture.proj.shape[0], L * batch_size)

    def test_vicreg_regularizer_uses_encoder_outputs_and_padding_mask(self):
        model = self._build_model(
            representation_regularizer="vicreg",
            sigreg_lambda=0.0,
            masked_token_loss_weight=0.0,
            multicrop_num_local_views=3,
        )

        class CaptureVICReg(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj_a = None
                self.proj_b = None
                self.valid_mask = None

            def forward(self, proj_a, proj_b, valid_mask=None):
                self.proj_a = proj_a.detach().clone()
                self.proj_b = proj_b.detach().clone()
                self.valid_mask = None if valid_mask is None else valid_mask.detach().clone()
                return proj_a.new_tensor(0.7)

        capture = CaptureVICReg()
        model.vicreg = capture

        batch = _make_fused_batch(num_views=model.num_views)
        batch_size = batch["fused_mz"].shape[0] // model.num_views
        model.forward_augmented(batch)

        # VICReg should receive the two local views encoder embeddings
        expected_valid_by_view = batch["fused_valid_mask"].reshape(model.num_views, batch_size, -1)
        expected_pair_valid = expected_valid_by_view[1] & expected_valid_by_view[2]  # local views
        self.assertIsNotNone(capture.proj_a)
        self.assertIsNotNone(capture.proj_b)
        self.assertIsNotNone(capture.valid_mask)
        self.assertTrue(torch.equal(capture.valid_mask, expected_pair_valid))
        # Shapes should be [B, N, D] — single view each
        self.assertEqual(capture.proj_a.shape[0], batch_size)

    def test_gco_constraint_uses_encoder_std(self):
        model = self._build_model(
            representation_regularizer="gco-sigreg",
            sigreg_lambda=0.0,
            masked_token_loss_weight=0.0,
            gco_std_target=100.0,
            gco_alpha=0.0,
            gco_eta=0.0,
            multicrop_num_local_views=3,
        )

        batch = _make_fused_batch(num_views=model.num_views)
        metrics = model.forward_augmented(batch)
        # GCO constraint = std_target - encoder_emb_std; with a high target
        # the constraint should be positive (encoder std << 100).
        self.assertGreater(float(metrics["gco_constraint"]), 0.0)
        self.assertGreater(float(metrics["gco_std_penalty"]), 0.0)
        # encoder_emb_std should be a reasonable positive value
        self.assertGreater(float(metrics["encoder_emb_std"]), 0.0)

    def test_forward_uses_local_and_global_paths(self):
        model = self._build_model(
            masked_token_loss_weight=1.0,
            representation_regularizer="sigreg",
        )
        batch = _make_fused_batch(num_views=model.num_views)
        metrics = model.forward_augmented(batch)

        self.assertEqual(model.latent_mask_token.shape[0], model.model_dim)
        self.assertTrue(torch.isfinite(metrics["loss"]).item())
        self.assertIn("local_global_loss", metrics)
        self.assertIn("regularizer_loss", metrics)
        self.assertIn("regularizer_term", metrics)

    def test_forward_without_anticollapse_regularizer(self):
        model = self._build_model(
            representation_regularizer=None,
            masked_token_loss_weight=1.0,
            sigreg_lambda=0.1,
            vicreg_beta=1e-3,
        )
        batch = _make_fused_batch(num_views=model.num_views)
        metrics = model.forward_augmented(batch)
        self.assertAlmostEqual(float(metrics["sigreg_loss"]), 0.0, places=7)
        self.assertAlmostEqual(float(metrics["token_sigreg_loss"]), 0.0, places=7)
        self.assertAlmostEqual(float(metrics["vicreg_loss"]), 0.0, places=7)
        self.assertAlmostEqual(float(metrics["sigreg_term"]), 0.0, places=7)
        self.assertAlmostEqual(float(metrics["vicreg_term"]), 0.0, places=7)
        self.assertAlmostEqual(float(metrics["regularizer_term"]), 0.0, places=7)
        self.assertAlmostEqual(float(metrics["regularizer_loss"]), 0.0, places=7)
        self.assertTrue(torch.allclose(metrics["loss"], metrics["jepa_term"]))

    def test_encode_output_shape(self):
        model = self._build_model()
        batch = _make_batch(batch_size=3)
        pooled = model.encode(batch, train=False)
        self.assertEqual(pooled.shape, (3, model.model_dim))

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
            pooling_type="mean",
            sigreg_lambda=0.0,
            multicrop_num_local_views=1,
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
        loss = model.forward_augmented(
            {
                "fused_mz": fused_mz,
                "fused_intensity": fused_intensity,
                "fused_valid_mask": fused_valid_mask,
                "fused_masked_positions": fused_masked_positions,
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
            )
            emb_with_rope = model_with_rope.encoder(
                batch["peak_mz"],
                batch["peak_intensity"],
                valid_mask=batch["peak_valid_mask"],
            )

        self.assertFalse(torch.allclose(emb_no_rope, emb_with_rope))

    def test_predictor_head_dim_is_compile_safe_when_possible(self):
        model = self._build_model(
            model_dim=512,
            encoder_num_heads=16,
        )
        self.assertGreaterEqual(model.model_dim // model.predictor_num_heads, 16)

    def test_ema_teacher_is_frozen_and_kept_in_eval(self):
        model = PeakSetSIGReg(
            num_peaks=60,
            model_dim=32,
            encoder_num_layers=1,
            encoder_num_heads=4,
            encoder_num_kv_heads=4,
            attention_mlp_multiple=2.0,
            feature_mlp_hidden_dim=16,
            multicrop_num_local_views=1,
            use_ema_teacher_target=True,
        )
        self.assertIsNotNone(model.teacher_encoder)
        self.assertTrue(all(not p.requires_grad for p in model.teacher_encoder.parameters()))
        model.train()
        self.assertFalse(model.teacher_encoder.training)

    def test_ema_teacher_update_matches_decay_formula(self):
        model = PeakSetSIGReg(
            num_peaks=60,
            model_dim=32,
            encoder_num_layers=1,
            encoder_num_heads=4,
            encoder_num_kv_heads=4,
            attention_mlp_multiple=2.0,
            feature_mlp_hidden_dim=16,
            multicrop_num_local_views=1,
            use_ema_teacher_target=True,
            teacher_ema_decay=0.5,
        )
        student_param = next(model.encoder.parameters())
        teacher_param = next(model.teacher_encoder.parameters())
        with torch.no_grad():
            # AveragedModel semantics: first update copies student -> teacher.
            model.update_teacher()
            teacher_before = teacher_param.clone()
            student_param.add_(1.0)
            expected = teacher_before * 0.5 + student_param * 0.5
        model.update_teacher()
        self.assertTrue(torch.allclose(teacher_param, expected))

    def test_ema_teacher_decay_warmup_schedule(self):
        model = PeakSetSIGReg(
            num_peaks=60,
            model_dim=32,
            encoder_num_layers=1,
            encoder_num_heads=4,
            encoder_num_kv_heads=4,
            attention_mlp_multiple=2.0,
            feature_mlp_hidden_dim=16,
            multicrop_num_local_views=1,
            use_ema_teacher_target=True,
            teacher_ema_decay=0.8,
            teacher_ema_decay_start=0.4,
            teacher_ema_decay_warmup_steps=4,
        )
        values = []
        for _ in range(6):
            model.advance_teacher_ema_decay_schedule()
            values.append(float(model.teacher_ema_decay_current))
        expected = [0.4, 0.5, 0.6, 0.7, 0.8, 0.8]
        for actual, target in zip(values, expected):
            self.assertAlmostEqual(actual, target, places=6)

    def test_teacher_branch_has_no_gradients(self):
        model = PeakSetSIGReg(
            num_peaks=60,
            model_dim=32,
            encoder_num_layers=1,
            encoder_num_heads=4,
            encoder_num_kv_heads=4,
            attention_mlp_multiple=2.0,
            feature_mlp_hidden_dim=16,
            sigreg_lambda=0.0,
            masked_token_loss_weight=1.0,
            multicrop_num_local_views=1,
            use_ema_teacher_target=True,
        )
        batch = _make_fused_batch(num_views=model.num_views)
        loss = model.forward_augmented(batch)["loss"]
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.encoder.parameters() if p.requires_grad))
        self.assertTrue(all(p.grad is None for p in model.teacher_encoder.parameters()))


class StandardRoPETests(unittest.TestCase):
    def _build_encoder_model(
        self,
        *,
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
            pooling_type="mean",
        )

    def test_standard_rope_inv_freq_range(self):
        model = self._build_encoder_model()
        inv_freq = model.encoder.rope_inv_freq
        half_dim = model.model_dim // model.encoder.blocks[0].attention.n_heads // 2
        expected_first = torch.tensor(1.0)
        expected_last = torch.tensor(10000.0 ** (-(half_dim - 1) / half_dim))
        self.assertTrue(torch.isclose(inv_freq[0], expected_first, rtol=1e-6, atol=1e-7))
        self.assertTrue(torch.isclose(inv_freq[-1], expected_last, rtol=1e-6, atol=1e-7))

    def test_predictor_standard_rope_inv_freq_shape(self):
        model = self._build_encoder_model()
        inv_freq = model.predictor_rope_inv_freq
        expected_half_dim = (model.model_dim // model.predictor_num_heads) // 2
        self.assertEqual(inv_freq.shape, (expected_half_dim,))
        self.assertTrue(torch.isclose(inv_freq[0], torch.tensor(1.0), rtol=1e-6, atol=1e-7))

    def test_rope_with_gqa_is_finite(self):
        model = self._build_encoder_model(
            encoder_num_kv_heads=2,
        )
        batch = _make_batch(batch_size=3)
        with torch.no_grad():
            emb = model.encoder(
                batch["peak_mz"],
                batch["peak_intensity"],
                valid_mask=batch["peak_valid_mask"],
            )
        self.assertTrue(torch.isfinite(emb).all())


class SIGRegLambdaScheduleTests(unittest.TestCase):
    def _build_model(
        self,
        *,
        sigreg_lambda: float = 0.2,
        sigreg_lambda_warmup_steps: int = 4,
        masked_token_loss_weight: float = 0.0,
    ) -> PeakSetSIGReg:
        return PeakSetSIGReg(
            num_peaks=60,
            model_dim=32,
            encoder_num_layers=1,
            encoder_num_heads=4,
            encoder_num_kv_heads=4,
            attention_mlp_multiple=2.0,
            feature_mlp_hidden_dim=16,
            sigreg_num_slices=32,
            sigreg_lambda=sigreg_lambda,
            sigreg_lambda_warmup_steps=sigreg_lambda_warmup_steps,
            multicrop_num_local_views=1,
            masked_token_loss_weight=masked_token_loss_weight,
        )

    def test_schedule_disabled_is_constant(self):
        model = self._build_model(sigreg_lambda=0.2, sigreg_lambda_warmup_steps=0)
        values = []
        for _ in range(3):
            model.advance_sigreg_lambda_schedule()
            values.append(float(model.sigreg_lambda_current))
        for value in values:
            self.assertAlmostEqual(value, 0.2, places=6)

    def test_schedule_linear_progression(self):
        model = self._build_model(sigreg_lambda=0.2, sigreg_lambda_warmup_steps=4)
        values = []
        for _ in range(6):
            model.advance_sigreg_lambda_schedule()
            values.append(float(model.sigreg_lambda_current))
        expected = [0.0, 0.05, 0.1, 0.15, 0.2, 0.2]
        for actual, target in zip(values, expected):
            self.assertAlmostEqual(actual, target, places=6)

    def test_forward_reports_current_lambda_metric(self):
        model = self._build_model(sigreg_lambda=0.2, sigreg_lambda_warmup_steps=4)
        batch = _make_fused_batch(num_views=model.num_views)
        model.advance_sigreg_lambda_schedule()
        metrics = model.forward_augmented(batch)
        self.assertIn("sigreg_lambda_current", metrics)
        self.assertTrue(torch.isfinite(metrics["sigreg_lambda_current"]).item())
        self.assertTrue(
            torch.allclose(
                metrics["sigreg_lambda_current"],
                model.sigreg_lambda_current.to(dtype=metrics["sigreg_lambda_current"].dtype),
            )
        )

    def test_loss_uses_scheduled_lambda(self):
        model = self._build_model(
            sigreg_lambda=0.2,
            sigreg_lambda_warmup_steps=4,
            masked_token_loss_weight=0.0,
        )
        batch = _make_fused_batch(num_views=model.num_views)

        model.advance_sigreg_lambda_schedule()
        early = model.forward_augmented(batch)
        self.assertAlmostEqual(float(early["sigreg_lambda_current"]), 0.0, places=7)
        self.assertAlmostEqual(float(early["loss"].detach()), 0.0, places=7)

        for _ in range(5):
            model.advance_sigreg_lambda_schedule()
        late = model.forward_augmented(batch)
        expected = late["sigreg_lambda_current"] * late["token_sigreg_loss"]
        self.assertTrue(torch.allclose(late["loss"], expected))
        self.assertGreater(float(late["loss"].detach()), float(early["loss"].detach()))

    def test_schedule_state_dict_roundtrip(self):
        model = self._build_model(sigreg_lambda=0.2, sigreg_lambda_warmup_steps=4)
        for _ in range(3):
            model.advance_sigreg_lambda_schedule()
        state = model.state_dict()

        restored = self._build_model(sigreg_lambda=0.2, sigreg_lambda_warmup_steps=4)
        restored.load_state_dict(state)
        self.assertTrue(torch.equal(restored.sigreg_lambda_step, model.sigreg_lambda_step))
        self.assertTrue(torch.equal(restored.sigreg_lambda_current, model.sigreg_lambda_current))


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
        # L1 loss is masked-slots-only, so ensure local views contain masked slots.
        bsz = batch["fused_mz"].shape[0] // model.num_views
        batch["fused_masked_positions"][bsz:, 0] = True
        loss = model.forward_augmented(batch)["loss"]
        loss.backward()

        self.assertIsNone(model.pool_query.grad)
        mha_grads = [p.grad for p in model.pool_mha.parameters() if p.requires_grad]
        self.assertTrue(all(g is None for g in mha_grads))

        predictor_params = list(model.masked_latent_predictor.parameters())
        predictor_grads = [p.grad for p in predictor_params if p.requires_grad]
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

    def test_sigreg_masked_matches_filtered_tokens(self):
        sigreg = SIGReg(num_slices=32)
        proj = torch.randn(2, 3, 5, 16)
        valid = torch.rand(2, 3, 5) > 0.3
        valid.view(-1)[0] = True
        torch.manual_seed(2026)
        result_masked = sigreg(proj, valid_mask=valid)
        torch.manual_seed(2026)
        result_filtered = sigreg(proj[valid])
        self.assertTrue(torch.allclose(result_masked, result_filtered))

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


class VICRegLossTests(unittest.TestCase):
    def test_vicreg_output_is_scalar(self):
        vicreg = VICRegLoss()
        proj_a = torch.randn(4, 8, 16)
        proj_b = torch.randn(4, 8, 16)
        result = vicreg(proj_a, proj_b)
        self.assertEqual(result.ndim, 0)
        self.assertTrue(torch.isfinite(result).item())

    def test_vicreg_accepts_any_leading_shape(self):
        vicreg = VICRegLoss()
        proj_a = torch.randn(2, 3, 5, 16)
        proj_b = torch.randn(2, 3, 5, 16)
        result = vicreg(proj_a, proj_b)
        self.assertEqual(result.ndim, 0)
        self.assertTrue(torch.isfinite(result).item())

    def test_vicreg_matches_flattened_input(self):
        vicreg = VICRegLoss()
        proj_a = torch.randn(2, 3, 5, 16)
        proj_b = torch.randn(2, 3, 5, 16)
        result_shaped = vicreg(proj_a, proj_b)
        result_flat = vicreg(
            proj_a.reshape(-1, proj_a.size(-1)),
            proj_b.reshape(-1, proj_b.size(-1)),
        )
        self.assertTrue(torch.allclose(result_shaped, result_flat))

    def test_vicreg_masked_matches_filtered_tokens(self):
        vicreg = VICRegLoss()
        proj_a = torch.randn(2, 3, 5, 16)
        proj_b = torch.randn(2, 3, 5, 16)
        valid = torch.rand(2, 3, 5) > 0.3
        valid.view(-1)[0] = True
        result_masked = vicreg(proj_a, proj_b, valid_mask=valid)
        result_filtered = vicreg(proj_a[valid], proj_b[valid])
        self.assertTrue(torch.allclose(result_masked, result_filtered))

    def test_vicreg_backpropagates(self):
        vicreg = VICRegLoss()
        proj_a = torch.randn(4, 8, 16, requires_grad=True)
        proj_b = torch.randn(4, 8, 16, requires_grad=True)
        result = vicreg(proj_a, proj_b)
        result.backward()
        self.assertIsNotNone(proj_a.grad)
        self.assertIsNotNone(proj_b.grad)
        self.assertGreater(float(proj_a.grad.abs().sum()), 0.0)
        self.assertGreater(float(proj_b.grad.abs().sum()), 0.0)

    def test_vicreg_penalizes_mismatch_between_views(self):
        vicreg = VICRegLoss()
        proj_a = torch.randn(4, 256, 32)
        proj_b_same = proj_a.clone()
        proj_b_random = torch.randn(4, 256, 32)
        loss_same = vicreg(proj_a, proj_b_same)
        loss_random = vicreg(proj_a, proj_b_random)
        self.assertGreater(float(loss_random), float(loss_same))


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
