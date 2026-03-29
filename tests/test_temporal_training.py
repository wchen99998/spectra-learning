"""Tests for temporal finetuning (frame -> next-frame prediction)."""

import torch
import pytest

from models.model import PeakSetSIGReg


def _small_model(**overrides) -> PeakSetSIGReg:
    kwargs = dict(
        model_dim=64,
        encoder_num_layers=2,
        encoder_num_heads=4,
        encoder_num_kv_heads=4,
        attention_mlp_multiple=2.0,
        feature_mlp_hidden_dim=32,
        masked_token_loss_weight=1.0,
        masked_token_loss_type="l2",
        representation_regularizer="none",
        masked_latent_predictor_num_layers=1,
        jepa_num_target_blocks=1,
        num_peaks=8,
        temporal_predictor_num_layers=2,
        use_ema_teacher_target=False,
    )
    kwargs.update(overrides)
    return PeakSetSIGReg(**kwargs)


def _temporal_batch(batch_size: int = 4, num_peaks: int = 8) -> dict[str, torch.Tensor]:
    return {
        "frame_peak_mz": torch.rand(batch_size, num_peaks),
        "frame_peak_intensity": torch.rand(batch_size, num_peaks),
        "frame_peak_valid_mask": torch.ones(batch_size, num_peaks, dtype=torch.bool),
        "frame_rt": torch.rand(batch_size) * 10,
        "frame_precursor_mz": torch.rand(batch_size),
        "next_frame_peak_mz": torch.rand(batch_size, num_peaks),
        "next_frame_peak_intensity": torch.rand(batch_size, num_peaks),
        "next_frame_peak_valid_mask": torch.ones(batch_size, num_peaks, dtype=torch.bool),
        "next_frame_rt": torch.rand(batch_size) * 10 + 10,
        "next_frame_precursor_mz": torch.rand(batch_size),
    }


class TestForwardTemporal:
    def test_finite_loss(self):
        model = _small_model()
        model.eval()
        batch = _temporal_batch()
        metrics = model.forward_temporal(batch)
        assert torch.isfinite(metrics["loss"])
        assert "next_frame_pred_loss" in metrics

    def test_backward(self):
        model = _small_model()
        model.train()
        batch = _temporal_batch()
        metrics = model.forward_temporal(batch)
        metrics["loss"].backward()

        # Check gradients flow to encoder
        encoder_grads = [
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.encoder.parameters()
            if p.requires_grad
        ]
        assert any(encoder_grads), "No gradients flowed to encoder"

        # Check gradients flow to temporal predictor
        tp_grads = [
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.temporal_predictor.parameters()
            if p.requires_grad
        ]
        assert any(tp_grads), "No gradients flowed to temporal predictor"

        # Check gradients flow to RT projection
        rt_grads = [
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.temporal_rt_proj.parameters()
            if p.requires_grad
        ]
        assert any(rt_grads), "No gradients flowed to RT projection"

        # Check gradients flow to temporal query tokens
        assert model.temporal_query_tokens.grad is not None
        assert model.temporal_query_tokens.grad.abs().sum() > 0

    def test_with_ema_teacher(self):
        model = _small_model(use_ema_teacher_target=True, teacher_ema_decay=0.99)
        model.eval()
        batch = _temporal_batch()
        teacher_embeddings = model.compute_next_frame_teacher_embeddings(batch)
        metrics = model.forward_temporal(batch, teacher_embeddings=teacher_embeddings)
        assert torch.isfinite(metrics["loss"])

    def test_partial_valid_mask(self):
        """Loss should be computed only over valid next-frame tokens."""
        model = _small_model()
        model.eval()
        batch = _temporal_batch()
        batch["next_frame_peak_valid_mask"][:, 4:] = False
        metrics = model.forward_temporal(batch)
        assert torch.isfinite(metrics["loss"])

    def test_different_loss_types(self):
        for loss_type in ("l1", "l2"):
            model = _small_model(masked_token_loss_type=loss_type)
            model.eval()
            batch = _temporal_batch()
            metrics = model.forward_temporal(batch)
            assert torch.isfinite(metrics["loss"]), f"Non-finite loss for {loss_type}"

    def test_temporal_predictor_absolute_positions_change_output(self):
        torch.manual_seed(0)
        model_without_pos = _small_model()
        torch.manual_seed(0)
        model_with_pos = _small_model()
        with torch.no_grad():
            model_without_pos.predictor_position_embedding.weight.zero_()
        model_without_pos.eval()
        model_with_pos.eval()
        batch = _temporal_batch(batch_size=2, num_peaks=8)

        with torch.no_grad():
            loss_without_pos = model_without_pos.forward_temporal(batch)["loss"]
            loss_with_pos = model_with_pos.forward_temporal(batch)["loss"]

        assert not torch.allclose(loss_without_pos, loss_with_pos, atol=1e-6)


class TestCheckpointPartialLoad:
    def test_partial_load_allows_temporal_keys_missing(self):
        """Pretrained state dict (no temporal keys) loads with strict=False."""
        # Build a model without temporal predictor (simulates pretrained)
        pretrained = _small_model(temporal_predictor_num_layers=0)
        sd = pretrained.state_dict()

        # Build full model with temporal predictor
        full = _small_model(temporal_predictor_num_layers=2)
        missing, unexpected = full.load_state_dict(sd, strict=False)

        # Only temporal keys should be missing
        allowed_prefixes = ("temporal_predictor.", "temporal_rt_proj.", "temporal_query_tokens")
        for key in missing:
            assert any(key.startswith(p) for p in allowed_prefixes), (
                f"Unexpected missing key: {key}"
            )
        assert len(unexpected) == 0

    def test_loaded_encoder_weights_match(self):
        """Encoder weights should be identical after partial load."""
        pretrained = _small_model(temporal_predictor_num_layers=0)
        sd = pretrained.state_dict()

        full = _small_model(temporal_predictor_num_layers=2)
        full.load_state_dict(sd, strict=False)

        for name, param in full.named_parameters():
            if name.startswith("encoder."):
                pretrained_param = sd[name]
                assert torch.equal(param.data, pretrained_param), (
                    f"Encoder param {name} doesn't match after load"
                )


class TestRTConditioningSensitivity:
    def test_different_delta_rt_different_predictions(self):
        """Different delta_rt values should produce different predictions."""
        model = _small_model()
        model.eval()

        batch1 = _temporal_batch()
        batch2 = {k: v.clone() for k, v in batch1.items()}
        batch2["next_frame_rt"] = batch1["next_frame_rt"] + 5.0

        with torch.no_grad():
            metrics1 = model.forward_temporal(batch1)
            metrics2 = model.forward_temporal(batch2)

        # Losses should differ due to different RT conditioning
        assert not torch.allclose(metrics1["loss"], metrics2["loss"], atol=1e-6), (
            "Predictions should differ for different delta_rt"
        )
