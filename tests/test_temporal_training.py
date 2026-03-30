"""Tests for temporal finetuning (frame -> next-frame prediction)."""

import tempfile

import torch

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
    def test_requires_temporal_predictor(self):
        model = _small_model(temporal_predictor_num_layers=0)
        batch = _temporal_batch()

        try:
            model.forward_temporal(batch)
        except ValueError as exc:
            assert "temporal_predictor_num_layers > 0" in str(exc)
        else:
            raise AssertionError("forward_temporal should require temporal layers")

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

        # Check gradients flow to the shared temporal query token
        assert model.temporal_query_token.grad is not None
        assert model.temporal_query_token.grad.abs().sum() > 0

    def test_with_target_override(self):
        model = _small_model()
        model.eval()
        batch = _temporal_batch()
        target_embeddings = model.compute_next_frame_target_embeddings(batch)
        metrics = model.forward_temporal(batch, target_embeddings=target_embeddings)
        assert torch.isfinite(metrics["loss"])

    def test_target_embeddings_require_grad(self):
        model = _small_model()
        batch = _temporal_batch()
        target_embeddings = model.compute_next_frame_target_embeddings(batch)
        assert target_embeddings.requires_grad

    def test_target_embeddings_use_next_frame_precursor_when_enabled(self):
        model = _small_model(use_precursor_token=True)
        model.eval()
        batch_a = _temporal_batch()
        batch_b = {k: v.clone() for k, v in batch_a.items()}
        batch_a["next_frame_precursor_mz"].fill_(0.1)
        batch_b["next_frame_precursor_mz"].fill_(0.9)

        emb_a = model.compute_next_frame_target_embeddings(batch_a)
        emb_b = model.compute_next_frame_target_embeddings(batch_b)

        assert not torch.allclose(emb_a, emb_b, atol=1e-6)

    def test_forward_temporal_uses_frame_precursor_when_enabled(self):
        model = _small_model(use_precursor_token=True)
        model.eval()
        batch_a = _temporal_batch()
        batch_b = {k: v.clone() for k, v in batch_a.items()}
        batch_a["frame_precursor_mz"].fill_(0.1)
        batch_b["frame_precursor_mz"].fill_(0.9)
        target_embeddings = model.compute_next_frame_target_embeddings(batch_a)

        loss_a = model.forward_temporal(
            batch_a,
            target_embeddings=target_embeddings,
        )["loss"]
        loss_b = model.forward_temporal(
            batch_b,
            target_embeddings=target_embeddings,
        )["loss"]

        assert not torch.allclose(loss_a, loss_b, atol=1e-6)

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
        allowed_prefixes = ("temporal_predictor.", "temporal_rt_proj.", "temporal_query_token")
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

    def test_temporal_checkpoint_loader_ignores_masked_latent_readout_mismatch(self):
        from train_temporal import _load_pretrained_checkpoint

        pretrained = _small_model(
            temporal_predictor_num_layers=0,
            jepa_target_layers=[1, 2],
        )
        full = _small_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/spatial.ckpt"
            torch.save({"model": pretrained.state_dict()}, path)
            _load_pretrained_checkpoint(full, path)

        for name, param in full.named_parameters():
            if name.startswith("encoder."):
                pretrained_param = pretrained.state_dict()[name]
                assert torch.equal(param.data, pretrained_param), (
                    f"Encoder param {name} doesn't match after temporal checkpoint load"
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
