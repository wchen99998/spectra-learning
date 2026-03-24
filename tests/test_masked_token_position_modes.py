import torch

from models.model import PeakSetSIGReg


def _build_model(
    *,
    num_target_blocks: int = 2,
    predictor_layers: int = 2,
    encoder_use_rope: bool = True,
) -> PeakSetSIGReg:
    torch.manual_seed(0)
    model = PeakSetSIGReg(
        model_dim=32,
        encoder_num_layers=2,
        encoder_num_heads=4,
        feature_mlp_hidden_dim=32,
        encoder_use_rope=encoder_use_rope,
        jepa_num_target_blocks=num_target_blocks,
        masked_token_loss_weight=1.0,
        masked_latent_predictor_num_layers=predictor_layers,
    )
    model.eval()
    return model


def _make_batch() -> dict[str, torch.Tensor]:
    peak_mz = torch.tensor(
        [
            [0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
            [0.11, 0.21, 0.31, 0.41, 0.51, 0.61],
        ],
        dtype=torch.float32,
    )
    peak_intensity = torch.tensor(
        [
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            [0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
        ],
        dtype=torch.float32,
    )
    peak_valid_mask = torch.ones_like(peak_mz, dtype=torch.bool)
    context_mask = torch.tensor(
        [
            [True, True, False, False, False, False],
            [False, True, True, False, False, False],
        ]
    )
    target_masks = torch.tensor(
        [
            [
                [False, False, True, False, False, False],
                [False, False, False, True, False, False],
            ],
            [
                [False, False, False, True, False, False],
                [False, False, False, False, True, False],
            ],
        ]
    )
    return {
        "peak_mz": peak_mz,
        "peak_intensity": peak_intensity,
        "peak_valid_mask": peak_valid_mask,
        "context_mask": context_mask,
        "target_masks": target_masks,
    }


@torch.no_grad()
def test_predictor_zero_layers_is_identity():
    model = _build_model(predictor_layers=0)
    predictor_input = torch.randn(2, 6, model.model_dim)
    context = torch.randn(2, 6, model.model_dim)
    target_mask = torch.ones(2, 6, dtype=torch.bool)
    context_mask = torch.ones(2, 6, dtype=torch.bool)

    out = model.predict_masked_latents(
        predictor_input,
        context,
        target_mask=target_mask,
        context_mask=context_mask,
    )

    assert torch.allclose(out, predictor_input)


@torch.no_grad()
def test_predictor_rope_is_independent_of_encoder_rope_toggle():
    torch.manual_seed(0)
    model_no_rope = _build_model(predictor_layers=2, encoder_use_rope=False)
    torch.manual_seed(0)
    model_with_rope = _build_model(predictor_layers=2, encoder_use_rope=True)
    predictor_input = torch.randn(1, 6, model_with_rope.model_dim)
    context = torch.randn(1, 6, model_with_rope.model_dim)
    target_mask = torch.ones(1, 6, dtype=torch.bool)
    context_mask = torch.ones(1, 6, dtype=torch.bool)

    out_1 = model_no_rope.predict_masked_latents(
        predictor_input,
        context,
        target_mask=target_mask,
        context_mask=context_mask,
    )
    out_2 = model_with_rope.predict_masked_latents(
        predictor_input,
        context,
        target_mask=target_mask,
        context_mask=context_mask,
    )

    assert torch.allclose(out_1, out_2, atol=1e-6, rtol=1e-6)


@torch.no_grad()
def test_packed_predictor_matches_dense_without_encoder_rope():
    torch.manual_seed(0)
    model = PeakSetSIGReg(
        model_dim=32,
        encoder_num_layers=2,
        encoder_num_heads=4,
        feature_mlp_hidden_dim=32,
        encoder_use_rope=False,
        jepa_num_target_blocks=2,
        masked_token_loss_weight=1.0,
        masked_latent_predictor_num_layers=2,
        num_peaks=6,
        jepa_target_fraction=0.25,
    )
    model.eval()
    predictor_input = torch.randn(2, 6, model.model_dim)
    context = torch.randn(2, 6, model.model_dim)
    target_mask = torch.tensor(
        [
            [False, False, True, True, False, False],
            [False, True, False, False, True, False],
        ]
    )
    context_mask = torch.tensor(
        [
            [True, True, False, False, False, False],
            [True, False, True, False, False, False],
        ]
    )

    out_dense = model.predict_masked_latents(
        predictor_input,
        context,
        target_mask=target_mask,
        context_mask=context_mask,
        pack_n=0,
    )
    out_packed = model.predict_masked_latents(
        predictor_input,
        context,
        target_mask=target_mask,
        context_mask=context_mask,
        pack_n=model._predictor_pack_n,
    )

    assert model._predictor_pack_n == 2
    assert torch.allclose(out_dense, out_packed, atol=1e-6, rtol=1e-6)


@torch.no_grad()
def test_forward_augmented_reports_loss_metrics():
    model = _build_model()
    metrics = model.forward_augmented(_make_batch())

    assert "local_global_loss" in metrics
    assert "context_fraction" in metrics
    assert "masked_fraction" in metrics
    assert torch.isfinite(metrics["local_global_loss"])
    assert float(metrics["masked_fraction"]) > 0.0


@torch.no_grad()
def test_local_global_loss_uses_full_view_teacher_targets():
    """Verify forward_augmented loss matches manual reconstruction with full-view teacher targets."""
    model = _build_model(num_target_blocks=2)
    model.sigreg_lambda = 0.0
    batch = _make_batch()

    metrics = model.forward_augmented(batch)

    peak_mz = batch["peak_mz"]
    peak_intensity = batch["peak_intensity"]
    peak_valid_mask = batch["peak_valid_mask"]
    context_mask = batch["context_mask"] & peak_valid_mask
    target_masks = batch["target_masks"] & peak_valid_mask.unsqueeze(1)
    target_masks_by_view = target_masks.permute(1, 0, 2)

    context_emb = model.encoder(
        peak_mz,
        peak_intensity,
        valid_mask=peak_valid_mask,
        visible_mask=context_mask,
    )
    B, K, N = target_masks.shape

    # Full encoder sees all valid peaks (shared encoder, no stop_grad in training)
    teacher_full = model.encoder(
        peak_mz,
        peak_intensity,
        valid_mask=peak_valid_mask,
        visible_mask=peak_valid_mask,
    ).detach()  # detach for this test's manual comparison only
    target_token_target = teacher_full.unsqueeze(1).expand(-1, K, -1, -1)
    target_token_target_by_view = target_token_target.permute(1, 0, 2, 3)

    predictor_queries = torch.zeros_like(context_emb.unsqueeze(0).expand(K, -1, -1, -1))
    latent_mask_token = model.latent_mask_token.view(1, 1, 1, -1).to(
        dtype=context_emb.dtype,
        device=context_emb.device,
    )
    predictor_queries = torch.where(
        target_masks_by_view.unsqueeze(-1),
        latent_mask_token,
        predictor_queries,
    )
    predictor_output = model.predict_masked_latents(
        predictor_queries.reshape(B * K, N, -1),
        context_emb.unsqueeze(0).expand(K, -1, -1, -1).reshape(B * K, N, -1),
        target_mask=target_masks_by_view.reshape(B * K, N),
        context_mask=context_mask.unsqueeze(0).expand(K, -1, -1).reshape(B * K, N),
    ).reshape(K, B, N, -1)

    per_token_mse = (predictor_output - target_token_target_by_view).square().mean(dim=-1)
    masked_only_loss = (
        per_token_mse * target_masks_by_view.float()
    ).sum() / target_masks_by_view.float().sum().clamp_min(1.0)

    assert torch.allclose(metrics["local_global_loss"], masked_only_loss)


@torch.no_grad()
def test_padding_positions_do_not_change_loss():
    """Padding (invalid) positions should not affect the loss."""
    model = _build_model(num_target_blocks=2)
    model.sigreg_lambda = 0.0

    batch_a = _make_batch()
    # Mark last 2 positions as padding
    batch_a["peak_valid_mask"] = batch_a["peak_valid_mask"].clone()
    batch_a["peak_valid_mask"][:, -2:] = False

    batch_b = {key: value.clone() for key, value in batch_a.items()}
    # Perturb padding positions — should have no effect
    padding = ~batch_b["peak_valid_mask"]
    batch_b["peak_intensity"] = batch_b["peak_intensity"].clone()
    batch_b["peak_intensity"][padding] = batch_b["peak_intensity"][padding] + 0.5
    batch_b["peak_mz"] = batch_b["peak_mz"].clone()
    batch_b["peak_mz"][padding] = batch_b["peak_mz"][padding] + 0.2

    metrics_a = model.forward_augmented(batch_a)
    metrics_b = model.forward_augmented(batch_b)

    assert torch.allclose(
        metrics_a["local_global_loss"],
        metrics_b["local_global_loss"],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(metrics_a["loss"], metrics_b["loss"], atol=1e-6, rtol=1e-6)
