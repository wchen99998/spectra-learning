import torch

from models.model import PeakSetEncoder, PeakSetSIGReg


def _build_model(
    *,
    num_target_blocks: int = 2,
    predictor_layers: int = 2,
    encoder_use_rope: bool = True,
    predictor_use_rope: bool | None = None,
    encoder_apply_final_norm: bool = True,
    predictor_apply_final_norm: bool = True,
) -> PeakSetSIGReg:
    torch.manual_seed(0)
    model = PeakSetSIGReg(
        model_dim=32,
        encoder_num_layers=2,
        encoder_num_heads=4,
        feature_mlp_hidden_dim=32,
        encoder_use_rope=encoder_use_rope,
        predictor_use_rope=predictor_use_rope,
        encoder_apply_final_norm=encoder_apply_final_norm,
        predictor_apply_final_norm=predictor_apply_final_norm,
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
    visible_mask = torch.ones(2, 6, dtype=torch.bool)

    out = model.predict_masked_latents(predictor_input, visible_mask)

    assert torch.allclose(out, predictor_input)


@torch.no_grad()
def test_predictor_rope_toggle_changes_output():
    torch.manual_seed(0)
    model_no_rope = _build_model(
        predictor_layers=2,
        encoder_use_rope=False,
        predictor_use_rope=False,
    )
    torch.manual_seed(0)
    model_with_rope = _build_model(
        predictor_layers=2,
        encoder_use_rope=False,
        predictor_use_rope=True,
    )
    predictor_input = torch.randn(1, 6, model_with_rope.model_dim)
    visible_mask = torch.ones(1, 6, dtype=torch.bool)

    out_1 = model_no_rope.predict_masked_latents(predictor_input, visible_mask)
    out_2 = model_with_rope.predict_masked_latents(predictor_input, visible_mask)

    diff = (out_1 - out_2).abs().mean()
    assert float(diff) > 1e-3


@torch.no_grad()
def test_encoder_final_norm_toggle_changes_output():
    torch.manual_seed(0)
    encoder_with_final_norm = PeakSetEncoder(
        model_dim=32,
        num_layers=2,
        num_heads=4,
        feature_mlp_hidden_dim=32,
        use_rope=False,
        norm_type="layernorm",
        apply_final_norm=True,
    ).eval()
    torch.manual_seed(0)
    encoder_without_final_norm = PeakSetEncoder(
        model_dim=32,
        num_layers=2,
        num_heads=4,
        feature_mlp_hidden_dim=32,
        use_rope=False,
        norm_type="layernorm",
        apply_final_norm=False,
    ).eval()
    peak_mz = torch.rand(2, 6)
    peak_intensity = torch.rand(2, 6)
    valid_mask = torch.ones(2, 6, dtype=torch.bool)

    out_1 = encoder_with_final_norm(
        peak_mz,
        peak_intensity,
        valid_mask=valid_mask,
        visible_mask=valid_mask,
    )
    out_2 = encoder_without_final_norm(
        peak_mz,
        peak_intensity,
        valid_mask=valid_mask,
        visible_mask=valid_mask,
    )

    diff = (out_1 - out_2).abs().mean()
    assert float(diff) > 1e-3


@torch.no_grad()
def test_predictor_final_norm_toggle_changes_output():
    model_with_final_norm = _build_model(
        predictor_layers=2,
        predictor_use_rope=False,
        predictor_apply_final_norm=True,
    )
    model_without_final_norm = _build_model(
        predictor_layers=2,
        predictor_use_rope=False,
        predictor_apply_final_norm=False,
    )
    predictor_input = torch.randn(1, 6, model_with_final_norm.model_dim)
    visible_mask = torch.ones(1, 6, dtype=torch.bool)

    out_1 = model_with_final_norm.predict_masked_latents(predictor_input, visible_mask)
    out_2 = model_without_final_norm.predict_masked_latents(
        predictor_input, visible_mask
    )

    diff = (out_1 - out_2).abs().mean()
    assert float(diff) > 1e-3


@torch.no_grad()
def test_predictor_rope_is_independent_from_encoder_rope():
    torch.manual_seed(0)
    model_encoder_no_rope = _build_model(
        predictor_layers=2,
        encoder_use_rope=False,
        predictor_use_rope=False,
    )
    torch.manual_seed(0)
    model_encoder_with_rope = _build_model(
        predictor_layers=2,
        encoder_use_rope=True,
        predictor_use_rope=False,
    )
    predictor_input = torch.randn(1, 6, model_encoder_no_rope.model_dim)
    visible_mask = torch.ones(1, 6, dtype=torch.bool)

    out_1 = model_encoder_no_rope.predict_masked_latents(predictor_input, visible_mask)
    out_2 = model_encoder_with_rope.predict_masked_latents(
        predictor_input,
        visible_mask,
    )

    assert torch.allclose(out_1, out_2)


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
def test_local_global_loss_uses_target_tokens_only():
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
        pack_n=model._context_pack_n,
    )
    B, K, N = target_masks.shape
    teacher_target = model.encoder(
        peak_mz,
        peak_intensity,
        valid_mask=peak_valid_mask,
        visible_mask=peak_valid_mask,
        pack_n=model._full_pack_n,
        prefix_pack=True,
    )

    predictor_union_mask = context_mask.unsqueeze(0) | target_masks_by_view
    predictor_input = torch.zeros_like(context_emb.unsqueeze(0).expand(K, -1, -1, -1))
    predictor_input = torch.where(
        context_mask.unsqueeze(0).unsqueeze(-1),
        context_emb.unsqueeze(0).expand(K, -1, -1, -1),
        predictor_input,
    )
    latent_mask_token = model.latent_mask_token.view(1, 1, 1, -1).to(
        dtype=context_emb.dtype,
        device=context_emb.device,
    )
    predictor_input = torch.where(
        target_masks_by_view.unsqueeze(-1),
        latent_mask_token,
        predictor_input,
    )
    predictor_output = model.predict_masked_latents(
        predictor_input.reshape(B * K, N, -1),
        predictor_union_mask.reshape(B * K, N),
        pack_n=model._predictor_pack_n,
    ).reshape(K, B, N, -1)

    per_token_l1 = (
        predictor_output
        - teacher_target.unsqueeze(0).expand(K, -1, -1, -1).detach()
    ).abs().mean(dim=-1)
    masked_only_loss = (
        per_token_l1 * target_masks_by_view.float()
    ).sum() / target_masks_by_view.float().sum().clamp_min(1.0)

    assert torch.allclose(metrics["local_global_loss"], masked_only_loss)


@torch.no_grad()
def test_positions_outside_union_do_not_change_loss():
    model = _build_model(num_target_blocks=2)
    model.sigreg_lambda = 0.0
    batch_a = _make_batch()
    batch_b = {key: value.clone() for key, value in batch_a.items()}

    ignored = ~(batch_a["context_mask"] | batch_a["target_masks"].any(dim=1))
    batch_b["peak_intensity"] = batch_b["peak_intensity"].clone()
    batch_b["peak_intensity"][ignored] = batch_b["peak_intensity"][ignored] + 0.5
    batch_b["peak_mz"] = batch_b["peak_mz"].clone()
    batch_b["peak_mz"][ignored] = batch_b["peak_mz"][ignored] + 0.2

    teacher_targets = model.encoder(
        batch_a["peak_mz"],
        batch_a["peak_intensity"],
        valid_mask=batch_a["peak_valid_mask"],
        visible_mask=batch_a["peak_valid_mask"],
        pack_n=model._full_pack_n,
        prefix_pack=True,
    ).unsqueeze(1).expand(-1, model.jepa_num_target_blocks, -1, -1)
    metrics_a = model.forward_augmented(batch_a, teacher_targets=teacher_targets)
    metrics_b = model.forward_augmented(batch_b, teacher_targets=teacher_targets)

    assert torch.allclose(
        metrics_a["local_global_loss"],
        metrics_b["local_global_loss"],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(metrics_a["loss"], metrics_b["loss"], atol=1e-6, rtol=1e-6)
