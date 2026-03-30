import torch

from models.model import PeakSetEncoder, PeakSetSIGReg


def _build_model(
    *,
    num_target_blocks: int = 2,
    predictor_layers: int = 2,
    predictor_num_register_tokens: int = 0,
    encoder_apply_final_norm: bool = True,
    predictor_apply_final_norm: bool = True,
    jepa_target_normalization: str = "none",
    jepa_target_layers: list[int] | None = None,
) -> PeakSetSIGReg:
    torch.manual_seed(0)
    model = PeakSetSIGReg(
        model_dim=32,
        encoder_num_layers=2,
        encoder_num_heads=4,
        num_peaks=6,
        feature_mlp_hidden_dim=32,
        encoder_apply_final_norm=encoder_apply_final_norm,
        predictor_apply_final_norm=predictor_apply_final_norm,
        predictor_num_register_tokens=predictor_num_register_tokens,
        jepa_num_target_blocks=num_target_blocks,
        masked_token_loss_weight=1.0,
        masked_latent_predictor_num_layers=predictor_layers,
        jepa_target_normalization=jepa_target_normalization,
        jepa_target_layers=jepa_target_layers,
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
def test_predictor_absolute_positions_change_output():
    torch.manual_seed(0)
    model_without_pos = _build_model(predictor_layers=2)
    torch.manual_seed(0)
    model_with_pos = _build_model(predictor_layers=2)
    with torch.no_grad():
        model_without_pos.predictor_position_embedding.weight.zero_()
    predictor_input = torch.randn(1, 6, model_with_pos.model_dim)
    visible_mask = torch.ones(1, 6, dtype=torch.bool)

    out_1 = model_without_pos.predict_masked_latents(predictor_input, visible_mask)
    out_2 = model_with_pos.predict_masked_latents(predictor_input, visible_mask)

    diff = (out_1 - out_2).abs().mean()
    assert float(diff) > 1e-3


@torch.no_grad()
def test_predictor_register_tokens_do_not_get_position_embeddings():
    torch.manual_seed(0)
    model_without_pos = _build_model(
        predictor_layers=2,
        predictor_num_register_tokens=2,
    )
    torch.manual_seed(0)
    model_with_pos = _build_model(
        predictor_layers=2,
        predictor_num_register_tokens=2,
    )
    with torch.no_grad():
        model_without_pos.predictor_position_embedding.weight.zero_()
    predictor_input = torch.randn(1, 6, model_with_pos.model_dim)
    visible_mask = torch.ones(1, 6, dtype=torch.bool)

    x_without_pos = model_without_pos._add_predictor_positions(predictor_input)
    x_without_pos, _ = model_without_pos._append_predictor_register_tokens(
        x_without_pos,
        visible_mask,
    )
    x_with_pos = model_with_pos._add_predictor_positions(predictor_input)
    x_with_pos, _ = model_with_pos._append_predictor_register_tokens(
        x_with_pos,
        visible_mask,
    )

    reg_count = model_with_pos.predictor_num_register_tokens
    torch.testing.assert_close(
        x_without_pos[:, -reg_count:],
        x_with_pos[:, -reg_count:],
        rtol=1e-6,
        atol=1e-6,
    )
    assert not torch.allclose(
        x_without_pos[:, :-reg_count],
        x_with_pos[:, :-reg_count],
        atol=1e-6,
        rtol=1e-6,
    )


@torch.no_grad()
def test_encoder_final_norm_toggle_changes_output():
    torch.manual_seed(0)
    encoder_with_final_norm = PeakSetEncoder(
        model_dim=32,
        num_layers=2,
        num_heads=4,
        num_peaks=6,
        feature_mlp_hidden_dim=32,
        norm_type="layernorm",
        apply_final_norm=True,
    ).eval()
    torch.manual_seed(0)
    encoder_without_final_norm = PeakSetEncoder(
        model_dim=32,
        num_layers=2,
        num_heads=4,
        num_peaks=6,
        feature_mlp_hidden_dim=32,
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
    out_1, _ = PeakSetEncoder.split_peak_and_cls(out_1)
    out_2, _ = PeakSetEncoder.split_peak_and_cls(out_2)

    diff = (out_1 - out_2).abs().mean()
    assert float(diff) > 1e-3


@torch.no_grad()
def test_predictor_final_norm_toggle_changes_output():
    model_with_final_norm = _build_model(
        predictor_layers=2,
        predictor_apply_final_norm=True,
    )
    model_without_final_norm = _build_model(
        predictor_layers=2,
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
def test_predictor_output_is_independent_from_encoder_positions():
    torch.manual_seed(0)
    model_without_encoder_pos = _build_model(predictor_layers=2)
    torch.manual_seed(0)
    model_with_encoder_pos = _build_model(predictor_layers=2)
    with torch.no_grad():
        model_without_encoder_pos.encoder.position_embedding.weight.zero_()
    predictor_input = torch.randn(1, 6, model_without_encoder_pos.model_dim)
    visible_mask = torch.ones(1, 6, dtype=torch.bool)

    out_1 = model_without_encoder_pos.predict_masked_latents(
        predictor_input,
        visible_mask,
    )
    out_2 = model_with_encoder_pos.predict_masked_latents(
        predictor_input,
        visible_mask,
    )

    assert torch.allclose(out_1, out_2)


@torch.no_grad()
def test_forward_augmented_reports_loss_metrics():
    model = _build_model()
    metrics = model.forward_augmented(_make_batch())

    assert "local_global_loss" in metrics
    assert "peak_recon_loss" in metrics
    assert "peak_recon_term" in metrics
    assert "context_fraction" in metrics
    assert "masked_fraction" in metrics
    assert torch.isfinite(metrics["local_global_loss"])
    assert torch.isfinite(metrics["peak_recon_loss"])
    assert float(metrics["masked_fraction"]) > 0.0


@torch.no_grad()
def test_multilayer_targets_widen_teacher_and_predictor_outputs():
    model = _build_model(
        num_target_blocks=2,
        jepa_target_layers=[1, 2],
    )
    batch = _make_batch()
    peak_mz = batch["peak_mz"]
    peak_intensity = batch["peak_intensity"]
    peak_valid_mask = batch["peak_valid_mask"]
    context_mask = batch["context_mask"] & peak_valid_mask
    target_masks = batch["target_masks"] & peak_valid_mask.unsqueeze(1)
    B, K, N = target_masks.shape

    target_latents = model._compute_jepa_online_targets(
        peak_mz,
        peak_intensity,
        peak_valid_mask,
    )
    assert target_latents.shape == (B, N, 2 * model.model_dim)

    context_encoded = model.encoder(
        peak_mz,
        peak_intensity,
        valid_mask=peak_valid_mask,
        visible_mask=context_mask,
    )
    context_emb, _ = PeakSetEncoder.split_peak_and_cls(context_encoded)
    predictor_input = torch.zeros_like(context_emb.unsqueeze(1).expand(-1, K, -1, -1))
    predictor_input = torch.where(
        context_mask.unsqueeze(1).unsqueeze(-1),
        context_emb.unsqueeze(1).expand(-1, K, -1, -1),
        predictor_input,
    )
    predictor_input = torch.where(
        target_masks.unsqueeze(-1),
        model.latent_mask_token.view(1, 1, 1, -1).to(context_emb),
        predictor_input,
    )
    predictor_output = model.predict_masked_targets(
        predictor_input.reshape(B * K, N, -1),
        (context_mask.unsqueeze(1) | target_masks).reshape(B * K, N),
    )
    assert predictor_output.shape == (B * K, N, 2 * model.model_dim)
    peak_output = model.predict_peak_values(
        predictor_input.reshape(B * K, N, -1),
        (context_mask.unsqueeze(1) | target_masks).reshape(B * K, N),
    )
    assert peak_output.shape == (B * K, N, 2)

    metrics = model.forward_augmented(batch)
    assert torch.isfinite(metrics["loss"])


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

    context_encoded = model.encoder(
        peak_mz,
        peak_intensity,
        valid_mask=peak_valid_mask,
        visible_mask=context_mask,
    )
    context_emb, _ = PeakSetEncoder.split_peak_and_cls(context_encoded)
    B, K, N = target_masks.shape
    teacher_target = model.encoder(
        peak_mz,
        peak_intensity,
        valid_mask=peak_valid_mask,
        visible_mask=peak_valid_mask,
    )
    teacher_target, _ = PeakSetEncoder.split_peak_and_cls(teacher_target)

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
    predictor_output = model.predict_masked_targets(
        predictor_input.reshape(B * K, N, -1),
        predictor_union_mask.reshape(B * K, N),
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
def test_local_global_loss_can_zscore_teacher_targets():
    model = _build_model(
        num_target_blocks=2,
        jepa_target_normalization="zscore",
    )
    model.sigreg_lambda = 0.0
    batch = _make_batch()

    metrics = model.forward_augmented(batch)

    peak_mz = batch["peak_mz"]
    peak_intensity = batch["peak_intensity"]
    peak_valid_mask = batch["peak_valid_mask"]
    context_mask = batch["context_mask"] & peak_valid_mask
    target_masks = batch["target_masks"] & peak_valid_mask.unsqueeze(1)
    target_masks_by_view = target_masks.permute(1, 0, 2)

    context_encoded = model.encoder(
        peak_mz,
        peak_intensity,
        valid_mask=peak_valid_mask,
        visible_mask=context_mask,
    )
    context_emb, _ = PeakSetEncoder.split_peak_and_cls(context_encoded)
    B, K, N = target_masks.shape
    teacher_target = model.encoder(
        peak_mz,
        peak_intensity,
        valid_mask=peak_valid_mask,
        visible_mask=peak_valid_mask,
    )
    teacher_target, _ = PeakSetEncoder.split_peak_and_cls(teacher_target)
    teacher_target = teacher_target.unsqueeze(1).expand(-1, K, -1, -1)
    teacher_target = (teacher_target - teacher_target.mean(dim=-1, keepdim=True)) / (
        teacher_target.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
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
    predictor_output = model.predict_masked_targets(
        predictor_input.reshape(B * K, N, -1),
        predictor_union_mask.reshape(B * K, N),
    ).reshape(K, B, N, -1)

    per_token_l1 = (
        predictor_output
        - teacher_target.permute(1, 0, 2, 3).detach()
    ).abs().mean(dim=-1)
    masked_only_loss = (
        per_token_l1 * target_masks_by_view.float()
    ).sum() / target_masks_by_view.float().sum().clamp_min(1.0)

    assert torch.allclose(metrics["local_global_loss"], masked_only_loss)


@torch.no_grad()
def test_multilayer_zscore_normalizes_each_target_slice_independently():
    model = _build_model(
        predictor_layers=2,
        jepa_target_normalization="zscore",
        jepa_target_layers=[1, 2],
    )
    x = torch.randn(2, 3, 2 * model.model_dim)

    normalized = model._apply_jepa_target_normalization(x)
    normalized = normalized.reshape(2, 3, 2, model.model_dim)

    assert torch.allclose(
        normalized.mean(dim=-1),
        torch.zeros(2, 3, 2),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        normalized.std(dim=-1, unbiased=False),
        torch.ones(2, 3, 2),
        atol=1e-4,
        rtol=1e-4,
    )


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

    target_latents = model._compute_jepa_online_targets(
        batch_a["peak_mz"],
        batch_a["peak_intensity"],
        batch_a["peak_valid_mask"],
    ).unsqueeze(1).expand(
        -1,
        model.jepa_num_target_blocks,
        -1,
        -1,
    )
    metrics_a = model.forward_augmented(batch_a, target_latents=target_latents)
    metrics_b = model.forward_augmented(batch_b, target_latents=target_latents)

    assert torch.allclose(
        metrics_a["local_global_loss"],
        metrics_b["local_global_loss"],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        metrics_a["peak_recon_loss"],
        metrics_b["peak_recon_loss"],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(metrics_a["loss"], metrics_b["loss"], atol=1e-6, rtol=1e-6)
