import torch

from models.model import PeakSetSIGReg


def _build_model(
    *,
    num_local_views: int = 0,
    predictor_layers: int = 2,
    encoder_use_rope: bool = True,
) -> PeakSetSIGReg:
    torch.manual_seed(0)
    model = PeakSetSIGReg(
        num_peaks=6,
        model_dim=32,
        encoder_num_layers=2,
        encoder_num_heads=4,
        feature_mlp_hidden_dim=32,
        encoder_use_rope=encoder_use_rope,
        pooling_type="mean",
        multicrop_num_local_views=num_local_views,
        masked_token_loss_weight=1.0,
        masked_latent_predictor_num_layers=predictor_layers,
    )
    model.eval()
    return model


@torch.no_grad()
def test_predictor_zero_layers_is_identity():
    model = _build_model(predictor_layers=0)
    local_token_emb = torch.randn(1, 6, model.model_dim)
    local_valid = torch.ones(1, 6, dtype=torch.bool)

    out = model.predict_masked_latents(local_token_emb, local_valid)

    assert torch.allclose(out, local_token_emb)


@torch.no_grad()
def test_predictor_rope_toggle_changes_output():
    torch.manual_seed(0)
    model_no_rope = _build_model(predictor_layers=2, encoder_use_rope=False)
    torch.manual_seed(0)
    model_with_rope = _build_model(predictor_layers=2, encoder_use_rope=True)
    local_token_emb = torch.randn(1, 6, model_with_rope.model_dim)
    local_valid = torch.ones(1, 6, dtype=torch.bool)

    out_1 = model_no_rope.predict_masked_latents(local_token_emb, local_valid)
    out_2 = model_with_rope.predict_masked_latents(local_token_emb, local_valid)

    diff = (out_1 - out_2).abs().mean()
    assert float(diff) > 1e-3


@torch.no_grad()
def test_predictor_outputs_finite():
    model = _build_model(predictor_layers=2)
    local_token_emb = torch.randn(1, 6, model.model_dim)
    local_valid = torch.ones(1, 6, dtype=torch.bool)

    out = model.predict_masked_latents(local_token_emb, local_valid)

    assert torch.isfinite(out).all()


@torch.no_grad()
def test_forward_augmented_reports_local_global_l1_loss():
    model = _build_model(num_local_views=1)

    fused_mz = torch.tensor(
        [
            [0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
            [0.11, 0.19, 0.29, 0.41, 0.49, 0.59],
            [0.15, 0.25, 0.35, 0.45, 0.55, 0.65],
            [0.14, 0.24, 0.34, 0.44, 0.54, 0.64],
        ],
        dtype=torch.float32,
    )
    fused_intensity = torch.tensor(
        [
            [0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
            [0.8, 0.6, 0.5, 0.4, 0.4, 0.2],
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            [0.9, 0.7, 0.6, 0.5, 0.5, 0.3],
        ],
        dtype=torch.float32,
    )
    fused_valid_mask = torch.ones_like(fused_mz, dtype=torch.bool)
    fused_masked_positions = torch.tensor(
        [
            [False, False, False, True, False, False],
            [False, True, False, False, False, False],
            [False, False, False, True, False, False],
            [False, False, True, False, False, False],
        ]
    )

    metrics = model.forward_augmented(
        {
            "fused_mz": fused_mz,
            "fused_intensity": fused_intensity,
            "fused_valid_mask": fused_valid_mask,
            "fused_masked_positions": fused_masked_positions,
        }
    )
    assert "local_global_l1_loss" in metrics
    assert "token_sigreg_loss" in metrics
    assert torch.isfinite(metrics["local_global_l1_loss"])
    assert float(metrics["masked_fraction"]) > 0.0


@torch.no_grad()
def test_local_global_l1_uses_masked_tokens_only():
    model = _build_model(num_local_views=1)
    model.sigreg_lambda = 0.0

    fused_mz = torch.tensor(
        [
            [0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
            [0.11, 0.19, 0.29, 0.41, 0.49, 0.59],
            [0.15, 0.25, 0.35, 0.45, 0.55, 0.65],
            [0.14, 0.24, 0.34, 0.44, 0.54, 0.64],
        ],
        dtype=torch.float32,
    )
    fused_intensity = torch.tensor(
        [
            [0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
            [0.8, 0.6, 0.5, 0.4, 0.4, 0.2],
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            [0.9, 0.7, 0.6, 0.5, 0.5, 0.3],
        ],
        dtype=torch.float32,
    )
    fused_valid_mask = torch.ones_like(fused_mz, dtype=torch.bool)
    fused_masked_positions = torch.tensor(
        [
            [False, False, False, True, False, False],
            [False, True, False, False, False, False],
            [False, False, False, True, False, False],
            [False, False, True, False, False, False],
        ]
    )

    metrics = model.forward_augmented(
        {
            "fused_mz": fused_mz,
            "fused_intensity": fused_intensity,
            "fused_valid_mask": fused_valid_mask,
            "fused_masked_positions": fused_masked_positions,
        }
    )

    fused_masked_positions = fused_masked_positions & fused_valid_mask
    fused_masked_positions = fused_masked_positions.reshape(model.num_views, -1, fused_masked_positions.shape[1])
    fused_masked_positions[0] = False
    fused_masked_positions = fused_masked_positions.reshape_as(fused_valid_mask)
    encoder_visible_mask = fused_valid_mask & (~fused_masked_positions)
    fused_emb = model.encoder(
        fused_mz,
        fused_intensity,
        valid_mask=encoder_visible_mask,
    )
    V = model.num_views
    B = fused_emb.shape[0] // V
    N = fused_emb.shape[1]
    token_emb = fused_emb.reshape(V, B, N, fused_emb.shape[2])
    token_valid = fused_valid_mask.reshape(V, B, N)
    token_masked = fused_masked_positions.reshape(V, B, N)
    global_token_emb = token_emb[0]
    local_token_emb = token_emb[1]
    local_valid = token_valid[1]
    local_masked = token_masked[1]
    latent_mask_token = model.latent_mask_token.view(1, 1, -1).to(
        dtype=fused_emb.dtype,
        device=fused_emb.device,
    )
    local_token_emb_remasked = torch.where(
        local_masked.unsqueeze(-1),
        latent_mask_token,
        local_token_emb,
    )
    local_token_pred = model.predict_masked_latents(
        local_token_emb_remasked,
        local_valid,
    )
    per_token_l1 = (local_token_pred - global_token_emb).abs().mean(dim=-1)
    all_valid_loss = (per_token_l1 * local_valid.float()).sum() / local_valid.float().sum()
    masked_only_loss = (
        (per_token_l1 * local_masked.float()).sum()
        / local_masked.float().sum().clamp_min(1.0)
    )

    assert torch.allclose(metrics["local_global_l1_loss"], masked_only_loss)
    assert float((metrics["local_global_l1_loss"] - all_valid_loss).abs()) > 1e-6


@torch.no_grad()
def test_local_global_l2_uses_masked_tokens_only():
    model = _build_model(num_local_views=1)
    model.sigreg_lambda = 0.0
    model.masked_token_loss_type = "l2"

    fused_mz = torch.tensor(
        [
            [0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
            [0.11, 0.19, 0.29, 0.41, 0.49, 0.59],
            [0.15, 0.25, 0.35, 0.45, 0.55, 0.65],
            [0.14, 0.24, 0.34, 0.44, 0.54, 0.64],
        ],
        dtype=torch.float32,
    )
    fused_intensity = torch.tensor(
        [
            [0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
            [0.8, 0.6, 0.5, 0.4, 0.4, 0.2],
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            [0.9, 0.7, 0.6, 0.5, 0.5, 0.3],
        ],
        dtype=torch.float32,
    )
    fused_valid_mask = torch.ones_like(fused_mz, dtype=torch.bool)
    fused_masked_positions = torch.tensor(
        [
            [False, False, False, True, False, False],
            [False, True, False, False, False, False],
            [False, False, False, True, False, False],
            [False, False, True, False, False, False],
        ]
    )

    metrics = model.forward_augmented(
        {
            "fused_mz": fused_mz,
            "fused_intensity": fused_intensity,
            "fused_valid_mask": fused_valid_mask,
            "fused_masked_positions": fused_masked_positions,
        }
    )

    fused_masked_positions = fused_masked_positions & fused_valid_mask
    fused_masked_positions = fused_masked_positions.reshape(model.num_views, -1, fused_masked_positions.shape[1])
    fused_masked_positions[0] = False
    fused_masked_positions = fused_masked_positions.reshape_as(fused_valid_mask)
    encoder_visible_mask = fused_valid_mask & (~fused_masked_positions)
    fused_emb = model.encoder(
        fused_mz,
        fused_intensity,
        valid_mask=encoder_visible_mask,
    )
    V = model.num_views
    B = fused_emb.shape[0] // V
    N = fused_emb.shape[1]
    token_emb = fused_emb.reshape(V, B, N, fused_emb.shape[2])
    token_valid = fused_valid_mask.reshape(V, B, N)
    token_masked = fused_masked_positions.reshape(V, B, N)
    global_token_emb = token_emb[0]
    local_token_emb = token_emb[1]
    local_valid = token_valid[1]
    local_masked = token_masked[1]
    latent_mask_token = model.latent_mask_token.view(1, 1, -1).to(
        dtype=fused_emb.dtype,
        device=fused_emb.device,
    )
    local_token_emb_remasked = torch.where(
        local_masked.unsqueeze(-1),
        latent_mask_token,
        local_token_emb,
    )
    local_token_pred = model.predict_masked_latents(
        local_token_emb_remasked,
        local_valid,
    )
    per_token_l2 = (local_token_pred - global_token_emb).square().mean(dim=-1)
    all_valid_loss = (per_token_l2 * local_valid.float()).sum() / local_valid.float().sum()
    masked_only_loss = (
        (per_token_l2 * local_masked.float()).sum()
        / local_masked.float().sum().clamp_min(1.0)
    )

    assert torch.allclose(metrics["local_global_loss"], masked_only_loss)
    assert torch.allclose(metrics["local_global_l1_loss"], masked_only_loss)
    assert float((metrics["local_global_loss"] - all_valid_loss).abs()) > 1e-6


@torch.no_grad()
def test_global_view_masked_positions_are_ignored_for_context():
    model = _build_model(num_local_views=1)
    model.sigreg_lambda = 0.0

    fused_mz = torch.tensor(
        [
            [0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
            [0.15, 0.25, 0.35, 0.45, 0.55, 0.65],
        ],
        dtype=torch.float32,
    )
    fused_intensity = torch.tensor(
        [
            [0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        ],
        dtype=torch.float32,
    )
    fused_valid_mask = torch.ones_like(fused_mz, dtype=torch.bool)

    masked_a = torch.tensor(
        [
            [False, True, False, True, False, False],
            [False, False, True, False, False, False],
        ]
    )
    masked_b = torch.tensor(
        [
            [False, False, False, False, False, False],
            [False, False, True, False, False, False],
        ]
    )

    metrics_a = model.forward_augmented(
        {
            "fused_mz": fused_mz,
            "fused_intensity": fused_intensity,
            "fused_valid_mask": fused_valid_mask,
            "fused_masked_positions": masked_a,
        }
    )
    metrics_b = model.forward_augmented(
        {
            "fused_mz": fused_mz,
            "fused_intensity": fused_intensity,
            "fused_valid_mask": fused_valid_mask,
            "fused_masked_positions": masked_b,
        }
    )

    assert torch.allclose(metrics_a["local_global_l1_loss"], metrics_b["local_global_l1_loss"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(metrics_a["loss"], metrics_b["loss"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(metrics_a["masked_fraction"], metrics_b["masked_fraction"], atol=1e-6, rtol=1e-6)
