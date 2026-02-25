import torch

from models.model import PeakSetSIGReg


def _build_model(
    position_mode: str,
    *,
    attention_mode: str = "bidirectional",
) -> PeakSetSIGReg:
    torch.manual_seed(0)
    model = PeakSetSIGReg(
        num_peaks=6,
        model_dim=32,
        encoder_num_layers=2,
        encoder_num_heads=4,
        feature_mlp_hidden_dim=32,
        encoder_use_rope=True,
        sigreg_use_projector=False,
        pooling_type="mean",
        multicrop_num_local_views=0,
        use_masked_token_input=True,
        masked_token_position_mode=position_mode,
        masked_token_attention_mode=attention_mode,
    )
    model.eval()
    return model


@torch.no_grad()
def test_masked_mz_position_mode_depends_on_masked_mz():
    model = _build_model("mz")
    peak_mz = torch.tensor([[0.10, 0.20, 0.30, 0.40, 0.50, 0.60]])
    peak_intensity = torch.tensor([[0.8, 0.7, 0.6, 0.5, 0.4, 0.3]])
    valid_mask = torch.ones_like(peak_mz, dtype=torch.bool)
    precursor_mz = torch.tensor([0.9])
    masked_positions = torch.tensor([[False, False, False, True, False, False]])

    peak_mz_alt = peak_mz.clone()
    peak_mz_alt[:, 3] = 0.95

    out_1 = model.encoder(
        peak_mz,
        peak_intensity,
        valid_mask=valid_mask,
        precursor_mz=precursor_mz,
        masked_positions=masked_positions,
        mask_token=model.mask_token,
    )
    out_2 = model.encoder(
        peak_mz_alt,
        peak_intensity,
        valid_mask=valid_mask,
        precursor_mz=precursor_mz,
        masked_positions=masked_positions,
        mask_token=model.mask_token,
    )

    masked_diff = (out_1[:, 3, :] - out_2[:, 3, :]).abs().mean()
    assert float(masked_diff) > 1e-3


@torch.no_grad()
def test_masked_index_position_mode_ignores_masked_mz():
    model = _build_model("index")
    peak_mz = torch.tensor([[0.10, 0.20, 0.30, 0.40, 0.50, 0.60]])
    peak_intensity = torch.tensor([[0.8, 0.7, 0.6, 0.5, 0.4, 0.3]])
    valid_mask = torch.ones_like(peak_mz, dtype=torch.bool)
    precursor_mz = torch.tensor([0.9])
    masked_positions = torch.tensor([[False, False, False, True, False, False]])

    peak_mz_alt = peak_mz.clone()
    peak_mz_alt[:, 3] = 0.95

    out_1 = model.encoder(
        peak_mz,
        peak_intensity,
        valid_mask=valid_mask,
        precursor_mz=precursor_mz,
        masked_positions=masked_positions,
        mask_token=model.mask_token,
    )
    out_2 = model.encoder(
        peak_mz_alt,
        peak_intensity,
        valid_mask=valid_mask,
        precursor_mz=precursor_mz,
        masked_positions=masked_positions,
        mask_token=model.mask_token,
    )

    masked_diff = (out_1[:, 3, :] - out_2[:, 3, :]).abs().mean()
    assert float(masked_diff) < 1e-6


@torch.no_grad()
def test_masked_query_to_unmasked_kv_blocks_mask_token_leakage():
    peak_mz = torch.tensor([[0.10, 0.20, 0.30, 0.40, 0.50, 0.60]])
    peak_intensity = torch.tensor([[0.8, 0.7, 0.6, 0.5, 0.4, 0.3]])
    valid_mask = torch.ones_like(peak_mz, dtype=torch.bool)
    precursor_mz = torch.tensor([0.9])
    masked_positions = torch.tensor([[False, False, False, True, False, False]])

    model = _build_model("index", attention_mode="masked_query_to_unmasked_kv")
    out_1 = model.encoder(
        peak_mz,
        peak_intensity,
        valid_mask=valid_mask,
        precursor_mz=precursor_mz,
        masked_positions=masked_positions,
        mask_token=model.mask_token,
    )
    model.mask_token.data.add_(5.0)
    out_2 = model.encoder(
        peak_mz,
        peak_intensity,
        valid_mask=valid_mask,
        precursor_mz=precursor_mz,
        masked_positions=masked_positions,
        mask_token=model.mask_token,
    )

    unmasked_positions = ~masked_positions
    unmasked_diff = (out_1[unmasked_positions] - out_2[unmasked_positions]).abs().mean()
    assert float(unmasked_diff) < 1e-6


@torch.no_grad()
def test_forward_augmented_reports_local_global_l1_loss():
    torch.manual_seed(0)
    model = PeakSetSIGReg(
        num_peaks=6,
        model_dim=32,
        encoder_num_layers=2,
        encoder_num_heads=4,
        feature_mlp_hidden_dim=32,
        encoder_use_rope=True,
        sigreg_use_projector=False,
        pooling_type="mean",
        multicrop_num_local_views=1,
        use_masked_token_input=True,
        masked_token_position_mode="index",
        masked_token_loss_weight=1.0,
    )
    model.eval()

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
    fused_precursor_mz = torch.tensor([0.9, 0.9, 0.95, 0.95], dtype=torch.float32)

    metrics = model.forward_augmented(
        {
            "fused_mz": fused_mz,
            "fused_intensity": fused_intensity,
            "fused_valid_mask": fused_valid_mask,
            "fused_masked_positions": fused_masked_positions,
            "fused_precursor_mz": fused_precursor_mz,
        }
    )
    assert "local_global_l1_loss" in metrics
    assert "token_sigreg_loss" in metrics
    assert torch.isfinite(metrics["local_global_l1_loss"])
    assert float(metrics["masked_fraction"]) > 0.0


@torch.no_grad()
def test_local_global_l1_uses_all_valid_tokens():
    torch.manual_seed(0)
    model = PeakSetSIGReg(
        num_peaks=6,
        model_dim=32,
        encoder_num_layers=2,
        encoder_num_heads=4,
        feature_mlp_hidden_dim=32,
        encoder_use_rope=True,
        sigreg_use_projector=False,
        pooling_type="mean",
        multicrop_num_local_views=1,
        use_masked_token_input=True,
        masked_token_position_mode="index",
        masked_token_loss_weight=1.0,
        sigreg_lambda=0.0,
    )
    model.eval()

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
    fused_precursor_mz = torch.tensor([0.9, 0.9, 0.95, 0.95], dtype=torch.float32)

    metrics = model.forward_augmented(
        {
            "fused_mz": fused_mz,
            "fused_intensity": fused_intensity,
            "fused_valid_mask": fused_valid_mask,
            "fused_masked_positions": fused_masked_positions,
            "fused_precursor_mz": fused_precursor_mz,
        }
    )

    fused_masked_positions = fused_masked_positions & fused_valid_mask
    fused_masked_positions = fused_masked_positions.reshape(model.num_views, -1, fused_masked_positions.shape[1])
    fused_masked_positions[0] = False
    fused_masked_positions = fused_masked_positions.reshape_as(fused_valid_mask)
    student_intensity = fused_intensity.masked_fill(fused_masked_positions, 0.0)
    fused_emb = model.encoder(
        fused_mz,
        student_intensity,
        valid_mask=fused_valid_mask,
        precursor_mz=fused_precursor_mz,
        masked_positions=fused_masked_positions,
        mask_token=model.mask_token,
    )
    V = model.num_views
    B = fused_emb.shape[0] // V
    N = fused_emb.shape[1]
    D = fused_emb.shape[2]
    token_emb = fused_emb.reshape(V, B, N, D)
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
    local_token_pred = model.masked_latent_predictor(local_token_emb_remasked)
    per_token_l1 = (local_token_pred - global_token_emb).abs().mean(dim=-1)
    all_valid_loss = (per_token_l1 * local_valid.float()).sum() / local_valid.float().sum()
    masked_only_loss = (
        (per_token_l1 * local_masked.float()).sum()
        / local_masked.float().sum().clamp_min(1.0)
    )

    assert torch.allclose(metrics["local_global_l1_loss"], all_valid_loss)
    assert float((metrics["local_global_l1_loss"] - masked_only_loss).abs()) > 1e-6


@torch.no_grad()
def test_global_view_masked_positions_are_ignored_for_context():
    torch.manual_seed(0)
    model = PeakSetSIGReg(
        num_peaks=6,
        model_dim=32,
        encoder_num_layers=2,
        encoder_num_heads=4,
        feature_mlp_hidden_dim=32,
        encoder_use_rope=True,
        sigreg_use_projector=False,
        pooling_type="mean",
        multicrop_num_local_views=1,
        use_masked_token_input=True,
        masked_token_position_mode="index",
        masked_token_loss_weight=1.0,
        sigreg_lambda=0.0,
    )
    model.eval()

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
    fused_precursor_mz = torch.tensor([0.9, 0.95], dtype=torch.float32)

    masked_a = torch.tensor(
        [
            [False, True, False, True, False, False],  # global row masks (should be ignored)
            [False, False, True, False, False, False],  # local row mask (used)
        ]
    )
    masked_b = torch.tensor(
        [
            [False, False, False, False, False, False],  # global row differs only here
            [False, False, True, False, False, False],   # local row identical
        ]
    )

    metrics_a = model.forward_augmented(
        {
            "fused_mz": fused_mz,
            "fused_intensity": fused_intensity,
            "fused_valid_mask": fused_valid_mask,
            "fused_masked_positions": masked_a,
            "fused_precursor_mz": fused_precursor_mz,
        }
    )
    metrics_b = model.forward_augmented(
        {
            "fused_mz": fused_mz,
            "fused_intensity": fused_intensity,
            "fused_valid_mask": fused_valid_mask,
            "fused_masked_positions": masked_b,
            "fused_precursor_mz": fused_precursor_mz,
        }
    )

    assert torch.allclose(metrics_a["local_global_l1_loss"], metrics_b["local_global_l1_loss"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(metrics_a["loss"], metrics_b["loss"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(metrics_a["masked_fraction"], metrics_b["masked_fraction"], atol=1e-6, rtol=1e-6)
