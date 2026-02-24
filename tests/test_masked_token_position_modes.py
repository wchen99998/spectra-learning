import torch

from models.model import PeakSetSIGReg


def _build_model(position_mode: str) -> PeakSetSIGReg:
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
        multicrop_num_global_views=2,
        multicrop_num_local_views=0,
        use_masked_token_input=True,
        masked_token_position_mode=position_mode,
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
def test_forward_augmented_reports_masked_latent_loss():
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
        multicrop_num_global_views=2,
        multicrop_num_local_views=0,
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
            [0.8, 0.7, 0.6, 0.0, 0.4, 0.3],
            [0.8, 0.6, 0.5, 0.0, 0.4, 0.2],
            [0.9, 0.8, 0.7, 0.0, 0.5, 0.4],
            [0.9, 0.7, 0.6, 0.0, 0.5, 0.3],
        ],
        dtype=torch.float32,
    )
    fused_valid_mask = torch.ones_like(fused_mz, dtype=torch.bool)
    fused_masked_positions = torch.tensor(
        [
            [False, False, False, True, False, False],
            [False, False, True, False, False, False],
            [False, True, False, False, False, False],
            [False, False, False, True, False, False],
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
    assert "masked_latent_loss" in metrics
    assert torch.isfinite(metrics["masked_latent_loss"])
    assert float(metrics["masked_fraction"]) > 0.0
