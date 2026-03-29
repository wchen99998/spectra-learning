import torch

from models.model import PeakSetEncoder, PeakSetSIGReg


def _assert_zero_at_mask(x: torch.Tensor, mask: torch.Tensor) -> None:
    masked = x.masked_select(mask.unsqueeze(-1).expand_as(x))
    assert torch.equal(masked, torch.zeros_like(masked))


def _place_visible_tokens(
    base_tokens: torch.Tensor,
    visible_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    peak_mz = torch.zeros_like(visible_mask, dtype=base_tokens.dtype)
    peak_intensity = torch.zeros_like(visible_mask, dtype=base_tokens.dtype)
    for row_idx in range(visible_mask.shape[0]):
        pos = visible_mask[row_idx].nonzero(as_tuple=False).squeeze(-1)
        peak_mz[row_idx, pos] = base_tokens[: pos.numel()]
        peak_intensity[row_idx, pos] = base_tokens[: pos.numel()] * 0.5 + 0.1
    return peak_mz, peak_intensity


@torch.no_grad()
def test_encoder_sparse_pack_matches_full_on_visible_tokens():
    torch.manual_seed(0)
    encoder = PeakSetEncoder(
        model_dim=32,
        num_layers=2,
        num_heads=4,
        num_peaks=8,
        feature_mlp_hidden_dim=16,
    ).eval()
    peak_mz = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0, 0.0],
            [0.7, 0.8, 0.9, 1.0, 1.1, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    peak_intensity = torch.tensor(
        [
            [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.0, 0.0],
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    valid_mask = torch.tensor(
        [
            [True, True, True, True, True, True, False, False],
            [True, True, True, True, True, False, False, False],
        ]
    )
    visible_mask = torch.tensor(
        [
            [False, True, False, True, True, False, False, False],
            [True, False, True, False, True, False, False, False],
        ]
    )
    out_full = encoder(
        peak_mz,
        peak_intensity,
        valid_mask=valid_mask,
        visible_mask=visible_mask,
    )
    out_packed = encoder(
        peak_mz,
        peak_intensity,
        valid_mask=valid_mask,
        visible_mask=visible_mask,
        pack_n=4,
    )

    visible = visible_mask & valid_mask
    torch.testing.assert_close(
        out_packed.masked_select(visible.unsqueeze(-1).expand_as(out_packed)),
        out_full.masked_select(visible.unsqueeze(-1).expand_as(out_full)),
        rtol=1e-6,
        atol=1e-6,
    )
    _assert_zero_at_mask(out_packed, ~visible)


@torch.no_grad()
def test_encoder_prefix_pack_matches_full_on_valid_prefix():
    torch.manual_seed(1)
    encoder = PeakSetEncoder(
        model_dim=32,
        num_layers=2,
        num_heads=4,
        num_peaks=8,
        feature_mlp_hidden_dim=16,
    ).eval()
    peak_mz = torch.rand(2, 8)
    peak_intensity = torch.rand(2, 8)
    valid_mask = torch.tensor(
        [
            [True, True, True, True, True, False, False, False],
            [True, True, True, False, False, False, False, False],
        ]
    )
    out_full = encoder(
        peak_mz,
        peak_intensity,
        valid_mask=valid_mask,
        visible_mask=valid_mask,
    )
    out_packed = encoder(
        peak_mz,
        peak_intensity,
        valid_mask=valid_mask,
        visible_mask=valid_mask,
        pack_n=8,
        prefix_pack=True,
    )

    torch.testing.assert_close(
        out_packed.masked_select(valid_mask.unsqueeze(-1).expand_as(out_packed)),
        out_full.masked_select(valid_mask.unsqueeze(-1).expand_as(out_full)),
        rtol=1e-6,
        atol=1e-6,
    )
    _assert_zero_at_mask(out_packed, ~valid_mask)


@torch.no_grad()
def test_forward_augmented_sparse_pack_matches_full_model_loss():
    torch.manual_seed(2)
    model = PeakSetSIGReg(
        model_dim=32,
        encoder_num_layers=2,
        encoder_num_heads=4,
        feature_mlp_hidden_dim=16,
        masked_token_loss_weight=1.0,
        representation_regularizer="none",
        jepa_num_target_blocks=2,
        num_peaks=8,
        jepa_context_fraction=0.5,
        jepa_target_fraction=0.25,
    ).eval()
    batch = {
        "peak_mz": torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0, 0.0],
                [0.7, 0.8, 0.9, 1.0, 1.1, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        "peak_intensity": torch.tensor(
            [
                [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.0, 0.0],
                [0.9, 0.8, 0.7, 0.6, 0.5, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        "peak_valid_mask": torch.tensor(
            [
                [True, True, True, True, True, True, False, False],
                [True, True, True, True, True, False, False, False],
            ]
        ),
        "context_mask": torch.tensor(
            [
                [False, True, True, False, False, False, False, False],
                [True, False, True, False, False, False, False, False],
            ]
        ),
        "target_masks": torch.tensor(
            [
                [
                    [False, False, False, True, False, False, False, False],
                    [False, False, False, False, True, False, False, False],
                ],
                [
                    [False, False, False, True, False, False, False, False],
                    [False, False, False, False, True, False, False, False],
                ],
            ]
        ),
    }

    packed_metrics = model.forward_augmented(batch)
    model._context_pack_n = 0
    model._predictor_pack_n = 0
    model._full_pack_n = 0
    full_metrics = model.forward_augmented(batch)

    torch.testing.assert_close(
        packed_metrics["local_global_loss"],
        full_metrics["local_global_loss"],
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        packed_metrics["loss"],
        full_metrics["loss"],
        rtol=1e-6,
        atol=1e-6,
    )


@torch.no_grad()
def test_encoder_sparse_pack_absolute_positions_use_original_positions():
    visible_mask = torch.tensor(
        [
            [True, True, False, False, True, False, False, False],
            [True, False, True, True, False, False, False, False],
        ]
    )
    base_tokens = torch.tensor([0.2, 0.4, 0.6], dtype=torch.float32)
    peak_mz, peak_intensity = _place_visible_tokens(base_tokens, visible_mask)

    torch.manual_seed(7)
    encoder_without_pos = PeakSetEncoder(
        model_dim=32,
        num_layers=2,
        num_heads=4,
        num_peaks=8,
        feature_mlp_hidden_dim=16,
    ).eval()
    torch.manual_seed(7)
    encoder_with_pos = PeakSetEncoder(
        model_dim=32,
        num_layers=2,
        num_heads=4,
        num_peaks=8,
        feature_mlp_hidden_dim=16,
    ).eval()
    with torch.no_grad():
        encoder_without_pos.position_embedding.weight.zero_()

    out_without_pos = encoder_without_pos(
        peak_mz,
        peak_intensity,
        valid_mask=visible_mask,
        visible_mask=visible_mask,
        pack_n=3,
    )
    out_with_pos = encoder_with_pos(
        peak_mz,
        peak_intensity,
        valid_mask=visible_mask,
        visible_mask=visible_mask,
        pack_n=3,
    )

    packed_row0_without_pos = out_without_pos[0, visible_mask[0]]
    packed_row1_without_pos = out_without_pos[1, visible_mask[1]]
    packed_row0_with_pos = out_with_pos[0, visible_mask[0]]
    packed_row1_with_pos = out_with_pos[1, visible_mask[1]]

    torch.testing.assert_close(
        packed_row0_without_pos,
        packed_row1_without_pos,
        rtol=1e-6,
        atol=1e-6,
    )
    assert not torch.allclose(
        packed_row0_with_pos,
        packed_row1_with_pos,
        atol=1e-4,
        rtol=1e-4,
    )


@torch.no_grad()
def test_predictor_sparse_pack_absolute_positions_use_original_positions():
    visible_mask = torch.tensor(
        [
            [True, True, False, False, True, False],
            [True, False, True, True, False, False],
        ]
    )
    base = torch.linspace(0.1, 0.9, steps=3)
    x = torch.zeros(2, 6, 32)
    for row_idx in range(visible_mask.shape[0]):
        pos = visible_mask[row_idx].nonzero(as_tuple=False).squeeze(-1)
        x[row_idx, pos] = base.unsqueeze(-1) * torch.linspace(1.0, 2.0, steps=32)

    torch.manual_seed(11)
    model_without_pos = PeakSetSIGReg(
        model_dim=32,
        encoder_num_layers=2,
        encoder_num_heads=4,
        feature_mlp_hidden_dim=16,
        representation_regularizer="none",
        masked_latent_predictor_num_layers=2,
        num_peaks=6,
    ).eval()
    torch.manual_seed(11)
    model_with_pos = PeakSetSIGReg(
        model_dim=32,
        encoder_num_layers=2,
        encoder_num_heads=4,
        feature_mlp_hidden_dim=16,
        representation_regularizer="none",
        masked_latent_predictor_num_layers=2,
        num_peaks=6,
    ).eval()
    with torch.no_grad():
        model_without_pos.predictor_position_embedding.weight.zero_()

    out_without_pos = model_without_pos.predict_masked_latents(
        x,
        visible_mask,
        pack_n=3,
    )
    out_with_pos = model_with_pos.predict_masked_latents(x, visible_mask, pack_n=3)

    packed_row0_without_pos = out_without_pos[0, visible_mask[0]]
    packed_row1_without_pos = out_without_pos[1, visible_mask[1]]
    packed_row0_with_pos = out_with_pos[0, visible_mask[0]]
    packed_row1_with_pos = out_with_pos[1, visible_mask[1]]

    torch.testing.assert_close(
        packed_row0_without_pos,
        packed_row1_without_pos,
        rtol=1e-6,
        atol=1e-6,
    )
    assert not torch.allclose(
        packed_row0_with_pos,
        packed_row1_with_pos,
        atol=1e-4,
        rtol=1e-4,
    )
