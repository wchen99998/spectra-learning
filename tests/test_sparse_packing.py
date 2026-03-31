import torch

from models.model import PackContext, PeakSetEncoder, PeakSetSIGReg, pack_sequence, unpack_sequence


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


def _split_peak_and_cls(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return PeakSetEncoder.split_peak_and_cls(x)


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
    out_full, cls_full = _split_peak_and_cls(out_full)
    out_packed, cls_packed = _split_peak_and_cls(out_packed)

    visible = visible_mask & valid_mask
    torch.testing.assert_close(
        out_packed.masked_select(visible.unsqueeze(-1).expand_as(out_packed)),
        out_full.masked_select(visible.unsqueeze(-1).expand_as(out_full)),
        rtol=1e-6,
        atol=1e-6,
    )
    _assert_zero_at_mask(out_packed, ~visible)
    torch.testing.assert_close(cls_packed, cls_full, rtol=1e-6, atol=1e-6)


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
    out_full, cls_full = _split_peak_and_cls(out_full)
    out_packed, cls_packed = _split_peak_and_cls(out_packed)

    torch.testing.assert_close(
        out_packed.masked_select(valid_mask.unsqueeze(-1).expand_as(out_packed)),
        out_full.masked_select(valid_mask.unsqueeze(-1).expand_as(out_full)),
        rtol=1e-6,
        atol=1e-6,
    )
    _assert_zero_at_mask(out_packed, ~valid_mask)
    torch.testing.assert_close(cls_packed, cls_full, rtol=1e-6, atol=1e-6)


@torch.no_grad()
def test_encoder_sparse_pack_matches_full_with_register_tokens():
    torch.manual_seed(3)
    encoder = PeakSetEncoder(
        model_dim=32,
        num_layers=2,
        num_heads=4,
        num_peaks=8,
        num_register_tokens=2,
        feature_mlp_hidden_dim=16,
    ).eval()
    peak_mz = torch.rand(2, 8)
    peak_intensity = torch.rand(2, 8)
    valid_mask = torch.tensor(
        [
            [True, True, True, True, True, False, False, False],
            [True, True, True, False, True, False, False, False],
        ]
    )
    visible_mask = torch.tensor(
        [
            [False, True, True, True, False, False, False, False],
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
    out_full, cls_full = _split_peak_and_cls(out_full)
    out_packed, cls_packed = _split_peak_and_cls(out_packed)

    visible = visible_mask & valid_mask
    torch.testing.assert_close(
        out_packed.masked_select(visible.unsqueeze(-1).expand_as(out_packed)),
        out_full.masked_select(visible.unsqueeze(-1).expand_as(out_full)),
        rtol=1e-6,
        atol=1e-6,
    )
    _assert_zero_at_mask(out_packed, ~visible)
    torch.testing.assert_close(cls_packed, cls_full, rtol=1e-6, atol=1e-6)


@torch.no_grad()
def test_forward_augmented_sparse_pack_matches_full_model_loss():
    torch.manual_seed(2)
    model = PeakSetSIGReg(
        model_dim=32,
        encoder_num_layers=2,
        encoder_num_heads=4,
        feature_mlp_hidden_dim=16,
        masked_token_loss_weight=1.0,
        jepa_num_target_blocks=2,
        jepa_target_layers=[1, 2],
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
    out_without_pos, _ = _split_peak_and_cls(out_without_pos)
    out_with_pos, _ = _split_peak_and_cls(out_with_pos)

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
        masked_latent_predictor_num_layers=2,
        num_peaks=6,
    ).eval()
    torch.manual_seed(11)
    model_with_pos = PeakSetSIGReg(
        model_dim=32,
        encoder_num_layers=2,
        encoder_num_heads=4,
        feature_mlp_hidden_dim=16,
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


# ---------------------------------------------------------------------------
# Pack/unpack utility round-trip tests
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_pack_unpack_round_trip_index_based():
    """pack_sequence -> unpack_sequence reproduces visible tokens at correct positions."""
    x = torch.randn(2, 8, 16)
    mask = torch.tensor([
        [True, False, True, False, True, False, False, False],
        [False, True, False, True, False, True, False, False],
    ])
    ctx, packed_x, packed_mask = pack_sequence(x, mask, pack_n=4, prefix_pack=False)
    restored = unpack_sequence(ctx, packed_x, packed_mask)

    for row in range(2):
        vis = mask[row]
        torch.testing.assert_close(restored[row, vis], x[row, vis])
        _assert_zero_at_mask(restored[row : row + 1], ~vis.unsqueeze(0))


@torch.no_grad()
def test_pack_unpack_round_trip_prefix():
    """Prefix pack round-trip with contiguous valid prefix."""
    x = torch.randn(2, 8, 16)
    mask = torch.tensor([
        [True, True, True, True, True, False, False, False],
        [True, True, True, False, False, False, False, False],
    ])
    ctx, packed_x, packed_mask = pack_sequence(x, mask, pack_n=5, prefix_pack=True)
    restored = unpack_sequence(ctx, packed_x, packed_mask)

    for row in range(2):
        vis = mask[row]
        torch.testing.assert_close(restored[row, vis], x[row, vis])
        _assert_zero_at_mask(restored[row : row + 1], ~vis.unsqueeze(0))


@torch.no_grad()
def test_pack_unpack_with_trailing_special_tokens():
    """unpack_sequence correctly ignores trailing tokens appended after packing."""
    x = torch.randn(2, 6, 16)
    mask = torch.tensor([
        [True, False, True, True, False, False],
        [False, True, True, False, True, False],
    ])
    ctx, packed_x, packed_mask = pack_sequence(x, mask, pack_n=4, prefix_pack=False)
    # Simulate appending 2 special tokens (cls + register)
    special_x = torch.randn(2, 2, 16)
    special_mask = torch.ones(2, 2, dtype=torch.bool)
    extended_x = torch.cat([packed_x, special_x], dim=1)
    extended_mask = torch.cat([packed_mask, special_mask], dim=1)

    restored = unpack_sequence(ctx, extended_x, extended_mask)
    assert restored.shape == (2, 6, 16)
    for row in range(2):
        vis = mask[row]
        torch.testing.assert_close(restored[row, vis], x[row, vis])
        _assert_zero_at_mask(restored[row : row + 1], ~vis.unsqueeze(0))


# ---------------------------------------------------------------------------
# Encoder forward: non-contiguous context-like masks
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_encoder_pack_matches_non_contiguous_every_other_token():
    """Every-other-token visible mask (context-like) with 3 batch rows."""
    torch.manual_seed(50)
    encoder = PeakSetEncoder(
        model_dim=32, num_layers=2, num_heads=4,
        num_peaks=10, feature_mlp_hidden_dim=16,
    ).eval()
    peak_mz = torch.rand(3, 10)
    peak_intensity = torch.rand(3, 10)
    valid_mask = torch.tensor([
        [True] * 8 + [False] * 2,
        [True] * 6 + [False] * 4,
        [True] * 10,
    ])
    visible_mask = torch.tensor([
        [True, False, True, False, True, False, True, False, False, False],
        [True, False, True, False, True, False, False, False, False, False],
        [True, False, True, False, True, False, True, False, True, False],
    ])
    out_full = encoder(peak_mz, peak_intensity, valid_mask=valid_mask, visible_mask=visible_mask)
    out_packed = encoder(
        peak_mz, peak_intensity, valid_mask=valid_mask, visible_mask=visible_mask, pack_n=6,
    )
    out_full, cls_full = _split_peak_and_cls(out_full)
    out_packed, cls_packed = _split_peak_and_cls(out_packed)

    visible = visible_mask & valid_mask
    torch.testing.assert_close(
        out_packed.masked_select(visible.unsqueeze(-1).expand_as(out_packed)),
        out_full.masked_select(visible.unsqueeze(-1).expand_as(out_full)),
        rtol=1e-6, atol=1e-6,
    )
    _assert_zero_at_mask(out_packed, ~visible)
    torch.testing.assert_close(cls_packed, cls_full, rtol=1e-6, atol=1e-6)


@torch.no_grad()
def test_encoder_pack_matches_single_visible_token():
    """Edge case: only one visible token per row."""
    torch.manual_seed(51)
    encoder = PeakSetEncoder(
        model_dim=32, num_layers=2, num_heads=4,
        num_peaks=8, feature_mlp_hidden_dim=16,
    ).eval()
    peak_mz = torch.rand(2, 8)
    peak_intensity = torch.rand(2, 8)
    valid_mask = torch.tensor([
        [True, True, True, True, False, False, False, False],
        [True, True, True, False, False, False, False, False],
    ])
    visible_mask = torch.tensor([
        [False, False, True, False, False, False, False, False],
        [True, False, False, False, False, False, False, False],
    ])
    out_full = encoder(peak_mz, peak_intensity, valid_mask=valid_mask, visible_mask=visible_mask)
    out_packed = encoder(
        peak_mz, peak_intensity, valid_mask=valid_mask, visible_mask=visible_mask, pack_n=2,
    )
    out_full, cls_full = _split_peak_and_cls(out_full)
    out_packed, cls_packed = _split_peak_and_cls(out_packed)

    visible = visible_mask & valid_mask
    torch.testing.assert_close(
        out_packed.masked_select(visible.unsqueeze(-1).expand_as(out_packed)),
        out_full.masked_select(visible.unsqueeze(-1).expand_as(out_full)),
        rtol=1e-6, atol=1e-6,
    )
    _assert_zero_at_mask(out_packed, ~visible)
    torch.testing.assert_close(cls_packed, cls_full, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# forward_peak_block_outputs: pack vs non-pack at every block
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_forward_peak_block_outputs_pack_matches_non_contiguous():
    """All intermediate + final block outputs match at visible positions."""
    torch.manual_seed(52)
    encoder = PeakSetEncoder(
        model_dim=32, num_layers=3, num_heads=4,
        num_peaks=8, feature_mlp_hidden_dim=16,
    ).eval()
    peak_mz = torch.rand(2, 8)
    peak_intensity = torch.rand(2, 8)
    valid_mask = torch.tensor([
        [True, True, True, True, True, True, False, False],
        [True, True, True, True, False, False, False, False],
    ])
    visible_mask = torch.tensor([
        [True, False, True, False, True, False, False, False],
        [False, True, False, True, False, False, False, False],
    ])
    block_indices = [1, 2, 3]

    outputs_full = encoder.forward_peak_block_outputs(
        peak_mz, peak_intensity,
        valid_mask=valid_mask, visible_mask=visible_mask,
        block_indices=block_indices,
    )
    outputs_packed = encoder.forward_peak_block_outputs(
        peak_mz, peak_intensity,
        valid_mask=valid_mask, visible_mask=visible_mask,
        pack_n=4, block_indices=block_indices,
    )

    visible = visible_mask & valid_mask
    for i, (full_out, packed_out) in enumerate(zip(outputs_full, outputs_packed)):
        torch.testing.assert_close(
            packed_out.masked_select(visible.unsqueeze(-1).expand_as(packed_out)),
            full_out.masked_select(visible.unsqueeze(-1).expand_as(full_out)),
            rtol=1e-6, atol=1e-6,
            msg=f"Block {block_indices[i]} mismatch",
        )
        _assert_zero_at_mask(packed_out, ~visible)


@torch.no_grad()
def test_forward_peak_block_outputs_prefix_pack_matches():
    """forward_peak_block_outputs with prefix_pack=True."""
    torch.manual_seed(53)
    encoder = PeakSetEncoder(
        model_dim=32, num_layers=3, num_heads=4,
        num_peaks=8, feature_mlp_hidden_dim=16,
    ).eval()
    peak_mz = torch.rand(2, 8)
    peak_intensity = torch.rand(2, 8)
    valid_mask = torch.tensor([
        [True, True, True, True, True, False, False, False],
        [True, True, True, False, False, False, False, False],
    ])
    block_indices = [1, 3]

    outputs_full = encoder.forward_peak_block_outputs(
        peak_mz, peak_intensity,
        valid_mask=valid_mask, visible_mask=valid_mask,
        block_indices=block_indices,
    )
    outputs_packed = encoder.forward_peak_block_outputs(
        peak_mz, peak_intensity,
        valid_mask=valid_mask, visible_mask=valid_mask,
        pack_n=8, prefix_pack=True, block_indices=block_indices,
    )

    for i, (full_out, packed_out) in enumerate(zip(outputs_full, outputs_packed)):
        torch.testing.assert_close(
            packed_out.masked_select(valid_mask.unsqueeze(-1).expand_as(packed_out)),
            full_out.masked_select(valid_mask.unsqueeze(-1).expand_as(full_out)),
            rtol=1e-6, atol=1e-6,
            msg=f"Block {block_indices[i]} mismatch",
        )
        _assert_zero_at_mask(packed_out, ~valid_mask)


# ---------------------------------------------------------------------------
# predict_masked_latents: pack vs non-pack
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_predict_masked_latents_pack_matches_non_contiguous():
    """Pack vs non-pack for predict_masked_latents with non-contiguous visible mask."""
    torch.manual_seed(54)
    model = PeakSetSIGReg(
        model_dim=32, encoder_num_layers=2, encoder_num_heads=4,
        feature_mlp_hidden_dim=16, masked_latent_predictor_num_layers=2,
        num_peaks=8,
    ).eval()
    visible_mask = torch.tensor([
        [True, False, True, False, True, True, False, False],
        [False, True, False, True, False, False, True, False],
    ])
    x = torch.randn(2, 8, 32)

    out_no_pack = model.predict_masked_latents(x, visible_mask, pack_n=0)
    out_packed = model.predict_masked_latents(x, visible_mask, pack_n=5)

    torch.testing.assert_close(
        out_packed.masked_select(visible_mask.unsqueeze(-1).expand_as(out_packed)),
        out_no_pack.masked_select(visible_mask.unsqueeze(-1).expand_as(out_no_pack)),
        rtol=1e-6, atol=1e-6,
    )
    _assert_zero_at_mask(out_packed, ~visible_mask)


@torch.no_grad()
def test_predict_masked_latents_pack_matches_with_register_tokens():
    """Pack vs non-pack with predictor register tokens and non-contiguous mask."""
    torch.manual_seed(55)
    model = PeakSetSIGReg(
        model_dim=32, encoder_num_layers=2, encoder_num_heads=4,
        feature_mlp_hidden_dim=16, masked_latent_predictor_num_layers=2,
        num_peaks=8,
        predictor_num_register_tokens=2,
    ).eval()
    visible_mask = torch.tensor([
        [True, False, True, True, False, False, True, False],
        [False, True, True, False, True, False, False, True],
    ])
    x = torch.randn(2, 8, 32)

    out_no_pack = model.predict_masked_latents(x, visible_mask, pack_n=0)
    out_packed = model.predict_masked_latents(x, visible_mask, pack_n=5)

    torch.testing.assert_close(
        out_packed.masked_select(visible_mask.unsqueeze(-1).expand_as(out_packed)),
        out_no_pack.masked_select(visible_mask.unsqueeze(-1).expand_as(out_no_pack)),
        rtol=1e-6, atol=1e-6,
    )
    _assert_zero_at_mask(out_packed, ~visible_mask)


@torch.no_grad()
def test_predict_masked_latents_pack_matches_with_predictor_dim():
    """Pack vs non-pack when predictor_dim != model_dim (proj is Linear, not Identity)."""
    torch.manual_seed(56)
    model = PeakSetSIGReg(
        model_dim=32, encoder_num_layers=2, encoder_num_heads=4,
        feature_mlp_hidden_dim=16, masked_latent_predictor_num_layers=2,
        masked_latent_predictor_num_heads=4,
        num_peaks=8,
        predictor_dim=16,
    ).eval()
    visible_mask = torch.tensor([
        [True, False, True, False, False, True, True, False],
        [False, True, False, True, True, False, False, True],
    ])
    x = torch.randn(2, 8, 32)

    out_no_pack = model.predict_masked_latents(x, visible_mask, pack_n=0)
    out_packed = model.predict_masked_latents(x, visible_mask, pack_n=5)

    torch.testing.assert_close(
        out_packed.masked_select(visible_mask.unsqueeze(-1).expand_as(out_packed)),
        out_no_pack.masked_select(visible_mask.unsqueeze(-1).expand_as(out_no_pack)),
        rtol=1e-6, atol=1e-6,
    )
    _assert_zero_at_mask(out_packed, ~visible_mask)


# ---------------------------------------------------------------------------
# Full forward_augmented: non-contiguous context + targets
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_forward_augmented_non_contiguous_context_and_targets():
    """forward_augmented with sparse non-contiguous context and target masks."""
    torch.manual_seed(57)
    model = PeakSetSIGReg(
        model_dim=32, encoder_num_layers=2, encoder_num_heads=4,
        feature_mlp_hidden_dim=16, masked_token_loss_weight=1.0,
        jepa_num_target_blocks=2, jepa_target_layers=[1, 2],
        num_peaks=10, jepa_context_fraction=0.5, jepa_target_fraction=0.3,
    ).eval()
    batch = {
        "peak_mz": torch.rand(2, 10),
        "peak_intensity": torch.rand(2, 10),
        "peak_valid_mask": torch.tensor([
            [True] * 7 + [False] * 3,
            [True] * 5 + [False] * 5,
        ]),
        "context_mask": torch.tensor([
            [True, False, True, False, True, False, False, False, False, False],
            [False, True, False, True, False, False, False, False, False, False],
        ]),
        "target_masks": torch.tensor([
            [
                [False, True, False, False, False, True, False, False, False, False],
                [False, False, False, True, False, False, True, False, False, False],
            ],
            [
                [True, False, False, False, True, False, False, False, False, False],
                [False, False, True, False, False, False, False, False, False, False],
            ],
        ]),
    }
    packed_metrics = model.forward_augmented(batch)

    model._context_pack_n = 0
    model._predictor_pack_n = 0
    model._full_pack_n = 0
    full_metrics = model.forward_augmented(batch)

    torch.testing.assert_close(
        packed_metrics["local_global_loss"],
        full_metrics["local_global_loss"],
        rtol=1e-6, atol=1e-6,
    )
    torch.testing.assert_close(
        packed_metrics["loss"],
        full_metrics["loss"],
        rtol=1e-6, atol=1e-6,
    )
