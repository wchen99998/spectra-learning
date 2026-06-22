from unittest import mock

import torch

from spectra_learning.models.encoder import PeakSetEncoder
from spectra_learning.models.model import PeakSetJEPA
from spectra_learning.models.pairmixer import (
    AttentionPairBias,
    MediatedTriangleMultiplicativeUpdate,
    PairMixerBlock,
    TriangleAttention,
    TriangleMultiplicativeUpdate,
)
from spectra_learning.models.peak_features import PeakFeatureEmbedder
from spectra_learning.models.settings import PeakSetJEPASettings


def _build_model(
    *,
    num_target_blocks: int = 2,
    predictor_layers: int = 2,
    encoder_use_position_embedding: bool = True,
    encoder_apply_final_norm: bool = True,
    predictor_apply_final_norm: bool = True,
    jepa_target_normalization: str = "none",
    jepa_mae_loss_weight: float = 0.0,
    predictor_dim: int | None = None,
    masked_token_input_mode: str = "latent_token",
    masked_mz_sentinel: float = -1.0,
) -> PeakSetJEPA:
    torch.manual_seed(0)
    model = PeakSetJEPA(
        model_dim=32,
        encoder_num_layers=2,
        encoder_num_heads=4,
        num_peaks=6,
        feature_mlp_hidden_dim=32,
        encoder_use_position_embedding=encoder_use_position_embedding,
        encoder_apply_final_norm=encoder_apply_final_norm,
        predictor_apply_final_norm=predictor_apply_final_norm,
        predictor_dim=predictor_dim,
        jepa_num_target_blocks=num_target_blocks,
        masked_token_loss_weight=1.0,
        masked_latent_predictor_num_layers=predictor_layers,
        jepa_target_normalization=jepa_target_normalization,
        jepa_mae_loss_weight=jepa_mae_loss_weight,
        masked_token_input_mode=masked_token_input_mode,
        masked_mz_sentinel=masked_mz_sentinel,
    )
    model.eval()
    return model


def _build_encoder_embedder() -> PeakFeatureEmbedder:
    return PeakFeatureEmbedder(model_dim=32, hidden_dim=32)


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


def _make_pair(model: PeakSetJEPA, single: torch.Tensor) -> torch.Tensor:
    return torch.randn(
        single.shape[0],
        single.shape[1],
        single.shape[1],
        model.predictor_pair_dim,
        device=single.device,
        dtype=single.dtype,
    )


def _expected_predictor_pair(
    model: PeakSetJEPA,
    context_pair: torch.Tensor,
    context_mask: torch.Tensor,
    target_masks: torch.Tensor,
) -> torch.Tensor:
    B, K, N = target_masks.shape
    context_mask_by_view = context_mask.unsqueeze(1)
    visible_mask = torch.cat(
        [
            context_mask_by_view | target_masks,
            torch.ones(B, K, 1, dtype=torch.bool, device=target_masks.device),
        ],
        dim=2,
    )
    expected_pair = context_pair.unsqueeze(1).expand(-1, K, -1, -1, -1)
    context_token_mask = torch.cat(
        [
            context_mask_by_view.expand(-1, K, -1),
            torch.ones(B, K, 1, dtype=torch.bool, device=target_masks.device),
        ],
        dim=2,
    )
    context_pair_mask = context_token_mask.unsqueeze(3) & context_token_mask.unsqueeze(2)
    expected_pair = expected_pair * context_pair_mask.unsqueeze(-1).to(
        dtype=expected_pair.dtype
    )
    target_token_mask = torch.cat(
        [
            target_masks,
            torch.zeros(B, K, 1, dtype=torch.bool, device=target_masks.device),
        ],
        dim=2,
    )
    target_pair_mask = target_token_mask.unsqueeze(3) | target_token_mask.unsqueeze(2)
    expected_pair = torch.where(
        target_pair_mask.unsqueeze(-1),
        model.pair_mask_token.view(1, 1, 1, 1, -1).to(context_pair),
        expected_pair,
    )
    visible_pair_mask = visible_mask.unsqueeze(3) & visible_mask.unsqueeze(2)
    expected_pair = expected_pair * visible_pair_mask.unsqueeze(-1).to(
        dtype=expected_pair.dtype
    )
    return expected_pair.reshape(B * K, N + 1, N + 1, -1)


@torch.no_grad()
def test_predictor_zero_layers_still_runs_projection_path():
    model = _build_model(predictor_layers=0, predictor_dim=24)
    predictor_input = torch.randn(2, 6, model.model_dim)
    predictor_pair = _make_pair(model, predictor_input)
    visible_mask = torch.ones(2, 6, dtype=torch.bool)
    out = model.predict_masked_latents(
        predictor_input,
        predictor_pair,
        visible_mask,
    )

    assert out.shape == (2, 6, model.predictor_dim)


@torch.no_grad()
def test_predictor_zero_layers_with_projection_supports_forward():
    model = _build_model(predictor_layers=0, predictor_dim=24)
    metrics = model.forward_augmented(_make_batch())

    assert torch.isfinite(metrics["loss"])


@torch.no_grad()
def test_predictor_absolute_positions_change_inputs():
    model = _build_model(predictor_layers=0, predictor_apply_final_norm=False)
    predictor_input = torch.randn(1, 6, model.model_dim)

    positioned = model._add_predictor_positions(predictor_input)
    expected_delta = model.predictor_position_embedding(
        torch.arange(predictor_input.shape[1])
    ).unsqueeze(0)
    torch.testing.assert_close(positioned - predictor_input, expected_delta)


@torch.no_grad()
def test_predictor_has_position_embedding():
    model = _build_model(predictor_layers=2)
    context_emb = torch.randn(1, 6, model.model_dim)
    context_pair = _make_pair(model, context_emb)
    context_mask = torch.ones(1, 6, dtype=torch.bool)

    out = model.predict_masked_latents(
        context_emb,
        context_pair,
        context_mask,
    )

    assert hasattr(model, "latent_mask_token")
    assert model.latent_mask_token.shape == (model.model_dim,)
    assert hasattr(model, "pair_mask_token")
    assert model.pair_mask_token.shape == (model.predictor_pair_dim,)
    assert not hasattr(model, "predictor_mask_token")
    assert not hasattr(model, "predictor_intensity_embed")
    assert hasattr(model, "predictor_position_embedding")
    assert hasattr(model, "predictor_pair_position_embedding")
    assert not hasattr(model, "predictor_slot_embedding")
    assert "latent_mask_token" in model.state_dict()
    assert "pair_mask_token" in model.state_dict()
    assert "predictor_position_embedding.weight" in model.state_dict()
    assert "predictor_pair_position_embedding.weight" in model.state_dict()
    assert out.shape == (1, 6, model.predictor_dim)


@torch.no_grad()
def test_encoder_position_embedding_toggle_matches_zeroed_table():
    torch.manual_seed(0)
    encoder_without_pos = PeakSetEncoder(
        model_dim=32,
        embedder=_build_encoder_embedder(),
        num_layers=2,
        num_heads=4,
        num_peaks=6,
        use_position_embedding=False,
    ).eval()
    torch.manual_seed(0)
    encoder_with_zeroed_pos = PeakSetEncoder(
        model_dim=32,
        embedder=_build_encoder_embedder(),
        num_layers=2,
        num_heads=4,
        num_peaks=6,
        use_position_embedding=True,
    ).eval()
    with torch.no_grad():
        encoder_with_zeroed_pos.position_embedding.weight.zero_()
    peak_mz = torch.rand(2, 6)
    peak_intensity = torch.rand(2, 6)
    valid_mask = torch.ones(2, 6, dtype=torch.bool)

    out_without_pos = encoder_without_pos(
        peak_mz,
        peak_intensity,
        valid_mask=valid_mask,
        visible_mask=valid_mask,
    )
    out_with_zeroed_pos = encoder_with_zeroed_pos(
        peak_mz,
        peak_intensity,
        valid_mask=valid_mask,
        visible_mask=valid_mask,
    )

    torch.testing.assert_close(out_without_pos, out_with_zeroed_pos)


@torch.no_grad()
def test_encoder_final_norm_toggle_changes_output():
    torch.manual_seed(0)
    encoder_with_final_norm = PeakSetEncoder(
        model_dim=32,
        embedder=_build_encoder_embedder(),
        num_layers=2,
        num_heads=4,
        num_peaks=6,
        apply_final_norm=True,
    ).eval()
    torch.manual_seed(0)
    encoder_without_final_norm = PeakSetEncoder(
        model_dim=32,
        embedder=_build_encoder_embedder(),
        num_layers=2,
        num_heads=4,
        num_peaks=6,
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
        predictor_apply_final_norm=True,
    )
    model_without_final_norm = _build_model(
        predictor_layers=2,
        predictor_apply_final_norm=False,
    )
    predictor_input = torch.randn(1, 6, model_with_final_norm.model_dim)
    predictor_pair = _make_pair(model_with_final_norm, predictor_input)
    visible_mask = torch.ones(1, 6, dtype=torch.bool)
    out_1 = model_with_final_norm.predict_masked_latents(
        predictor_input,
        predictor_pair,
        visible_mask,
    )
    out_2 = model_without_final_norm.predict_masked_latents(
        predictor_input,
        predictor_pair,
        visible_mask,
    )

    diff = (out_1 - out_2).abs().mean()
    assert float(diff) > 1e-3


def test_encoder_and_predictor_final_norms_are_non_affine():
    model = _build_model()

    assert list(model.encoder.final_norm.parameters()) == []
    assert list(model.predictor_final_norm.parameters()) == []
    encoder_block = model.encoder.blocks[0]
    predictor_block = model.masked_latent_predictor[0]
    assert isinstance(encoder_block, PairMixerBlock)
    assert isinstance(predictor_block, PairMixerBlock)
    assert list(encoder_block.single_attention.single_norm.parameters())
    assert list(predictor_block.single_attention.single_norm.parameters())

    encoder = PeakSetEncoder(
        model_dim=32,
        embedder=_build_encoder_embedder(),
        num_layers=2,
        num_heads=4,
        num_peaks=6,
        apply_final_norm=True,
    )
    assert list(encoder.final_norm.parameters()) == []


def test_backbone_uses_pairmixer_with_pair_bias_attention_by_construction():
    model = _build_model()
    block = model.encoder.blocks[0]

    assert isinstance(block, PairMixerBlock)
    assert not hasattr(block, "refresh_pair")
    assert not hasattr(block, "tri_att_start")
    assert not hasattr(block, "tri_att_end")
    assert isinstance(block.single_attention, AttentionPairBias)
    assert not hasattr(block, "single_attention_norm")


@torch.no_grad()
def test_pairmixer_uses_pair_bias_attention_by_construction():
    model = PeakSetJEPA(
        model_dim=32,
        encoder_num_layers=2,
        encoder_num_heads=4,
        num_peaks=6,
        feature_mlp_hidden_dim=32,
    )
    block = model.encoder.blocks[0]
    predictor_block = model.masked_latent_predictor[0]
    batch = _make_batch()

    encoded = model.encoder(
        batch["peak_mz"],
        batch["peak_intensity"],
        valid_mask=batch["peak_valid_mask"],
        visible_mask=batch["peak_valid_mask"],
    )

    assert isinstance(block, PairMixerBlock)
    assert isinstance(block.single_attention, AttentionPairBias)
    assert not hasattr(block, "single_attention_norm")
    assert isinstance(predictor_block, PairMixerBlock)
    assert isinstance(predictor_block.single_attention, AttentionPairBias)
    assert not hasattr(predictor_block, "single_attention_norm")
    assert encoded.shape == (2, 7, 32)


def test_bi_dense_block_type_is_configurable():
    settings = PeakSetJEPASettings.from_config(
        {"pairmixer_block_type": "bi-dense"}
    )

    assert settings.pairmixer_block_type == "bi-dense"


@torch.no_grad()
def test_bi_dense_pairmixer_adds_gated_single_to_pair_update():
    model = PeakSetJEPA(
        model_dim=32,
        encoder_num_layers=2,
        encoder_num_heads=4,
        num_peaks=6,
        feature_mlp_hidden_dim=32,
        pairmixer_block_type="bi-dense",
    )
    block = model.encoder.blocks[0]
    predictor_block = model.masked_latent_predictor[0]
    batch = _make_batch()

    encoded, pair = model.encoder.forward_with_pair(
        batch["peak_mz"],
        batch["peak_intensity"],
        valid_mask=batch["peak_valid_mask"],
        visible_mask=batch["peak_valid_mask"],
    )

    assert isinstance(block, PairMixerBlock)
    assert isinstance(block.single_attention, AttentionPairBias)
    assert hasattr(block, "single_to_pair_update")
    assert isinstance(predictor_block, PairMixerBlock)
    assert hasattr(predictor_block, "single_to_pair_update")
    assert encoded.shape == (2, 7, 32)
    assert pair.shape == (2, 7, 7, 32)


@torch.no_grad()
def test_triangle_mediator_pairmixer_uses_dense_pair_state():
    model = PeakSetJEPA(
        model_dim=32,
        encoder_num_layers=2,
        encoder_num_heads=4,
        num_peaks=6,
        feature_mlp_hidden_dim=32,
        pairmixer_block_type="triangle_mediator",
        pairmixer_triangle_mediator_num_mediators=3,
    )
    block = model.encoder.blocks[0]
    predictor_block = model.masked_latent_predictor[0]
    batch = _make_batch()

    encoded, pair = model.encoder.forward_with_pair(
        batch["peak_mz"],
        batch["peak_intensity"],
        valid_mask=batch["peak_valid_mask"],
        visible_mask=batch["peak_valid_mask"],
    )

    assert isinstance(block, PairMixerBlock)
    assert block.use_triangle_mediator
    assert block.triangle_mediator_assignment.num_mediators == 3
    assert isinstance(predictor_block, PairMixerBlock)
    assert predictor_block.use_triangle_mediator
    assert isinstance(pair, torch.Tensor)
    assert encoded.shape == (2, 7, 32)
    assert pair.shape == (2, 7, 7, 32)


@torch.no_grad()
def test_mediated_triangle_matches_dense_with_identity_mediator():
    torch.manual_seed(17)
    x = torch.randn(2, 4, 4, 6)
    token_mask = torch.ones(2, 4, dtype=torch.bool)
    pair_mask = token_mask.unsqueeze(2) & token_mask.unsqueeze(1)
    mediator_assignment = torch.eye(4).expand(2, -1, -1)

    for direction in ("outgoing", "incoming"):
        dense = TriangleMultiplicativeUpdate(
            6,
            direction=direction,
            norm_eps=1e-5,
        )
        mediated = MediatedTriangleMultiplicativeUpdate(
            6,
            direction=direction,
            norm_eps=1e-5,
            mediator_eps=0.0,
        )
        mediated.load_state_dict(dense.state_dict())

        expected = dense(x, pair_mask)
        actual = mediated(x, token_mask, pair_mask, mediator_assignment)

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@torch.no_grad()
def test_triangle_attention_is_retained_for_pair_features():
    pair = torch.randn(2, 5, 5, 16)
    peak_mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, False],
        ]
    )
    pair_mask = peak_mask.unsqueeze(2) & peak_mask.unsqueeze(1)

    start = TriangleAttention(
        16,
        num_heads=4,
        ending=False,
        norm_eps=1e-5,
    )
    end = TriangleAttention(
        16,
        num_heads=4,
        ending=True,
        norm_eps=1e-5,
    )

    start_out = start(pair, peak_mask, pair_mask)
    end_out = end(pair, peak_mask, pair_mask)

    assert start_out.shape == pair.shape
    assert end_out.shape == pair.shape
    torch.testing.assert_close(
        start_out[~pair_mask],
        torch.zeros_like(start_out[~pair_mask]),
    )
    torch.testing.assert_close(
        end_out[~pair_mask],
        torch.zeros_like(end_out[~pair_mask]),
    )

def test_predictor_uses_pairmixer_with_pair_bias_attention():
    model = PeakSetJEPA(
        model_dim=32,
        encoder_num_layers=2,
        encoder_num_heads=4,
        num_peaks=6,
        feature_mlp_hidden_dim=32,
        jepa_num_target_blocks=2,
        masked_token_loss_weight=1.0,
        masked_latent_predictor_num_layers=3,
    )

    for block in model.masked_latent_predictor:
        assert isinstance(block, PairMixerBlock)
        assert isinstance(block.single_attention, AttentionPairBias)
        assert not hasattr(block, "refresh_pair")
        assert not hasattr(block, "tri_att_start")
        assert not hasattr(block, "tri_att_end")


def test_masked_latent_predictor_uses_pairmixer_blocks():
    model = _build_model(predictor_layers=2)
    block = model.masked_latent_predictor[0]

    assert isinstance(block, PairMixerBlock)
    assert hasattr(block, "single_attention")
    assert hasattr(block, "tri_mul_out")
    assert not hasattr(block, "tri_att_start")
    assert not hasattr(block, "tri_att_end")


@torch.no_grad()
def test_predictor_output_is_independent_from_encoder_positions():
    torch.manual_seed(0)
    model_without_encoder_pos = _build_model(predictor_layers=2)
    torch.manual_seed(0)
    model_with_encoder_pos = _build_model(predictor_layers=2)
    with torch.no_grad():
        model_without_encoder_pos.encoder.position_embedding.weight.zero_()
    predictor_input = torch.randn(1, 6, model_without_encoder_pos.model_dim)
    predictor_pair = _make_pair(model_without_encoder_pos, predictor_input)
    visible_mask = torch.ones(1, 6, dtype=torch.bool)
    out_1 = model_without_encoder_pos.predict_masked_latents(
        predictor_input,
        predictor_pair,
        visible_mask,
    )
    out_2 = model_with_encoder_pos.predict_masked_latents(
        predictor_input,
        predictor_pair,
        visible_mask,
    )

    assert torch.allclose(out_1, out_2)


@torch.no_grad()
def test_forward_augmented_reports_loss_metrics():
    model = _build_model()
    metrics = model.forward_augmented(_make_batch())

    assert "masked_prediction_loss" in metrics
    assert "context_fraction" in metrics
    assert "target_fraction" in metrics
    assert "target_fraction_per_view" in metrics
    assert "target_union_fraction" in metrics
    assert "target_entry_fraction" in metrics
    assert "target_overlap_entries" in metrics
    assert torch.isfinite(metrics["masked_prediction_loss"])
    assert float(metrics["target_fraction"]) > 0.0
    assert torch.allclose(metrics["target_fraction"], metrics["target_fraction_per_view"])
    assert float(metrics["target_entry_fraction"]) >= float(metrics["target_fraction"])


@torch.no_grad()
def test_mz_sentinel_mode_masks_target_mz_for_context_encoder():
    model = _build_model(
        masked_token_input_mode="mz_sentinel",
        masked_mz_sentinel=-0.5,
    )
    batch = _make_batch()

    context_mz, context_intensity, context_visible_mask = model._context_encoder_inputs(
        batch["peak_mz"],
        batch["peak_intensity"],
        batch["context_mask"],
        batch["target_masks"],
    )

    target_union = batch["target_masks"].any(dim=1)
    torch.testing.assert_close(
        context_mz[target_union],
        torch.full_like(context_mz[target_union], -0.5),
    )
    torch.testing.assert_close(
        context_mz[~target_union],
        batch["peak_mz"][~target_union],
    )
    torch.testing.assert_close(context_intensity, batch["peak_intensity"])
    assert torch.equal(context_visible_mask, batch["context_mask"] | target_union)


@torch.no_grad()
def test_mz_sentinel_predictor_receives_context_and_target_memory_mask():
    model = _build_model(masked_token_input_mode="mz_sentinel")
    batch = _make_batch()
    context_emb = torch.randn(
        batch["peak_mz"].shape[0],
        batch["peak_mz"].shape[1] + 1,
        model.model_dim,
    )
    context_pair = _make_pair(model, context_emb)
    captured: dict[str, torch.Tensor] = {}

    def fake_predict_masked_target_features_with_pair(
        x: torch.Tensor,
        pair: torch.Tensor,
        visible_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        captured["x"] = x.detach().clone()
        captured["pair"] = pair.detach().clone()
        captured["visible_mask"] = visible_mask.detach().clone()
        return x.new_zeros(x.shape[0], x.shape[1], model.jepa_target_dim), pair

    with mock.patch.object(
        model,
        "predict_masked_target_features_with_pair",
        side_effect=fake_predict_masked_target_features_with_pair,
    ):
        model._predict_augmented_targets(
            context_emb,
            context_pair,
            batch["context_mask"],
            batch["target_masks"],
        )

    B, K, N = batch["target_masks"].shape
    expected_context_emb = context_emb[:, :N].unsqueeze(1).expand(-1, K, -1, -1)
    expected_input = expected_context_emb * batch["context_mask"].unsqueeze(1).unsqueeze(
        -1
    )
    expected_input = torch.where(
        batch["target_masks"].unsqueeze(-1),
        model.latent_mask_token.view(1, 1, 1, -1).to(context_emb),
        expected_input,
    )
    expected_input = torch.cat(
        [
            expected_input,
            context_emb[:, N : N + 1].unsqueeze(1).expand(-1, K, -1, -1),
        ],
        dim=2,
    ).reshape(B * K, N + 1, -1)
    expected_visible_mask = torch.cat(
        [
            batch["context_mask"].unsqueeze(1) | batch["target_masks"],
            torch.ones(B, K, 1, dtype=torch.bool),
        ],
        dim=2,
    ).reshape(B * K, N + 1)
    expected_pair = _expected_predictor_pair(
        model,
        context_pair,
        batch["context_mask"],
        batch["target_masks"],
    )
    assert captured["x"].shape == (B * K, N + 1, model.model_dim)
    assert captured["pair"].shape == (B * K, N + 1, N + 1, model.predictor_pair_dim)
    torch.testing.assert_close(captured["x"], expected_input)
    torch.testing.assert_close(captured["pair"], expected_pair)
    assert torch.equal(captured["visible_mask"], expected_visible_mask)


@torch.no_grad()
def test_latent_token_predictor_receives_per_view_target_masks():
    model = _build_model(masked_token_input_mode="latent_token")
    batch = _make_batch()
    context_emb = torch.randn(
        batch["peak_mz"].shape[0],
        batch["peak_mz"].shape[1] + 1,
        model.model_dim,
    )
    context_pair = _make_pair(model, context_emb)
    captured: dict[str, torch.Tensor] = {}

    def fake_predict_masked_target_features_with_pair(
        x: torch.Tensor,
        pair: torch.Tensor,
        visible_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        captured["x"] = x.detach().clone()
        captured["pair"] = pair.detach().clone()
        captured["visible_mask"] = visible_mask.detach().clone()
        return x.new_zeros(x.shape[0], x.shape[1], model.jepa_target_dim), pair

    with mock.patch.object(
        model,
        "predict_masked_target_features_with_pair",
        side_effect=fake_predict_masked_target_features_with_pair,
    ):
        model._predict_augmented_targets(
            context_emb,
            context_pair,
            batch["context_mask"],
            batch["target_masks"],
        )

    B, K, N = batch["target_masks"].shape
    expected_context_emb = context_emb[:, :N].unsqueeze(1).expand(-1, K, -1, -1)
    expected_input = expected_context_emb * batch["context_mask"].unsqueeze(1).unsqueeze(
        -1
    )
    expected_input = torch.where(
        batch["target_masks"].unsqueeze(-1),
        model.latent_mask_token.view(1, 1, 1, -1).to(context_emb),
        expected_input,
    )
    expected_input = torch.cat(
        [
            expected_input,
            context_emb[:, N : N + 1].unsqueeze(1).expand(-1, K, -1, -1),
        ],
        dim=2,
    ).reshape(B * K, N + 1, -1)
    expected_visible_mask = torch.cat(
        [
            batch["context_mask"].unsqueeze(1) | batch["target_masks"],
            torch.ones(B, K, 1, dtype=torch.bool),
        ],
        dim=2,
    ).reshape(B * K, N + 1)
    expected_pair = _expected_predictor_pair(
        model,
        context_pair,
        batch["context_mask"],
        batch["target_masks"],
    )
    assert captured["x"].shape == (B * K, N + 1, model.model_dim)
    assert captured["pair"].shape == (B * K, N + 1, N + 1, model.predictor_pair_dim)
    torch.testing.assert_close(captured["x"], expected_input)
    torch.testing.assert_close(captured["pair"], expected_pair)
    assert torch.equal(captured["visible_mask"], expected_visible_mask)


@torch.no_grad()
def test_mz_sentinel_mode_value_objective_predicts_mz_only():
    model = _build_model(
        masked_token_input_mode="mz_sentinel",
        jepa_mae_loss_weight=1.0,
    )
    batch = _make_batch()
    predicted_latents = torch.randn(
        batch["target_masks"].shape[0],
        batch["target_masks"].shape[1],
        batch["target_masks"].shape[2],
        model.target_projector_dim,
    )

    (
        value_loss,
        mz_loss,
        intensity_loss,
        _mz_accuracy,
        intensity_accuracy,
    ) = model._jepa_mae_value_prediction_loss(
        predicted_latents,
        batch["peak_mz"],
        batch["peak_intensity"],
        batch["target_masks"],
    )

    torch.testing.assert_close(value_loss, mz_loss)
    torch.testing.assert_close(intensity_loss, torch.zeros_like(intensity_loss))
    torch.testing.assert_close(intensity_accuracy, torch.zeros_like(intensity_accuracy))


@torch.no_grad()
def test_teacher_and_predictor_outputs_use_final_encoder_dim():
    model = _build_model(num_target_blocks=2)
    batch = _make_batch()
    peak_mz = batch["peak_mz"]
    peak_intensity = batch["peak_intensity"]
    peak_valid_mask = batch["peak_valid_mask"]
    context_mask = batch["context_mask"] & peak_valid_mask
    target_masks = batch["target_masks"] & peak_valid_mask.unsqueeze(1)
    B, K, N = target_masks.shape

    teacher_target_features = model._compute_jepa_teacher_target_features(
        peak_mz,
        peak_intensity,
        peak_valid_mask,
    )
    assert teacher_target_features.shape == (B, N, model.model_dim)
    teacher_targets = model._compute_jepa_teacher_targets(
        peak_mz,
        peak_intensity,
        peak_valid_mask,
    )
    assert teacher_targets.shape == (B, N, model.target_projector_dim)

    context_encoded, context_pair = model.encoder.forward_with_pair(
        peak_mz,
        peak_intensity,
        valid_mask=peak_valid_mask,
        visible_mask=context_mask,
    )
    context_emb = context_encoded
    target_union = target_masks.any(dim=1)
    predictor_input = context_emb[:, :N] * context_mask.unsqueeze(-1)
    predictor_input = torch.where(
        target_union.unsqueeze(-1),
        model.latent_mask_token.view(1, 1, -1).to(context_emb),
        predictor_input,
    )
    predictor_input = torch.cat([predictor_input, context_emb[:, N : N + 1]], dim=1)
    predictor_visible_mask = torch.cat(
        [
            context_mask | target_union,
            torch.ones(B, 1, dtype=torch.bool),
        ],
        dim=1,
    )
    predictor_output_features = model.predict_masked_target_features(
        predictor_input,
        context_pair,
        predictor_visible_mask,
    )
    assert predictor_output_features.shape == (B, N + 1, model.model_dim)
    predictor_output = model.predict_masked_targets(
        predictor_input,
        context_pair,
        predictor_visible_mask,
    )
    assert predictor_output.shape == (B, N + 1, model.target_projector_dim)
    predictor_output_features_by_view, predictor_output_by_view = (
        model._predict_augmented_targets(
            context_emb,
            context_pair,
            context_mask,
            target_masks,
        )
    )
    assert predictor_output_features_by_view.shape == (
        B,
        K,
        N,
        model.model_dim,
    )
    assert predictor_output_by_view.shape == (B, K, N, model.target_projector_dim)

    metrics = model.forward_augmented(batch)
    assert torch.isfinite(metrics["loss"])


@torch.no_grad()
def test_masked_prediction_loss_uses_target_tokens_only():
    model = _build_model(num_target_blocks=2)
    batch = _make_batch()

    metrics = model.forward_augmented(batch)

    peak_mz = batch["peak_mz"]
    peak_intensity = batch["peak_intensity"]
    peak_valid_mask = batch["peak_valid_mask"]
    context_mask = batch["context_mask"] & peak_valid_mask
    target_masks = batch["target_masks"] & peak_valid_mask.unsqueeze(1)

    context_encoded, context_pair = model.encoder.forward_with_pair(
        peak_mz,
        peak_intensity,
        valid_mask=peak_valid_mask,
        visible_mask=context_mask,
    )
    context_emb = context_encoded
    B, K, N = target_masks.shape
    teacher_target = model._compute_jepa_teacher_targets(
        peak_mz,
        peak_intensity,
        peak_valid_mask,
    )

    _, predictor_output = model._predict_augmented_targets(
        context_emb,
        context_pair,
        context_mask,
        target_masks,
    )
    masked_only_loss = (
        model._embedding_loss(predictor_output, teacher_target.unsqueeze(1))
        * target_masks.float()
    ).sum() / target_masks.float().sum().clamp_min(1.0)

    assert torch.allclose(metrics["masked_prediction_loss"], masked_only_loss)


@torch.no_grad()
def test_masked_prediction_loss_can_zscore_teacher_targets():
    model = _build_model(
        num_target_blocks=2,
        jepa_target_normalization="zscore",
    )
    batch = _make_batch()

    metrics = model.forward_augmented(batch)

    peak_mz = batch["peak_mz"]
    peak_intensity = batch["peak_intensity"]
    peak_valid_mask = batch["peak_valid_mask"]
    context_mask = batch["context_mask"] & peak_valid_mask
    target_masks = batch["target_masks"] & peak_valid_mask.unsqueeze(1)

    context_encoded, context_pair = model.encoder.forward_with_pair(
        peak_mz,
        peak_intensity,
        valid_mask=peak_valid_mask,
        visible_mask=context_mask,
    )
    context_emb = context_encoded
    B, K, N = target_masks.shape
    teacher_target = model._compute_jepa_teacher_targets(
        peak_mz,
        peak_intensity,
        peak_valid_mask,
    )

    _, predictor_output = model._predict_augmented_targets(
        context_emb,
        context_pair,
        context_mask,
        target_masks,
    )
    masked_only_loss = (
        model._embedding_loss(predictor_output, teacher_target.unsqueeze(1))
        * target_masks.float()
    ).sum() / target_masks.float().sum().clamp_min(1.0)

    assert torch.allclose(metrics["masked_prediction_loss"], masked_only_loss)


@torch.no_grad()
def test_single_layer_zscore_normalizes_final_target_slice():
    model = _build_model(
        predictor_layers=2,
        jepa_target_normalization="zscore",
    )
    x = torch.randn(2, 3, model.model_dim)

    normalized = model._apply_jepa_target_normalization(x)

    assert torch.allclose(
        normalized.mean(dim=-1),
        torch.zeros(2, 3),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        normalized.std(dim=-1, unbiased=False),
        torch.ones(2, 3),
        atol=1e-4,
        rtol=1e-4,
    )


@torch.no_grad()
def test_positions_outside_union_do_not_change_context_conditioning_with_fixed_teacher_targets():
    model = _build_model(num_target_blocks=2)
    batch_a = _make_batch()
    batch_b = {key: value.clone() for key, value in batch_a.items()}

    ignored = ~(batch_a["context_mask"] | batch_a["target_masks"].any(dim=1))
    batch_b["peak_intensity"] = batch_b["peak_intensity"].clone()
    batch_b["peak_intensity"][ignored] = batch_b["peak_intensity"][ignored] + 0.5
    batch_b["peak_mz"] = batch_b["peak_mz"].clone()
    batch_b["peak_mz"][ignored] = batch_b["peak_mz"][ignored] + 0.2

    teacher_targets = model._compute_jepa_teacher_targets(
        batch_a["peak_mz"],
        batch_a["peak_intensity"],
        batch_a["peak_valid_mask"],
    )

    def masked_prediction_loss(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        peak_mz = batch["peak_mz"]
        peak_intensity = batch["peak_intensity"]
        peak_valid_mask = batch["peak_valid_mask"]
        context_mask = batch["context_mask"] & peak_valid_mask
        target_masks = batch["target_masks"] & peak_valid_mask.unsqueeze(1)
        context_encoded, context_pair = model.encoder.forward_with_pair(
            peak_mz,
            peak_intensity,
            valid_mask=peak_valid_mask,
            visible_mask=context_mask,
        )
        context_emb = context_encoded
        _, predictor_output = model._predict_augmented_targets(
            context_emb,
            context_pair,
            context_mask,
            target_masks,
        )
        return (
            model._embedding_loss(predictor_output, teacher_targets.unsqueeze(1))
            * target_masks.float()
        ).sum() / target_masks.float().sum().clamp_min(1.0)

    loss_a = masked_prediction_loss(batch_a)
    loss_b = masked_prediction_loss(batch_b)

    assert torch.allclose(loss_a, loss_b, atol=1e-6, rtol=1e-6)
