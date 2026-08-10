from unittest import mock

import pytest
import torch

from spectra_learning.models.model import PeakSetJEPA
from spectra_learning.models.pairmixer import PairMixerBlock
from spectra_learning.models.transformer import CrossAttentionBlock


def _model(**overrides) -> PeakSetJEPA:
    torch.manual_seed(0)
    values = {
        "training_mode": "mae",
        "model_dim": 32,
        "predictor_dim": 32,
        "encoder_num_layers": 1,
        "encoder_num_heads": 4,
        "masked_latent_predictor_num_layers": 2,
        "masked_latent_predictor_num_heads": 4,
        "num_peaks": 6,
        "feature_mlp_hidden_dim": 32,
        "pairmixer_pair_dim": 16,
        "pairmixer_pair_feature_hidden_dim": 32,
        "pairmixer_fourier_num_freqs": 1,
        "predictor_target_max_tokens": 2,
        "distogram_loss_weight": 0.0,
        "mae_intensity_loss_weight": 0.0,
        "target_projector_dim": -1,
        "predictor_dropout": 0.0,
    }
    values.update(overrides)
    return PeakSetJEPA(**values).eval()


def _batch() -> dict[str, torch.Tensor]:
    return {
        "peak_mz": torch.tensor(
            [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]
        ),
        "peak_intensity": torch.ones(2, 6),
        "peak_valid_mask": torch.ones(2, 6, dtype=torch.bool),
        "context_mask": torch.tensor(
            [[True, True, False, False, False, False], [False, True, True, False, False, False]]
        ),
        "target_masks": torch.tensor(
            [
                [[False, False, True, True, False, False]],
                [[False, False, False, True, True, False]],
            ]
        ),
    }


def _predictor_inputs(model: PeakSetJEPA):
    query = model.latent_mask_token.view(1, 1, -1).expand(2, 3, -1)
    memory = torch.randn(2, 4, model.model_dim)
    query_positions = torch.tensor([[1, 3, 5], [0, 2, 4]])
    memory_positions = torch.tensor([[0, 2, 4, 6], [1, 3, 5, 6]])
    query_mask = torch.tensor([[True, True, False], [True, True, True]])
    memory_mask = torch.tensor([[True, True, True, True], [True, True, False, True]])
    return (
        query,
        memory,
        query_positions,
        memory_positions,
        query_mask,
        memory_mask,
    )


def test_predictor_is_cross_attention_only():
    model = _model()

    assert isinstance(model.encoder.blocks[0], PairMixerBlock)
    assert all(
        isinstance(block, CrossAttentionBlock)
        for block in model.masked_latent_predictor
    )
    assert not hasattr(model, "pair_mask_token")
    assert not hasattr(model, "predictor_position_embedding")
    assert not hasattr(model.masked_latent_predictor[0], "single_attention")


@torch.no_grad()
def test_rope_uses_original_target_positions():
    model = _model()
    inputs = list(_predictor_inputs(model))
    expected = model.predict_masked_latents(*inputs)
    inputs[2] = inputs[2].roll(1, dims=1)
    actual = model.predict_masked_latents(*inputs)

    assert not torch.allclose(actual[:, :2], expected[:, :2])


@torch.no_grad()
def test_memory_padding_is_masked():
    model = _model()
    inputs = list(_predictor_inputs(model))
    expected = model.predict_masked_latents(*inputs)
    inputs[1] = inputs[1].clone()
    inputs[1][1, 2] = 1_000_000
    actual = model.predict_masked_latents(*inputs)

    torch.testing.assert_close(actual, expected)


@torch.no_grad()
def test_query_padding_is_zero():
    model = _model()
    inputs = _predictor_inputs(model)
    output = model.predict_masked_latents(*inputs)

    torch.testing.assert_close(output[0, 2], torch.zeros_like(output[0, 2]))


@pytest.mark.parametrize(
    ("mode", "memory_tokens"),
    [("latent_token", 3), ("mz_sentinel", 5)],
)
@torch.no_grad()
def test_augmented_predictor_packs_memory_and_targets(mode, memory_tokens):
    model = _model(masked_token_input_mode=mode)
    batch = _batch()
    context_emb = torch.randn(2, 7, model.model_dim)
    captured = {}
    original = model.predict_masked_target_features

    def capture(query, memory, query_positions, memory_positions, query_mask, memory_mask):
        captured.update(
            query=query,
            memory=memory,
            query_positions=query_positions,
            memory_positions=memory_positions,
            query_mask=query_mask,
            memory_mask=memory_mask,
        )
        return original(
            query,
            memory,
            query_positions,
            memory_positions,
            query_mask,
            memory_mask,
        )

    with mock.patch.object(model, "predict_masked_target_features", side_effect=capture):
        model._predict_augmented_targets(
            context_emb,
            batch["context_mask"],
            batch["target_masks"],
        )

    assert captured["query"].shape == (2, 2, model.predictor_dim)
    assert captured["query_mask"].all()
    assert captured["memory_mask"].sum(dim=1).tolist() == [memory_tokens, memory_tokens]
    assert captured["query_positions"].tolist() == [[2, 3], [3, 4]]


@torch.no_grad()
def test_augmented_output_is_scattered_only_to_targets():
    model = _model()
    batch = _batch()
    context_emb = torch.randn(2, 7, model.model_dim)

    features, output = model._predict_augmented_targets(
        context_emb,
        batch["context_mask"],
        batch["target_masks"],
    )

    assert features.shape == (2, 1, 6, model.jepa_target_dim)
    assert output.shape == (2, 1, 6, model.target_projector_dim)
    torch.testing.assert_close(
        features[~batch["target_masks"]],
        torch.zeros_like(features[~batch["target_masks"]]),
    )


@torch.no_grad()
def test_mae_forward_reports_finite_loss():
    metrics = _model().forward_mae(_batch())

    assert torch.isfinite(metrics["loss"])
    assert metrics["target_fraction"] > 0


def test_pair_prediction_losses_are_rejected():
    with pytest.raises(ValueError, match="cross-attention predictor"):
        _model(distogram_loss_weight=0.1)
