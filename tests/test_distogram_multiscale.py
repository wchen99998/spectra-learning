"""The multi-scale distogram objective and the projected pair representation."""

import numpy as np
import pytest
from ml_collections import config_dict

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from flax import nnx  # noqa: E402

from spectra_learning.models.factory_jax import (  # noqa: E402
    build_model_from_config,
)
from spectra_learning.models.model_jax import (  # noqa: E402
    _coarse_marginal_ce_rows,
    _cross_entropy_from_logits,
    _grouped_marginal_ce_rows,
)


WIDTHS = (2, 4, 8, 16, 32, 64)


def _config(**overrides) -> config_dict.ConfigDict:
    cfg = config_dict.ConfigDict()
    cfg.device_backend = "jax"
    cfg.training_mode = "mae"
    cfg.model_dim = 32
    cfg.predictor_dim = 32
    cfg.encoder_num_layers = 2
    cfg.encoder_num_heads = 4
    cfg.attention_mlp_multiple = 2.0
    cfg.feature_mlp_hidden_dim = 16
    cfg.encoder_fourier_mlp_hidden_dim = 32
    cfg.num_peaks = 8
    cfg.masked_latent_predictor_num_layers = 1
    cfg.masked_latent_predictor_num_heads = 4
    cfg.jepa_mae_mz_bin_size = 2.5
    cfg.jepa_num_target_blocks = 1
    cfg.encoder_use_pair_path = False
    for key, value in overrides.items():
        cfg[key] = value
    return cfg


def _batch(seed: int = 0, batch: int = 3, peaks: int = 8) -> dict:
    rng = np.random.default_rng(seed)
    context = rng.random((batch, peaks)) < 0.6
    target = (~context) & (rng.random((batch, peaks)) < 0.9)
    target[:, -1] = ~context[:, -1]
    return {
        "peak_mz": jnp.asarray(
            np.sort(rng.random((batch, peaks)), axis=-1), jnp.float32
        ),
        "peak_intensity": jnp.asarray(rng.random((batch, peaks)), jnp.float32),
        "peak_valid_mask": jnp.ones((batch, peaks), bool),
        "context_mask": jnp.asarray(context),
        "target_masks": jnp.asarray(target[:, None, :]),
        "precursor_mz": jnp.asarray(rng.random(batch), jnp.float32),
    }


def test_width_one_marginal_is_the_exact_cross_entropy():
    """The exact term is width 1, so one log-softmax serves every scale."""
    rng = np.random.default_rng(3)
    logits = jnp.asarray(rng.normal(size=(2, 5, 7, 64)), jnp.float32)
    targets = jnp.asarray(rng.integers(0, 64, size=(2, 5, 7)), jnp.int32)

    rows = _coarse_marginal_ce_rows(logits, targets, (1,))

    np.testing.assert_allclose(
        np.asarray(rows[1]),
        np.asarray(_cross_entropy_from_logits(logits, targets)),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("num_bins", (2000, 64, 100, 63))
def test_grouped_marginal_matches_the_coarse_form(num_bins: int):
    """Scoring only the target's group must equal scoring every group.

    Widths 32 and 64 do not divide 2,000, so this also covers the -inf padding
    the coarse form adds and the index masking that replaces it.
    """
    rng = np.random.default_rng(7)
    logits = jnp.asarray(rng.normal(size=(2, 5, 7, num_bins)) * 3, jnp.float32)
    targets = jnp.asarray(rng.integers(0, num_bins, size=(2, 5, 7)), jnp.int32)
    widths = (1, *WIDTHS)

    coarse = _coarse_marginal_ce_rows(logits, targets, widths)
    grouped = _grouped_marginal_ce_rows(logits, targets, widths)

    for width in widths:
        np.testing.assert_allclose(
            np.asarray(grouped[width]),
            np.asarray(coarse[width]),
            rtol=1e-5,
            atol=1e-5,
        )
    np.testing.assert_allclose(
        np.asarray(grouped[1]),
        np.asarray(_cross_entropy_from_logits(logits, targets)),
        rtol=1e-6,
        atol=1e-6,
    )


def test_multiscale_distogram_adds_one_term_per_width():
    model = build_model_from_config(
        _config(
            distogram_loss_weight=1.0,
            distogram_multiscale_widths=WIDTHS,
            distogram_multiscale_weight=1.0,
        )
    )

    metrics = model(_batch())

    coarse = [f"distogram_ce_{width}bins" for width in WIDTHS]
    assert set(coarse) <= set(metrics)
    # Marginalising onto coarser bins can only raise the probability of the
    # bucket that holds the target, so each coarser scale scores lower.
    values = [float(metrics[name]) for name in coarse]
    assert float(metrics["distogram_loss"]) > values[0]
    assert values == sorted(values, reverse=True)
    np.testing.assert_allclose(
        float(metrics["distogram_multiscale_loss"]),
        sum(values) / len(values),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        float(metrics["distogram_objective"]),
        float(metrics["distogram_loss"])
        + float(metrics["distogram_multiscale_loss"]),
        rtol=1e-5,
    )


def test_multiscale_is_off_without_widths_or_weight():
    for overrides in (
        {"distogram_multiscale_widths": (), "distogram_multiscale_weight": 1.0},
        {"distogram_multiscale_widths": WIDTHS, "distogram_multiscale_weight": 0.0},
    ):
        model = build_model_from_config(
            _config(distogram_loss_weight=1.0, **overrides)
        )
        metrics = model(_batch())
        assert not [key for key in metrics if key.startswith("distogram_ce_")]
        assert "distogram_multiscale_loss" not in metrics
        np.testing.assert_allclose(
            float(metrics["distogram_objective"]),
            float(metrics["distogram_loss"]),
            rtol=1e-6,
        )


def test_pair_projection_sits_between_the_latents_and_the_head():
    projected = build_model_from_config(
        _config(distogram_loss_weight=1.0, distogram_pair_dim=8)
    )
    unprojected = build_model_from_config(_config(distogram_loss_weight=1.0))

    assert projected.distogram_pair_proj is not None
    assert projected.distogram_pair_proj.weight.shape == (8, 32)
    assert projected.distogram_head.weight.shape[1] == 8
    assert unprojected.distogram_pair_proj is None
    assert unprojected.distogram_head.weight.shape[1] == 32


def test_zero_mae_weight_drops_the_value_heads():
    model = build_model_from_config(
        _config(mae_loss_weight=0.0, distogram_loss_weight=1.0)
    )

    metrics = model(_batch())

    assert not model.uses_mae_value_heads
    assert model.jepa_mae_mz_head is None
    assert model.jepa_mae_intensity_head is None
    assert not [key for key in metrics if key.startswith("mae_")]
    np.testing.assert_allclose(
        float(metrics["loss"]), float(metrics["distogram_term"]), rtol=1e-6
    )


def test_mae_run_keeps_its_value_heads():
    model = build_model_from_config(_config(mae_loss_weight=1.0))

    assert model.uses_mae_value_heads
    assert model.jepa_mae_mz_head is not None
    assert "mae_mz_loss" in model(_batch())


def test_torch_and_jax_agree_on_the_projected_multiscale_distogram():
    """The two backends must score the new head identically."""
    import torch

    from spectra_learning.models.model import PeakSetJEPA
    from spectra_learning.models.model_jax import PeakSetJEPAJax

    kwargs = dict(
        training_mode="mae",
        model_dim=32,
        predictor_dim=32,
        encoder_num_layers=2,
        encoder_num_heads=4,
        attention_mlp_multiple=2.0,
        feature_mlp_hidden_dim=16,
        encoder_fourier_mlp_hidden_dim=32,
        num_peaks=8,
        masked_latent_predictor_num_layers=1,
        masked_latent_predictor_num_heads=4,
        jepa_mae_mz_bin_size=2.5,
        jepa_num_target_blocks=1,
        encoder_use_pair_path=False,
        mae_intensity_loss_weight=0.0,
        target_projector_dim=-1,
        mae_loss_weight=0.0,
        distogram_loss_weight=1.0,
        distogram_pair_dim=8,
        distogram_multiscale_widths=WIDTHS,
        distogram_multiscale_weight=1.0,
    )
    torch.manual_seed(11)
    torch_model = PeakSetJEPA(**kwargs).eval()
    jax_model = PeakSetJEPAJax(**kwargs)
    jax_model.load_torch_state_dict(torch_model.state_dict())

    assert torch_model.distogram_pair_proj is not None
    jax_batch = _batch(seed=5)
    torch_batch = {
        key: torch.from_numpy(np.asarray(value)) for key, value in jax_batch.items()
    }

    with torch.no_grad():
        torch_metrics = torch_model(torch_batch)
    jax_metrics = jax_model(jax_batch)

    shared = set(torch_metrics) & set(jax_metrics)
    assert {"distogram_loss", "distogram_objective", *(
        f"distogram_ce_{width}bins" for width in WIDTHS
    )} <= shared
    for name in sorted(shared):
        np.testing.assert_allclose(
            float(jax_metrics[name]),
            float(torch_metrics[name]),
            rtol=2e-4,
            atol=2e-5,
            err_msg=name,
        )


def test_distogram_remat_is_exact_in_value_and_gradient():
    """Recomputing the logits must not move the loss or its gradients."""
    import jax as _jax

    shared = dict(
        distogram_loss_weight=1.0,
        distogram_pair_dim=8,
        distogram_multiscale_widths=WIDTHS,
        distogram_multiscale_weight=1.0,
        mae_loss_weight=0.0,
    )
    stored = build_model_from_config(_config(**shared))
    recomputed = build_model_from_config(
        _config(
            **shared,
            activation_checkpoint_mode="full",
            activation_checkpoint_modules=("distogram",),
        )
    )
    nnx.update(recomputed, nnx.state(stored))
    batch = _batch(seed=2)

    stored_metrics = stored(batch)
    recomputed_metrics = recomputed(batch)
    assert set(stored_metrics) == set(recomputed_metrics)
    for name in stored_metrics:
        assert float(stored_metrics[name]) == float(recomputed_metrics[name]), name

    def gradients(model):
        graphdef, params, static = nnx.split(model, nnx.Param, ...)

        def loss(values):
            return nnx.merge(graphdef, values, static)(batch)["loss"]

        return _jax.tree.leaves(_jax.grad(loss)(params))

    for stored_leaf, recomputed_leaf in zip(
        gradients(stored), gradients(recomputed), strict=True
    ):
        np.testing.assert_array_equal(
            np.asarray(stored_leaf), np.asarray(recomputed_leaf)
        )


@pytest.mark.parametrize("widths", [(), WIDTHS])
def test_chunked_distogram_ce_matches_unchunked(widths):
    """Scoring pairs in padded chunks must give the same loss and gradients."""
    shared = dict(
        distogram_loss_weight=1.0,
        distogram_pair_dim=8,
        distogram_multiscale_widths=widths,
        distogram_multiscale_weight=1.0 if widths else 0.0,
        mae_loss_weight=0.0,
    )
    whole = build_model_from_config(_config(**shared))
    # 5 does not divide the number of scored pairs, so the last chunk is padded.
    chunked = build_model_from_config(_config(**shared, distogram_ce_chunk_rows=5))
    nnx.update(chunked, nnx.state(whole))
    batch = _batch(seed=3)

    whole_metrics = whole(batch)
    chunked_metrics = chunked(batch)
    assert set(whole_metrics) == set(chunked_metrics)
    for name in whole_metrics:
        np.testing.assert_allclose(
            float(chunked_metrics[name]), float(whole_metrics[name]),
            rtol=1e-6, err_msg=name,
        )

    def gradients(model):
        graphdef, params, static = nnx.split(model, nnx.Param, ...)

        def loss(values):
            return nnx.merge(graphdef, values, static)(batch)["loss"]

        return jax.tree.leaves(jax.grad(loss)(params))

    for whole_leaf, chunked_leaf in zip(
        gradients(whole), gradients(chunked), strict=True
    ):
        # The head gradient sums over chunks in a different order.
        whole_leaf = np.asarray(whole_leaf)
        np.testing.assert_allclose(
            np.asarray(chunked_leaf), whole_leaf,
            rtol=0, atol=1e-6 * max(np.abs(whole_leaf).max(), 1e-6),
        )
