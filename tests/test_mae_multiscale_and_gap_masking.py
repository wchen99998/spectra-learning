import numpy as np
import pytest
import torch

from spectra_learning.config import load_config
from spectra_learning.data.gems.gap_aware import (
    bucket_sampling_weights,
    gap_to_nearest,
    sample_gap_aware_masks_torch,
)
from spectra_learning.models.model import (
    _cross_entropy_from_logits as torch_cross_entropy_from_logits,
    _sparse_ordinal_cross_entropy_from_logits as torch_sparse_ordinal_ce,
)

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from spectra_learning.models.factory_jax import build_model_from_config  # noqa: E402
from spectra_learning.models.model_jax import (  # noqa: E402
    _cross_entropy_from_logits as jax_cross_entropy_from_logits,
    _sparse_ordinal_cross_entropy_from_logits as jax_sparse_ordinal_ce,
)


BASE_OVERRIDES = {
    "model_dim": 64,
    "encoder_num_layers": 2,
    "encoder_num_heads": 2,
    "feature_mlp_hidden_dim": 64,
    "encoder_fourier_mlp_hidden_dim": 64,
    "predictor_dim": 64,
    "masked_latent_predictor_num_layers": 1,
    "masked_latent_predictor_num_heads": 2,
    "num_peaks": 16,
    "encoder_metadata_schema": "",
    "encoder_metadata_conditioning": "none",
}
CONFIG = "configs/1b_singlemixer_dense_muon_metadata_adaln.py"
GAP_AWARE_CONFIG = "configs/1b_singlemixer_multiscale_gapaware.py"
MULTISCALE_N128_CONFIG = (
    "configs/1b_singlemixer_dense_muon_metadata_adaln_multiscale_n128_b2048_1m.py"
)
SPARSE_ORDINAL_N128_CONFIG = (
    "configs/1b_singlemixer_dense_muon_metadata_adaln_"
    "sparse_ordinal_n128_b2048_1m.py"
)
SPARSE_ORDINAL_N128_500K_CONFIG = (
    "configs/1b_singlemixer_dense_muon_metadata_adaln_"
    "sparse_ordinal_n128_b2048_500k.py"
)
WIDTHS = (2, 4, 8, 16, 32, 64)


def _batch(seed: int = 0, batch: int = 4, peaks: int = 16) -> dict:
    rng = np.random.default_rng(seed)
    mz = np.sort(rng.uniform(0.02, 0.9, size=(batch, peaks)).astype(np.float32), axis=1)
    context = rng.random((batch, peaks)) < 0.6
    target = (~context) & (rng.random((batch, peaks)) < 0.9)
    target[:, 0] = ~context[:, 0]
    return {
        "peak_mz": jnp.asarray(mz),
        "peak_intensity": jnp.asarray(
            rng.uniform(0.01, 1.0, size=(batch, peaks)).astype(np.float32)
        ),
        "peak_valid_mask": jnp.ones((batch, peaks), dtype=bool),
        "context_mask": jnp.asarray(context),
        "target_masks": jnp.asarray(target[:, None, :]),
        "precursor_mz": jnp.asarray(
            rng.uniform(0.3, 1.0, size=(batch,)).astype(np.float32)
        ),
    }


def _model(**overrides):
    return build_model_from_config(load_config(CONFIG, {**BASE_OVERRIDES, **overrides}))


def _dense_sparse_ordinal_target(
    targets: torch.Tensor,
    classes: int,
    radius: int,
    epsilon: float,
) -> torch.Tensor:
    distributions = torch.zeros(targets.shape[0], classes)
    offsets = torch.arange(-radius, radius + 1)
    triangular = (radius + 1 - offsets.abs()).float()
    for row, target in enumerate(targets):
        indices = target + offsets
        valid = (indices >= 0) & (indices < classes)
        local = triangular[valid]
        local = local / local.sum()
        distributions[row, indices[valid]] = epsilon * local
        distributions[row, target] += 1.0 - epsilon
    return distributions


def test_sparse_ordinal_ce_matches_dense_soft_target_and_jax():
    logits = torch.tensor(
        [
            [0.2, -0.4, 1.1, 0.7, -1.2, 0.3, 0.5],
            [-0.7, 0.6, 0.1, 1.2, -0.3, 0.8, -0.2],
            [0.9, -0.1, 0.4, -0.8, 0.2, 0.3, 1.0],
        ]
    )
    targets = torch.tensor([0, 3, 6])
    dense_target = _dense_sparse_ordinal_target(targets, 7, radius=2, epsilon=0.2)
    expected = -(dense_target * torch.log_softmax(logits, dim=-1)).sum(dim=-1)

    torch.testing.assert_close(
        torch_sparse_ordinal_ce(logits, targets, radius=2, epsilon=0.2),
        expected,
    )
    np.testing.assert_allclose(
        np.asarray(
            jax_sparse_ordinal_ce(
                jnp.asarray(logits.numpy()),
                jnp.asarray(targets.numpy()),
                radius=2,
                epsilon=0.2,
            )
        ),
        expected.numpy(),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        dense_target[0, :3],
        torch.tensor([0.9, 1.0 / 15.0, 1.0 / 30.0]),
    )
    torch.testing.assert_close(
        dense_target[1, 1:6],
        torch.tensor([1, 2, 39, 2, 1], dtype=torch.float32) / 45.0,
    )


def test_sparse_ordinal_epsilon_zero_matches_exact_ce():
    logits = torch.randn(2, 3, 11)
    targets = torch.randint(0, 11, (2, 3))
    torch.testing.assert_close(
        torch_sparse_ordinal_ce(logits, targets, radius=4, epsilon=0.0),
        torch_cross_entropy_from_logits(logits, targets),
    )
    np.testing.assert_allclose(
        np.asarray(
            jax_sparse_ordinal_ce(
                jnp.asarray(logits.numpy()),
                jnp.asarray(targets.numpy()),
                radius=4,
                epsilon=0.0,
            )
        ),
        np.asarray(
            jax_cross_entropy_from_logits(
                jnp.asarray(logits.numpy()),
                jnp.asarray(targets.numpy()),
            )
        ),
        rtol=1e-6,
        atol=1e-6,
    )


def test_sparse_ordinal_ce_has_finite_gradients():
    logits = torch.randn(4, 13, requires_grad=True)
    targets = torch.tensor([0, 4, 8, 12])
    torch_sparse_ordinal_ce(logits, targets, radius=2, epsilon=0.2).mean().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0.0

    jax_logits = jnp.asarray(logits.detach().numpy())
    jax_gradient = jax.grad(
        lambda values: jax_sparse_ordinal_ce(
            values,
            jnp.asarray(targets.numpy()),
            radius=2,
            epsilon=0.2,
        ).mean()
    )(jax_logits)
    assert np.all(np.isfinite(np.asarray(jax_gradient)))
    assert float(jnp.abs(jax_gradient).sum()) > 0.0


def test_multiscale_disabled_by_default_leaves_loss_unchanged():
    from spectra_learning.models.settings import PeakSetJEPASettings

    settings = PeakSetJEPASettings.from_config(load_config(CONFIG, BASE_OVERRIDES))
    assert settings.mae_mz_objective == "exact_ce"
    assert tuple(settings.mae_mz_multiscale_widths) == ()
    assert float(settings.mae_mz_multiscale_weight) == 0.0
    metrics = _model()(_batch())
    assert "mae_mz_multiscale_loss" not in metrics
    assert float(metrics["mae_loss"]) == pytest.approx(
        float(metrics["mae_mz_loss"]), rel=1e-6
    )


def test_legacy_multiscale_config_resolves_multiscale_objective():
    from spectra_learning.models.settings import PeakSetJEPASettings

    config = load_config(
        CONFIG,
        {
            **BASE_OVERRIDES,
            "mae_mz_multiscale_widths": WIDTHS,
            "mae_mz_multiscale_weight": 1.0,
        },
    )
    assert "mae_mz_objective" not in config
    assert PeakSetJEPASettings.from_config(config).mae_mz_objective == "multiscale_ce"


def test_multiscale_adds_weighted_coarse_terms():
    batch = _batch()
    baseline = _model()(batch)
    metrics = _model(
        mae_mz_multiscale_widths=WIDTHS, mae_mz_multiscale_weight=1.0
    )(batch)
    per_width = [float(metrics[f"mae_mz_ce_{width}bins"]) for width in WIDTHS]
    assert len(per_width) == len(WIDTHS)
    # coarser bins are strictly easier than finer bins
    assert per_width == sorted(per_width, reverse=True)
    assert all(value < float(metrics["mae_mz_loss"]) for value in per_width)
    assert float(metrics["mae_mz_multiscale_loss"]) == pytest.approx(
        sum(per_width) / len(per_width), rel=1e-5
    )
    assert float(metrics["mae_loss"]) == pytest.approx(
        float(metrics["mae_mz_loss"]) + float(metrics["mae_mz_multiscale_loss"]),
        rel=1e-5,
    )
    # the fine term is unaffected by enabling the auxiliary losses
    assert float(metrics["mae_mz_loss"]) == pytest.approx(
        float(baseline["mae_mz_loss"]), rel=1e-5
    )


def test_multiscale_weight_scales_the_objective():
    batch = _batch()
    half = _model(mae_mz_multiscale_widths=WIDTHS, mae_mz_multiscale_weight=0.5)(batch)
    assert float(half["mae_loss"]) == pytest.approx(
        float(half["mae_mz_loss"]) + 0.5 * float(half["mae_mz_multiscale_loss"]),
        rel=1e-5,
    )


def test_multiscale_loss_is_differentiable():
    from flax import nnx

    model = _model(mae_mz_multiscale_widths=WIDTHS, mae_mz_multiscale_weight=1.0)
    batch = _batch()
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)

    def loss_fn(p):
        return nnx.merge(graphdef, p, rest)(batch, loss_only=True)["loss"]

    grads = jax.grad(loss_fn)(nnx.as_pure(params))
    leaves = [np.asarray(v) for v in jax.tree.leaves(grads)]
    assert leaves
    assert all(np.all(np.isfinite(leaf)) for leaf in leaves)
    assert any(np.abs(leaf).sum() > 0 for leaf in leaves)


def test_loss_only_path_matches_full_path():
    model = _model(mae_mz_multiscale_widths=WIDTHS, mae_mz_multiscale_weight=1.0)
    batch = _batch()
    assert float(model(batch, loss_only=True)["loss"]) == pytest.approx(
        float(model(batch)["loss"]), rel=1e-5
    )


def test_sparse_ordinal_replaces_exact_and_multiscale_objectives():
    model = _model(
        mae_mz_objective="sparse_ordinal_ce",
        mae_mz_ordinal_radius_bins=2,
        mae_mz_ordinal_epsilon=0.2,
        mae_mz_multiscale_widths=WIDTHS,
        mae_mz_multiscale_weight=1.0,
    )
    batch = _batch()
    metrics = model(batch)

    assert "mae_mz_multiscale_loss" not in metrics
    assert float(metrics["mae_loss"]) == pytest.approx(
        float(metrics["mae_mz_loss"]), rel=1e-6
    )
    assert float(model(batch, loss_only=True)["loss"]) == pytest.approx(
        float(metrics["loss"]), rel=1e-5
    )


def test_sparse_ordinal_model_loss_is_differentiable():
    from flax import nnx

    model = _model(
        mae_mz_objective="sparse_ordinal_ce",
        mae_mz_ordinal_radius_bins=2,
        mae_mz_ordinal_epsilon=0.2,
    )
    batch = _batch()
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)

    def loss_fn(p):
        return nnx.merge(graphdef, p, rest)(batch, loss_only=True)["loss"]

    grads = jax.grad(loss_fn)(nnx.as_pure(params))
    leaves = [np.asarray(value) for value in jax.tree.leaves(grads)]
    assert leaves
    assert all(np.all(np.isfinite(leaf)) for leaf in leaves)
    assert any(np.abs(leaf).sum() > 0 for leaf in leaves)


def test_gap_bucket_metrics_partition_the_targets():
    metrics = _model()(_batch())
    fractions = [
        float(value)
        for key, value in metrics.items()
        if key.startswith("mae_gap_fraction_")
    ]
    assert len(fractions) == 3
    assert sum(fractions) == pytest.approx(1.0, abs=1e-5)
    assert all(f"mae_mz_loss_gap_{label}da" in metrics for label in ("0_2", "2_5", "5_inf"))


def test_gap_aware_experiment_hard_excludes_anchored_targets():
    config = load_config(GAP_AWARE_CONFIG)
    assert config.training_max_steps == 1_000_000
    assert tuple(config.jepa_gap_aware_bucket_weights) == (0.0, 0.4, 0.6)
    assert "gapaware-0-40-60" in config.run_name_suffix


def test_dense_metadata_adaln_multiscale_n128_experiment():
    base = load_config(CONFIG)
    config = load_config(MULTISCALE_N128_CONFIG)

    assert config.batch_size == 2_048
    assert config.gradient_accumulation_steps == 2
    assert config.num_peaks + int(config.encoder_use_cls_token) == 128
    assert tuple(config.mae_mz_multiscale_widths) == WIDTHS
    assert config.mae_mz_multiscale_weight == 1.0
    assert config.training_max_steps == 1_000_000
    assert config.max_duration_hours == 167.0
    assert config.jepa_mask_strategy == base.jepa_mask_strategy == ["random"]
    assert config.optimizer == base.optimizer == "muon"
    assert config.learning_rate == base.learning_rate == 3e-4
    assert config.warmup_steps == base.warmup_steps == 20_000
    assert config.pairmixer_use_pair_bias is base.pairmixer_use_pair_bias is True


def test_dense_metadata_adaln_sparse_ordinal_n128_experiments():
    base = load_config(CONFIG)
    config = load_config(SPARSE_ORDINAL_N128_CONFIG)
    short = load_config(SPARSE_ORDINAL_N128_500K_CONFIG)

    assert config.batch_size == 2_048
    assert config.gradient_accumulation_steps == 2
    assert config.num_peaks + int(config.encoder_use_cls_token) == 128
    assert config.mae_mz_objective == "sparse_ordinal_ce"
    assert config.mae_mz_ordinal_radius_bins == 2
    assert config.mae_mz_ordinal_epsilon == pytest.approx(0.2)
    assert tuple(config.mae_mz_multiscale_widths) == ()
    assert config.mae_mz_multiscale_weight == 0.0
    assert config.training_max_steps == 1_000_000
    assert config.max_duration_hours == 167.0
    assert "sparseord-r2-eps0p2" in config.run_name_suffix
    assert config.jepa_mask_strategy == base.jepa_mask_strategy == ["random"]
    assert config.optimizer == base.optimizer == "muon"

    assert short.training_max_steps == 500_000
    assert short.max_duration_hours == 95.0
    assert short.min_learning_rate == pytest.approx(short.learning_rate / 10)
    assert short.run_name_suffix.endswith("-500k")


def test_torch_gap_bucket_metrics_partition_the_targets():
    from spectra_learning.models.model import PeakSetJEPA

    model = PeakSetJEPA(
        training_mode="mae",
        model_dim=32,
        predictor_dim=32,
        encoder_num_layers=1,
        encoder_num_heads=4,
        masked_latent_predictor_num_layers=1,
        masked_latent_predictor_num_heads=4,
        num_peaks=6,
        feature_mlp_hidden_dim=32,
        pairmixer_pair_dim=16,
        pairmixer_pair_feature_hidden_dim=32,
        pairmixer_fourier_num_freqs=1,
        predictor_target_max_tokens=2,
        distogram_loss_weight=0.0,
        mae_intensity_loss_weight=0.0,
        target_projector_dim=-1,
        predictor_dropout=0.0,
    ).eval()
    batch = {
        "peak_mz": torch.tensor(
            [[0.10, 0.20, 0.30, 0.40, 0.50, 0.60]] * 2,
        ),
        "peak_intensity": torch.ones(2, 6),
        "peak_valid_mask": torch.ones(2, 6, dtype=torch.bool),
        "context_mask": torch.tensor(
            [[True, True, False, False, False, False]] * 2,
        ),
        "target_masks": torch.tensor(
            [[[False, False, True, True, False, False]]] * 2,
        ),
    }
    with torch.no_grad():
        metrics = model(batch)
    fractions = [
        float(value)
        for key, value in metrics.items()
        if key.startswith("mae_gap_fraction_")
    ]
    assert len(fractions) == 3
    assert sum(fractions) == pytest.approx(1.0)


def test_gap_to_nearest_measures_distance_to_context_only():
    mz = torch.tensor([100.0, 101.0, 200.0, 400.0])
    context = torch.tensor([True, False, True, False])
    gaps = gap_to_nearest(mz, context)
    assert gaps.tolist() == pytest.approx([0.0, 1.0, 0.0, 200.0])


def test_bucket_weights_ignore_unavailable_peaks():
    gap = torch.tensor([0.5, 3.0, 50.0, 0.5])
    available = torch.tensor([True, True, True, False])
    weights = bucket_sampling_weights(
        gap, available, bucket_edges_da=(2.0, 5.0), bucket_weights=(0.2, 0.4, 0.4)
    )
    assert float(weights[3]) == 0.0
    assert float(weights[0]) == pytest.approx(0.2)
    assert float(weights[1]) == pytest.approx(0.4)
    assert float(weights[2]) == pytest.approx(0.4)


def test_bucket_boundaries_match_reported_gap_metrics():
    weights = bucket_sampling_weights(
        torch.tensor([1.0, 2.0, 5.0]),
        torch.ones(3, dtype=torch.bool),
        bucket_edges_da=(2.0, 5.0),
        bucket_weights=(0.1, 0.2, 0.3),
    )
    assert weights.tolist() == pytest.approx([0.1, 0.2, 0.3])


def test_zero_weight_bucket_is_a_hard_exclusion():
    gap = torch.tensor([0.5, 0.7])
    available = torch.tensor([True, True])
    weights = bucket_sampling_weights(
        gap, available, bucket_edges_da=(2.0, 5.0), bucket_weights=(0.0, 0.5, 0.5)
    )
    assert float(weights.sum()) == 0.0


def test_gap_aware_masking_zero_weight_bucket_is_avoided():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    mz = torch.from_numpy(
        np.sort(rng.uniform(20.0, 900.0, size=(256, 40)), axis=1).astype(np.float32)
    )
    valid = torch.ones(256, 40, dtype=torch.bool)
    context, targets = sample_gap_aware_masks_torch(
        valid,
        mz,
        num_target_blocks=1,
        context_fraction=0.6,
        target_fraction=0.15,
        block_min_len=1,
        allow_target_overlap=False,
        bucket_edges_da=(2.0, 5.0),
        bucket_weights=(0.0, 0.5, 0.5),
    )
    anchored = 0
    total = 0
    for row in range(mz.shape[0]):
        selected = targets[row, 0]
        if not bool(selected.any()):
            continue
        gaps = gap_to_nearest(mz[row], context[row])[selected]
        anchored += int((gaps < 2.0).sum())
        total += int(selected.sum())
    assert total > 0
    assert anchored == 0


def test_gap_aware_masking_underfills_instead_of_backfilling_excluded_bucket():
    torch.manual_seed(0)
    valid = torch.ones(8, 8, dtype=torch.bool)
    mz = torch.zeros(8, 8)
    context, targets = sample_gap_aware_masks_torch(
        valid,
        mz,
        num_target_blocks=1,
        context_fraction=0.5,
        target_fraction=0.25,
        block_min_len=1,
        allow_target_overlap=False,
        bucket_edges_da=(2.0, 5.0),
        bucket_weights=(0.0, 0.4, 0.6),
    )
    assert int(context.sum()) == 32
    assert int(targets.sum()) == 0


def test_gap_aware_masking_keeps_context_and_targets_disjoint():
    torch.manual_seed(1)
    rng = np.random.default_rng(1)
    mz = torch.from_numpy(
        np.sort(rng.uniform(20.0, 900.0, size=(64, 24)), axis=1).astype(np.float32)
    )
    valid = torch.zeros(64, 24, dtype=torch.bool)
    valid[:, :18] = True
    context, targets = sample_gap_aware_masks_torch(
        valid,
        mz,
        num_target_blocks=2,
        context_fraction=0.5,
        target_fraction=0.2,
        block_min_len=1,
        allow_target_overlap=False,
    )
    assert not bool((context & ~valid).any())
    assert not bool((targets & ~valid[:, None, :]).any())
    assert not bool((targets & context[:, None, :]).any())
    assert not bool((targets[:, 0] & targets[:, 1]).any())


def test_gap_aware_masking_rejects_mismatched_weights():
    mz = torch.zeros(2, 4)
    valid = torch.ones(2, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match="one more bucket weight"):
        sample_gap_aware_masks_torch(
            valid,
            mz,
            num_target_blocks=1,
            context_fraction=0.5,
            target_fraction=0.25,
            block_min_len=1,
            allow_target_overlap=False,
            bucket_edges_da=(2.0, 5.0),
            bucket_weights=(0.5, 0.5),
        )


def test_collator_dispatches_gap_aware_strategy():
    from spectra_learning.data.gems.collate import GemsBatchCollator

    collator = GemsBatchCollator(
        augment=True,
        num_target_blocks=1,
        context_fraction=0.6,
        target_fraction=0.25,
        block_min_len=1,
        num_peaks=16,
        max_precursor_mz=1000.0,
        min_peak_intensity=0.0001,
        peak_drop_min_intensity=0.0001,
        peak_ordering="mz",
        precursor_peak_exclusion_window_da=0.0,
        mask_strategy=["gap_aware"],
        gap_aware_bucket_weights=(0.0, 0.5, 0.5),
        output_format="torch",
    )
    rng = np.random.default_rng(3)
    samples = []
    for _ in range(8):
        mz = np.sort(rng.uniform(50.0, 900.0, size=128)).astype(np.float32)
        intensity = rng.uniform(0.01, 1.0, size=128).astype(np.float32)
        samples.append(
            {
                "spectra": np.stack([mz, intensity]),
                "precursor_mz_raw": np.float32(950.0),
                "collision_energy": np.float32(30.0),
                "charge": np.float32(1.0),
            }
        )
    batch = collator(samples)
    assert batch["target_masks"].shape[1] == 1
    assert bool(batch["target_masks"].any())
    assert not bool(
        (batch["target_masks"] & batch["context_mask"][:, None, :]).any()
    )
