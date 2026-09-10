from spectra_learning.training.naming import (
    _architecture_parts,
    _ema_parts,
    _objective_parts,
    _target_parts,
)


def test_ema_parts_label_momentum_points_explicitly() -> None:
    parts = _ema_parts(
        {
            "use_ema_teacher": True,
            "ema_teacher_schedule": "cosine",
            "ema_teacher_momentum_start": 0.995,
            "ema_teacher_momentum_mid": 0.99,
            "ema_teacher_momentum_final": 0.9995,
        }
    )

    assert parts == [
        "ema",
        "cosine",
        "mstart0.9950",
        "mmid0.9900",
        "mfinal0.9995",
    ]


def test_ema_parts_are_suppressed_for_mae_mode() -> None:
    assert _ema_parts({"training_mode": "mae", "use_ema_teacher": True}) == []


def test_mae_mode_names_primary_binned_objective() -> None:
    parts = _objective_parts(
        {
            "training_mode": "mae",
            "mae_loss_weight": 0.5,
            "jepa_mae_loss_weight": 1.0,
            "jepa_mae_mz_bin_size": 2.5,
            "jepa_mae_intensity_bin_size": 0.1,
        }
    )

    assert parts == ["maew5e-01", "mzbin2.5", "intbin0.1"]


def test_mae_mode_names_disabled_intensity_objective() -> None:
    parts = _objective_parts(
        {
            "training_mode": "mae",
            "mae_loss_weight": 0.5,
            "jepa_mae_mz_bin_size": 2.5,
            "jepa_mae_intensity_bin_size": 0.1,
            "mae_intensity_loss_weight": 0.0,
        }
    )

    assert parts == ["maew5e-01", "mzbin2.5", "nointensity"]


def test_sparse_ordinal_objective_names_radius_and_epsilon() -> None:
    parts = _objective_parts(
        {
            "training_mode": "mae",
            "mae_loss_weight": 1.0,
            "jepa_mae_mz_bin_size": 0.5,
            "mae_mz_objective": "sparse_ordinal_ce",
            "mae_mz_ordinal_radius_bins": 2,
            "mae_mz_ordinal_epsilon": 0.2,
            "mae_intensity_loss_weight": 0.0,
        }
    )

    assert parts == [
        "maew1e+00",
        "mzbin0.5",
        "sparseord",
        "ordr2",
        "ordeps0.2",
        "nointensity",
    ]


def test_legacy_multiscale_objective_is_named() -> None:
    parts = _objective_parts(
        {
            "training_mode": "mae",
            "mae_loss_weight": 1.0,
            "jepa_mae_mz_bin_size": 0.5,
            "mae_mz_multiscale_widths": (2, 4),
            "mae_mz_multiscale_weight": 0.5,
            "mae_intensity_loss_weight": 0.0,
        }
    )

    assert parts == [
        "maew1e+00",
        "mzbin0.5",
        "multiscale",
        "multiw5e-01",
        "nointensity",
    ]


def test_distogram_objective_uses_shared_mz_bin_name() -> None:
    parts = _objective_parts(
        {
            "distogram_loss_weight": 1.0,
            "jepa_mae_mz_bin_size": 0.1,
        }
    )

    assert parts == ["disto", "distow1e+00", "mzbin0.1"]


def test_distogram_objective_does_not_duplicate_mae_mz_bin_name() -> None:
    parts = _objective_parts(
        {
            "jepa_mae_loss_weight": 1.0,
            "distogram_loss_weight": 1.0,
            "jepa_mae_mz_bin_size": 0.1,
            "jepa_mae_intensity_bin_size": 0.1,
        }
    )

    assert parts == [
        "jepamae",
        "maew1e+00",
        "mzbin0.1",
        "intbin0.1",
        "disto",
        "distow1e+00",
    ]


def test_latent_pair_objective_is_named_when_enabled() -> None:
    parts = _objective_parts({"latent_pair_loss_weight": 0.5})

    assert parts == ["latentpair", "pairw5e-01"]


def test_architecture_parts_name_token_mz_embedding() -> None:
    parts = _architecture_parts(
        {
            "encoder_mz_embedding": "token",
            "encoder_mz_token_bin_size": 0.02,
            "encoder_mz_token_embedding_dim": 77,
            "encoder_fourier_mlp_num_layers": 4,
            "encoder_fourier_mlp_hidden_dim": 64,
        }
    )

    assert "mztok0.02x77" in parts
    assert "fmlp4x64" in parts


def test_jax_parameter_count_uses_abstract_trainable_state(monkeypatch):
    import pytest
    jax = pytest.importorskip("jax")
    from flax import nnx
    from spectra_learning.models import factory_jax
    from spectra_learning.training import naming

    class Model(nnx.Module):
        def __init__(self):
            self.weight = nnx.Param(jax.numpy.ones((2, 3)))
            self.b = nnx.Param(jax.numpy.ones((7,)))
            self.teacher_encoder = nnx.Dict(weight=nnx.Param(jax.numpy.ones((10,))))

    monkeypatch.setattr(factory_jax, "build_model_from_config", lambda config: Model())
    def torch_build(config):
        raise AssertionError("JAX naming must not build the torch model")
    monkeypatch.setattr(naming, "build_model_from_config", torch_build)
    assert naming._parameter_count({"device_backend": "jax"}) == 6


def test_pair_only_architecture_names_pair_width_instead_of_unused_peak_mlp():
    assert _architecture_parts({"training_mode": "pair_distogram", "model_dim": 2048,
        "pairmixer_pair_dim": 256, "encoder_fourier_mlp_hidden_dim": 4096}) == ["pairdim256"]
