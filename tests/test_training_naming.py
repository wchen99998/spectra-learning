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


def test_distogram_objective_uses_shared_mz_bin_name() -> None:
    parts = _objective_parts(
        {
            "distogram_loss_weight": 1.0,
            "jepa_mae_mz_bin_size": 0.1,
        }
    )

    assert parts == ["disto", "distow1e+00", "mzbin0.1"]


def test_pair_latent_objective_names_weight() -> None:
    parts = _objective_parts({"pair_latent_loss_weight": 1.0})

    assert parts == ["pairlat", "pairw1e+00"]


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


def test_mae_mode_does_not_name_jepa_target_layers() -> None:
    assert _target_parts(
        {"training_mode": "mae", "jepa_target_layers": [1, 2], "model_dim": 32}
    ) == []


def test_mae_teacher_jepa_mode_does_not_name_jepa_target_layers() -> None:
    assert _target_parts(
        {
            "training_mode": "mae_teacher_jepa",
            "jepa_target_layers": [1, 2],
            "model_dim": 32,
        }
    ) == []


def test_architecture_parts_name_disabled_fourier_features() -> None:
    parts = _architecture_parts(
        {
            "encoder_use_fourier_features": False,
            "encoder_fourier_mlp_num_layers": 4,
            "encoder_fourier_mlp_hidden_dim": 64,
        }
    )

    assert "no-fourier" in parts
    assert "fmlp4x64" in parts
