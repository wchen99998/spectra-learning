from spectra_learning.training.naming import _ema_parts, _regularizer_parts, _target_parts


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
    parts = _regularizer_parts(
        {
            "training_mode": "mae",
            "mae_loss_weight": 0.5,
            "jepa_mae_loss_weight": 1.0,
            "jepa_mae_mz_bin_size": 2.5,
            "jepa_mae_intensity_bin_size": 0.1,
        }
    )

    assert parts == ["maew5e-01", "mzbin2.5", "intbin0.1"]


def test_mae_mode_does_not_name_jepa_target_layers() -> None:
    assert _target_parts(
        {"training_mode": "mae", "jepa_target_layers": [1, 2], "model_dim": 32}
    ) == []
