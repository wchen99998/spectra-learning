from spectra_learning.training.naming import _ema_parts


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
