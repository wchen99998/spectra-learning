from spectra_learning.training.pretrain_jax import _epoch_progress_steps


def test_epoch_progress_starts_at_restored_global_step() -> None:
    assert _epoch_progress_steps(
        global_step=150_000,
        total_steps=500_000,
        train_steps=800_000,
        epoch=0,
    ) == (150_000, 500_000)


def test_epoch_progress_is_local_to_later_epoch() -> None:
    assert _epoch_progress_steps(
        global_step=225_000,
        total_steps=250_000,
        train_steps=100_000,
        epoch=2,
    ) == (25_000, 50_000)
