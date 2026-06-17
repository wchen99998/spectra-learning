from spectra_learning.training.cadence import should_run_at_step_or_final


def test_should_run_at_step_or_final_adds_final_probe_without_duplicates():
    assert should_run_at_step_or_final(
        100_000,
        100_000,
        total_steps=250_000,
        run_at_final_step=True,
    )
    assert should_run_at_step_or_final(
        100_000,
        200_000,
        total_steps=250_000,
        run_at_final_step=True,
    )
    assert should_run_at_step_or_final(
        100_000,
        250_000,
        total_steps=250_000,
        run_at_final_step=True,
    )
    assert not should_run_at_step_or_final(
        100_000,
        150_000,
        total_steps=250_000,
        run_at_final_step=True,
    )
    assert not should_run_at_step_or_final(
        100_000,
        250_000,
        total_steps=250_000,
        run_at_final_step=False,
    )
