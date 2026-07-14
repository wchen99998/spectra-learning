from types import SimpleNamespace

from ml_collections import config_dict

from spectra_learning.training.cadence import (
    should_run_at_step_or_final,
    total_training_steps,
)


def test_total_training_steps_applies_optional_positive_cap():
    config = config_dict.ConfigDict({"num_epochs": 2.5})
    datamodule = SimpleNamespace(train_steps=10)

    assert total_training_steps(config, datamodule) == 25

    config.training_max_steps = 12
    assert total_training_steps(config, datamodule) == 12

    config.training_max_steps = 0
    assert total_training_steps(config, datamodule) == 1


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
