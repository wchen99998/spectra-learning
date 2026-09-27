from __future__ import annotations

import pytest
from ml_collections import config_dict

from spectra_learning.training import pretrain_jax


def _forbid_storage_access(monkeypatch) -> None:
    def unexpected_access(*args, **kwargs):
        pytest.fail("This rank/configuration must not access deadline storage")

    for name in ("storage_exists", "read_text", "write_text"):
        monkeypatch.setattr(pretrain_jax, name, unexpected_access)


def test_training_deadline_survives_restart(monkeypatch, tmp_path) -> None:
    path = tmp_path / "training_deadline.txt"
    cfg = config_dict.ConfigDict(
        {"max_duration_hours": 1.0, "jax_training_deadline_path": str(path)}
    )
    monkeypatch.setattr(pretrain_jax.jax, "process_index", lambda: 0)
    monkeypatch.setattr(pretrain_jax, "_jax_wall_time", lambda: 100.0)

    assert pretrain_jax._jax_training_deadline(cfg) == 3700.0
    assert float(path.read_text()) == 3700.0

    # A restart after expiry must not grant another hour or rewrite the budget.
    monkeypatch.setattr(pretrain_jax, "_jax_wall_time", lambda: 4000.0)
    assert pretrain_jax._jax_training_deadline(cfg) == 3700.0
    assert float(path.read_text()) == 3700.0


def test_nonzero_rank_keeps_collective_participation_without_storage(
    monkeypatch,
) -> None:
    cfg = config_dict.ConfigDict(
        {"max_duration_hours": 1.0, "jax_training_deadline_path": "gs://deadline"}
    )
    monkeypatch.setattr(pretrain_jax.jax, "process_index", lambda: 1)
    monkeypatch.setattr(pretrain_jax, "_jax_wall_time", lambda: 100.0)
    _forbid_storage_access(monkeypatch)

    assert pretrain_jax._jax_training_deadline(cfg) == 3700.0


@pytest.mark.parametrize("hours", [None, "missing"])
def test_disabled_deadline_does_not_access_storage(monkeypatch, hours) -> None:
    cfg = config_dict.ConfigDict({"jax_training_deadline_path": "gs://deadline"})
    if hours is None:
        cfg.max_duration_hours = None
    _forbid_storage_access(monkeypatch)

    assert pretrain_jax._jax_training_deadline(cfg) is None


def test_default_deadline_starts_fresh_without_storage(monkeypatch) -> None:
    cfg = config_dict.ConfigDict({"max_duration_hours": 1.0})
    monkeypatch.setattr(pretrain_jax.jax, "process_index", lambda: 0)
    _forbid_storage_access(monkeypatch)
    wall_times = iter((100.0, 4000.0))
    monkeypatch.setattr(pretrain_jax, "_jax_wall_time", lambda: next(wall_times))

    assert pretrain_jax._jax_training_deadline(cfg) == 3700.0
    assert pretrain_jax._jax_training_deadline(cfg) == 7600.0


@pytest.mark.parametrize("saved", ["nan", "inf", "-inf", "not a timestamp"])
def test_invalid_saved_deadline_is_rejected(monkeypatch, tmp_path, saved) -> None:
    path = tmp_path / "training_deadline.txt"
    path.write_text(saved)
    cfg = config_dict.ConfigDict(
        {"max_duration_hours": 1.0, "jax_training_deadline_path": str(path)}
    )
    monkeypatch.setattr(pretrain_jax.jax, "process_index", lambda: 0)
    monkeypatch.setattr(pretrain_jax, "_jax_wall_time", lambda: 100.0)

    with pytest.raises(ValueError):
        pretrain_jax._jax_training_deadline(cfg)
    assert path.read_text() == saved
