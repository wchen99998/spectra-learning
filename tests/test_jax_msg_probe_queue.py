from __future__ import annotations

import jax.numpy as jnp

from spectra_learning.probes.massspec import msg_probe_jax
from spectra_learning.probes.massspec.msg_settings import MsgProbeTaskSpec


def test_pending_predictions_use_bounded_async_window(monkeypatch):
    task_spec = MsgProbeTaskSpec(
        regression_tasks=(),
        binary_tasks=("fluorine",),
        maccs_bits=0,
        regression_means={},
        regression_stds={},
    )
    epoch_state = msg_probe_jax._new_epoch_state(task_spec)
    pending = []
    copied = []

    def fake_copy_to_host_async(tree):
        copied.append(tree)

    monkeypatch.setattr(
        msg_probe_jax,
        "_copy_tree_to_host_async",
        fake_copy_to_host_async,
    )

    batch = {
        "probe_valid_mol": jnp.asarray([True]),
        "probe_fluorine": jnp.asarray([1.0], dtype=jnp.float32),
    }
    appended = msg_probe_jax.MAX_PENDING_PREDICTIONS + 2

    for idx in range(appended):
        logits = {"fluorine": jnp.asarray([[float(idx)]], dtype=jnp.float32)}
        msg_probe_jax._append_pending_prediction(
            pending,
            logits=logits,
            batch=batch,
            task_spec=task_spec,
        )
        msg_probe_jax._flush_pending_predictions_if_full(
            epoch_state,
            pending,
            task_spec,
        )
        assert len(pending) < msg_probe_jax.MAX_PENDING_PREDICTIONS

    assert len(copied) == appended
    assert epoch_state["count"] == appended - (
        msg_probe_jax.MAX_PENDING_PREDICTIONS - 1
    )

    msg_probe_jax._flush_pending_predictions(epoch_state, pending, task_spec)

    assert not pending
    assert epoch_state["count"] == appended
    assert sum(
        len(chunk) for chunk in epoch_state["predictions"]["fluorine"]
    ) == appended
