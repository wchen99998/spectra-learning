from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from ml_collections import config_dict
from types import SimpleNamespace

from spectra_learning.data.gems.collate import GemsBatchCollator
from spectra_learning.models.model_jax import PeakSetJEPAJax
from spectra_learning.models.settings import PeakSetJEPASettings
from spectra_learning.training.checkpointing_jax import (
    EmergencyCheckpointMonitor,
    build_jax_checkpoint_manager,
    jax_training_checkpoint_metadata,
    restore_frozen_teacher_encoder,
    restore_jax_training_state,
    save_jax_training_state,
)
from spectra_learning.training.logging import MetricLogger
from spectra_learning.training.pretrain_jax import (
    _jax_process_bool_broadcast,
    _run_jax_training_loop,
    build_jax_optax_transform,
    init_pure_optax_train_state,
    initialize_jax_model_from_torch_seed,
    trainable_param_filter,
)


CHECKPOINT_METADATA = jax_training_checkpoint_metadata("test", {})


class _FakeDistributedClient:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def key_value_set(self, key: str, value: str) -> None:
        self.values[key] = value

    def blocking_key_value_get(self, key: str, timeout_in_ms: int) -> str:
        assert timeout_in_ms == 60_000
        return self.values[key]


def test_jax_training_rejects_removed_probe_sharding_path() -> None:
    from spectra_learning.training.pretrain_jax import prepare_jax_training_config

    config = config_dict.ConfigDict({"jax_msg_probe_shard_batches": True})

    with pytest.raises(ValueError, match="jax_msg_probe_shard_batches has been removed"):
        prepare_jax_training_config(config)


def _tiny_mae_kwargs() -> dict[str, object]:
    return {
        "training_mode": "mae",
        "model_dim": 4,
        "encoder_num_layers": 1,
        "encoder_num_heads": 1,
        "attention_mlp_multiple": 1.0,
        "feature_mlp_hidden_dim": 4,
        "encoder_fourier_num_freqs": 1,
        "pairmixer_fourier_num_freqs": 1,
        "pairmixer_pair_dim": 4,
        "pairmixer_pair_feature_hidden_dim": 4,
        "masked_latent_predictor_num_layers": 1,
        "masked_latent_predictor_num_heads": 1,
        "num_peaks": 3,
        "jepa_num_target_blocks": 1,
        "distogram_loss_weight": 0.0,
        "predictor_dropout": 0.0,
        "target_projector_dim": -1,
        "jepa_mae_mz_bin_size": 100.0,
        "jepa_mae_intensity_bin_size": 0.5,
    }


def _zeros_like_with_sharding(value: jax.Array) -> jax.Array:
    return jax.device_put(jnp.zeros_like(value), value.sharding)


def _tiny_numpy_batch() -> dict[str, np.ndarray]:
    torch.manual_seed(123)

    def sample(mz, intensity, precursor_mz, collision_energy, charge):
        spectra = np.zeros((2, 128), dtype=np.float32)
        spectra[0, : len(mz)] = np.asarray(mz, dtype=np.float32)
        spectra[1, : len(intensity)] = np.asarray(intensity, dtype=np.float32)
        return {
            "spectra": spectra,
            "precursor_mz_raw": np.asarray(precursor_mz, dtype=np.float32),
            "collision_energy": np.asarray(collision_energy, dtype=np.float32),
            "charge": np.asarray(charge, dtype=np.float32),
        }

    collator = GemsBatchCollator(
        augment=True,
        num_target_blocks=1,
        context_fraction=0.4,
        target_fraction=0.35,
        block_min_len=1,
        num_peaks=3,
        max_precursor_mz=1000.0,
        min_peak_intensity=0.0,
        peak_drop_min_intensity=0.0,
        peak_ordering="mz",
        precursor_peak_exclusion_window_da=0.0,
        mask_strategy="contiguous",
        mask_lengths=(1, 2, 3),
        mask_round_from=2,
        output_format="numpy",
    )
    return collator(
        [
            sample([100.0, 125.0, 150.0], [1.0, 0.8, 0.4], 500.0, 20.0, 1.0),
            sample([220.0, 240.0, 300.0], [0.9, 0.3, 0.2], 620.0, 40.0, 2.0),
        ]
    )


class _FakeDataModule:
    def __init__(
        self,
        batch: dict[str, np.ndarray],
        train_steps: int,
        gradient_accumulation_steps: int = 1,
        val_batch: dict[str, np.ndarray] | None = None,
    ) -> None:
        self._batch = batch
        self._val_batch = batch if val_batch is None else val_batch
        self._accum = gradient_accumulation_steps
        self.train_steps = train_steps
        self.global_batch_size = int(batch["peak_mz"].shape[0])
        self.batch_size = self.global_batch_size
        self.gradient_accumulation_steps = 1
        self.loader_calls: list[tuple[int, int]] = []
        self.mask_fraction_calls: list[tuple[float, float]] = []
        self.accumulation_calls: list[int] = []

    def train_loader_for_epoch(self, epoch: int, start_batch: int = 0):
        self.loader_calls.append((epoch, start_batch))
        return [self._batch] * ((self.train_steps - start_batch) * self._accum)

    @property
    def val_loader(self):
        return [self._val_batch]

    def val_loader_for_eval(self, *, augment: bool):
        del augment
        return [self._val_batch]

    def set_mask_fractions(
        self,
        context_fraction: float,
        target_fraction: float,
    ) -> None:
        self.mask_fraction_calls.append((context_fraction, target_fraction))

    def set_gradient_accumulation_steps(self, steps: int) -> None:
        self._accum = steps
        self.gradient_accumulation_steps = steps
        self.accumulation_calls.append(steps)


class _RecordingLogger(MetricLogger):
    def __init__(self) -> None:
        self.logs = []

    def log_metrics(self, metrics, step=None) -> None:
        self.logs.append((dict(metrics), step))


class _EmergencyAfterFirstStep:
    reason = "test termination"

    def __init__(self) -> None:
        self.checks = 0
        self.waited = False

    @property
    def requested(self) -> bool:
        self.checks += 1
        return self.checks >= 2

    def wait_for_forced_termination(self) -> None:
        self.waited = True


def test_emergency_checkpoint_monitor_records_first_reason():
    monitor = EmergencyCheckpointMonitor(watch_gce_metadata=False)

    monitor.request("preemption")
    monitor.request("later signal")

    assert monitor.requested is True
    assert monitor.reason == "preemption"


def test_jax_checkpoint_roundtrip_preserves_values_and_sharding(tmp_path):
    mesh = Mesh(np.asarray(jax.devices()), ("data",))
    replicated = NamedSharding(mesh, P())
    params = {
        "weight": jax.device_put(jnp.arange(8.0).reshape(2, 4), replicated),
        "blocks": {0: jnp.full((3,), 2.5)},
    }
    optimizer = build_jax_optax_transform(
        {
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "optimizer_state_dtype": "bf16",
        }
    )
    opt_state = optimizer.init(params)
    state = {"trainable_params": params, "opt_state": opt_state}

    manager = build_jax_checkpoint_manager(tmp_path / "checkpoints", max_to_keep=2)
    save_jax_training_state(manager, 10, state, metadata=CHECKPOINT_METADATA)
    manager.close()

    reopened = build_jax_checkpoint_manager(tmp_path / "checkpoints", max_to_keep=2)
    assert reopened.latest_step() == 10
    template = jax.tree.map(_zeros_like_with_sharding, state)
    restored = restore_jax_training_state(
        reopened,
        10,
        template,
        expected_metadata=CHECKPOINT_METADATA,
    )
    reopened.close()

    assert restored["trainable_params"]["weight"].sharding == replicated
    expected_leaves = jax.tree.leaves(state)
    restored_leaves = jax.tree.leaves(restored)
    for expected, actual in zip(expected_leaves, restored_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual))
    adam_state = restored["opt_state"][0]
    assert {value.dtype for value in jax.tree.leaves(adam_state.mu)} == {
        jnp.dtype(jnp.bfloat16)
    }
    assert {value.dtype for value in jax.tree.leaves(adam_state.nu)} == {
        jnp.dtype(jnp.bfloat16)
    }


def test_jax_checkpoint_restore_releases_template_before_loading():
    template = {"value": jnp.ones((4,), dtype=jnp.float32)}

    class _RestoreManager:
        def restore(self, step, *, args):
            del step, args
            assert template["value"].is_deleted()
            return SimpleNamespace(
                state={"value": jnp.zeros((4,), dtype=jnp.float32)},
                metadata=CHECKPOINT_METADATA,
            )

    restored = restore_jax_training_state(
        _RestoreManager(),
        1,
        template,
        expected_metadata=CHECKPOINT_METADATA,
    )

    np.testing.assert_array_equal(restored["value"], np.zeros((4,)))


def test_restore_frozen_teacher_encoder_from_native_jax_checkpoint(tmp_path):
    source = PeakSetJEPAJax(**_tiny_mae_kwargs(), rngs=nnx.Rngs(7))
    _, source_trainable, source_static = nnx.split(
        source,
        trainable_param_filter,
        ...,
    )
    manager = build_jax_checkpoint_manager(
        tmp_path / "source",
        enable_async_checkpointing=False,
    )
    save_jax_training_state(
        manager,
        300_000,
        {
            "trainable_params": nnx.as_pure(source_trainable),
            "static_state": nnx.as_pure(source_static),
            "opt_state": {},
        },
        metadata=CHECKPOINT_METADATA,
    )
    manager.close()

    target = PeakSetJEPAJax(**_tiny_mae_kwargs(), rngs=nnx.Rngs(19))
    target_state = nnx.as_pure(nnx.state(target.encoder))
    restored = restore_frozen_teacher_encoder(
        tmp_path / "source" / "orbax" / "300000",
        target_state,
    )

    expected = dict(nnx.to_flat_state(nnx.as_pure(nnx.state(source.encoder))))
    actual = dict(nnx.to_flat_state(restored))
    target_values = dict(nnx.to_flat_state(target_state))
    assert actual.keys() == expected.keys()
    for path in expected:
        np.testing.assert_array_equal(
            np.asarray(actual[path]),
            np.asarray(expected[path]),
        )
        assert actual[path].sharding == target_values[path].sharding


def test_jax_checkpoint_manager_keeps_all_steps_when_max_to_keep_is_none(tmp_path):
    manager = build_jax_checkpoint_manager(
        tmp_path / "checkpoints",
        max_to_keep=None,
        enable_async_checkpointing=False,
    )
    for step in (1, 2, 3):
        save_jax_training_state(
            manager,
            step,
            {"value": jnp.asarray(step)},
            metadata=CHECKPOINT_METADATA,
        )

    assert manager.all_steps() == [1, 2, 3]
    manager.close()


def test_jax_checkpoint_restore_rejects_training_contract_mismatch(tmp_path):
    state = {"value": jnp.asarray(1.0)}
    manager = build_jax_checkpoint_manager(
        tmp_path / "checkpoints",
        enable_async_checkpointing=False,
    )
    save_jax_training_state(
        manager,
        1,
        state,
        metadata=jax_training_checkpoint_metadata(
            "ar_spectra",
            {"tokenizer": {"mz_bin_widths": [50.0, 25.0, 5.0, 1.0]}},
        ),
    )
    manager.close()

    reopened = build_jax_checkpoint_manager(
        tmp_path / "checkpoints",
        enable_async_checkpointing=False,
    )
    with pytest.raises(ValueError, match="training contract mismatch"):
        restore_jax_training_state(
            reopened,
            1,
            state,
            expected_metadata=jax_training_checkpoint_metadata(
                "ar_spectra",
                {"tokenizer": {"mz_bin_widths": [10.0, 1.0]}},
            ),
        )
    reopened.close()


def test_jax_checkpoint_restore_allows_explicit_config_changes(tmp_path):
    state = {"value": jnp.asarray(1.0)}
    manager = build_jax_checkpoint_manager(
        tmp_path / "checkpoints",
        enable_async_checkpointing=False,
    )
    save_jax_training_state(
        manager,
        1,
        state,
        metadata=jax_training_checkpoint_metadata(
            "pretrain",
            {"config": {"learning_rate": 1e-3, "model_dim": 8}},
        ),
    )
    manager.close()

    reopened = build_jax_checkpoint_manager(
        tmp_path / "checkpoints",
        enable_async_checkpointing=False,
    )
    restored = restore_jax_training_state(
        reopened,
        1,
        {"value": jnp.asarray(0.0)},
        expected_metadata=jax_training_checkpoint_metadata(
            "pretrain",
            {"config": {"learning_rate": 1e-4, "model_dim": 8}},
        ),
        allowed_config_keys=("learning_rate",),
    )
    reopened.close()

    assert float(restored["value"]) == 1.0


def test_jax_training_loop_saves_periodically_and_resumes(tmp_path):
    kwargs = _tiny_mae_kwargs()
    cfg = config_dict.ConfigDict(kwargs)
    cfg.seed = 5
    cfg.num_epochs = 2
    cfg.learning_rate = 1e-3
    cfg.jax_mesh_devices = "1"
    cfg.checkpoint_every_steps = 2
    cfg.log_every_n_steps = 0
    cfg.msg_probe_every_n_steps = -1
    batch = _tiny_numpy_batch()

    model = PeakSetJEPAJax(**kwargs)
    initialize_jax_model_from_torch_seed(cfg, model)
    datamodule = _FakeDataModule(batch, train_steps=2)
    manager = build_jax_checkpoint_manager(tmp_path / "checkpoints")
    metrics = _run_jax_training_loop(
        config=cfg,
        datamodule=datamodule,
        model=model,
        logger=MetricLogger(),
        total_steps=3,
        checkpoint_manager=manager,
        resume_step=None,
        checkpoint_metadata=CHECKPOINT_METADATA,
        metric_reduction="mean",
        enable_msg_probe=False,
    )
    assert metrics["run/final_global_step"] == 3.0
    assert sorted(manager.all_steps()) == [2, 3]
    manager.close()

    resumed_model = PeakSetJEPAJax(**kwargs)
    resumed_datamodule = _FakeDataModule(batch, train_steps=2)
    resumed_manager = build_jax_checkpoint_manager(tmp_path / "checkpoints")
    assert resumed_manager.latest_step() == 3
    resumed_metrics = _run_jax_training_loop(
        config=cfg,
        datamodule=resumed_datamodule,
        model=resumed_model,
        logger=MetricLogger(),
        total_steps=4,
        checkpoint_manager=resumed_manager,
        resume_step=resumed_manager.latest_step(),
        checkpoint_metadata=CHECKPOINT_METADATA,
        metric_reduction="mean",
        enable_msg_probe=False,
    )
    assert resumed_metrics["run/final_global_step"] == 4.0
    assert resumed_datamodule.loader_calls[-1] == (1, 1)
    assert resumed_manager.latest_step() == 4

    _graphdef, trainable_params, static_state, opt_state, _opt = (
        init_pure_optax_train_state(cfg, model, total_steps=4)
    )
    expected_trainable_params = jax.tree.map(np.asarray, trainable_params)
    restored = restore_jax_training_state(
        resumed_manager,
        3,
        {
            "trainable_params": trainable_params,
            "static_state": static_state,
            "opt_state": opt_state,
        },
        expected_metadata=CHECKPOINT_METADATA,
    )
    resumed_manager.close()
    for expected, actual in zip(
        jax.tree.leaves(expected_trainable_params),
        jax.tree.leaves(restored["trainable_params"]),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual))


def test_jax_training_loop_honors_wall_clock_budget(monkeypatch, tmp_path):
    from spectra_learning.training import pretrain_jax

    kwargs = _tiny_mae_kwargs()
    cfg = config_dict.ConfigDict(kwargs)
    cfg.seed = 5
    cfg.num_epochs = 1
    cfg.learning_rate = 1e-3
    cfg.jax_mesh_devices = "1"
    cfg.checkpoint_every_steps = 0
    cfg.log_every_n_steps = 0
    cfg.msg_probe_every_n_steps = -1
    cfg.max_duration_hours = 1.0
    cfg.jax_time_limit_check_every_steps = 1
    wall_times = iter((100.0, 100.0, 4000.0))
    monkeypatch.setattr(pretrain_jax, "_jax_wall_time", lambda: next(wall_times))
    batch = _tiny_numpy_batch()

    model = PeakSetJEPAJax(**kwargs)
    initialize_jax_model_from_torch_seed(cfg, model)
    datamodule = _FakeDataModule(batch, train_steps=3)
    manager = build_jax_checkpoint_manager(tmp_path / "checkpoints")
    metrics = _run_jax_training_loop(
        config=cfg,
        datamodule=datamodule,
        model=model,
        logger=MetricLogger(),
        total_steps=3,
        checkpoint_manager=manager,
        resume_step=None,
        checkpoint_metadata=CHECKPOINT_METADATA,
        metric_reduction="mean",
        enable_msg_probe=False,
    )

    assert metrics["run/final_global_step"] == 1.0
    assert metrics["run/stopped_for_time_limit"] == 1.0
    assert manager.latest_step() == 1
    manager.close()


def test_jax_training_loop_writes_emergency_checkpoint_before_waiting(tmp_path):
    kwargs = _tiny_mae_kwargs()
    cfg = config_dict.ConfigDict(kwargs)
    cfg.seed = 5
    cfg.num_epochs = 1
    cfg.learning_rate = 1e-3
    cfg.jax_mesh_devices = "1"
    cfg.checkpoint_every_steps = 0
    cfg.log_every_n_steps = 0
    cfg.msg_probe_every_n_steps = -1
    batch = _tiny_numpy_batch()

    model = PeakSetJEPAJax(**kwargs)
    initialize_jax_model_from_torch_seed(cfg, model)
    datamodule = _FakeDataModule(batch, train_steps=3)
    manager = build_jax_checkpoint_manager(tmp_path / "checkpoints")
    emergency = _EmergencyAfterFirstStep()

    metrics = _run_jax_training_loop(
        config=cfg,
        datamodule=datamodule,
        model=model,
        logger=MetricLogger(),
        total_steps=3,
        checkpoint_manager=manager,
        resume_step=None,
        checkpoint_metadata=CHECKPOINT_METADATA,
        metric_reduction="mean",
        enable_msg_probe=False,
        emergency_checkpoint=emergency,
    )

    assert emergency.waited is True
    assert metrics["run/final_global_step"] == 1.0
    assert metrics["run/stopped_for_termination"] == 1.0
    assert manager.latest_step() == 1
    manager.close()


def test_jax_process_bool_broadcast_uses_distributed_runtime(monkeypatch):
    from spectra_learning.training import pretrain_jax

    client = _FakeDistributedClient()
    monkeypatch.setattr(pretrain_jax.jax, "process_count", lambda: 2)
    monkeypatch.setattr(pretrain_jax.jax, "process_index", lambda: 0)
    monkeypatch.setattr(
        pretrain_jax.jax_distributed.global_state,
        "client",
        client,
    )

    assert _jax_process_bool_broadcast(True, key="stop_100") is True
    assert client.values == {"stop_100": "1"}

    monkeypatch.setattr(pretrain_jax.jax, "process_index", lambda: 1)
    assert _jax_process_bool_broadcast(False, key="stop_100") is True


def test_jax_training_loop_pure_optax_saves_and_resumes(tmp_path):
    kwargs = _tiny_mae_kwargs()
    cfg = config_dict.ConfigDict(kwargs)
    cfg.seed = 5
    cfg.num_epochs = 2
    cfg.learning_rate = 1e-3
    cfg.jax_mesh_devices = "1"
    cfg.gradient_accumulation_steps = 2
    cfg.checkpoint_every_steps = 2
    cfg.log_every_n_steps = 0
    cfg.msg_probe_every_n_steps = -1
    batch = _tiny_numpy_batch()

    model = PeakSetJEPAJax(**kwargs)
    initialize_jax_model_from_torch_seed(cfg, model)
    datamodule = _FakeDataModule(batch, train_steps=2, gradient_accumulation_steps=2)
    manager = build_jax_checkpoint_manager(tmp_path / "checkpoints")
    metrics = _run_jax_training_loop(
        config=cfg,
        datamodule=datamodule,
        model=model,
        logger=MetricLogger(),
        total_steps=3,
        checkpoint_manager=manager,
        resume_step=None,
        checkpoint_metadata=CHECKPOINT_METADATA,
        metric_reduction="mean",
        enable_msg_probe=False,
    )
    assert metrics["run/final_global_step"] == 3.0
    assert sorted(manager.all_steps()) == [2, 3]
    manager.close()

    resumed_model = PeakSetJEPAJax(**kwargs)
    resumed_datamodule = _FakeDataModule(
        batch,
        train_steps=2,
        gradient_accumulation_steps=2,
    )
    resumed_manager = build_jax_checkpoint_manager(tmp_path / "checkpoints")
    assert resumed_manager.latest_step() == 3
    resumed_metrics = _run_jax_training_loop(
        config=cfg,
        datamodule=resumed_datamodule,
        model=resumed_model,
        logger=MetricLogger(),
        total_steps=4,
        checkpoint_manager=resumed_manager,
        resume_step=resumed_manager.latest_step(),
        checkpoint_metadata=CHECKPOINT_METADATA,
        metric_reduction="mean",
        enable_msg_probe=False,
    )
    assert resumed_metrics["run/final_global_step"] == 4.0
    assert resumed_datamodule.loader_calls[-1] == (1, 1)
    assert resumed_manager.latest_step() == 4

    # The first-run model holds the step-3 params (the loop merges the pure
    # state back); the step-3 checkpoint must match them.
    _graphdef, trainable_params, static_state, opt_state, _opt = (
        init_pure_optax_train_state(cfg, model, total_steps=4)
    )
    expected_trainable_params = jax.tree.map(np.asarray, trainable_params)
    restored = restore_jax_training_state(
        resumed_manager,
        3,
        {
            "trainable_params": trainable_params,
            "static_state": static_state,
            "opt_state": opt_state,
        },
        expected_metadata=CHECKPOINT_METADATA,
    )
    resumed_manager.close()
    for expected, actual in zip(
        jax.tree.leaves(expected_trainable_params),
        jax.tree.leaves(restored["trainable_params"]),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual))


def test_jax_training_loop_switches_mask_stage_graphs_and_loaders(tmp_path):
    kwargs = {
        **_tiny_mae_kwargs(),
        "pairmixer_block_type": "fastmixer-dense",
    }
    cfg = config_dict.ConfigDict(kwargs)
    cfg.seed = 5
    cfg.num_epochs = 1
    cfg.learning_rate = 1e-3
    cfg.jax_mesh_devices = "1"
    cfg.gradient_accumulation_steps = 1
    cfg.checkpoint_every_steps = 0
    cfg.log_every_n_steps = 0
    cfg.msg_probe_every_n_steps = -1
    cfg.jepa_mask_strategy = ["random"]
    cfg.jepa_context_fraction = 0.35
    cfg.jepa_target_fraction = 0.50
    cfg.jepa_context_fraction_schedule = (0.35, 0.55, 0.75)
    cfg.jepa_target_fraction_schedule = (0.50, 0.30, 0.10)
    cfg.jepa_mask_schedule_step_fractions = (1 / 3, 2 / 3)
    cfg.gradient_accumulation_steps_schedule = (1, 2, 4)
    batch = _tiny_numpy_batch()

    model = PeakSetJEPAJax(PeakSetJEPASettings.from_config(cfg))
    datamodule = _FakeDataModule(batch, train_steps=3)
    manager = build_jax_checkpoint_manager(tmp_path / "checkpoints")
    metrics = _run_jax_training_loop(
        config=cfg,
        datamodule=datamodule,
        model=model,
        logger=MetricLogger(),
        total_steps=3,
        checkpoint_manager=manager,
        resume_step=None,
        checkpoint_metadata=CHECKPOINT_METADATA,
        metric_reduction="mean",
        enable_msg_probe=False,
    )
    manager.close()

    assert metrics["run/final_global_step"] == 3.0
    assert datamodule.mask_fraction_calls == [
        (0.35, 0.50),
        (0.55, 0.30),
        (0.75, 0.10),
    ]
    assert datamodule.loader_calls == [(0, 0), (0, 1), (0, 2)]
    assert datamodule.accumulation_calls == [1, 2, 4]


def test_jax_training_loop_logs_validation_and_online_probe(monkeypatch, tmp_path):
    from spectra_learning.training import pretrain_jax

    kwargs = _tiny_mae_kwargs()
    cfg = config_dict.ConfigDict(kwargs)
    cfg.seed = 5
    cfg.num_epochs = 1
    cfg.learning_rate = 1e-3
    cfg.jax_mesh_devices = "1"
    cfg.checkpoint_every_steps = 1000
    cfg.log_every_n_steps = 0
    cfg.val_every_n_steps = 2
    cfg.val_num_steps = 1
    cfg.msg_probe_every_n_steps = 2
    cfg.msg_probe_variants = ["mean"]
    batch = _tiny_numpy_batch()

    probe_calls = []

    def fake_run_msg_probe_jax(
        *,
        config,
        model,
        data_mesh=None,
        online_maccs_only=False,
    ):
        assert online_maccs_only is False
        probe_calls.append((config, model, data_mesh))
        pretrain_jax.time.sleep(0.01)
        return {"msg_probe/mean/test/auc_fluorine": 0.5}

    monkeypatch.setattr(pretrain_jax, "run_msg_probe_jax", fake_run_msg_probe_jax)

    model = PeakSetJEPAJax(**kwargs)
    initialize_jax_model_from_torch_seed(cfg, model)
    datamodule = _FakeDataModule(batch, train_steps=2)
    manager = build_jax_checkpoint_manager(tmp_path / "checkpoints")
    logger = _RecordingLogger()
    metrics = _run_jax_training_loop(
        config=cfg,
        datamodule=datamodule,
        model=model,
        logger=logger,
        total_steps=2,
        checkpoint_manager=manager,
        resume_step=None,
        checkpoint_metadata=CHECKPOINT_METADATA,
        metric_reduction="mean",
        enable_msg_probe=True,
    )
    manager.close()

    assert len(probe_calls) == 1
    assert probe_calls[0][2] is None
    assert metrics["run/final_global_step"] == 2.0
    assert metrics["run/wall_elapsed_seconds"] >= metrics["run/train_elapsed_seconds"]
    assert metrics["run/wall_samples_per_second"] <= metrics["run/samples_per_second"]
    assert metrics["run/measured_wall_elapsed_seconds"] >= metrics["run/measured_elapsed_seconds"]
    assert metrics["run/measured_wall_samples_per_second"] <= metrics["run/measured_samples_per_second"]
    assert metrics["run/msg_probe_seconds"] >= 0.01
    assert metrics["run/non_train_elapsed_seconds"] >= metrics["run/msg_probe_seconds"]
    assert metrics["run/measured_non_train_elapsed_seconds"] >= metrics["run/msg_probe_seconds"]
    assert np.isfinite(metrics["val/loss"])
    assert metrics["msg_probe/mean/test/auc_fluorine"] == 0.5
    assert any(step == 2 and "val/loss" in payload for payload, step in logger.logs)
    assert any(
        step == 2 and payload.get("msg_probe/mean/test/auc_fluorine") == 0.5
        for payload, step in logger.logs
    )


@pytest.mark.parametrize(
    ("early_stopping", "expected_splits"),
    (
        (
            False,
            [
                "massspec_train",
                "massspec_train",
                "massspec_val",
                "massspec_test",
            ],
        ),
        (
            True,
            [
                "massspec_train",
                "massspec_train",
                "massspec_val",
                "massspec_test",
            ],
        ),
    ),
    ids=("full-epochs", "early-stop"),
)
def test_run_msg_probe_jax_uses_jax_dataset_and_optimizer(
    monkeypatch,
    early_stopping,
    expected_splits,
):
    from spectra_learning.probes.massspec import msg_probe_jax

    def to_jax_batch(batch: dict[str, np.ndarray]) -> dict[str, object]:
        converted = {key: jnp.asarray(value) for key, value in batch.items()}
        converted["probe_valid_mol"] = jnp.asarray([True, True])
        converted["probe_fluorine"] = jnp.asarray([0.0, 1.0], dtype=jnp.float32)
        converted["probe_sulfur"] = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
        converted["probe_maccs"] = jnp.asarray(
            [[1, 0], [0, 1]],
            dtype=jnp.int32,
        )
        return converted

    class FakeProbeData:
        batch_size = 2
        info = {
            "massspec_train_size": 2,
            "massspec_val_size": 2,
            "massspec_test_size": 2,
            "probe_maccs_bits": 2,
        }

        def __init__(self, batch: dict[str, object]) -> None:
            self.batch = batch
            self.calls = []

        def build_dataset(self, split: str, **kwargs):
            self.calls.append((split, kwargs))
            assert kwargs["output_format"] == "jax"
            return [self.batch]

    class FakeMassSpecProbeData:
        @staticmethod
        def from_config(config, **kwargs):
            assert config is cfg
            assert kwargs == {
                "distributed_world_size": 1,
                "distributed_rank": 0,
                "distributed_local_rank": 0,
            }
            return fake_probe_data

    kwargs = _tiny_mae_kwargs()
    cfg = config_dict.ConfigDict(kwargs)
    cfg.seed = 9
    cfg.msg_probe_batch_size = 2
    cfg.msg_probe_num_epochs = 1
    cfg.msg_probe_learning_rate = 0.01
    cfg.msg_probe_weight_decay = 0.0
    cfg.msg_probe_warmup_steps = 0
    cfg.msg_probe_mlp_hidden_dim = 4
    cfg.msg_probe_variants = ["mean"]
    cfg.msg_probe_early_stopping = early_stopping
    cfg.msg_probe_num_repeats = 1
    cfg.peak_ordering = "mz"
    fake_probe_data = FakeProbeData(to_jax_batch(_tiny_numpy_batch()))
    monkeypatch.setattr(
        msg_probe_jax,
        "MassSpecProbeData",
        FakeMassSpecProbeData,
    )

    model = PeakSetJEPAJax(**kwargs)
    initialize_jax_model_from_torch_seed(cfg, model)

    metrics = msg_probe_jax.run_msg_probe_jax(config=cfg, model=model)

    assert "torch" not in msg_probe_jax.__dict__
    assert metrics["msg_probe/repeats"] == 1.0
    assert metrics["msg_probe/mean/epoch"] == 1.0
    assert "msg_probe/mean/val/auc_fluorine" in metrics
    assert "msg_probe/mean/test/auc_fluorine" in metrics
    assert "msg_probe/mean/test/pr_curve_fluorine" in metrics
    assert "msg_probe/mean/test/pr_curve_sulfur" in metrics
    assert [call[0] for call in fake_probe_data.calls] == expected_splits


def test_msg_probe_jax_masks_distributed_sampler_padding_rows():
    from spectra_learning.probes.massspec import msg_probe_jax

    def batch(value: float) -> dict[str, np.ndarray]:
        return {
            "peak_mz": np.asarray([[value, 0.0, 0.0]], dtype=np.float32),
            "peak_intensity": np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            "peak_valid_mask": np.asarray([[True, False, False]], dtype=bool),
            "probe_valid_mol": np.asarray([True], dtype=bool),
            "probe_fluorine": np.asarray([0.0], dtype=np.float32),
            "probe_sulfur": np.asarray([1.0], dtype=np.float32),
            "probe_maccs": np.asarray([[1, 0]], dtype=np.int32),
        }

    class FakeProbeData:
        batch_size = 4

        def __init__(self, size: int, batches: list[dict[str, np.ndarray]]) -> None:
            self.info = {"massspec_test_size": size}
            self.batches = batches
            self.calls = []

        def build_dataset(self, split: str, **kwargs):
            self.calls.append((split, kwargs))
            return self.batches

    probe_data = FakeProbeData(5, [batch(1.0), batch(2.0)])
    batches = list(
        msg_probe_jax.iter_massspec_probe_jax(
            probe_data=probe_data,
            split="massspec_test",
            seed=7,
            peak_ordering="mz",
            drop_remainder=False,
            distributed_world_size=4,
            distributed_rank=1,
        )
    )

    assert probe_data.calls[0][1]["pad_distributed"] is True
    assert [np.asarray(item["probe_valid_mol"]).tolist() for item in batches] == [
        [True],
        [False],
    ]

    empty_rank_data = FakeProbeData(3, [batch(3.0)])
    empty_rank_batches = list(
        msg_probe_jax.iter_massspec_probe_jax(
            probe_data=empty_rank_data,
            split="massspec_test",
            seed=7,
            peak_ordering="mz",
            drop_remainder=False,
            distributed_world_size=4,
            distributed_rank=3,
        )
    )

    assert len(empty_rank_batches) == 1
    assert np.asarray(empty_rank_batches[0]["probe_valid_mol"]).tolist() == [False]


def test_msg_probe_jax_distributed_helpers_shard_steps_and_merge_states(monkeypatch):
    from spectra_learning.probes.massspec import msg_probe_jax

    probe_data = SimpleNamespace(
        batch_size=4,
        info={"massspec_train_size": 5},
    )
    assert (
        msg_probe_jax.probe_steps_per_epoch_jax(
            probe_data,
            split="massspec_train",
            drop_remainder=False,
            distributed_world_size=2,
        )
        == 2
    )

    task_spec = msg_probe_jax.MsgProbeTaskSpec(
        regression_tasks=(),
        binary_tasks=("fluorine",),
        maccs_bits=2,
        regression_means={},
        regression_stds={},
        fingerprint_task="maccs",
    )
    local_states = {
        "mean": {
            "count": 1,
            "predictions": {
                "fluorine": [np.asarray([0.25], dtype=np.float32)],
                "maccs": [np.asarray([[0.1, 0.9]], dtype=np.float32)],
            },
            "targets": {
                "fluorine": [np.asarray([0.0], dtype=np.float32)],
                "maccs": [np.asarray([[0, 1]], dtype=np.int32)],
            },
        },
    }
    other_states = {
        "mean": {
            "count": 1,
            "predictions": {
                "fluorine": [np.asarray([0.75], dtype=np.float32)],
                "maccs": [np.asarray([[0.8, 0.2]], dtype=np.float32)],
            },
            "targets": {
                "fluorine": [np.asarray([1.0], dtype=np.float32)],
                "maccs": [np.asarray([[1, 0]], dtype=np.int32)],
            },
        },
    }

    monkeypatch.setattr(msg_probe_jax.jax, "process_count", lambda: 2)
    monkeypatch.setattr(
        msg_probe_jax,
        "_all_gather_object_jax",
        lambda value: [value, other_states],
    )

    gathered = msg_probe_jax._gather_variant_states_jax(local_states, task_spec)

    assert gathered["mean"]["count"] == 2
    np.testing.assert_array_equal(
        np.concatenate(gathered["mean"]["targets"]["fluorine"]),
        np.asarray([0.0, 1.0], dtype=np.float32),
    )


def test_msg_probe_jax_mean_tree_averages_host_local_leaves(monkeypatch):
    from spectra_learning.probes.massspec import msg_probe_jax

    calls = []

    def fake_process_allgather(value, *, tiled=False):
        calls.append(tiled)
        return np.stack(
            [
                np.asarray(value),
                np.asarray(value) + 2.0,
            ],
            axis=0,
        )

    monkeypatch.setattr(msg_probe_jax.jax, "process_count", lambda: 2)
    monkeypatch.setattr(
        msg_probe_jax.multihost_utils,
        "process_allgather",
        fake_process_allgather,
    )

    averaged = msg_probe_jax._mean_tree_across_processes(
        {"weight": jnp.asarray([1.0, 3.0])}
    )

    assert calls == [False]
    np.testing.assert_allclose(np.asarray(averaged["weight"]), np.asarray([2.0, 4.0]))


def test_msg_probe_jax_mean_tree_keeps_global_sharded_leaf_shape(monkeypatch):
    from spectra_learning.probes.massspec import msg_probe_jax

    calls = []
    local_leaf = jnp.asarray([[1.0, 3.0], [5.0, 7.0]])
    gathered_leaf = np.asarray(local_leaf)

    def fake_process_allgather(value, *, tiled=False):
        calls.append(tiled)
        assert value is local_leaf
        return gathered_leaf

    monkeypatch.setattr(msg_probe_jax.jax, "process_count", lambda: 2)
    monkeypatch.setattr(
        msg_probe_jax,
        "_is_global_non_fully_addressable_array",
        lambda value: value is local_leaf,
    )
    monkeypatch.setattr(
        msg_probe_jax.multihost_utils,
        "process_allgather",
        fake_process_allgather,
    )

    averaged = msg_probe_jax._mean_tree_across_processes({"weight": local_leaf})

    assert calls == [True]
    np.testing.assert_allclose(np.asarray(averaged["weight"]), gathered_leaf)


def test_msg_probe_jax_single_pair_covariance_params_are_differentiable():
    from spectra_learning.probes.massspec import msg_probe_jax

    cfg = config_dict.ConfigDict(
        {
            "model_dim": 4,
            "pairmixer_pair_dim": 6,
            "covariance_pooling_dim": 3,
            "msg_probe_mlp_hidden_dim": 4,
            "msg_probe_mlp_num_layers": 1,
            "msg_probe_single_pair_covariance_include_diagonal": False,
        }
    )
    task_spec = msg_probe_jax.MsgProbeTaskSpec(
        regression_tasks=("mol_weight",),
        binary_tasks=("fluorine",),
        maccs_bits=2,
        regression_means={"mol_weight": 100.0},
        regression_stds={"mol_weight": 10.0},
        fingerprint_task="maccs",
        single_pair_covariance_include_diagonal=False,
    )

    params = msg_probe_jax._init_probe_params(
        jax.random.PRNGKey(0),
        variant="single_pair_covariance",
        config=cfg,
        task_spec=task_spec,
    )

    assert jax.tree.leaves(params)
    for leaf in jax.tree.leaves(params):
        assert jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.inexact)
