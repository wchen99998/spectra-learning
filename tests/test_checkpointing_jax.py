from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from ml_collections import config_dict

from spectra_learning.data.gems.collate import GemsBatchCollator
from spectra_learning.models.model_jax import PeakSetJEPAJax
from spectra_learning.training.checkpointing_jax import (
    build_jax_checkpoint_manager,
    restore_jax_training_state,
    save_jax_training_state,
)
from spectra_learning.training.logging import MetricLogger
from spectra_learning.training.pretrain_jax import (
    _run_jax_training_loop,
    build_jax_optimizer,
    init_pure_optax_train_state,
    initialize_jax_model_from_torch_seed,
)


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
        "pairmixer_use_pair_bias_attention": False,
        "predictor_dropout": 0.0,
        "target_projector_dim": -1,
        "jepa_mae_mz_bin_size": 100.0,
        "jepa_mae_intensity_bin_size": 0.5,
    }


def _zeros_like_with_sharding(value: jax.Array) -> jax.Array:
    return jax.device_put(jnp.zeros_like(value), value.sharding)


def _tiny_torch_batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(123)

    def sample(mz, intensity, precursor_mz):
        spectra = torch.zeros(2, 128, dtype=torch.float32)
        spectra[0, : len(mz)] = torch.tensor(mz, dtype=torch.float32)
        spectra[1, : len(intensity)] = torch.tensor(intensity, dtype=torch.float32)
        return {
            "spectra": spectra,
            "precursor_mz_raw": torch.tensor(precursor_mz, dtype=torch.float32),
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
    )
    return collator(
        [
            sample([100.0, 125.0, 150.0], [1.0, 0.8, 0.4], 500.0),
            sample([220.0, 240.0, 300.0], [0.9, 0.3, 0.2], 620.0),
        ]
    )


class _FakeDataModule:
    def __init__(
        self,
        batch: dict[str, torch.Tensor],
        train_steps: int,
        gradient_accumulation_steps: int = 1,
        val_batch: dict[str, torch.Tensor] | None = None,
    ) -> None:
        self._batch = batch
        self._val_batch = batch if val_batch is None else val_batch
        self._accum = gradient_accumulation_steps
        self.train_steps = train_steps
        self.global_batch_size = int(batch["peak_mz"].shape[0])
        self.batch_size = self.global_batch_size
        self.loader_calls: list[tuple[int, int]] = []

    def train_loader_for_epoch(self, epoch: int, start_batch: int = 0):
        self.loader_calls.append((epoch, start_batch))
        return [self._batch] * ((self.train_steps - start_batch) * self._accum)

    @property
    def val_loader(self):
        return [self._val_batch]


class _RecordingLogger(MetricLogger):
    def __init__(self) -> None:
        self.logs = []

    def log_metrics(self, metrics, step=None) -> None:
        self.logs.append((dict(metrics), step))


def test_jax_checkpoint_roundtrip_preserves_values_and_sharding(tmp_path):
    mesh = Mesh(np.asarray(jax.devices()), ("data",))
    replicated = NamedSharding(mesh, P())
    params = {
        "weight": jax.device_put(jnp.arange(8.0).reshape(2, 4), replicated),
        "blocks": {0: jnp.full((3,), 2.5)},
    }
    opt_state = optax.adamw(1e-3).init(params)
    state = {"trainable_params": params, "opt_state": opt_state}

    manager = build_jax_checkpoint_manager(tmp_path / "checkpoints", max_to_keep=2)
    save_jax_training_state(manager, 10, state)
    manager.close()

    reopened = build_jax_checkpoint_manager(tmp_path / "checkpoints", max_to_keep=2)
    assert reopened.latest_step() == 10
    template = jax.tree.map(_zeros_like_with_sharding, state)
    restored = restore_jax_training_state(reopened, 10, template)
    reopened.close()

    assert restored["trainable_params"]["weight"].sharding == replicated
    expected_leaves = jax.tree.leaves(state)
    restored_leaves = jax.tree.leaves(restored)
    for expected, actual in zip(expected_leaves, restored_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual))


def test_jax_checkpoint_roundtrip_restores_model_and_optimizer_state(tmp_path):
    kwargs = _tiny_mae_kwargs()
    cfg = config_dict.ConfigDict(kwargs)
    cfg.learning_rate = 1e-3
    cfg.weight_decay = 0.0
    cfg.seed = 11

    source_model = PeakSetJEPAJax(**kwargs)
    initialize_jax_model_from_torch_seed(cfg, source_model)
    source_optimizer = build_jax_optimizer(cfg, source_model)
    bumped = jax.tree.map(
        lambda value: value + jnp.ones((), dtype=value.dtype),
        nnx.as_pure(nnx.state(source_optimizer)),
    )
    nnx.update(source_optimizer, bumped)

    manager = build_jax_checkpoint_manager(tmp_path / "checkpoints")
    save_jax_training_state(
        manager,
        3,
        {
            "model": nnx.as_pure(nnx.state(source_model)),
            "optimizer": nnx.as_pure(nnx.state(source_optimizer)),
        },
    )
    manager.close()

    cfg.seed = 99
    target_model = PeakSetJEPAJax(**kwargs)
    initialize_jax_model_from_torch_seed(cfg, target_model)
    target_optimizer = build_jax_optimizer(cfg, target_model)
    reopened = build_jax_checkpoint_manager(tmp_path / "checkpoints")
    assert reopened.latest_step() == 3
    restored = restore_jax_training_state(
        reopened,
        3,
        {
            "model": nnx.as_pure(nnx.state(target_model)),
            "optimizer": nnx.as_pure(nnx.state(target_optimizer)),
        },
    )
    reopened.close()
    nnx.update(target_model, restored["model"])
    nnx.update(target_optimizer, restored["optimizer"])

    source_leaves = jax.tree.leaves(nnx.as_pure(nnx.state(source_model)))
    target_leaves = jax.tree.leaves(nnx.as_pure(nnx.state(target_model)))
    for expected, actual in zip(source_leaves, target_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual))
    source_opt_leaves = jax.tree.leaves(nnx.as_pure(nnx.state(source_optimizer)))
    target_opt_leaves = jax.tree.leaves(nnx.as_pure(nnx.state(target_optimizer)))
    for expected, actual in zip(source_opt_leaves, target_opt_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual))


def test_jax_training_loop_saves_periodically_and_resumes(tmp_path):
    kwargs = _tiny_mae_kwargs()
    cfg = config_dict.ConfigDict(kwargs)
    cfg.seed = 5
    cfg.num_epochs = 2
    cfg.learning_rate = 1e-3
    cfg.checkpoint_every_steps = 2
    cfg.log_every_n_steps = 0
    cfg.msg_probe_every_n_steps = -1
    batch = _tiny_torch_batch()

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
    )
    assert resumed_metrics["run/final_global_step"] == 4.0
    assert resumed_datamodule.loader_calls[-1] == (1, 1)
    assert resumed_manager.latest_step() == 4

    _graphdef, trainable_params, static_state, opt_state, _opt = (
        init_pure_optax_train_state(cfg, model, total_steps=4)
    )
    restored = restore_jax_training_state(
        resumed_manager,
        3,
        {
            "trainable_params": trainable_params,
            "static_state": static_state,
            "opt_state": opt_state,
        },
    )
    resumed_manager.close()
    for expected, actual in zip(
        jax.tree.leaves(trainable_params),
        jax.tree.leaves(restored["trainable_params"]),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual))


def test_jax_training_loop_pure_optax_saves_and_resumes(tmp_path):
    kwargs = _tiny_mae_kwargs()
    cfg = config_dict.ConfigDict(kwargs)
    cfg.seed = 5
    cfg.num_epochs = 2
    cfg.learning_rate = 1e-3
    cfg.gradient_accumulation_steps = 2
    cfg.checkpoint_every_steps = 2
    cfg.log_every_n_steps = 0
    cfg.msg_probe_every_n_steps = -1
    batch = _tiny_torch_batch()

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
    )
    assert resumed_metrics["run/final_global_step"] == 4.0
    assert resumed_datamodule.loader_calls[-1] == (1, 1)
    assert resumed_manager.latest_step() == 4

    # The first-run model holds the step-3 params (the loop merges the pure
    # state back); the step-3 checkpoint must match them.
    _graphdef, trainable_params, static_state, opt_state, _opt = (
        init_pure_optax_train_state(cfg, model, total_steps=4)
    )
    restored = restore_jax_training_state(
        resumed_manager,
        3,
        {
            "trainable_params": trainable_params,
            "static_state": static_state,
            "opt_state": opt_state,
        },
    )
    resumed_manager.close()
    for expected, actual in zip(
        jax.tree.leaves(trainable_params),
        jax.tree.leaves(restored["trainable_params"]),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual))


def test_jax_training_loop_logs_validation_and_online_probe(monkeypatch, tmp_path):
    from spectra_learning.training import pretrain_jax

    kwargs = _tiny_mae_kwargs()
    cfg = config_dict.ConfigDict(kwargs)
    cfg.seed = 5
    cfg.num_epochs = 1
    cfg.learning_rate = 1e-3
    cfg.checkpoint_every_steps = 1000
    cfg.log_every_n_steps = 0
    cfg.jax_precompile_train_steps = False
    cfg.val_every_n_steps = 2
    cfg.val_num_steps = 1
    cfg.msg_probe_every_n_steps = 2
    cfg.msg_probe_variants = ["mean"]
    batch = _tiny_torch_batch()

    probe_calls = []

    def fake_run_msg_probe_jax(*, config, model):
        probe_calls.append((config, model))
        return {"msg_probe/mean/test/auc_maccs_mean": 0.5}

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
    )
    manager.close()

    assert len(probe_calls) == 1
    assert metrics["run/final_global_step"] == 2.0
    assert np.isfinite(metrics["val/loss"])
    assert metrics["msg_probe/mean/test/auc_maccs_mean"] == 0.5
    assert any(step == 2 and "val/loss" in payload for payload, step in logger.logs)
    assert any(
        step == 2 and payload.get("msg_probe/mean/test/auc_maccs_mean") == 0.5
        for payload, step in logger.logs
    )


def test_run_msg_probe_jax_uses_jax_dataset_and_optimizer(monkeypatch):
    from spectra_learning.probes.massspec import msg_probe_jax

    def to_jax_batch(batch: dict[str, torch.Tensor]) -> dict[str, object]:
        converted = {
            key: jnp.asarray(value.detach().cpu().numpy())
            for key, value in batch.items()
        }
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
            "massspec_mcebio_test_size": 2,
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
        def from_config(config):
            assert config is cfg
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
    cfg.msg_probe_early_stopping = False
    cfg.msg_probe_num_repeats = 1
    cfg.peak_ordering = "mz"
    fake_probe_data = FakeProbeData(to_jax_batch(_tiny_torch_batch()))
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
    assert "msg_probe/mean/test/auc_fluorine" in metrics
    assert "msg_probe/mean/test/pr_curve_fluorine" in metrics
    assert "msg_probe/mean/test/pr_curve_sulfur" in metrics
    assert "msg_probe/mean/mcebio_sulfur_test/auc_sulfur" in metrics
    assert "msg_probe/mean/mcebio_sulfur_test/pr_curve_sulfur" in metrics
    assert "msg_probe/mean/mcebio_sulfur_test/auc_fluorine" not in metrics
    assert "msg_probe/mean/mcebio_sulfur_test/pr_curve_fluorine" not in metrics
    assert [call[0] for call in fake_probe_data.calls] == [
        "massspec_train",
        "massspec_val",
        "massspec_test",
        "massspec_train",
        "massspec_test",
        "massspec_mcebio_test",
    ]
