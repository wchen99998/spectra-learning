import json
import sys
import tempfile
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import fsspec
import numpy as np
import pytest
import torch
from ml_collections import config_dict

from spectra_learning.config import config_to_dict
from spectra_learning.models.factory import build_model_from_config
from spectra_learning.models.model import PeakSetJEPA
from spectra_learning.models.pooling import CovariancePool
from spectra_learning.training.checkpointing import (
    AsyncCheckpointWriter,
    covariance_pooler_checkpoint_path,
    is_training_checkpoint_path,
    latest_ckpt_path,
    load_torch_checkpoint,
    load_grad_scaler_state,
    load_resume_covariance_pooler_state,
    load_resume_model_state,
    save_checkpoint,
)
from spectra_learning.training import pretrain
from spectra_learning.training import checkpointing as checkpointing_module
from spectra_learning.probes.massspec import checkpoint_probe
from spectra_learning.probes.massspec.pr_curves import PrecisionRecallCurve
from spectra_learning.training.optimization import (
    build_optimizers,
    is_weight_decay_target,
)
from spectra_learning.training.schedules import learning_rate_at_step
from spectra_learning.training.logging import (
    WandbMetricLogger,
    _build_wandb_init_kwargs,
    _serialise_metrics,
    log_msg_probe_metrics,
)
from spectra_learning.training.runtime import (
    build_grad_scaler,
    cumulative_training_flops,
    estimate_training_flops_per_optimizer_step,
    parse_autocast_dtype,
)


def _clear_memory_fs(prefix: str) -> None:
    fs = fsspec.filesystem("memory")
    if fs.exists(prefix):
        fs.rm(prefix, recursive=True)


def _small_model(**overrides) -> PeakSetJEPA:
    kwargs = dict(
        model_dim=64,
        encoder_num_layers=2,
        encoder_num_heads=4,
        attention_mlp_multiple=2.0,
        feature_mlp_hidden_dim=32,
        masked_token_loss_weight=1.0,
        masked_latent_predictor_num_layers=1,
        jepa_num_target_blocks=1,
        num_peaks=8,
    )
    kwargs.update(overrides)
    return PeakSetJEPA(**kwargs)


def _optimizer_param_ids(optimizer: torch.optim.Optimizer) -> set[int]:
    return {
        id(param)
        for group in optimizer.param_groups
        for param in group["params"]
    }


def _optimizer_config(**overrides) -> config_dict.ConfigDict:
    cfg = config_dict.ConfigDict()
    cfg.learning_rate = 1e-3
    cfg.min_learning_rate = 1e-4
    cfg.warmup_steps = 0
    cfg.b2 = 0.999
    cfg.weight_decay = 0.01
    cfg.optimizer = "adamw"
    cfg.optimizer_fused = False
    cfg.update(overrides)
    return cfg


def test_msg_probe_interval_zero_runs_at_final_training_step():
    cfg = config_dict.ConfigDict()
    cfg.msg_probe_every_n_steps = 0
    cfg.num_epochs = 2

    datamodule = SimpleNamespace(train_steps=10)

    assert pretrain.msg_probe_interval(cfg, datamodule, total_steps=23) == 23


def test_msg_probe_interval_negative_disables_probe():
    cfg = config_dict.ConfigDict()
    cfg.msg_probe_every_n_steps = -1
    cfg.num_epochs = 2

    datamodule = SimpleNamespace(train_steps=10)

    assert pretrain.msg_probe_interval(cfg, datamodule, total_steps=23) == -1


class _FakePbar:
    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        self.postfix = None

    def set_postfix(self, **kwargs) -> None:
        self.postfix = kwargs

    def update(self, steps: int) -> None:
        del steps

    def close(self) -> None:
        pass


class _FakeLogger:
    def __init__(self) -> None:
        self.logs = []
        self.finished = False

    def log_metrics(self, metrics, step=None) -> None:
        self.logs.append((dict(metrics), step))

    def finish(self) -> None:
        self.finished = True


class _FakeCheckpointManager:
    def latest_step(self) -> int | None:
        return None

    def close(self) -> None:
        pass


class _CompileRecorder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.compile_kwargs = None

    def compile(self, **kwargs) -> None:
        self.compile_kwargs = kwargs


def test_compile_forward_disables_shape_padding_for_max_autotune():
    original = pretrain.inductor_config.shape_padding
    try:
        pretrain.inductor_config.shape_padding = True
        model = _CompileRecorder()

        pretrain.compile_forward(model, {"compile_mode": "max-autotune"})

        assert pretrain.inductor_config.shape_padding is False
        assert model.compile_kwargs == {
            "mode": "max-autotune",
            "fullgraph": False,
        }
    finally:
        pretrain.inductor_config.shape_padding = original


def test_compile_forward_uses_no_cudagraphs_for_max_autotune_with_accumulation():
    original = pretrain.inductor_config.shape_padding
    try:
        pretrain.inductor_config.shape_padding = True
        model = _CompileRecorder()

        pretrain.compile_forward(
            model,
            {
                "compile_mode": "max-autotune",
                "gradient_accumulation_steps": 2,
            },
        )

        assert pretrain.inductor_config.shape_padding is False
        assert model.compile_kwargs == {
            "mode": "max-autotune-no-cudagraphs",
            "fullgraph": False,
        }
    finally:
        pretrain.inductor_config.shape_padding = original


def test_compile_forward_keeps_explicit_no_cudagraphs_mode():
    model = _CompileRecorder()

    pretrain.compile_forward(
        model,
        {
            "compile_mode": "max-autotune-no-cudagraphs",
            "gradient_accumulation_steps": 2,
        },
    )

    assert model.compile_kwargs == {
        "mode": "max-autotune-no-cudagraphs",
        "fullgraph": False,
    }


def test_compile_forward_enables_shape_padding_for_reduce_overhead():
    original = pretrain.inductor_config.shape_padding
    try:
        pretrain.inductor_config.shape_padding = False
        model = _CompileRecorder()

        pretrain.compile_forward(model, {"compile_mode": "reduce-overhead"})

        assert pretrain.inductor_config.shape_padding is True
        assert model.compile_kwargs == {
            "mode": "reduce-overhead",
            "fullgraph": False,
        }
    finally:
        pretrain.inductor_config.shape_padding = original


def test_jax_device_backend_uses_native_jax():
    assert pretrain._use_jax_backend({"device_backend": "jax"})
    assert not pretrain._use_jax_backend({"device_backend": "auto"})
    assert not pretrain._use_jax_backend({"device_backend": "torch"})


def test_jax_backend_applies_tpu_flags_before_dispatch(monkeypatch, tmp_path):
    cfg = config_dict.ConfigDict()
    cfg.device_backend = "jax"
    calls = []

    def fake_configure_jax_tpu_xla_flags():
        calls.append("flags")

    def fake_train_and_evaluate_jax(config, workdir):
        assert config is cfg
        assert workdir == tmp_path
        calls.append("train")
        return {"run/device_backend": "jax"}

    fake_pretrain_jax = ModuleType("spectra_learning.training.pretrain_jax")
    fake_pretrain_jax.train_and_evaluate_jax = fake_train_and_evaluate_jax
    monkeypatch.setattr(
        pretrain,
        "configure_jax_tpu_xla_flags",
        fake_configure_jax_tpu_xla_flags,
    )
    monkeypatch.setitem(
        sys.modules,
        "spectra_learning.training.pretrain_jax",
        fake_pretrain_jax,
    )

    assert pretrain.train_and_evaluate(cfg, tmp_path) == {"run/device_backend": "jax"}
    assert calls == ["flags", "train"]


def test_jax_train_metrics_logging_materializes_non_main_without_logging(monkeypatch):
    from spectra_learning.training import pretrain_jax

    cfg = config_dict.ConfigDict()
    logger = _FakeLogger()
    pbar = _FakePbar()
    device_get_calls = 0

    def fake_device_get(value):
        nonlocal device_get_calls
        device_get_calls += 1
        return value

    monkeypatch.setattr(pretrain_jax.jax, "process_index", lambda: 1)
    monkeypatch.setattr(pretrain_jax.jax, "device_get", fake_device_get)

    staged = pretrain_jax._StagedJaxTrainMetrics(
        metrics={"loss": pretrain_jax.np.asarray(1.25)},
        epoch=0,
        global_step=10,
        total_steps=100,
    )
    pretrain_jax._log_jax_train_metrics(
        cfg,
        logger,
        pbar,
        staged,
    )

    assert device_get_calls == 1
    assert logger.logs == []
    assert pbar.postfix is None


def test_jax_train_metrics_staging_copies_to_host_async(monkeypatch):
    from spectra_learning.training import pretrain_jax

    metrics = {"loss": pretrain_jax.np.asarray(1.25)}
    copy_calls = []

    def fake_copy_to_host_async(value):
        copy_calls.append(value)
        return value

    monkeypatch.setattr(
        pretrain_jax.jax,
        "copy_to_host_async",
        fake_copy_to_host_async,
    )

    staged = pretrain_jax._stage_jax_train_metrics(
        metrics,
        epoch=2,
        global_step=10,
        total_steps=100,
    )

    assert copy_calls == [metrics]
    assert staged.metrics is metrics
    assert staged.epoch == 2
    assert staged.global_step == 10
    assert staged.total_steps == 100


def test_jax_train_metrics_logging_materializes_once_on_main(monkeypatch):
    from spectra_learning.training import pretrain_jax

    cfg = config_dict.ConfigDict()
    cfg.learning_rate = 0.003
    cfg.min_learning_rate = 0.0003
    cfg.warmup_steps = 20
    logger = _FakeLogger()
    pbar = _FakePbar()
    device_get_calls = 0

    def fake_device_get(value):
        nonlocal device_get_calls
        device_get_calls += 1
        return value

    monkeypatch.setattr(pretrain_jax.jax, "process_index", lambda: 0)
    monkeypatch.setattr(pretrain_jax.jax, "device_get", fake_device_get)

    staged = pretrain_jax._StagedJaxTrainMetrics(
        metrics={
            "loss": pretrain_jax.np.asarray(1.25),
            "ema_teacher_momentum": pretrain_jax.np.asarray(0.99),
        },
        epoch=2,
        global_step=10,
        total_steps=100,
    )
    pretrain_jax._log_jax_train_metrics(
        cfg,
        logger,
        pbar,
        staged,
    )

    expected_lr = learning_rate_at_step(
        10,
        base_lr=0.003,
        total_steps=100,
        warmup_steps=20,
        min_learning_rate=0.0003,
    )
    assert device_get_calls == 1
    assert pbar.postfix == {"loss": "1.2500", "step": 10}
    assert logger.logs[0][1] == 10
    assert logger.logs[0][0]["train/loss"] == pytest.approx(1.25)
    assert logger.logs[0][0]["train/ema_teacher_momentum"] == pytest.approx(0.99)
    assert logger.logs[0][0]["train/learning_rate"] == pytest.approx(expected_lr)
    assert logger.logs[0][0]["epoch"] == 2.0
    assert logger.logs[0][0]["global_step"] == 10.0


def test_distributed_jax_msg_probe_runs_on_all_processes_and_returns_only_main(
    monkeypatch,
):
    from spectra_learning.training import pretrain_jax

    current_rank = {"value": 0}
    updated_models = []
    probe_ranks = []
    barriers = []
    mesh_contexts = []

    class FakeMeshContext:
        def __init__(self, mesh):
            self.mesh = mesh

        def __enter__(self):
            mesh_contexts.append((current_rank["value"], "enter", self.mesh))

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            mesh_contexts.append((current_rank["value"], "exit", self.mesh))

    def fake_run_and_log(**kwargs):
        del kwargs
        probe_ranks.append(current_rank["value"])
        return {"msg_probe/mean/test/auc_maccs_mean": 0.75 + current_rank["value"]}

    monkeypatch.setattr(
        pretrain_jax.jax,
        "process_index",
        lambda: current_rank["value"],
    )
    monkeypatch.setattr(
        pretrain_jax.jax,
        "block_until_ready",
        lambda value: pytest.fail("probe wrapper should defer explicit JAX sync"),
    )
    monkeypatch.setattr(
        pretrain_jax.nnx,
        "update",
        lambda model, params: updated_models.append((model, params)),
    )
    monkeypatch.setattr(
        pretrain_jax,
        "run_and_log_msg_probe_jax",
        fake_run_and_log,
    )
    monkeypatch.setattr(
        pretrain_jax.multihost_utils,
        "sync_global_devices",
        lambda name: barriers.append((current_rank["value"], name)),
    )
    monkeypatch.setattr(
        pretrain_jax.jax,
        "set_mesh",
        lambda mesh: FakeMeshContext(mesh),
    )

    outputs = []
    for rank in (0, 1):
        current_rank["value"] = rank
        outputs.append(
            pretrain_jax._run_distributed_msg_probe_jax(
                config=config_dict.ConfigDict(),
                model="model",
                logger=_FakeLogger(),
                variants=("mean",),
                global_step=100,
                trainable_params=f"params-{rank}",
                data_mesh="mesh",
            )
        )

    assert probe_ranks == [0, 1]
    assert updated_models == [("model", "params-0"), ("model", "params-1")]
    assert barriers == [
        (0, "spectra_learning_jax_msg_probe_100"),
        (1, "spectra_learning_jax_msg_probe_100"),
    ]
    assert mesh_contexts == [
        (0, "enter", "mesh"),
        (0, "exit", "mesh"),
        (1, "enter", "mesh"),
        (1, "exit", "mesh"),
    ]
    assert outputs == [
        {"msg_probe/mean/test/auc_maccs_mean": 0.75},
        {},
    ]


def test_run_and_log_jax_msg_probe_skips_logging_on_non_main(monkeypatch):
    from spectra_learning.training import pretrain_jax

    log_calls = []

    monkeypatch.setattr(pretrain_jax.jax, "process_index", lambda: 1)
    monkeypatch.setattr(
        pretrain_jax,
        "run_msg_probe_jax",
        lambda **kwargs: {"msg_probe/mean/test/auc_maccs_mean": 0.75},
    )
    monkeypatch.setattr(
        pretrain_jax,
        "log_msg_probe_metrics",
        lambda *args, **kwargs: log_calls.append((args, kwargs)),
    )

    metrics = pretrain_jax.run_and_log_msg_probe_jax(
        config=config_dict.ConfigDict({"enable_wandb": True}),
        model="model",
        logger=_FakeLogger(),
        variants=("mean",),
        global_step=100,
    )

    assert metrics == {"msg_probe/mean/test/auc_maccs_mean": 0.75}
    assert log_calls == []


def test_jax_validation_loss_uses_augmented_validation_loader(monkeypatch):
    from spectra_learning.training import pretrain_jax

    calls = []

    class FakeDataModule:
        def val_loader_for_eval(self, *, augment: bool):
            calls.append(augment)
            return [
                {"peak_mz": pretrain_jax.np.asarray([2.0])},
                {"peak_mz": pretrain_jax.np.asarray([4.0])},
            ]

    def fake_eval_step(trainable_params, static_state, batch):
        del trainable_params, static_state
        return {"loss": batch["peak_mz"].mean()}

    monkeypatch.setattr(pretrain_jax.jax, "device_get", lambda value: value)

    metrics = pretrain_jax._evaluate_jax_validation_loss(
        datamodule=FakeDataModule(),
        trainable_params={},
        static_state={},
        eval_step=fake_eval_step,
        max_steps=2,
        use_sharded_step=False,
        data_mesh=None,
        metric_reduction="mean",
    )

    assert calls == [True]
    assert metrics["val/loss"] == pytest.approx(3.0)


def test_jax_optax_transform_uses_learning_rate_schedule(monkeypatch):
    from spectra_learning.training import pretrain_jax

    calls = []
    sentinel = object()

    def fake_adamw(**kwargs):
        calls.append(kwargs)
        return sentinel

    cfg = config_dict.ConfigDict()
    cfg.learning_rate = 0.004
    cfg.min_learning_rate = 0.0004
    cfg.warmup_steps = 20
    cfg.b1 = 0.8
    cfg.b2 = 0.95
    cfg.weight_decay = 0.1

    monkeypatch.setattr(pretrain_jax.optax, "adamw", fake_adamw)

    transform = pretrain_jax.build_jax_optax_transform(cfg, total_steps=100)

    assert transform is sentinel
    assert callable(calls[0]["learning_rate"])
    assert callable(calls[0]["mask"])
    assert calls[0]["b1"] == pytest.approx(0.8)
    assert calls[0]["b2"] == pytest.approx(0.95)
    assert calls[0]["weight_decay"] == pytest.approx(0.1)


def test_jax_optax_transform_applies_global_norm_clipping(monkeypatch):
    from spectra_learning.training import pretrain_jax

    adamw_transform = object()
    clip_transform = object()
    chained_transform = object()
    adamw_calls = []
    clip_calls = []
    chain_calls = []

    def fake_adamw(**kwargs):
        adamw_calls.append(kwargs)
        return adamw_transform

    def fake_clip_by_global_norm(max_norm):
        clip_calls.append(max_norm)
        return clip_transform

    def fake_chain(*transforms):
        chain_calls.append(transforms)
        return chained_transform

    monkeypatch.setattr(pretrain_jax.optax, "adamw", fake_adamw)
    monkeypatch.setattr(
        pretrain_jax.optax,
        "clip_by_global_norm",
        fake_clip_by_global_norm,
    )
    monkeypatch.setattr(pretrain_jax.optax, "chain", fake_chain)
    cfg = config_dict.ConfigDict(
        {
            "learning_rate": 3e-4,
            "b1": 0.85,
            "b2": 0.95,
            "weight_decay": 0.01,
            "grad_clip_norm": 1.0,
        }
    )

    transform = pretrain_jax.build_jax_optax_transform(cfg, total_steps=100)

    assert transform is chained_transform
    assert clip_calls == [1.0]
    assert chain_calls == [(clip_transform, adamw_transform)]
    assert adamw_calls[0]["b1"] == pytest.approx(0.85)


def test_jax_optax_transform_supports_muon(monkeypatch):
    from spectra_learning.training import pretrain_jax

    calls = []
    sentinel = object()

    def fake_muon(**kwargs):
        calls.append(kwargs)
        return pretrain_jax.optax.GradientTransformation(
            lambda params: sentinel,
            lambda updates, state, params=None: (updates, state),
        )

    cfg = config_dict.ConfigDict()
    cfg.optimizer = "muon"
    cfg.learning_rate = 0.02
    cfg.min_learning_rate = 0.002
    cfg.warmup_steps = 20
    cfg.b2 = 0.95
    cfg.weight_decay = 0.05
    cfg.muon_beta = 0.95
    cfg.muon_ns_steps = 5
    cfg.muon_ns_coeffs = (3.4445, -4.7750, 2.0315)
    cfg.muon_eps = 1e-8
    cfg.muon_mu_dtype = "float32"
    cfg.muon_nesterov = True
    cfg.muon_adaptive = False
    cfg.muon_preconditioning = "frobenius"
    cfg.muon_adam_learning_rate = 0.0004
    cfg.muon_adam_min_learning_rate = 0.00004
    cfg.muon_adam_b1 = 0.9
    cfg.muon_adam_b2 = 0.95
    cfg.muon_adam_eps_root = 0.0
    cfg.muon_adam_weight_decay = 0.0
    cfg.muon_consistent_rms = None

    monkeypatch.setattr(pretrain_jax.optax.contrib, "muon", fake_muon)

    transform = pretrain_jax.build_jax_optax_transform(cfg, total_steps=100)

    assert transform.init({}) is sentinel
    assert callable(calls[0]["learning_rate"])
    assert callable(calls[0]["adam_learning_rate"])
    assert calls[0]["ns_coeffs"] == pytest.approx((3.4445, -4.7750, 2.0315))
    assert calls[0]["ns_steps"] == 5
    assert calls[0]["beta"] == pytest.approx(0.95)
    assert calls[0]["weight_decay"] == pytest.approx(0.05)
    assert calls[0]["weight_decay_mask"] is pretrain_jax._jax_weight_decay_mask
    assert calls[0]["muon_weight_dimension_numbers"] is (
        pretrain_jax._jax_muon_weight_dimension_numbers
    )
    assert calls[0]["mu_dtype"] == "float32"
    assert calls[0]["nesterov"] is True
    assert calls[0]["adaptive"] is False
    assert calls[0]["preconditioning"] == "frobenius"
    assert calls[0]["adam_b1"] == pytest.approx(0.9)
    assert calls[0]["adam_b2"] == pytest.approx(0.95)
    assert calls[0]["adam_eps_root"] == pytest.approx(0.0)
    assert calls[0]["adam_weight_decay"] == pytest.approx(0.0)
    assert calls[0]["consistent_rms"] is None


def test_jax_muon_adjust_lr_match_rms_adamw_maps_to_consistent_rms(monkeypatch):
    from spectra_learning.training import pretrain_jax

    calls = []

    def fake_muon(**kwargs):
        calls.append(kwargs)
        return pretrain_jax.optax.GradientTransformation(
            lambda params: None,
            lambda updates, state, params=None: (updates, state),
        )

    cfg = config_dict.ConfigDict()
    cfg.optimizer = "muon"
    cfg.learning_rate = 0.0004
    cfg.min_learning_rate = 0.00004
    cfg.warmup_steps = 20
    cfg.weight_decay = 0.05
    cfg.muon_adjust_lr_fn = "match_rms_adamw"

    monkeypatch.setattr(pretrain_jax.optax.contrib, "muon", fake_muon)

    pretrain_jax.build_jax_optax_transform(cfg, total_steps=100)

    assert calls[0]["consistent_rms"] == pytest.approx(0.2)


def test_jax_weight_decay_mask_matches_torch_matrix_weight_rule():
    from spectra_learning.training import pretrain_jax

    params = {
        "linear": {
            "weight": pretrain_jax.np.ones((4, 3)),
            "bias": pretrain_jax.np.ones((4,)),
        },
        "norm": {
            "weight": pretrain_jax.np.ones((4,)),
            "bias": pretrain_jax.np.ones((4,)),
        },
        "token": pretrain_jax.np.ones((4,)),
    }

    mask = pretrain_jax._jax_weight_decay_mask(params)

    assert mask == {
        "linear": {"weight": True, "bias": False},
        "norm": {"weight": False, "bias": False},
        "token": False,
    }


def test_jax_muon_weight_dimension_numbers_matches_matrix_weight_rule():
    from spectra_learning.training import pretrain_jax

    params = {
        "linear": {
            "weight": pretrain_jax.np.ones((4, 3)),
            "bias": pretrain_jax.np.ones((4,)),
        },
        "norm": {
            "weight": pretrain_jax.np.ones((4,)),
            "bias": pretrain_jax.np.ones((4,)),
        },
        "token": pretrain_jax.np.ones((4,)),
    }

    dim_numbers = pretrain_jax._jax_muon_weight_dimension_numbers(params)

    assert isinstance(
        dim_numbers["linear"]["weight"],
        pretrain_jax.optax.contrib.MuonDimensionNumbers,
    )
    assert dim_numbers["linear"]["weight"].reduction_axis == 1
    assert dim_numbers["linear"]["weight"].output_axis == 0
    assert dim_numbers["linear"]["bias"] is None
    assert dim_numbers["norm"]["weight"] is None
    assert dim_numbers["norm"]["bias"] is None
    assert dim_numbers["token"] is None


def test_jax_muon_weight_dimension_numbers_splits_qkv_blocks():
    from spectra_learning.training import pretrain_jax

    params = {
        "attention": {
            "wqkv": {"weight": pretrain_jax.np.ones((3, 4, 5))},
            "wo": {"weight": pretrain_jax.np.ones((4, 5))},
        },
        "pair_attention": {
            "qkv": {"weight": pretrain_jax.np.ones((3, 4, 5))},
        },
    }

    dim_numbers = pretrain_jax._jax_muon_weight_dimension_numbers(params)

    assert dim_numbers["attention"]["wqkv"]["weight"].reduction_axis == 2
    assert dim_numbers["attention"]["wqkv"]["weight"].output_axis == 1
    assert dim_numbers["attention"]["wo"]["weight"].reduction_axis == 1
    assert dim_numbers["attention"]["wo"]["weight"].output_axis == 0
    assert dim_numbers["pair_attention"]["qkv"]["weight"].reduction_axis == 2
    assert dim_numbers["pair_attention"]["qkv"]["weight"].output_axis == 1


def test_jax_muon_split_qkv_transform_preserves_model_update_shapes():
    from spectra_learning.training import pretrain_jax

    calls = {}

    def init_fn(params):
        calls["init_wqkv_shape"] = params["attention"]["wqkv"]["weight"].shape
        calls["init_wo_shape"] = params["attention"]["wo"]["weight"].shape
        return "state"

    def update_fn(updates, state, params=None):
        calls["update_wqkv_shape"] = updates["attention"]["wqkv"]["weight"].shape
        calls["param_wqkv_shape"] = params["attention"]["wqkv"]["weight"].shape
        return updates, state

    transform = pretrain_jax._jax_split_qkv_transform(
        pretrain_jax.optax.GradientTransformation(init_fn, update_fn)
    )
    params = {
        "attention": {
            "wqkv": {"weight": pretrain_jax.np.ones((12, 5))},
            "wo": {"weight": pretrain_jax.np.ones((4, 5))},
        },
    }

    state = transform.init(params)
    updates, state = transform.update(params, state, params)

    assert state == "state"
    assert calls == {
        "init_wqkv_shape": (3, 4, 5),
        "init_wo_shape": (4, 5),
        "update_wqkv_shape": (3, 4, 5),
        "param_wqkv_shape": (3, 4, 5),
    }
    assert updates["attention"]["wqkv"]["weight"].shape == (12, 5)
    assert updates["attention"]["wo"]["weight"].shape == (4, 5)


def test_scheduled_jax_learning_rate_matches_torch_schedule():
    from spectra_learning.training import pretrain_jax

    cfg = config_dict.ConfigDict()
    cfg.learning_rate = 0.004
    cfg.min_learning_rate = 0.0004
    cfg.warmup_steps = 20

    for step in (0, 10, 20, 60, 100):
        assert pretrain_jax._scheduled_jax_learning_rate(
            cfg,
            global_step=step,
            total_steps=100,
        ) == pytest.approx(
            learning_rate_at_step(
                step,
                base_lr=0.004,
                total_steps=100,
                warmup_steps=20,
                min_learning_rate=0.0004,
            )
        )


def test_train_and_evaluate_jax_logs_final_metrics_on_main_process(
    monkeypatch,
    tmp_path: Path,
):
    from spectra_learning.training import pretrain_jax

    logger = _FakeLogger()
    datamodule_kwargs = {}

    class FakeDataModule:
        train_steps = 3
        global_batch_size = 32
        batch_size = 16

        def __init__(self, config, **kwargs) -> None:
            del config
            datamodule_kwargs.update(kwargs)

    def fake_run_jax_training_loop(**kwargs):
        assert kwargs["logger"] is logger
        return {"run/final_global_step": 3.0, "train/loss": 1.5}

    param_metrics = {
        "model/params_total": 123.0,
        "model/params_trainable": 120.0,
        "model/params_non_trainable": 3.0,
    }
    cfg = config_dict.ConfigDict()
    cfg.seed = 7
    cfg.num_epochs = 1
    cfg.enable_wandb = True

    monkeypatch.setattr(pretrain_jax, "configure_jax_runtime", lambda config: None)
    monkeypatch.setattr(pretrain_jax, "initialize_jax_distributed", lambda config: None)
    monkeypatch.setattr(
        pretrain_jax,
        "build_jax_checkpoint_manager",
        lambda checkpoint_dir, *, max_to_keep, enable_async_checkpointing: (
            _FakeCheckpointManager()
        ),
    )
    monkeypatch.setattr(pretrain_jax.jax, "process_index", lambda: 0)
    monkeypatch.setattr(pretrain_jax.jax, "process_count", lambda: 2)
    monkeypatch.setattr(pretrain_jax.jax, "device_count", lambda: 8)
    monkeypatch.setattr(pretrain_jax.jax, "local_device_count", lambda: 4)
    monkeypatch.setattr(
        pretrain_jax.multihost_utils,
        "sync_global_devices",
        lambda name: None,
    )
    monkeypatch.setattr(pretrain_jax, "storage_mkdir", lambda path: None)
    monkeypatch.setattr(pretrain_jax, "GemsDataModule", FakeDataModule)
    monkeypatch.setattr(
        pretrain_jax,
        "build_model_from_config",
        lambda config: SimpleNamespace(use_ema_teacher=False),
    )
    monkeypatch.setattr(
        pretrain_jax,
        "initialize_jax_model_from_torch_seed",
        lambda config, model: None,
    )
    monkeypatch.setattr(
        pretrain_jax,
        "collect_jax_param_metrics",
        lambda model: param_metrics,
    )
    monkeypatch.setattr(pretrain_jax, "build_logger", lambda config, workdir: logger)
    monkeypatch.setattr(
        pretrain_jax,
        "_run_jax_training_loop",
        fake_run_jax_training_loop,
    )

    results = pretrain_jax.train_and_evaluate_jax(cfg, tmp_path)

    assert datamodule_kwargs["distributed_world_size"] == 2
    assert datamodule_kwargs["distributed_rank"] == 0
    assert datamodule_kwargs["distributed_local_rank"] == 0
    assert cfg.dataloader_pin_memory is False
    assert cfg.dataloader_persistent_workers is False
    assert cfg.dataloader_output_format == "numpy"
    assert results["run/jax_process_count"] == 2.0
    assert results["run/jax_data_parallel_devices"] == 8.0
    assert results["run/device_microbatch_size"] == 4.0
    assert logger.logs == [
        (
            {
                "global_step": 0.0,
                **param_metrics,
            },
            0,
        ),
        (
            {
                "global_step": 3.0,
                "run/final_global_step": 3.0,
                "train/loss": 1.5,
                "run/world_size": 2.0,
                "run/jax_device_count": 8.0,
                "run/jax_process_index": 0.0,
                "run/jax_process_count": 2.0,
                "run/jax_data_parallel_devices": 8.0,
                "run/global_batch_size": 32.0,
                "run/local_batch_size": 16.0,
                "run/device_microbatch_size": 4.0,
                "run/gradient_accumulation_steps": 1.0,
                "run/device_backend": "jax",
                "run/training_task": "pretrain",
                **param_metrics,
            },
            3,
        )
    ]
    assert logger.finished is True


def test_train_and_evaluate_jax_skips_logger_on_worker_process(
    monkeypatch,
    tmp_path: Path,
):
    from spectra_learning.training import pretrain_jax

    class FakeDataModule:
        train_steps = 3
        global_batch_size = 32
        batch_size = 16

        def __init__(self, config, **kwargs) -> None:
            del config, kwargs

    def fake_run_jax_training_loop(**kwargs):
        assert isinstance(kwargs["logger"], pretrain_jax.MetricLogger)
        return {"run/final_global_step": 3.0, "train/loss": float("nan")}

    cfg = config_dict.ConfigDict()
    cfg.seed = 7
    cfg.num_epochs = 1
    cfg.enable_wandb = True

    monkeypatch.setattr(pretrain_jax, "configure_jax_runtime", lambda config: None)
    monkeypatch.setattr(pretrain_jax, "initialize_jax_distributed", lambda config: None)
    monkeypatch.setattr(
        pretrain_jax,
        "build_jax_checkpoint_manager",
        lambda checkpoint_dir, *, max_to_keep, enable_async_checkpointing: (
            _FakeCheckpointManager()
        ),
    )
    monkeypatch.setattr(pretrain_jax.jax, "process_index", lambda: 1)
    monkeypatch.setattr(pretrain_jax.jax, "process_count", lambda: 2)
    monkeypatch.setattr(pretrain_jax.jax, "device_count", lambda: 8)
    monkeypatch.setattr(pretrain_jax.jax, "local_device_count", lambda: 4)
    monkeypatch.setattr(
        pretrain_jax.multihost_utils,
        "sync_global_devices",
        lambda name: None,
    )
    monkeypatch.setattr(pretrain_jax, "storage_mkdir", lambda path: None)
    monkeypatch.setattr(pretrain_jax, "GemsDataModule", FakeDataModule)
    monkeypatch.setattr(
        pretrain_jax,
        "build_model_from_config",
        lambda config: SimpleNamespace(use_ema_teacher=False),
    )
    monkeypatch.setattr(
        pretrain_jax,
        "initialize_jax_model_from_torch_seed",
        lambda config, model: None,
    )
    monkeypatch.setattr(
        pretrain_jax,
        "collect_jax_param_metrics",
        lambda model: {"model/params_total": 123.0},
    )
    monkeypatch.setattr(
        pretrain_jax,
        "build_logger",
        lambda config, workdir: pytest.fail("worker process initialized logger"),
    )
    monkeypatch.setattr(
        pretrain_jax,
        "_run_jax_training_loop",
        fake_run_jax_training_loop,
    )

    results = pretrain_jax.train_and_evaluate_jax(cfg, tmp_path)

    assert results["run/jax_process_index"] == 1.0
    assert results["run/jax_process_count"] == 2.0


def test_save_checkpoint_persists_optimizer_state():
    model = _small_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    loss = torch.stack([param.sum() for param in model.parameters()]).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/resume.pt"
        save_checkpoint(
            path=path,
            model=model,
            optimizers=[optimizer],
            schedulers=[],
            global_step=12,
            epoch=1,
            loss=float(loss.detach()),
            wandb_run_id="wandb-run-123",
        )
        ckpt = torch.load(path, map_location="cpu", weights_only=True)

    saved_optimizer = ckpt["optimizers"][0]
    assert ckpt["wandb_run_id"] == "wandb-run-123"
    assert saved_optimizer["state"]
    assert "scalar_optimizer_state" not in saved_optimizer


def test_fp16_autocast_parses_with_cpu_scaler_disabled():
    assert parse_autocast_dtype("fp16") == torch.float16
    assert not build_grad_scaler(torch.float16, torch.device("cpu")).is_enabled()


def test_save_checkpoint_persists_grad_scaler_state():
    model = _small_model()
    grad_scaler = torch.amp.GradScaler(
        device="cpu",
        init_scale=16.0,
        enabled=True,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/resume.pt"
        save_checkpoint(
            path=path,
            model=model,
            optimizers=[],
            schedulers=[],
            global_step=12,
            epoch=1,
            loss=0.5,
            grad_scaler=grad_scaler,
        )
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        restored = torch.amp.GradScaler(device="cpu", enabled=True)
        load_grad_scaler_state(restored, ckpt["grad_scaler"])

    assert ckpt["grad_scaler"]["scale"] == 16.0
    assert restored.state_dict()["scale"] == 16.0


def test_save_checkpoint_writes_covariance_pooler_sibling_pt():
    model = _small_model()
    pooler = CovariancePool(input_dim=model.model_dim, compressed_dim=4)
    optimizer = torch.optim.AdamW(
        [*model.parameters(), *pooler.parameters()],
        lr=0.01,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "step-00000012.pt"
        save_checkpoint(
            path=path,
            model=model,
            covariance_pooler=pooler,
            optimizers=[optimizer],
            schedulers=[],
            global_step=12,
            epoch=1,
            loss=0.5,
        )
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        pooler_path = covariance_pooler_checkpoint_path(path)
        pooler_ckpt = torch.load(pooler_path, map_location="cpu", weights_only=True)

    assert pooler_path.name == "covariance-pooler-step-00000012.pt"
    assert ckpt["covariance_pooler_checkpoint"] == pooler_path.name
    assert "covariance_pooler.left_proj.weight" not in ckpt["model"]
    assert set(pooler_ckpt["pooler"]) == set(pooler.state_dict())
    assert pooler_ckpt["global_step"] == 12


def test_save_checkpoint_writes_fsspec_uri_checkpoint():
    _clear_memory_fs("/spectra-remote-save")
    model = _small_model()
    path = "memory://spectra-remote-save/run/checkpoints/step-00000012.pt"

    save_checkpoint(
        path=path,
        model=model,
        optimizers=[],
        schedulers=[],
        global_step=12,
        epoch=1,
        loss=0.5,
        wandb_run_id="wandb-run-123",
    )
    ckpt = load_torch_checkpoint(path, map_location="cpu", weights_only=True)

    assert ckpt["global_step"] == 12
    assert ckpt["wandb_run_id"] == "wandb-run-123"


def test_remote_covariance_pooler_checkpoint_uses_sibling_uri():
    _clear_memory_fs("/spectra-remote-pooler")
    model = _small_model()
    pooler = CovariancePool(input_dim=model.model_dim, compressed_dim=4)
    path = "memory://spectra-remote-pooler/run/checkpoints/step-00000012.pt"

    save_checkpoint(
        path=path,
        model=model,
        covariance_pooler=pooler,
        optimizers=[],
        schedulers=[],
        global_step=12,
        epoch=1,
        loss=0.5,
    )
    ckpt = load_torch_checkpoint(path, map_location="cpu", weights_only=True)
    pooler_path = covariance_pooler_checkpoint_path(path)
    pooler_ckpt = load_torch_checkpoint(
        pooler_path,
        map_location="cpu",
        weights_only=True,
    )
    restored = CovariancePool(input_dim=model.model_dim, compressed_dim=4)
    load_resume_covariance_pooler_state(restored, path, ckpt)

    assert pooler_path == (
        "memory://spectra-remote-pooler/run/checkpoints/"
        "covariance-pooler-step-00000012.pt"
    )
    assert ckpt["covariance_pooler_checkpoint"] == "covariance-pooler-step-00000012.pt"
    assert set(pooler_ckpt["pooler"]) == set(pooler.state_dict())
    for key, value in pooler.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)


def test_async_checkpoint_writer_returns_before_torch_save_finishes(monkeypatch, tmp_path: Path):
    model = _small_model()
    save_entered = threading.Event()
    release_save = threading.Event()
    original_save = checkpointing_module.torch.save

    def slow_save(obj, path):
        save_entered.set()
        release_save.wait(timeout=10.0)
        original_save(obj, path)

    monkeypatch.setattr(checkpointing_module.torch, "save", slow_save)
    writer = AsyncCheckpointWriter()
    path = tmp_path / "step-00000012.pt"

    writer.save_checkpoint(
        path=path,
        model=model,
        optimizers=[],
        schedulers=[],
        global_step=12,
        epoch=1,
        loss=0.5,
    )

    assert save_entered.wait(timeout=1.0)
    assert not path.exists()

    release_save.set()
    writer.close()
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    assert ckpt["global_step"] == 12


def test_async_checkpoint_writer_raises_background_failures(monkeypatch, tmp_path: Path):
    def fail_write_job(*args, **kwargs):
        raise OSError("upload failed")

    monkeypatch.setattr(
        checkpointing_module,
        "_write_training_checkpoint_job",
        fail_write_job,
    )
    writer = AsyncCheckpointWriter()

    writer.save_checkpoint(
        path=tmp_path / "step-00000012.pt",
        model=_small_model(),
        optimizers=[],
        schedulers=[],
        global_step=12,
        epoch=1,
        loss=0.5,
    )
    with pytest.raises(OSError, match="upload failed"):
        writer.close()


def test_latest_ckpt_path_ignores_non_training_checkpoints(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    step_path = checkpoint_dir / "step-00000012.pt"
    probe_path = checkpoint_dir / "probe-step-00000013.pt"
    pooler_path = checkpoint_dir / "covariance-pooler-step-00000012.pt"

    step_path.write_bytes(b"step")
    probe_path.write_bytes(b"probe")
    pooler_path.write_bytes(b"pooler")

    assert is_training_checkpoint_path(step_path)
    assert not is_training_checkpoint_path(probe_path)
    assert not is_training_checkpoint_path(pooler_path)
    assert latest_ckpt_path(tmp_path) == str(step_path)


def test_latest_ckpt_path_supports_fsspec_uri():
    _clear_memory_fs("/spectra-remote-latest")
    fs = fsspec.filesystem("memory")
    fs.pipe_file(
        "/spectra-remote-latest/run/checkpoints/probe-step-00000013.pt",
        b"probe",
    )
    fs.pipe_file(
        "/spectra-remote-latest/run/checkpoints/covariance-pooler-step-00000012.pt",
        b"pooler",
    )
    fs.pipe_file(
        "/spectra-remote-latest/run/checkpoints/step-00000012.pt",
        b"step",
    )

    assert latest_ckpt_path("memory://spectra-remote-latest/run") == (
        "memory://spectra-remote-latest/run/checkpoints/step-00000012.pt"
    )


def test_load_resume_covariance_pooler_state_reads_sibling_pt():
    model = _small_model()
    pooler = CovariancePool(input_dim=model.model_dim, compressed_dim=4)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "step-00000012.pt"
        save_checkpoint(
            path=path,
            model=model,
            covariance_pooler=pooler,
            optimizers=[],
            schedulers=[],
            global_step=12,
            epoch=1,
            loss=0.5,
        )
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        restored = CovariancePool(input_dim=model.model_dim, compressed_dim=4)
        load_resume_covariance_pooler_state(restored, path, ckpt)

    for key, value in pooler.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)


def test_restore_training_state_loads_canonical_checkpoint(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    source = _small_model()
    torch.save(
        {
            "model": source.state_dict(),
            "optimizers": [],
            "schedulers": [],
            "grad_scaler": None,
            "global_step": 3,
            "epoch": 0,
            "loss": 0.5,
            "wandb_run_id": "wandb-run-123",
            "covariance_pooler_checkpoint": None,
        },
        checkpoint_dir / "step-00000003.pt",
    )
    restored = _small_model()
    config = config_dict.ConfigDict()

    start_epoch, global_step, resume_offset = pretrain.restore_training_state(
        config=config,
        checkpoint_dir=checkpoint_dir,
        model=restored,
        optimizers=[],
        schedulers=[],
        steps_per_epoch=5,
        device=torch.device("cpu"),
    )

    assert (start_epoch, global_step, resume_offset) == (0, 3, 3)
    assert config.wandb_resume_id == "wandb-run-123"
    for key, value in source.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)


@pytest.mark.parametrize("missing_key", ["grad_scaler", "loss"])
def test_restore_training_state_rejects_missing_checkpoint_keys(
    tmp_path: Path,
    missing_key: str,
):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    state = {
        "model": _small_model().state_dict(),
        "optimizers": [],
        "schedulers": [],
        "grad_scaler": None,
        "global_step": 3,
        "epoch": 0,
        "loss": 0.5,
        "wandb_run_id": None,
        "covariance_pooler_checkpoint": None,
    }
    state.pop(missing_key)
    torch.save(state, checkpoint_dir / "step-00000003.pt")

    with pytest.raises(KeyError, match=missing_key):
        pretrain.restore_training_state(
            config=config_dict.ConfigDict(),
            checkpoint_dir=checkpoint_dir,
            model=_small_model(),
            optimizers=[],
            schedulers=[],
            steps_per_epoch=5,
            device=torch.device("cpu"),
        )


def test_training_loop_resumes_with_offset_loader(monkeypatch, tmp_path: Path):
    cfg = config_dict.ConfigDict()
    cfg.autocast_dtype = "bf16"
    cfg.log_every_n_steps = 0
    cfg.collapse_metrics_every_n_steps = 0
    cfg.checkpoint_every_steps = 1000
    cfg.msg_probe_every_n_steps = -1
    cfg.device_prefetch_size = 1

    class FakeDataModule:
        train_steps = 5
        global_batch_size = 1

        def __init__(self) -> None:
            self.calls = []

        def train_loader_for_epoch(self, epoch: int, start_batch: int = 0):
            self.calls.append((epoch, start_batch))
            return [
                {"peak_mz": torch.tensor([float(step)])}
                for step in range(start_batch, self.train_steps)
            ]

    def fake_train_step_impl(*args, **kwargs):
        return {"loss": torch.tensor(1.0)}

    datamodule = FakeDataModule()
    monkeypatch.setattr(pretrain, "train_step_impl", fake_train_step_impl)

    metrics = pretrain.run_training_loop(
        config=cfg,
        datamodule=datamodule,
        model=torch.nn.Linear(1, 1),
        optimizers=[],
        schedulers=[],
        logger=SimpleNamespace(experiment=None, log_metrics=lambda *args, **kwargs: None),
        checkpoint_dir=tmp_path,
        start_epoch=0,
        loop_epochs=1,
        resume_offset=3,
        global_step=3,
        total_steps=5,
        device=torch.device("cpu"),
    )

    assert datamodule.calls == [(0, 3)]
    assert metrics["run/final_global_step"] == 5.0


def test_training_loop_requests_fresh_loader_for_each_epoch(monkeypatch, tmp_path: Path):
    cfg = config_dict.ConfigDict()
    cfg.autocast_dtype = "bf16"
    cfg.log_every_n_steps = 0
    cfg.collapse_metrics_every_n_steps = 0
    cfg.checkpoint_every_steps = 1000
    cfg.msg_probe_every_n_steps = -1
    cfg.device_prefetch_size = 1

    class FakeDataModule:
        train_steps = 2
        global_batch_size = 1

        def __init__(self) -> None:
            self.calls = []

        def train_loader_for_epoch(self, epoch: int, start_batch: int = 0):
            self.calls.append((epoch, start_batch))
            return [
                {"peak_mz": torch.tensor([float(epoch * self.train_steps + step)])}
                for step in range(start_batch, self.train_steps)
            ]

    def fake_train_step_impl(*args, **kwargs):
        return {"loss": torch.tensor(1.0)}

    datamodule = FakeDataModule()
    monkeypatch.setattr(pretrain, "train_step_impl", fake_train_step_impl)

    metrics = pretrain.run_training_loop(
        config=cfg,
        datamodule=datamodule,
        model=torch.nn.Linear(1, 1),
        optimizers=[],
        schedulers=[],
        logger=SimpleNamespace(experiment=None, log_metrics=lambda *args, **kwargs: None),
        checkpoint_dir=tmp_path,
        start_epoch=0,
        loop_epochs=2,
        resume_offset=0,
        global_step=0,
        total_steps=4,
        device=torch.device("cpu"),
    )

    assert datamodule.calls == [(0, 0), (1, 0)]
    assert metrics["run/final_global_step"] == 4.0


def test_training_loop_counts_optimizer_steps_with_gradient_accumulation(
    monkeypatch,
    tmp_path: Path,
):
    cfg = config_dict.ConfigDict()
    cfg.autocast_dtype = "bf16"
    cfg.log_every_n_steps = 0
    cfg.collapse_metrics_every_n_steps = 0
    cfg.checkpoint_every_steps = 1000
    cfg.msg_probe_every_n_steps = -1
    cfg.device_prefetch_size = 1
    cfg.gradient_accumulation_steps = 2

    class FakeDataModule:
        train_steps = 2
        global_batch_size = 4

        def train_loader_for_epoch(self, epoch: int, start_batch: int = 0):
            return [
                {"peak_mz": torch.tensor([float(step)])}
                for step in range(start_batch, self.train_steps * 2)
            ]

    accumulation_steps = []

    def fake_train_step_impl(*args, **kwargs):
        accumulation_step = int(kwargs["accumulation_step"])
        accumulation_steps.append(accumulation_step)
        return {
            "loss": torch.tensor(1.0),
            "optimizer_step": torch.tensor(float((accumulation_step + 1) % 2 == 0)),
        }

    monkeypatch.setattr(pretrain, "train_step_impl", fake_train_step_impl)

    metrics = pretrain.run_training_loop(
        config=cfg,
        datamodule=FakeDataModule(),
        model=torch.nn.Linear(1, 1),
        optimizers=[],
        schedulers=[],
        logger=SimpleNamespace(experiment=None, log_metrics=lambda *args, **kwargs: None),
        checkpoint_dir=tmp_path,
        start_epoch=0,
        loop_epochs=1,
        resume_offset=0,
        global_step=0,
        total_steps=2,
        device=torch.device("cpu"),
    )

    assert accumulation_steps == [0, 1, 2, 3]
    assert metrics["run/final_global_step"] == 2.0


def test_training_loop_preserves_optimizer_boundary_event_order(
    monkeypatch,
    tmp_path: Path,
):
    cfg = config_dict.ConfigDict()
    cfg.autocast_dtype = "bf16"
    cfg.log_every_n_steps = 0
    cfg.collapse_metrics_every_n_steps = 0
    cfg.checkpoint_every_steps = 1
    cfg.msg_probe_every_n_steps = 1
    cfg.msg_probe_variants = ["mean"]
    cfg.val_every_n_steps = 1
    cfg.val_num_steps = 1
    cfg.device_prefetch_size = 1
    cfg.throughput_warmup_steps = 1000
    cfg.gradient_accumulation_steps = 2

    class FakeDataModule:
        train_steps = 1
        global_batch_size = 2

        def train_loader_for_epoch(self, epoch: int, start_batch: int = 0):
            del epoch, start_batch
            return [
                {"peak_mz": torch.tensor([0.0])},
                {"peak_mz": torch.tensor([1.0])},
            ]

    events = []

    class FakeProfiler:
        def start(self) -> None:
            events.append("profiler_start")

        def step(self) -> None:
            events.append("profiler_step")

        def stop(self) -> None:
            events.append("profiler_stop")

    class FakeCheckpointWriter:
        def save_checkpoint(self, *args, **kwargs) -> None:
            del args, kwargs
            events.append("checkpoint")

        def log_completed_failures(self) -> None:
            events.append("checkpoint_failure_poll")

        def close(self) -> None:
            raise AssertionError("the loop does not own this writer")

    def fake_train_step_impl(*args, **kwargs):
        del args
        accumulation_step = int(kwargs["accumulation_step"])
        events.append(f"microbatch_{accumulation_step}")
        return {
            "loss": torch.tensor(1.0),
            "optimizer_step": torch.tensor(float(accumulation_step == 1)),
        }

    def fake_validation(**kwargs):
        del kwargs
        events.append("validation")
        return {"loss": torch.tensor(2.0)}

    def fake_probe(*args, **kwargs):
        del args, kwargs
        events.append("probe")
        return {"msg_probe/mean/test/auc_maccs_mean": 0.75}

    barrier_events = iter(
        ["checkpoint_barrier", "probe_barrier", "final_barrier"]
    )

    monkeypatch.setattr(pretrain, "tqdm", _FakePbar)
    monkeypatch.setattr(pretrain, "train_step_impl", fake_train_step_impl)
    monkeypatch.setattr(
        pretrain,
        "make_torch_profiler",
        lambda *args, **kwargs: FakeProfiler(),
    )
    monkeypatch.setattr(pretrain, "evaluate_validation_loss", fake_validation)
    monkeypatch.setattr(pretrain, "run_and_log_msg_probe", fake_probe)
    monkeypatch.setattr(
        pretrain,
        "barrier",
        lambda distributed: events.append(next(barrier_events)),
    )
    monkeypatch.setattr(
        pretrain,
        "synchronize_device",
        lambda device: events.append("final_sync"),
    )

    metrics = pretrain.run_training_loop(
        config=cfg,
        datamodule=FakeDataModule(),
        model=torch.nn.Linear(1, 1),
        optimizers=[],
        schedulers=[],
        logger=_FakeLogger(),
        checkpoint_dir=tmp_path,
        start_epoch=0,
        loop_epochs=1,
        resume_offset=0,
        global_step=0,
        total_steps=1,
        device=torch.device("cpu"),
        checkpoint_writer=FakeCheckpointWriter(),
        flops_per_optimizer_step=1.0,
    )

    assert events == [
        "profiler_start",
        "microbatch_0",
        "microbatch_1",
        "profiler_step",
        "checkpoint",
        "checkpoint_barrier",
        "validation",
        "probe",
        "probe_barrier",
        "checkpoint_failure_poll",
        "final_sync",
        "final_barrier",
        "profiler_stop",
    ]
    assert metrics["run/final_global_step"] == 1.0
    assert metrics["val/loss"] == 2.0
    assert metrics["msg_probe/mean/test/auc_maccs_mean"] == 0.75


def test_training_loop_measures_resumed_steps_after_optimizer_warmup(
    monkeypatch,
    tmp_path: Path,
):
    cfg = config_dict.ConfigDict()
    cfg.autocast_dtype = "bf16"
    cfg.log_every_n_steps = 0
    cfg.collapse_metrics_every_n_steps = 0
    cfg.checkpoint_every_steps = 1000
    cfg.msg_probe_every_n_steps = -1
    cfg.device_prefetch_size = 1
    cfg.throughput_warmup_steps = 1
    cfg.gradient_accumulation_steps = 2

    class FakeDataModule:
        train_steps = 4
        global_batch_size = 8

        def __init__(self) -> None:
            self.calls = []

        def train_loader_for_epoch(self, epoch: int, start_batch: int = 0):
            self.calls.append((epoch, start_batch))
            return [
                {"peak_mz": torch.tensor([float(step)])}
                for step in range(4)
            ]

    class FakeCheckpointWriter:
        def save_checkpoint(self, *args, **kwargs) -> None:
            del args, kwargs
            raise AssertionError("no checkpoint is due")

        def log_completed_failures(self) -> None:
            pass

        def close(self) -> None:
            raise AssertionError("the loop does not own this writer")

    timeline = []

    def fake_train_step_impl(*args, **kwargs):
        del args
        accumulation_step = int(kwargs["accumulation_step"])
        timeline.append(f"microbatch_{accumulation_step}")
        return {
            "loss": torch.tensor(1.0),
            "optimizer_step": torch.tensor(
                float((accumulation_step + 1) % 2 == 0)
            ),
        }

    clock = iter([100.0, 110.0, 140.0, 140.0])
    datamodule = FakeDataModule()
    monkeypatch.setattr(pretrain, "tqdm", _FakePbar)
    monkeypatch.setattr(pretrain, "train_step_impl", fake_train_step_impl)
    monkeypatch.setattr(pretrain, "make_torch_profiler", lambda *args: None)
    monkeypatch.setattr(
        pretrain,
        "time",
        SimpleNamespace(perf_counter=lambda: next(clock)),
    )
    monkeypatch.setattr(
        pretrain,
        "synchronize_device",
        lambda device: timeline.append("sync"),
    )
    monkeypatch.setattr(
        pretrain,
        "barrier",
        lambda distributed: timeline.append("barrier"),
    )

    metrics = pretrain.run_training_loop(
        config=cfg,
        datamodule=datamodule,
        model=torch.nn.Linear(1, 1),
        optimizers=[],
        schedulers=[],
        logger=_FakeLogger(),
        checkpoint_dir=tmp_path,
        start_epoch=1,
        loop_epochs=2,
        resume_offset=1,
        global_step=5,
        total_steps=7,
        device=torch.device("cpu"),
        checkpoint_writer=FakeCheckpointWriter(),
        flops_per_optimizer_step=1.0,
    )

    assert datamodule.calls == [(1, 1)]
    assert timeline == [
        "microbatch_0",
        "microbatch_1",
        "sync",
        "barrier",
        "microbatch_2",
        "microbatch_3",
        "sync",
        "barrier",
    ]
    assert metrics["run/final_global_step"] == 7.0
    assert metrics["run/train_elapsed_seconds"] == 40.0
    assert metrics["run/steps_per_second"] == pytest.approx(2.0 / 40.0)
    assert metrics["run/samples_per_second"] == pytest.approx(16.0 / 40.0)
    assert metrics["run/measured_steps"] == 1.0
    assert metrics["run/measured_elapsed_seconds"] == 30.0
    assert metrics["run/measured_steps_per_second"] == pytest.approx(1.0 / 30.0)
    assert metrics["run/measured_samples_per_second"] == pytest.approx(8.0 / 30.0)


def test_training_loop_continues_while_checkpoint_save_is_pending(monkeypatch, tmp_path: Path):
    cfg = config_dict.ConfigDict()
    cfg.autocast_dtype = "bf16"
    cfg.log_every_n_steps = 0
    cfg.collapse_metrics_every_n_steps = 0
    cfg.checkpoint_every_steps = 1
    cfg.msg_probe_every_n_steps = -1
    cfg.device_prefetch_size = 1
    cfg.throughput_warmup_steps = 1000

    class FakeDataModule:
        train_steps = 2
        global_batch_size = 1

        def train_loader_for_epoch(self, epoch: int, start_batch: int = 0):
            return [
                {"peak_mz": torch.tensor([float(step)])}
                for step in range(start_batch, self.train_steps)
            ]

    train_steps = []
    save_entered = threading.Event()
    release_save = threading.Event()
    original_save = checkpointing_module.torch.save

    def fake_train_step_impl(*args, **kwargs):
        train_steps.append(len(train_steps))
        return {"loss": torch.tensor(1.0)}

    def slow_save(obj, path):
        save_entered.set()
        release_save.wait(timeout=10.0)
        original_save(obj, path)

    monkeypatch.setattr(pretrain, "train_step_impl", fake_train_step_impl)
    monkeypatch.setattr(checkpointing_module.torch, "save", slow_save)

    writer = AsyncCheckpointWriter()
    metrics = pretrain.run_training_loop(
        config=cfg,
        datamodule=FakeDataModule(),
        model=_small_model(),
        optimizers=[],
        schedulers=[],
        logger=SimpleNamespace(experiment=None, log_metrics=lambda *args, **kwargs: None),
        checkpoint_dir=tmp_path,
        start_epoch=0,
        loop_epochs=1,
        resume_offset=0,
        global_step=0,
        total_steps=2,
        device=torch.device("cpu"),
        checkpoint_writer=writer,
    )

    assert save_entered.wait(timeout=1.0)
    assert train_steps == [0, 1]
    assert metrics["run/final_global_step"] == 2.0
    assert not (tmp_path / "step-00000001.pt").exists()

    release_save.set()
    writer.close()
    assert (tmp_path / "step-00000001.pt").exists()
    assert (tmp_path / "step-00000002.pt").exists()


def test_training_loop_writes_fsspec_checkpoint_dir(monkeypatch):
    _clear_memory_fs("/spectra-loop-remote")
    cfg = config_dict.ConfigDict()
    cfg.autocast_dtype = "bf16"
    cfg.log_every_n_steps = 0
    cfg.collapse_metrics_every_n_steps = 0
    cfg.checkpoint_every_steps = 1
    cfg.msg_probe_every_n_steps = -1
    cfg.device_prefetch_size = 1
    cfg.throughput_warmup_steps = 1000

    class FakeDataModule:
        train_steps = 1
        global_batch_size = 1

        def train_loader_for_epoch(self, epoch: int, start_batch: int = 0):
            return [{"peak_mz": torch.tensor([0.0])}]

    def fake_train_step_impl(*args, **kwargs):
        return {"loss": torch.tensor(1.0)}

    monkeypatch.setattr(pretrain, "train_step_impl", fake_train_step_impl)

    metrics = pretrain.run_training_loop(
        config=cfg,
        datamodule=FakeDataModule(),
        model=_small_model(),
        optimizers=[],
        schedulers=[],
        logger=SimpleNamespace(experiment=None, log_metrics=lambda *args, **kwargs: None),
        checkpoint_dir="memory://spectra-loop-remote/run/checkpoints",
        start_epoch=0,
        loop_epochs=1,
        resume_offset=0,
        global_step=0,
        total_steps=1,
        device=torch.device("cpu"),
    )
    ckpt = load_torch_checkpoint(
        "memory://spectra-loop-remote/run/checkpoints/step-00000001.pt",
        map_location="cpu",
        weights_only=True,
    )

    assert metrics["run/final_global_step"] == 1.0
    assert ckpt["global_step"] == 1


def test_training_loop_stops_when_signal_requested(monkeypatch, tmp_path: Path):
    cfg = config_dict.ConfigDict()
    cfg.autocast_dtype = "bf16"
    cfg.log_every_n_steps = 0
    cfg.collapse_metrics_every_n_steps = 0
    cfg.checkpoint_every_steps = 1000
    cfg.msg_probe_every_n_steps = -1
    cfg.device_prefetch_size = 1

    class FakeDataModule:
        train_steps = 2
        global_batch_size = 1

        def train_loader_for_epoch(self, epoch: int, start_batch: int = 0):
            return [
                {"peak_mz": torch.tensor([float(step)])}
                for step in range(start_batch, self.train_steps)
            ]

    train_step_calls = []

    def fake_train_step_impl(*args, **kwargs):
        train_step_calls.append((args, kwargs))
        return {"loss": torch.tensor(1.0)}

    monkeypatch.setattr(pretrain, "train_step_impl", fake_train_step_impl)
    monkeypatch.setattr(pretrain, "_STOP_REQUESTED", True)

    metrics = pretrain.run_training_loop(
        config=cfg,
        datamodule=FakeDataModule(),
        model=torch.nn.Linear(1, 1),
        optimizers=[],
        schedulers=[],
        logger=SimpleNamespace(experiment=None, log_metrics=lambda *args, **kwargs: None),
        checkpoint_dir=tmp_path,
        start_epoch=0,
        loop_epochs=1,
        resume_offset=0,
        global_step=0,
        total_steps=2,
        device=torch.device("cpu"),
    )

    assert train_step_calls == []
    assert metrics["run/stopped_for_signal"] == 1.0
    assert metrics["run/final_global_step"] == 0.0


def test_training_loop_logs_limited_validation_loss(monkeypatch, tmp_path: Path):
    cfg = config_dict.ConfigDict()
    cfg.autocast_dtype = "bf16"
    cfg.log_every_n_steps = 0
    cfg.collapse_metrics_every_n_steps = 0
    cfg.checkpoint_every_steps = 1000
    cfg.msg_probe_every_n_steps = -1
    cfg.val_every_n_steps = 2
    cfg.val_num_steps = 2
    cfg.device_prefetch_size = 1
    cfg.throughput_warmup_steps = 1000

    class FakeDataModule:
        train_steps = 2
        global_batch_size = 1

        def train_loader_for_epoch(self, epoch: int, start_batch: int = 0):
            return [
                {"peak_mz": torch.tensor([float(step)])}
                for step in range(start_batch, self.train_steps)
            ]

        @property
        def val_loader(self):
            return [
                {"peak_mz": torch.tensor([2.0])},
                {"peak_mz": torch.tensor([4.0])},
                {"peak_mz": torch.tensor([100.0])},
            ]

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

        def forward(self, batch):
            loss = batch["peak_mz"].float().mean() + self.weight * 0.0
            return {"loss": loss}

    def fake_train_step_impl(*args, **kwargs):
        return {"loss": torch.tensor(1.0)}

    logger = _FakeLogger()
    monkeypatch.setattr(pretrain, "train_step_impl", fake_train_step_impl)

    metrics = pretrain.run_training_loop(
        config=cfg,
        datamodule=FakeDataModule(),
        model=FakeModel(),
        optimizers=[],
        schedulers=[],
        logger=logger,
        checkpoint_dir=tmp_path,
        start_epoch=0,
        loop_epochs=1,
        resume_offset=0,
        global_step=0,
        total_steps=2,
        device=torch.device("cpu"),
    )

    assert logger.logs == [
        ({"val/loss": pytest.approx(3.0), "global_step": 2}, 2)
    ]
    assert metrics["val/loss"] == pytest.approx(3.0)
    assert metrics["run/final_global_step"] == 2.0


def test_training_loop_runs_distributed_online_probe_and_logs_on_main(monkeypatch, tmp_path: Path):
    cfg = config_dict.ConfigDict()
    cfg.autocast_dtype = "bf16"
    cfg.log_every_n_steps = 0
    cfg.collapse_metrics_every_n_steps = 0
    cfg.checkpoint_every_steps = 1000
    cfg.msg_probe_every_n_steps = 2
    cfg.msg_probe_variants = ["mean"]
    cfg.device_prefetch_size = 1
    cfg.throughput_warmup_steps = 1000

    class FakeDataModule:
        train_steps = 2
        global_batch_size = 4

        def train_loader_for_epoch(self, epoch: int, start_batch: int = 0):
            return [
                {"peak_mz": torch.tensor([float(step)])}
                for step in range(start_batch, self.train_steps)
            ]

    class FakeLogger:
        experiment = None

        def __init__(self) -> None:
            self.logs = []

        def log_metrics(self, metrics, step=None) -> None:
            self.logs.append((dict(metrics), step))

    probe_calls = []
    barriers = []

    def fake_train_step_impl(*args, **kwargs):
        return {"loss": torch.tensor(1.0)}

    def fake_run_msg_probe(
        *,
        config,
        model,
        device,
        distributed,
        covariance_pooler=None,
        online_maccs_only=False,
    ):
        assert online_maccs_only is True
        probe_calls.append((model, device))
        return {"msg_probe/mean/test/auc_maccs_mean": 0.75}

    def fake_barrier(distributed):
        barriers.append((distributed.rank, distributed.world_size))

    monkeypatch.setattr(pretrain, "train_step_impl", fake_train_step_impl)
    monkeypatch.setattr(pretrain, "run_msg_probe", fake_run_msg_probe)
    monkeypatch.setattr(pretrain, "barrier", fake_barrier)

    device = torch.device("cpu")
    model = torch.nn.Linear(1, 1)
    main_logger = FakeLogger()
    main_metrics = pretrain.run_training_loop(
        config=cfg,
        datamodule=FakeDataModule(),
        model=model,
        optimizers=[],
        schedulers=[],
        logger=main_logger,
        checkpoint_dir=tmp_path,
        start_epoch=0,
        loop_epochs=1,
        resume_offset=0,
        global_step=0,
        total_steps=2,
        device=device,
        distributed=pretrain.DistributedContext(
            rank=0,
            local_rank=0,
            world_size=2,
            device=device,
        ),
    )
    worker_logger = FakeLogger()
    worker_metrics = pretrain.run_training_loop(
        config=cfg,
        datamodule=FakeDataModule(),
        model=model,
        optimizers=[],
        schedulers=[],
        logger=worker_logger,
        checkpoint_dir=tmp_path,
        start_epoch=0,
        loop_epochs=1,
        resume_offset=0,
        global_step=0,
        total_steps=2,
        device=device,
        distributed=pretrain.DistributedContext(
            rank=1,
            local_rank=1,
            world_size=2,
            device=device,
        ),
    )

    assert probe_calls == [(model, device), (model, device)]
    assert main_logger.logs == [({"msg_probe/mean/test/auc_maccs_mean": 0.75}, 2)]
    assert worker_logger.logs == []
    assert barriers == [(0, 2), (0, 2), (1, 2), (1, 2)]
    assert main_metrics["msg_probe/mean/test/auc_maccs_mean"] == 0.75
    assert main_metrics["run/final_global_step"] == 2.0
    assert worker_metrics["run/final_global_step"] == 2.0


def test_run_checkpoint_msg_probe_loads_checkpoint_and_logs_metrics(monkeypatch, tmp_path: Path):
    cfg = config_dict.ConfigDict()
    cfg.model_dim = 32
    cfg.encoder_num_layers = 1
    cfg.encoder_num_heads = 4
    cfg.attention_mlp_multiple = 2.0
    cfg.feature_mlp_hidden_dim = 16
    cfg.masked_token_loss_weight = 1.0
    cfg.masked_latent_predictor_num_layers = 1
    cfg.jepa_num_target_blocks = 1
    cfg.num_peaks = 8
    cfg.enable_wandb = False
    cfg.seed = 3
    model = build_model_from_config(cfg)
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizers=[],
        schedulers=[],
        global_step=9,
        epoch=1,
        loss=0.1,
        wandb_run_id="wandb-run-1",
    )
    calls = []
    logger_configs = []
    logger_logs = []

    def fake_run_msg_probe(**kwargs):
        calls.append(kwargs)
        return {
            "msg_probe/mean/test/auc_maccs_mean": 0.5,
            "msg_probe/mean/test/pr_curve_sulfur": PrecisionRecallCurve(
                label="sulfur",
                targets=torch.tensor([0, 1]).numpy(),
                probabilities=torch.tensor([0.2, 0.8]).numpy(),
                title="sulfur pr",
            ),
        }

    class FakeLogger:
        def log_metrics(self, metrics, step=None) -> None:
            logger_logs.append((metrics, step))

    def fake_build_logger(config, workdir):
        logger_configs.append(config.copy_and_resolve_references())
        return FakeLogger()

    monkeypatch.setattr(checkpoint_probe, "run_msg_probe", fake_run_msg_probe)
    monkeypatch.setattr(checkpoint_probe, "build_logger", fake_build_logger)
    monkeypatch.setattr(checkpoint_probe.torch.cuda, "is_available", lambda: False)

    metrics = checkpoint_probe.run_checkpoint_msg_probe(
        config_json=json.dumps(cfg.to_dict()),
        checkpoint_path=checkpoint_path,
        workdir=tmp_path / "probe",
        global_step=9,
    )

    assert len(calls) == 1
    assert calls[0]["device"].type == "cpu"
    assert logger_configs[0].enable_wandb is True
    assert logger_configs[0].wandb_project == "msg-probe-evaluations"
    assert logger_configs[0].wandb_resume_id == ""
    assert logger_configs[0].wandb_resume_from_env is False
    assert logger_configs[0].wandb_shared_mode is False
    assert logger_configs[0].source_wandb_run_id == "wandb-run-1"
    assert logger_configs[0].msg_probe_checkpoint_path == str(checkpoint_path)
    assert logger_configs[0].msg_probe_global_step == 9
    assert logger_configs[0].wandb_kwargs["name"] == "msg_probe_step-00000009_checkpoint"
    assert logger_logs[0][0]["global_step"] == 9.0
    assert logger_logs[0][1] is None
    assert metrics["msg_probe/mean/test/auc_maccs_mean"] == 0.5
    metrics_json = json.loads(
        (tmp_path / "probe" / "msg_probe_step-00000009.json").read_text()
    )
    assert metrics_json == {"msg_probe/mean/test/auc_maccs_mean": 0.5}


def test_run_msg_probe_evaluation_logs_supplied_model_to_standalone_wandb(
    monkeypatch,
    tmp_path: Path,
):
    from spectra_learning.config import load_config

    cfg = load_config("configs/100m_pairmixer_dense_adamw.py")
    fake_model = torch.nn.Linear(1, 1)
    run_calls = []

    def fake_run_msg_probe(**kwargs):
        run_calls.append(kwargs)
        return {
            "msg_probe/mean/test/auc_sulfur": 0.75,
            "msg_probe/mean/test/pr_curve_sulfur": PrecisionRecallCurve(
                label="sulfur",
                targets=torch.tensor([0, 1, 1]).numpy(),
                probabilities=torch.tensor([0.1, 0.8, 0.6]).numpy(),
                title="sulfur pr",
            ),
        }

    init_calls = []
    wandb_logs = []

    class FakeRun:
        def __init__(self) -> None:
            self.config = SimpleNamespace(update=lambda *args, **kwargs: None)

        def define_metric(self, *args, **kwargs) -> None:
            pass

        def log(self, metrics, step=None) -> None:
            wandb_logs.append((metrics, step))

    def fake_init(**kwargs):
        init_calls.append(kwargs)
        return FakeRun()

    def fake_pr_curve(**kwargs):
        return {"native_pr": kwargs}

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(
            init=fake_init,
            Image=lambda figure: {"image": figure.__class__.__name__},
            plot=SimpleNamespace(pr_curve=fake_pr_curve),
        ),
    )
    monkeypatch.setattr(checkpoint_probe, "run_msg_probe", fake_run_msg_probe)

    metrics = checkpoint_probe.run_msg_probe_evaluation(
        config=cfg,
        model=fake_model,
        workdir=tmp_path / "probe",
        global_step=123,
        checkpoint_path="gs://bucket/checkpoints/step-00000123.pt",
        wandb_project="standalone-probes",
    )

    assert run_calls[0]["model"] is fake_model
    assert init_calls[0]["project"] == "standalone-probes"
    assert init_calls[0]["name"] == "msg_probe_step-00000123_step-00000123"
    assert "id" not in init_calls[0]
    assert wandb_logs[0][1] is None
    payload = wandb_logs[0][0]
    assert payload["global_step"] == 123.0
    assert payload["msg_probe/mean/test/auc_sulfur"] == 0.75
    assert "msg_probe/mean/test/pr_curve_sulfur/image" in payload
    assert "msg_probe/mean/test/pr_curve_sulfur/native" in payload
    assert metrics["msg_probe/mean/test/auc_sulfur"] == 0.75
    metrics_json = json.loads(
        (tmp_path / "probe" / "msg_probe_step-00000123.json").read_text()
    )
    assert metrics_json == {"msg_probe/mean/test/auc_sulfur": 0.75}


def test_run_checkpoint_msg_probe_script_infers_step_and_applies_overrides(monkeypatch):
    from scripts import run_checkpoint_msg_probe as script

    cfg = config_dict.ConfigDict()
    cfg.seed = 3
    cfg.msg_probe_num_epochs = 100
    calls = []

    def fake_load_config(path, overrides):
        assert path == Path("configs/base.py")
        cfg.update(overrides)
        return cfg

    def fake_load_torch_checkpoint(path, *, map_location, weights_only):
        calls.append(("load_checkpoint", path, map_location, weights_only))
        return {"global_step": 12}

    def fake_run_checkpoint_msg_probe(**kwargs):
        calls.append(("run_probe", kwargs))
        config_payload = json.loads(kwargs["config_json"])
        assert config_payload["seed"] == 3
        assert config_payload["msg_probe_num_epochs"] == 2
        return {"msg_probe/mean/test/auc_maccs_mean": 0.75}

    monkeypatch.setattr(script, "load_config", fake_load_config)
    monkeypatch.setattr(script, "load_torch_checkpoint", fake_load_torch_checkpoint)
    monkeypatch.setattr(
        script,
        "run_checkpoint_msg_probe",
        fake_run_checkpoint_msg_probe,
    )

    metrics = script.main(
        [
            "--config",
            "configs/base.py",
            "--checkpoint",
            "checkpoints/step-00000012.pt",
            "--workdir",
            "experiments/probe",
            "--overrides-json",
            '{"msg_probe_num_epochs": 2}',
        ]
    )

    assert metrics["msg_probe/mean/test/auc_maccs_mean"] == 0.75
    assert calls[0][0] == "load_checkpoint"
    assert calls[1][0] == "run_probe"
    run_kwargs = calls[1][1]
    assert json.loads(run_kwargs["config_json"]) == {
        "seed": 3,
        "msg_probe_num_epochs": 2,
    }
    assert run_kwargs["checkpoint_path"] == "checkpoints/step-00000012.pt"
    assert run_kwargs["workdir"] == "experiments/probe"
    assert run_kwargs["global_step"] == 12
    assert run_kwargs["wandb_project"] == script.DEFAULT_STANDALONE_WANDB_PROJECT


def test_build_optimizers_uses_single_adamw_optimizer_by_default():
    model = _small_model()
    cfg = _optimizer_config()

    optimizers, schedulers = build_optimizers(
        cfg,
        model,
        total_steps=10,
        device=torch.device("cpu"),
    )

    assert len(optimizers) == 1
    assert len(schedulers) == 1


def test_estimate_training_flops_per_optimizer_step_uses_trainable_params():
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 3),
        torch.nn.Linear(3, 2),
    )
    model[1].requires_grad_(False)
    cfg = config_dict.ConfigDict()

    flops_per_step = estimate_training_flops_per_optimizer_step(
        cfg,
        model,
        global_batch_size=8,
    )

    assert flops_per_step == 6.0 * sum(p.numel() for p in model[0].parameters()) * 8
    assert cumulative_training_flops(5, flops_per_step) == 5 * flops_per_step


def test_estimate_training_flops_per_optimizer_step_supports_config_override():
    model = torch.nn.Linear(2, 2)
    cfg = config_dict.ConfigDict()
    cfg.training_flops_per_optimizer_step = 123.0

    assert (
        estimate_training_flops_per_optimizer_step(
            cfg,
            model,
            global_batch_size=8,
        )
        == 123.0
    )


def test_log_train_metrics_includes_cumulative_flops():
    cfg = _optimizer_config()
    logger = _FakeLogger()
    pbar = _FakePbar()
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(()))], lr=0.1)

    pretrain.log_train_metrics(
        cfg,
        logger,
        pbar,
        {"loss": torch.tensor(2.0)},
        [optimizer],
        epoch=3,
        global_step=5,
        every_n_steps=1,
        flops_per_optimizer_step=1.5e12,
    )

    payload, step = logger.logs[-1]
    assert step == 5
    assert payload["train/cumulative_flops"] == 7.5e12
    assert payload["train/cumulative_peta_flops"] == 0.0075
    assert payload["train/flops_per_optimizer_step"] == 1.5e12
    assert payload["global_step"] == 5
    assert pbar.postfix == {"loss": "2.0000", "step": 5}


def test_build_optimizers_do_not_include_standalone_covariance_pooler():
    cfg = _optimizer_config()
    model = _small_model()
    pooler = CovariancePool(
        input_dim=model.model_dim,
        compressed_dim=4,
    )
    optimizers, _ = build_optimizers(
        cfg,
        model,
        total_steps=10,
        device=torch.device("cpu"),
    )

    pooler_param_ids = {id(param) for param in pooler.parameters()}
    optimizer_param_ids = set().union(
        *(_optimizer_param_ids(optimizer) for optimizer in optimizers)
    )

    assert pooler_param_ids.isdisjoint(optimizer_param_ids)


def test_build_optimizers_uses_single_adamw_optimizer():
    model = _small_model()
    cfg = _optimizer_config()

    optimizers, schedulers = build_optimizers(
        cfg,
        model,
        total_steps=10,
        device=torch.device("cpu"),
    )
    optimizer = optimizers[0]
    optimized_param_ids = _optimizer_param_ids(optimizer)
    trainable_param_ids = {
        id(param) for param in model.parameters() if param.requires_grad
    }

    assert len(optimizers) == 1
    assert len(schedulers) == 1
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimized_param_ids == trainable_param_ids
    assert {group["weight_decay"] for group in optimizer.param_groups} == {0.0, 0.01}


def test_load_resume_model_state_rejects_removed_cls_predictor_keys():
    model = _small_model()
    resume_state = model.state_dict()
    resume_state["cls_predictor.0.weight"] = torch.ones(model.model_dim)
    resume_state["cls_predictor.0.bias"] = torch.zeros(model.model_dim)
    resume_state["cls_predictor.1.weight"] = torch.randn(
        model.predictor_dim,
        model.model_dim,
    )
    resume_state["cls_predictor.3.weight"] = torch.randn(
        model.model_dim,
        model.predictor_dim,
    )

    restored = _small_model()
    with pytest.raises(RuntimeError, match="Unexpected key"):
        load_resume_model_state(restored, resume_state)


def test_load_resume_model_state_rejects_removed_target_projector():
    model = _small_model()
    resume_state = model.state_dict()

    restored = _small_model(target_projector_dim=-1)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        load_resume_model_state(restored, resume_state)


def test_load_resume_model_state_rejects_missing_ema_target_projector():
    model = _small_model()
    resume_state = model.state_dict()

    restored = _small_model(use_ema_teacher=True)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_resume_model_state(restored, resume_state)


def test_load_resume_model_state_rejects_embedded_covariance_pooler_keys():
    model = _small_model()
    resume_state = model.state_dict()
    resume_state["covariance_pooler.left_proj.weight"] = torch.randn(4, model.model_dim)

    restored = _small_model()
    with pytest.raises(RuntimeError, match="Unexpected key"):
        load_resume_model_state(restored, resume_state)


def test_load_resume_covariance_pooler_state_requires_sidecar_key():
    model = _small_model()
    pooler = CovariancePool(input_dim=model.model_dim, compressed_dim=4)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "step-00000012.pt"
        ckpt = {"model": model.state_dict()}
        with pytest.raises(KeyError, match="covariance_pooler_checkpoint"):
            load_resume_covariance_pooler_state(pooler, path, ckpt)


def test_wandb_logger_defines_msg_probe_global_step(monkeypatch, tmp_path: Path):
    class FakeRun:
        def __init__(self) -> None:
            self.config = SimpleNamespace(update=lambda *args, **kwargs: None)
            self.definitions = []
            self.finished = False

        def define_metric(self, *args, **kwargs) -> None:
            self.definitions.append((args, kwargs))

        def finish(self) -> None:
            self.finished = True

    fake_run = FakeRun()
    init_calls = []

    def fake_init(**kwargs):
        init_calls.append(kwargs)
        return fake_run

    class FakeSettings:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(init=fake_init, Settings=FakeSettings),
    )
    cfg = config_dict.ConfigDict()
    cfg.enable_wandb = True
    cfg.wandb_project = "test-project"

    logger = WandbMetricLogger(cfg, tmp_path)
    logger.finish()

    assert logger.experiment is fake_run
    assert init_calls[0]["project"] == "test-project"
    assert init_calls[0]["config"] == config_to_dict(cfg)
    assert fake_run.finished is True
    assert "settings" not in init_calls[0]
    assert fake_run.definitions == [
        (("global_step",), {}),
        (("train/*",), {"step_metric": "global_step"}),
        (("val/*",), {"step_metric": "global_step"}),
        (("msg_probe/*",), {"step_metric": "global_step"}),
        (("run/*",), {"step_metric": "global_step"}),
        (("model/*",), {"step_metric": "global_step"}),
    ]


def test_wandb_logger_uses_non_primary_shared_settings(monkeypatch, tmp_path: Path):
    class FakeRun:
        config = SimpleNamespace(update=lambda *args, **kwargs: None)

        def define_metric(self, *args, **kwargs) -> None:
            pass

    init_calls = []

    def fake_init(**kwargs):
        init_calls.append(kwargs)
        return FakeRun()

    class FakeSettings:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(init=fake_init, Settings=FakeSettings),
    )
    cfg = config_dict.ConfigDict()
    cfg.enable_wandb = True
    cfg.wandb_project = "test-project"
    cfg.wandb_shared_mode = True
    cfg.wandb_shared_primary = False
    cfg.wandb_shared_label = "probe_step_100"
    cfg.wandb_shared_update_finish_state = False

    WandbMetricLogger(cfg, tmp_path)

    assert init_calls[0]["settings"].kwargs == {
        "mode": "shared",
        "x_label": "probe_step_100",
        "x_primary": False,
        "x_update_finish_state": False,
    }


def test_serialise_metrics_expands_pr_curves_for_wandb(monkeypatch):
    from spectra_learning.training import logging as logging_module

    curve = PrecisionRecallCurve(
        label="sulfur",
        targets=torch.tensor([0, 1]).numpy(),
        probabilities=torch.tensor([0.2, 0.8]).numpy(),
        title="sulfur pr",
    )
    monkeypatch.setattr(
        logging_module,
        "_wandb_precision_recall_image",
        lambda value: f"image:{value.label}",
    )
    monkeypatch.setattr(
        logging_module,
        "_wandb_precision_recall_native",
        lambda value: f"native:{value.label}",
    )

    csv_metrics = _serialise_metrics({"metric": 1.0, "curve": curve})
    wandb_metrics = _serialise_metrics(
        {"metric": 1.0, "curve": curve},
        enable_wandb_artifacts=True,
    )

    assert csv_metrics == {"metric": 1.0}
    assert wandb_metrics == {
        "metric": 1.0,
        "curve/image": "image:sulfur",
        "curve/native": "native:sulfur",
    }


def test_train_main_writes_json_safe_probe_metrics(monkeypatch, tmp_path: Path):
    import train as train_script

    curve = PrecisionRecallCurve(
        label="sulfur",
        targets=np.asarray([0, 1]),
        probabilities=np.asarray([0.2, 0.8]),
        title="sulfur pr",
    )
    metrics_path = tmp_path / "metrics.json"
    cfg = config_dict.ConfigDict()

    monkeypatch.setattr(train_script, "load_config", lambda path, overrides: cfg)
    monkeypatch.setattr(
        train_script,
        "train_and_evaluate",
        lambda config, workdir: {
            "run/jax_process_index": 0.0,
            "metric": np.float32(1.25),
            "vector": np.asarray([1.0, 2.0]),
            "msg_probe/mean/test/pr_curve_sulfur": curve,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            "config.py",
            "--workdir",
            str(tmp_path / "work"),
            "--metrics-json",
            str(metrics_path),
        ],
    )

    train_script.main()

    payload = json.loads(metrics_path.read_text())
    assert payload["metric"] == pytest.approx(1.25)
    assert payload["vector"] == [1.0, 2.0]
    assert "msg_probe/mean/test/pr_curve_sulfur" not in payload


def test_log_msg_probe_metrics_uses_custom_global_step_for_wandb():
    class FakeLogger:
        def __init__(self) -> None:
            self.logs = []

        def log_metrics(self, metrics, step=None) -> None:
            self.logs.append((metrics, step))

    logger = FakeLogger()

    log_msg_probe_metrics(
        logger,
        {"msg_probe/test/auc_maccs_mean": 0.5},
        200,
        enable_wandb=True,
    )

    assert logger.logs == [
        (
            {
                "global_step": 200.0,
                "msg_probe/test/auc_maccs_mean": 0.5,
            },
            None,
        )
    ]


def test_log_msg_probe_metrics_uses_explicit_step_for_csv_logger():
    class FakeLogger:
        def __init__(self) -> None:
            self.logs = []

        def log_metrics(self, metrics, step=None) -> None:
            self.logs.append((metrics, step))

    logger = FakeLogger()

    log_msg_probe_metrics(
        logger,
        {"msg_probe/test/auc_maccs_mean": 0.5},
        200,
        enable_wandb=False,
    )

    assert logger.logs == [
        ({"msg_probe/test/auc_maccs_mean": 0.5}, 200),
    ]


def test_build_wandb_init_kwargs_prefers_config_resume_id(monkeypatch):
    monkeypatch.delenv("WANDB_RESUME_ID", raising=False)
    cfg = config_dict.ConfigDict()
    cfg.wandb_kwargs = {"name": "fresh-run"}
    cfg.wandb_resume_id = "resume-123"

    kwargs = _build_wandb_init_kwargs(cfg)

    assert kwargs["id"] == "resume-123"
    assert kwargs["resume"] == "must"
    assert "name" not in kwargs


def test_build_wandb_init_kwargs_can_ignore_resume_env(monkeypatch):
    monkeypatch.setenv("WANDB_RESUME_ID", "resume-from-env")
    cfg = config_dict.ConfigDict()
    cfg.wandb_resume_from_env = False
    cfg.wandb_kwargs = {"name": "fresh-standalone-run"}

    kwargs = _build_wandb_init_kwargs(cfg)

    assert kwargs == {"name": "fresh-standalone-run"}


def test_is_weight_decay_target_matches_pretrain_expectation():
    model = _small_model()
    assert is_weight_decay_target(
        "encoder.embedder.output_proj.weight",
        model.encoder.embedder.output_proj.weight,
    )
    assert is_weight_decay_target(
        "encoder.embedder.fourier_ffn.0.weight",
        model.encoder.embedder.fourier_ffn[0].weight,
    )
    assert not is_weight_decay_target(
        "encoder.embedder.mz_fourier.b",
        model.encoder.embedder.mz_fourier.b,
    )


def test_jepa_mae_mz_scale_follows_peak_mz_preprocessing_scale():
    cfg = config_dict.ConfigDict()
    cfg.model_dim = 32
    cfg.encoder_num_layers = 1
    cfg.encoder_num_heads = 4
    cfg.attention_mlp_multiple = 2.0
    cfg.feature_mlp_hidden_dim = 16
    cfg.num_peaks = 8
    cfg.peak_mz_max = 750.0
    cfg.encoder_fourier_input_scale = 1000.0
    cfg.max_precursor_mz = 2000.0
    cfg.jepa_mae_loss_weight = 1.0
    cfg.jepa_mae_mz_bin_size = 2.5

    model = build_model_from_config(cfg)

    assert model.jepa_mae_mz_max == 750.0
    assert model.jepa_mae_num_mz_bins == 300
    assert model.encoder.pair_embedder.mz_scale == 750.0
    assert model.encoder.pair_embedder.precursor_mz_scale == 2000.0
    assert model.encoder.pair_embedder.pair_fourier.num_freqs == 16
    torch.testing.assert_close(
        model.encoder.pair_embedder.pair_fourier.b[0, -1],
        torch.tensor(1.0 / 750.0),
    )


def test_config_can_disable_encoder_fourier_features():
    cfg = config_dict.ConfigDict()
    cfg.model_dim = 32
    cfg.encoder_num_layers = 1
    cfg.encoder_num_heads = 4
    cfg.attention_mlp_multiple = 2.0
    cfg.feature_mlp_hidden_dim = 16
    cfg.num_peaks = 8
    cfg.encoder_use_fourier_features = False
    cfg.encoder_fourier_mlp_hidden_dim = 64
    cfg.encoder_fourier_mlp_num_layers = 4

    model = build_model_from_config(cfg)

    assert not model.encoder.embedder.use_fourier_features
    assert not hasattr(model.encoder.embedder, "mz_fourier")
    raw_layers = [
        layer
        for layer in model.encoder.embedder.raw_ffn
        if isinstance(layer, torch.nn.Linear)
    ]
    assert len(raw_layers) == 4
    assert raw_layers[0].out_features == 64
    assert raw_layers[-1].out_features == model.model_dim
