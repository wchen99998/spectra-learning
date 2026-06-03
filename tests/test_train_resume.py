import tempfile
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import fsspec
import pytest
import torch
from ml_collections import config_dict

from spectra_learning.models.model import PeakSetJEPA
from spectra_learning.models.pooling import CovariancePool
from spectra_learning.training.checkpointing import (
    AsyncCheckpointWriter,
    covariance_pooler_checkpoint_path,
    is_training_checkpoint_path,
    latest_ckpt_path,
    load_pretrained_weights,
    load_torch_checkpoint,
    load_grad_scaler_state,
    load_resume_covariance_pooler_state,
    load_resume_model_state,
    save_checkpoint,
)
from spectra_learning.training import pretrain
from spectra_learning.training import checkpointing as checkpointing_module
from spectra_learning.probes.massspec import checkpoint_probe
from spectra_learning.training.modules import PretrainModule
from spectra_learning.training.optimization import (
    build_optimizers,
    is_weight_decay_target,
)
from spectra_learning.training.api import (
    _build_wandb_init_kwargs,
    build_grad_scaler,
    build_model_from_config,
    parse_autocast_dtype,
)
from spectra_learning.training.logging import WandbMetricLogger, log_msg_probe_metrics


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


def test_training_loop_resumes_with_offset_loader(monkeypatch, tmp_path: Path):
    cfg = config_dict.ConfigDict()
    cfg.autocast_dtype = "bf16"
    cfg.log_every_n_steps = 0
    cfg.collapse_metrics_every_n_steps = 0
    cfg.checkpoint_every_steps = 1000
    cfg.msg_probe_every_n_steps = 0
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


def test_training_loop_continues_while_checkpoint_save_is_pending(monkeypatch, tmp_path: Path):
    cfg = config_dict.ConfigDict()
    cfg.autocast_dtype = "bf16"
    cfg.log_every_n_steps = 0
    cfg.collapse_metrics_every_n_steps = 0
    cfg.checkpoint_every_steps = 1
    cfg.msg_probe_every_n_steps = 0
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
    cfg.msg_probe_every_n_steps = 0
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
    cfg.msg_probe_every_n_steps = 0
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

    def fake_run_msg_probe(*, config, model, device, distributed, covariance_pooler=None):
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

    def fake_run_msg_probe(**kwargs):
        calls.append(kwargs)
        return {"msg_probe/mean/test/auc_maccs_mean": 0.5}

    class FakeLogger:
        def log_metrics(self, metrics, step=None) -> None:
            pass

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
    assert logger_configs[0].wandb_resume_id == "wandb-run-1"
    assert logger_configs[0].wandb_shared_mode is True
    assert logger_configs[0].wandb_shared_primary is False
    assert logger_configs[0].wandb_shared_label == "probe_step_9"
    assert logger_configs[0].wandb_shared_update_finish_state is False
    assert metrics["msg_probe/mean/test/auc_maccs_mean"] == 0.5
    assert (tmp_path / "probe" / "msg_probe_step-00000009.json").exists()


def test_run_checkpoint_msg_probe_script_infers_step_and_applies_overrides(monkeypatch):
    from scripts import run_checkpoint_msg_probe as script

    cfg = config_dict.ConfigDict()
    cfg.seed = 3
    cfg.msg_probe_num_epochs = 100
    calls = []

    def fake_load_config(path):
        assert path == Path("configs/base.py")
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


def test_build_optimizers_do_not_include_standalone_covariance_pooler():
    cfg = _optimizer_config()
    model = _small_model()
    pooler = CovariancePool(
        input_dim=model.model_dim,
        compressed_dim=4,
    )
    module = PretrainModule(model)

    optimizers, _ = build_optimizers(
        cfg,
        module,
        total_steps=10,
        device=torch.device("cpu"),
    )

    pooler_param_ids = {id(param) for param in pooler.parameters()}
    optimizer_param_ids = set().union(
        *(_optimizer_param_ids(optimizer) for optimizer in optimizers)
    )

    assert pooler_param_ids.isdisjoint(optimizer_param_ids)


def test_build_optimizers_uses_official_torch_muon_and_adamw():
    model = _small_model()
    cfg = _optimizer_config(optimizer="muon")

    optimizers, schedulers = build_optimizers(
        cfg,
        model,
        total_steps=10,
        device=torch.device("cpu"),
    )
    qkv = model.encoder.blocks[0].single_attention.wqkv.weight
    muon_optimizer = next(opt for opt in optimizers if isinstance(opt, torch.optim.Muon))
    adamw_optimizer = next(opt for opt in optimizers if isinstance(opt, torch.optim.AdamW))
    muon_param_ids = _optimizer_param_ids(muon_optimizer)
    adamw_param_ids = _optimizer_param_ids(adamw_optimizer)

    assert len(optimizers) == 2
    assert len(schedulers) == 2
    assert not hasattr(muon_optimizer, "scalar_optimizer")
    assert id(qkv) in muon_param_ids
    assert all(param.ndim == 2 for group in muon_optimizer.param_groups for param in group["params"])
    assert muon_param_ids.isdisjoint(adamw_param_ids)


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


def test_load_resume_model_state_ignores_legacy_distogram_head_when_disabled():
    model = _small_model(distogram_loss_weight=1.0)
    resume_state = model.state_dict()

    restored = _small_model(distogram_loss_weight=0.0)
    load_resume_model_state(restored, resume_state)

    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, resume_state[key])


def test_load_resume_model_state_accepts_legacy_checkpoint_without_pair_mask_token():
    model = _small_model()
    resume_state = dict(model.state_dict())
    resume_state.pop("pair_mask_token")

    restored = _small_model()
    initial_pair_mask_token = restored.pair_mask_token.detach().clone()
    load_resume_model_state(restored, resume_state)

    restored_state = restored.state_dict()
    for key, value in resume_state.items():
        torch.testing.assert_close(restored_state[key], value)
    torch.testing.assert_close(restored.pair_mask_token, initial_pair_mask_token)


def test_load_pretrained_weights_accepts_legacy_checkpoint_without_pair_mask_token(tmp_path: Path):
    model = _small_model()
    state = dict(model.state_dict())
    state.pop("pair_mask_token")
    path = tmp_path / "legacy.pt"
    torch.save({"model": state}, path)

    restored = _small_model()
    initial_pair_mask_token = restored.pair_mask_token.detach().clone()
    load_pretrained_weights(restored, path)

    restored_state = restored.state_dict()
    for key, value in state.items():
        torch.testing.assert_close(restored_state[key], value)
    torch.testing.assert_close(restored.pair_mask_token, initial_pair_mask_token)


def test_load_pretrained_weights_accepts_old_checkpoint_with_stale_distogram_head(
    tmp_path: Path,
):
    model = _small_model(distogram_loss_weight=1.0)
    state = dict(model.state_dict())
    state.pop("pair_mask_token")
    path = tmp_path / "old.pt"
    torch.save({"model": state}, path)

    restored = _small_model(distogram_loss_weight=0.0)
    initial_pair_mask_token = restored.pair_mask_token.detach().clone()
    load_pretrained_weights(restored, path)

    restored_state = restored.state_dict()
    for key, value in state.items():
        if key.startswith("distogram_head."):
            continue
        torch.testing.assert_close(restored_state[key], value)
    assert not any(key.startswith("distogram_head.") for key in restored_state)
    torch.testing.assert_close(restored.pair_mask_token, initial_pair_mask_token)


def test_restore_training_state_accepts_legacy_optimizer_without_pair_mask_token(tmp_path: Path):
    cfg = _optimizer_config()
    source = _small_model()
    optimizers, schedulers = build_optimizers(cfg, source, 10, torch.device("cpu"))
    optimizer = optimizers[0]
    for param in source.parameters():
        param.grad = torch.ones_like(param)
    optimizer.step()

    sentinel_name = "encoder.embedder.fourier_ffn.0.bias"
    source_param = next(
        param for name, param in source.named_parameters() if name == sentinel_name
    )
    optimizer.state[source_param]["exp_avg"].fill_(7.0)
    optimizer.state[source_param]["exp_avg_sq"].fill_(11.0)

    optimizer_state = optimizer.state_dict()
    pair_group_idx = next(
        group_idx
        for group_idx, group in enumerate(optimizer.param_groups)
        if any(param is source.pair_mask_token for param in group["params"])
    )
    pair_param_idx = next(
        param_idx
        for param_idx, param in enumerate(optimizer.param_groups[pair_group_idx]["params"])
        if param is source.pair_mask_token
    )
    pair_state_key = optimizer_state["param_groups"][pair_group_idx]["params"].pop(
        pair_param_idx,
    )
    optimizer_state["state"].pop(pair_state_key)

    model_state = dict(source.state_dict())
    model_state.pop("pair_mask_token")
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    torch.save(
        {
            "model": model_state,
            "optimizers": [optimizer_state],
            "schedulers": [schedulers[0].state_dict()],
            "global_step": 3,
            "epoch": 0,
            "loss": 1.0,
        },
        checkpoint_dir / "step-00000003.pt",
    )

    restored = _small_model()
    restored_optimizers, restored_schedulers = build_optimizers(
        cfg,
        restored,
        10,
        torch.device("cpu"),
    )
    start_epoch, global_step, resume_offset = pretrain.restore_training_state(
        config=cfg,
        checkpoint_dir=checkpoint_dir,
        model=restored,
        optimizers=restored_optimizers,
        schedulers=restored_schedulers,
        steps_per_epoch=10,
        device=torch.device("cpu"),
    )

    restored_optimizer = restored_optimizers[0]
    restored_param = next(
        param for name, param in restored.named_parameters() if name == sentinel_name
    )
    assert (start_epoch, global_step, resume_offset) == (0, 3, 3)
    assert restored.pair_mask_token not in restored_optimizer.state
    torch.testing.assert_close(
        restored_optimizer.state[restored_param]["exp_avg"],
        torch.full_like(restored_optimizer.state[restored_param]["exp_avg"], 7.0),
    )
    torch.testing.assert_close(
        restored_optimizer.state[restored_param]["exp_avg_sq"],
        torch.full_like(restored_optimizer.state[restored_param]["exp_avg_sq"], 11.0),
    )


def test_load_resume_model_state_rejects_missing_distogram_head_when_enabled():
    model = _small_model(distogram_loss_weight=0.0)
    resume_state = model.state_dict()

    restored = _small_model(distogram_loss_weight=1.0)
    with pytest.raises(RuntimeError, match="Missing key"):
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


def test_wandb_logger_defines_msg_probe_global_step(monkeypatch, tmp_path: Path):
    class FakeRun:
        def __init__(self) -> None:
            self.config = SimpleNamespace(update=lambda *args, **kwargs: None)
            self.definitions = []

        def define_metric(self, *args, **kwargs) -> None:
            self.definitions.append((args, kwargs))

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

    assert logger.experiment is fake_run
    assert init_calls[0]["project"] == "test-project"
    assert "settings" not in init_calls[0]
    assert fake_run.definitions == [
        (("global_step",), {}),
        (("train/*",), {"step_metric": "global_step"}),
        (("msg_probe/*",), {"step_metric": "global_step"}),
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
