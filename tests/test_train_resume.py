import tempfile
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from ml_collections import config_dict

from spectra_learning.models.model import PeakSetSIGReg
from spectra_learning.models.pooling import CovariancePool
from spectra_learning.training.checkpointing import (
    covariance_pooler_checkpoint_path,
    is_training_checkpoint_path,
    latest_ckpt_path,
    load_grad_scaler_state,
    load_resume_covariance_pooler_state,
    load_resume_model_state,
    save_checkpoint,
)
from spectra_learning.training import pretrain
from spectra_learning.training import modal_probe as modal_probe_module
from spectra_learning.probes.massspec import checkpoint_probe
from spectra_learning.training.modules import PretrainModule
from spectra_learning.training.modal_probe import save_and_submit_modal_msg_probe
from spectra_learning.training.modal_probe import _path_is_modal_volume
from spectra_learning.training.modal_probe import _spawn_modal_probe_from_volume
from spectra_learning.training.optimization import (
    build_optimizers,
    is_predictor_parameter,
    is_weight_decay_target,
)
from spectra_learning.training.api import (
    _build_wandb_init_kwargs,
    build_grad_scaler,
    build_model_from_config,
    parse_autocast_dtype,
)
from spectra_learning.training.logging import WandbMetricLogger, log_msg_probe_metrics
from spectra_learning.training.schedules import WarmupCosineSchedule


def _small_model(**overrides) -> PeakSetSIGReg:
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
    return PeakSetSIGReg(**kwargs)


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


def test_latest_ckpt_path_ignores_modal_probe_checkpoints(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    step_path = checkpoint_dir / "step-00000012.pt"
    modal_probe_path = checkpoint_dir / "modal-probe-step-00000013.pt"
    pooler_path = checkpoint_dir / "covariance-pooler-step-00000012.pt"

    step_path.write_bytes(b"step")
    modal_probe_path.write_bytes(b"probe")
    pooler_path.write_bytes(b"pooler")

    assert is_training_checkpoint_path(step_path)
    assert not is_training_checkpoint_path(modal_probe_path)
    assert not is_training_checkpoint_path(pooler_path)
    assert latest_ckpt_path(tmp_path) == str(step_path)


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


def test_training_loop_modal_probe_submits_on_main_without_inline_probe(monkeypatch, tmp_path: Path):
    cfg = config_dict.ConfigDict()
    cfg.autocast_dtype = "bf16"
    cfg.log_every_n_steps = 0
    cfg.collapse_metrics_every_n_steps = 0
    cfg.checkpoint_every_steps = 1000
    cfg.msg_probe_every_n_steps = 2
    cfg.msg_probe_backend = "modal"
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

    submit_calls = []
    barriers = []

    def fake_train_step_impl(*args, **kwargs):
        return {"loss": torch.tensor(1.0)}

    def fake_run_msg_probe(**kwargs):
        raise AssertionError("inline probe should not run in modal mode")

    def fake_submit_modal_probe(**kwargs):
        submit_calls.append(kwargs)
        return {
            "msg_probe/modal/submitted": 1.0,
            "msg_probe/modal/call_id": "fc-test",
        }

    def fake_barrier(distributed):
        barriers.append((distributed.rank, distributed.world_size))

    monkeypatch.setattr(pretrain, "train_step_impl", fake_train_step_impl)
    monkeypatch.setattr(pretrain, "run_msg_probe", fake_run_msg_probe)
    monkeypatch.setattr(
        pretrain,
        "save_and_submit_modal_msg_probe",
        fake_submit_modal_probe,
    )
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

    assert len(submit_calls) == 1
    assert submit_calls[0]["global_step"] == 2
    assert main_logger.logs == [({"msg_probe/modal/submitted": 1.0}, 2)]
    assert worker_logger.logs == []
    assert barriers == [(0, 2), (0, 2), (1, 2), (1, 2)]
    assert main_metrics["msg_probe/modal/submitted"] == 1.0
    assert main_metrics["run/modal_probe_call_ids"] == ["fc-test"]
    assert main_metrics["run/modal_probe_calls"] == 1.0
    assert main_metrics["run/final_global_step"] == 2.0
    assert worker_metrics["run/final_global_step"] == 2.0


def test_save_and_submit_modal_msg_probe_writes_checkpoint_and_manifest(tmp_path: Path):
    cfg = config_dict.ConfigDict()
    cfg.seed = 7
    cfg.enable_wandb = False
    model = _small_model()
    calls = []

    def submitter(config_json, checkpoint_path, workdir, global_step):
        calls.append(
            {
                "config": json.loads(config_json),
                "checkpoint_path": checkpoint_path,
                "workdir": workdir,
                "global_step": global_step,
            }
        )
        return "call-123"

    metrics = save_and_submit_modal_msg_probe(
        config=cfg,
        model=model,
        covariance_pooler=None,
        checkpoint_dir=tmp_path,
        workdir=tmp_path,
        global_step=12,
        epoch=3,
        loss=0.25,
        wandb_run_id="wandb-run-1",
        submitter=submitter,
    )

    checkpoint_path = tmp_path / "modal-probe-step-00000012.pt"
    manifest_path = tmp_path / "modal-probe-step-00000012.submission.json"
    config_path = tmp_path / "modal-probe-step-00000012.config.json"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    manifest = json.loads(manifest_path.read_text())

    assert calls[0]["config"]["seed"] == 7
    assert calls[0]["checkpoint_path"] == checkpoint_path
    assert calls[0]["global_step"] == 12
    assert checkpoint["global_step"] == 12
    assert checkpoint["wandb_run_id"] == "wandb-run-1"
    assert "optimizers" not in checkpoint
    assert "schedulers" not in checkpoint
    assert config_path.exists()
    assert manifest["call_id"] == "call-123"
    assert metrics["msg_probe/modal/submitted"] == 1.0
    assert metrics["msg_probe/modal/call_id"] == "call-123"


def test_save_and_submit_modal_msg_probe_uses_detached_submit_process(monkeypatch, tmp_path: Path):
    cfg = config_dict.ConfigDict()
    cfg.seed = 7
    cfg.enable_wandb = False
    cfg.modal_probe_cli = "modal"
    model = _small_model()
    popen_calls = []

    class FakeProcess:
        pid = 1234

    def fake_popen(command, **kwargs):
        popen_calls.append((command, kwargs))
        return FakeProcess()

    monkeypatch.setattr(modal_probe_module.subprocess, "Popen", fake_popen)

    metrics = save_and_submit_modal_msg_probe(
        config=cfg,
        model=model,
        covariance_pooler=None,
        checkpoint_dir=tmp_path,
        workdir=tmp_path,
        global_step=12,
        epoch=3,
        loss=0.25,
    )

    command, kwargs = popen_calls[0]
    manifest_path = tmp_path / "modal-probe-step-00000012.submission.json"
    manifest = json.loads(manifest_path.read_text())

    assert command[:4] == ["modal", "run", "--detach", "modal_train.py"]
    assert "--submit-probe-checkpoint-path" in command
    assert "--submit-probe-config-json-path" in command
    assert kwargs["start_new_session"] is True
    assert manifest["call_id"] == "local-submit-pid-1234"
    assert metrics["msg_probe/modal/submitted"] == 1.0
    assert metrics["msg_probe/modal/call_id"] == "local-submit-pid-1234"


def test_modal_probe_volume_path_detection_uses_mount_path_without_resolving_symlinks():
    cfg = config_dict.ConfigDict()

    assert _path_is_modal_volume(
        Path("/vol/experiments/run/checkpoints/modal-probe-step-00000100.pt"),
        cfg,
    )


def test_spawn_modal_probe_from_volume_commits_before_spawn(monkeypatch):
    calls = []

    class FakeVolume:
        def commit(self):
            calls.append("commit")

    class FakeProbeFunction:
        def spawn(self, **kwargs):
            calls.append(("spawn", kwargs))
            return SimpleNamespace(object_id="fc-test")

    monkeypatch.setitem(
        sys.modules,
        "modal_train",
        SimpleNamespace(volume=FakeVolume(), run_probe_checkpoint=FakeProbeFunction()),
    )

    call_id = _spawn_modal_probe_from_volume(
        config_json="{}",
        checkpoint_path=Path("/vol/run/checkpoint.pt"),
        workdir=Path("/vol/run"),
        global_step=100,
    )

    assert call_id == "fc-test"
    assert calls[0] == "commit"
    assert calls[1][0] == "spawn"


def test_modal_train_waits_for_probe_function_calls(monkeypatch):
    import modal_train

    calls = []

    class FakeFunctionCall:
        @staticmethod
        def from_id(call_id):
            calls.append(("from_id", call_id))
            return f"call:{call_id}"

        @staticmethod
        def gather(*function_calls):
            calls.append(("gather", function_calls))
            return []

    monkeypatch.setattr(modal_train.modal, "FunctionCall", FakeFunctionCall)

    modal_train._wait_for_modal_probe_calls(
        {"run/modal_probe_call_ids": ["fc-1", "fc-2"]}
    )

    assert calls == [
        ("from_id", "fc-1"),
        ("from_id", "fc-2"),
        ("gather", ("call:fc-1", "call:fc-2")),
    ]


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


def test_build_optimizers_applies_predictor_learning_rate_ratio():
    model = _small_model()
    cfg = _optimizer_config(predictor_learning_rate_ratio=3.0)

    optimizers, schedulers = build_optimizers(
        cfg,
        model,
        total_steps=10,
        device=torch.device("cpu"),
    )

    assert len(optimizers) == 2
    assert len(schedulers) == 2
    assert all(
        float(group["lr"]) == pytest.approx(1e-3)
        for group in optimizers[0].param_groups
    )
    assert all(
        float(group["lr"]) == pytest.approx(3e-3)
        for group in optimizers[1].param_groups
    )
    assert cast(WarmupCosineSchedule, schedulers[0]).eta_min == pytest.approx(1e-4)
    assert cast(WarmupCosineSchedule, schedulers[1]).eta_min == pytest.approx(3e-4)

    predictor_param_ids = {
        id(param)
        for name, param in model.named_parameters()
        if param.requires_grad and is_predictor_parameter(name)
    }
    base_param_ids = _optimizer_param_ids(optimizers[0])
    actual_predictor_param_ids = _optimizer_param_ids(optimizers[1])
    all_trainable_param_ids = {
        id(param) for param in model.parameters() if param.requires_grad
    }

    assert actual_predictor_param_ids == predictor_param_ids
    assert base_param_ids.isdisjoint(actual_predictor_param_ids)
    assert base_param_ids | actual_predictor_param_ids == all_trainable_param_ids


def test_mae_value_heads_use_predictor_learning_rate_group():
    model = _small_model(training_mode="mae")
    cfg = _optimizer_config(predictor_learning_rate_ratio=2.0)

    optimizers, _ = build_optimizers(
        cfg,
        model,
        total_steps=10,
        device=torch.device("cpu"),
    )

    predictor_param_ids = {
        id(param)
        for name, param in model.named_parameters()
        if param.requires_grad and is_predictor_parameter(name)
    }
    actual_predictor_param_ids = _optimizer_param_ids(optimizers[1])
    value_head_names = {
        name
        for name, param in model.named_parameters()
        if id(param) in actual_predictor_param_ids and name.startswith("jepa_mae_")
    }

    assert actual_predictor_param_ids == predictor_param_ids
    assert value_head_names == {
        "jepa_mae_mz_head.weight",
        "jepa_mae_mz_head.bias",
        "jepa_mae_intensity_head.weight",
        "jepa_mae_intensity_head.bias",
    }


def test_build_optimizers_respects_frozen_covariance_pooling():
    cfg = _optimizer_config()
    trainable_model = _small_model(covariance_pooling_dim=4)
    trainable_pooler = CovariancePool(
        input_dim=trainable_model.model_dim,
        compressed_dim=4,
    )
    trainable_module = PretrainModule(trainable_model, trainable_pooler)
    frozen_model = _small_model(
        covariance_pooling_dim=4,
        train_covariance_pooling=False,
    )
    frozen_pooler = CovariancePool(input_dim=frozen_model.model_dim, compressed_dim=4)
    frozen_pooler.requires_grad_(False)
    frozen_module = PretrainModule(frozen_model, frozen_pooler)

    trainable_optimizers, _ = build_optimizers(
        cfg,
        trainable_module,
        total_steps=10,
        device=torch.device("cpu"),
    )
    frozen_optimizers, _ = build_optimizers(
        cfg,
        frozen_module,
        total_steps=10,
        device=torch.device("cpu"),
    )

    trainable_cov_ids = {id(param) for param in trainable_pooler.parameters()}
    frozen_cov_ids = {id(param) for param in frozen_pooler.parameters()}
    trainable_optimizer_ids = set().union(
        *(_optimizer_param_ids(optimizer) for optimizer in trainable_optimizers)
    )
    frozen_optimizer_ids = set().union(
        *(_optimizer_param_ids(optimizer) for optimizer in frozen_optimizers)
    )

    assert trainable_cov_ids <= trainable_optimizer_ids
    assert frozen_cov_ids.isdisjoint(frozen_optimizer_ids)


def test_build_optimizers_uses_official_torch_muon_and_adamw():
    model = _small_model()
    cfg = _optimizer_config(optimizer="muon")

    optimizers, schedulers = build_optimizers(
        cfg,
        model,
        total_steps=10,
        device=torch.device("cpu"),
    )
    qkv = model.encoder.blocks[0].single_attention.qkv.weight
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


def test_build_optimizers_splits_official_muon_predictor_lr():
    model = _small_model()
    cfg = _optimizer_config(optimizer="muon", predictor_learning_rate_ratio=3.0)

    optimizers, schedulers = build_optimizers(
        cfg,
        model,
        total_steps=10,
        device=torch.device("cpu"),
    )

    labels = [getattr(optimizer, "_spectra_lr_label") for optimizer in optimizers]
    lrs = [float(optimizer.param_groups[0]["lr"]) for optimizer in optimizers]
    predictor_param_ids = {
        id(param)
        for name, param in model.named_parameters()
        if param.requires_grad and is_predictor_parameter(name)
    }
    predictor_muon_ids = _optimizer_param_ids(optimizers[1])
    predictor_adamw_ids = _optimizer_param_ids(optimizers[3])

    assert labels == ["muon", "predictor_muon", "adamw", "predictor_adamw"]
    assert lrs == pytest.approx([1e-3, 3e-3, 1e-3, 3e-3])
    assert len(schedulers) == 4
    assert isinstance(optimizers[0], torch.optim.Muon)
    assert isinstance(optimizers[1], torch.optim.Muon)
    assert isinstance(optimizers[2], torch.optim.AdamW)
    assert isinstance(optimizers[3], torch.optim.AdamW)
    assert predictor_muon_ids | predictor_adamw_ids == predictor_param_ids


def test_load_resume_model_state_rejects_sigreg_checkpoint_drift():
    model = _small_model(representation_regularizer="sigreg", sigreg_lambda=0.02)
    resume_state = model.state_dict()
    for key in ("sigreg.t", "sigreg.phi", "sigreg.weights"):
        resume_state.pop(key)
    for key in tuple(resume_state):
        if key.startswith("target_projector."):
            resume_state.pop(key)
    resume_state["sigreg_lambda_target"] = torch.tensor(0.02)
    resume_state["sigreg_lambda_current"] = torch.tensor(0.02)
    resume_state["sigreg_lambda_step"] = torch.tensor(0)

    restored = _small_model(representation_regularizer="sigreg", sigreg_lambda=0.02)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_resume_model_state(restored, resume_state)


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


def test_load_resume_model_state_rejects_removed_special_tokens():
    model = _small_model(
        encoder_num_register_tokens=2,
        predictor_num_register_tokens=2,
    )
    resume_state = model.state_dict()

    restored = _small_model(
        encoder_use_cls_token=False,
        encoder_num_register_tokens=0,
        predictor_num_register_tokens=0,
    )
    with pytest.raises(RuntimeError, match="Unexpected key"):
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
    cfg.msg_probe_backend = "modal"

    logger = WandbMetricLogger(cfg, tmp_path)

    assert logger.experiment is fake_run
    assert init_calls[0]["project"] == "test-project"
    assert init_calls[0]["settings"].kwargs == {
        "mode": "shared",
        "x_label": "train",
        "x_primary": True,
    }
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
