import copy
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch

from spectra_learning.data.contracts import validate_peak_preprocessing_contract
from spectra_learning.models.model import PeakSetJEPA
from spectra_learning.training.schedules import LRSchedulerLike
from spectra_learning.training.storage import (
    StoragePath,
    is_remote_path,
    list_storage_files,
    local_cache_path,
    local_path_for_read,
    storage_delete,
    storage_delete_if_exists,
    storage_join,
    storage_name,
    storage_with_name,
    upload_local_file,
)

COVARIANCE_POOLER_CHECKPOINT_PREFIX = "covariance-pooler-"


def covariance_pooler_checkpoint_path(path: StoragePath) -> StoragePath:
    return storage_with_name(
        path,
        f"{COVARIANCE_POOLER_CHECKPOINT_PREFIX}{storage_name(path)}",
    )


def is_main_checkpoint_path(path: StoragePath) -> bool:
    return not storage_name(path).startswith(COVARIANCE_POOLER_CHECKPOINT_PREFIX)


def is_training_checkpoint_path(path: StoragePath) -> bool:
    name = storage_name(path)
    return is_main_checkpoint_path(path) and (
        name == "last.pt" or name.startswith("step-")
    )


def optimizer_state_dict(optimizer: torch.optim.Optimizer) -> dict:
    return optimizer.state_dict()


def grad_scaler_state_dict(grad_scaler: torch.amp.GradScaler | None) -> dict | None:
    if grad_scaler is None or not grad_scaler.is_enabled():
        return None
    return grad_scaler.state_dict()


def _snapshot_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {key: _snapshot_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot_value(item) for item in value)
    return copy.deepcopy(value)


def _training_checkpoint_state(
    path: StoragePath,
    model: PeakSetJEPA,
    optimizers: list[torch.optim.Optimizer],
    schedulers: list[LRSchedulerLike],
    global_step: int,
    epoch: int,
    loss: float,
    wandb_run_id: str | None,
    covariance_pooler: torch.nn.Module | None,
    grad_scaler: torch.amp.GradScaler | None,
    peak_preprocessing: dict[str, Any],
    data_provenance: dict[str, Any],
) -> tuple[dict[str, Any], StoragePath | None, dict[str, Any] | None]:
    pooler_path = (
        covariance_pooler_checkpoint_path(path)
        if covariance_pooler is not None
        else None
    )
    state = {
        "model": model.state_dict(),
        "optimizers": [optimizer_state_dict(opt) for opt in optimizers],
        "schedulers": [sched.state_dict() for sched in schedulers],
        "grad_scaler": grad_scaler_state_dict(grad_scaler),
        "global_step": global_step,
        "epoch": epoch,
        "loss": loss,
        "wandb_run_id": wandb_run_id,
        "peak_preprocessing": peak_preprocessing,
        "data_provenance": data_provenance,
        "covariance_pooler_checkpoint": (
            storage_name(pooler_path) if pooler_path is not None else None
        ),
    }
    pooler_state = (
        {
            "pooler": covariance_pooler.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
        }
        if covariance_pooler is not None
        else None
    )
    return _snapshot_value(state), pooler_path, _snapshot_value(pooler_state)


def _write_torch_checkpoint(state: dict[str, Any], path: StoragePath) -> None:
    local_path = local_cache_path(path) if is_remote_path(path) else Path(path).expanduser()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = local_path.with_name(f".{local_path.name}.tmp")
    torch.save(_snapshot_value(state), tmp_path)
    tmp_path.replace(local_path)
    if is_remote_path(path):
        upload_local_file(local_path, path)


def save_torch_checkpoint(state: dict[str, Any], path: StoragePath) -> None:
    _write_torch_checkpoint(state, path)


def load_torch_checkpoint(
    path: StoragePath,
    *,
    map_location: torch.device | str = "cpu",
    weights_only: bool = True,
) -> dict[str, Any]:
    return torch.load(
        local_path_for_read(path),
        map_location=map_location,
        weights_only=weights_only,
    )


def _write_training_checkpoint_job(
    path: StoragePath,
    state: dict[str, Any],
    pooler_path: StoragePath | None,
    pooler_state: dict[str, Any] | None,
    prune_checkpoint_dir: StoragePath | None,
    keep_top_k: int | None,
) -> None:
    if pooler_path is not None and pooler_state is not None:
        _write_torch_checkpoint(pooler_state, pooler_path)
    _write_torch_checkpoint(state, path)
    if prune_checkpoint_dir is not None and keep_top_k is not None:
        prune_checkpoints(prune_checkpoint_dir, keep_top_k=keep_top_k)


class AsyncCheckpointWriter:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="checkpoint-writer",
        )
        self._futures: list[Future[None]] = []

    def save_checkpoint(
        self,
        path: Path | str,
        model: PeakSetJEPA,
        optimizers: list[torch.optim.Optimizer],
        schedulers: list[LRSchedulerLike],
        global_step: int,
        epoch: int,
        loss: float,
        wandb_run_id: str | None = None,
        *,
        peak_preprocessing: dict[str, Any],
        data_provenance: dict[str, Any],
        covariance_pooler: torch.nn.Module | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
        prune_checkpoint_dir: StoragePath | None = None,
        keep_top_k: int | None = None,
    ) -> None:
        state, pooler_path, pooler_state = _training_checkpoint_state(
            path,
            model,
            optimizers,
            schedulers,
            global_step,
            epoch,
            loss,
            wandb_run_id,
            covariance_pooler,
            grad_scaler,
            peak_preprocessing,
            data_provenance,
        )
        future = self._executor.submit(
            _write_training_checkpoint_job,
            path,
            state,
            pooler_path,
            pooler_state,
            prune_checkpoint_dir,
            keep_top_k,
        )
        self._futures.append(future)

    def log_completed_failures(self) -> None:
        pending = []
        for future in self._futures:
            if future.done():
                self._log_future_result(future)
            else:
                pending.append(future)
        self._futures = pending

    def wait(self) -> None:
        for future in self._futures:
            self._log_future_result(future)
        self._futures.clear()

    def close(self) -> None:
        self.wait()
        self._executor.shutdown(wait=True)

    def _log_future_result(self, future: Future[None]) -> None:
        future.result()


def prune_checkpoints(checkpoint_dir: StoragePath, keep_top_k: int = 5) -> None:
    pts = sorted(
        (
            entry
            for entry in list_storage_files(checkpoint_dir)
            if entry.name.startswith("step-") and is_main_checkpoint_path(entry.path)
        ),
        key=lambda entry: entry.mtime,
    )
    if len(pts) <= keep_top_k:
        return
    losses = [
        (
            load_torch_checkpoint(entry.path, map_location="cpu", weights_only=True)[
                "loss"
            ],
            entry,
        )
        for entry in pts
    ]
    losses.sort(key=lambda x: x[0])
    keep = {entry.path for _, entry in losses[:keep_top_k]}
    keep.add(pts[-1].path)
    for entry in pts:
        path = entry.path
        if path not in keep:
            storage_delete(path)
            storage_delete_if_exists(covariance_pooler_checkpoint_path(path))


def load_resume_model_state(
    model: PeakSetJEPA,
    state_dict: dict[str, torch.Tensor],
) -> None:
    model.load_state_dict(state_dict)


def load_resume_covariance_pooler_state(
    covariance_pooler: torch.nn.Module | None,
    checkpoint_path: StoragePath,
    checkpoint: dict,
) -> None:
    if covariance_pooler is None:
        return
    pooler_path = storage_with_name(
        checkpoint_path,
        checkpoint["covariance_pooler_checkpoint"],
    )
    pooler_ckpt = load_torch_checkpoint(
        pooler_path,
        map_location="cpu",
        weights_only=True,
    )
    covariance_pooler.load_state_dict(pooler_ckpt["pooler"])


def load_optimizer_state(optimizer: torch.optim.Optimizer, state: dict) -> None:
    optimizer.load_state_dict(state)


def load_grad_scaler_state(
    grad_scaler: torch.amp.GradScaler | None,
    state: dict | None,
) -> None:
    if grad_scaler is None or not grad_scaler.is_enabled() or not state:
        return
    grad_scaler.load_state_dict(state)


def load_pretrained_weights(
    model: PeakSetJEPA,
    checkpoint_path: StoragePath,
    *,
    config: Any,
) -> None:
    ckpt = load_torch_checkpoint(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    validate_peak_preprocessing_contract(ckpt, config)
    model.load_state_dict(ckpt["model"])


def load_frozen_teacher_weights(
    model: PeakSetJEPA,
    checkpoint_path: StoragePath,
    *,
    config: Any,
) -> None:
    ckpt = load_torch_checkpoint(checkpoint_path, map_location="cpu", weights_only=True)
    validate_peak_preprocessing_contract(ckpt, config)
    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in ckpt["model"].items()
        if key.startswith("encoder.")
    }
    teacher_encoder = model.teacher_encoder
    assert teacher_encoder is not None
    teacher_encoder.load_state_dict(encoder_state)
    teacher_target_projector = model.teacher_target_projector
    if teacher_target_projector is not None:
        projector_state = {
            key.removeprefix("target_projector."): value
            for key, value in ckpt["model"].items()
            if key.startswith("target_projector.")
        }
        teacher_target_projector.load_state_dict(projector_state)
    teacher_encoder.requires_grad_(False)
    if teacher_target_projector is not None:
        teacher_target_projector.requires_grad_(False)


def training_checkpoint_paths(checkpoint_dir: StoragePath) -> list[StoragePath]:
    checkpoints = sorted(
        (
            entry
            for entry in list_storage_files(checkpoint_dir)
            if entry.name.endswith(".pt") and is_training_checkpoint_path(entry.path)
        ),
        key=lambda entry: entry.mtime,
    )
    return [entry.path for entry in checkpoints]


def latest_ckpt_path(directory: StoragePath) -> str | None:
    checkpoint_dir = storage_join(directory, "checkpoints")
    ckpts = _latest_checkpoint_entries(checkpoint_dir)
    return str(ckpts[-1].path) if ckpts else None


def _latest_checkpoint_entries(directory: StoragePath) -> list:
    return sorted(
        (
            entry
            for entry in list_storage_files(directory)
            if entry.name.endswith(".pt") and is_training_checkpoint_path(entry.path)
        ),
        key=lambda entry: entry.mtime,
    )
