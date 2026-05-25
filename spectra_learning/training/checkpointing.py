from pathlib import Path

import torch

from spectra_learning.models.model import PeakSetJEPA
from spectra_learning.training.schedules import LRSchedulerLike


COVARIANCE_POOLER_PREFIX = "covariance_pooler."
COVARIANCE_POOLER_CHECKPOINT_PREFIX = "covariance-pooler-"


def covariance_pooler_checkpoint_path(path: Path | str) -> Path:
    path = Path(path)
    return path.with_name(f"{COVARIANCE_POOLER_CHECKPOINT_PREFIX}{path.name}")


def is_main_checkpoint_path(path: Path) -> bool:
    return not path.name.startswith(COVARIANCE_POOLER_CHECKPOINT_PREFIX)


def is_training_checkpoint_path(path: Path) -> bool:
    return is_main_checkpoint_path(path) and (
        path.name == "last.pt" or path.name.startswith("step-")
    )


def _model_state_without_legacy_pooler(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in state_dict.items()
        if not key.startswith(COVARIANCE_POOLER_PREFIX)
    }


def _legacy_pooler_state(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        key.removeprefix(COVARIANCE_POOLER_PREFIX): value
        for key, value in state_dict.items()
        if key.startswith(COVARIANCE_POOLER_PREFIX)
    }


def optimizer_state_dict(optimizer: torch.optim.Optimizer) -> dict:
    return optimizer.state_dict()


def grad_scaler_state_dict(grad_scaler: torch.amp.GradScaler | None) -> dict | None:
    if grad_scaler is None or not grad_scaler.is_enabled():
        return None
    return grad_scaler.state_dict()


def save_checkpoint(
    path: Path | str,
    model: PeakSetJEPA,
    optimizers: list[torch.optim.Optimizer],
    schedulers: list[LRSchedulerLike],
    global_step: int,
    epoch: int,
    loss: float,
    wandb_run_id: str | None = None,
    covariance_pooler: torch.nn.Module | None = None,
    grad_scaler: torch.amp.GradScaler | None = None,
) -> None:
    path = Path(path)
    pooler_path = (
        covariance_pooler_checkpoint_path(path)
        if covariance_pooler is not None
        else None
    )
    torch.save(
        {
            "model": model.state_dict(),
            "optimizers": [optimizer_state_dict(opt) for opt in optimizers],
            "schedulers": [sched.state_dict() for sched in schedulers],
            "grad_scaler": grad_scaler_state_dict(grad_scaler),
            "global_step": global_step,
            "epoch": epoch,
            "loss": loss,
            "wandb_run_id": wandb_run_id,
            "covariance_pooler_checkpoint": (
                pooler_path.name if pooler_path is not None else None
            ),
        },
        path,
    )
    if covariance_pooler is not None:
        pooler_save_path = covariance_pooler_checkpoint_path(path)
        torch.save(
            {
                "pooler": covariance_pooler.state_dict(),
                "global_step": global_step,
                "epoch": epoch,
            },
            pooler_save_path,
        )


def save_probe_checkpoint(
    path: Path | str,
    model: PeakSetJEPA,
    global_step: int,
    epoch: int,
    loss: float,
    wandb_run_id: str | None = None,
    covariance_pooler: torch.nn.Module | None = None,
) -> None:
    path = Path(path)
    pooler_path = (
        covariance_pooler_checkpoint_path(path)
        if covariance_pooler is not None
        else None
    )
    torch.save(
        {
            "model": model.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
            "loss": loss,
            "wandb_run_id": wandb_run_id,
            "covariance_pooler_checkpoint": (
                pooler_path.name if pooler_path is not None else None
            ),
        },
        path,
    )
    if covariance_pooler is not None:
        pooler_save_path = covariance_pooler_checkpoint_path(path)
        torch.save(
            {
                "pooler": covariance_pooler.state_dict(),
                "global_step": global_step,
                "epoch": epoch,
            },
            pooler_save_path,
        )


def prune_checkpoints(checkpoint_dir: Path, keep_top_k: int = 5) -> None:
    pts = sorted(
        (p for p in checkpoint_dir.glob("step-*.pt") if is_main_checkpoint_path(p)),
        key=lambda p: p.stat().st_mtime,
    )
    if len(pts) <= keep_top_k:
        return
    losses = [(torch.load(p, map_location="cpu", weights_only=True).get("loss", float("inf")), p) for p in pts]
    losses.sort(key=lambda x: x[0])
    keep = {p for _, p in losses[:keep_top_k]}
    keep.add(pts[-1])
    for path in pts:
        if path not in keep:
            path.unlink()
            pooler_path = covariance_pooler_checkpoint_path(path)
            if pooler_path.exists():
                pooler_path.unlink()


def load_resume_model_state(
    model: PeakSetJEPA,
    state_dict: dict[str, torch.Tensor],
) -> None:
    model.load_state_dict(_model_state_without_legacy_pooler(state_dict))


def load_resume_covariance_pooler_state(
    covariance_pooler: torch.nn.Module | None,
    checkpoint_path: Path | str,
    checkpoint: dict,
) -> None:
    if covariance_pooler is None:
        return
    checkpoint_path = Path(checkpoint_path)
    pooler_name = checkpoint.get("covariance_pooler_checkpoint", None)
    pooler_path = (
        checkpoint_path.with_name(pooler_name)
        if pooler_name
        else covariance_pooler_checkpoint_path(checkpoint_path)
    )
    if pooler_path.exists():
        pooler_ckpt = torch.load(pooler_path, map_location="cpu", weights_only=True)
        covariance_pooler.load_state_dict(pooler_ckpt["pooler"])
        return
    legacy_state = _legacy_pooler_state(checkpoint["model"])
    if legacy_state:
        covariance_pooler.load_state_dict(legacy_state)


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
    checkpoint_path: str,
) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt["state_dict"]
    model.load_state_dict(_model_state_without_legacy_pooler(state_dict))


def load_frozen_teacher_weights(
    model: PeakSetJEPA,
    checkpoint_path: str,
) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt["state_dict"]
    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in state_dict.items()
        if key.startswith("encoder.")
    }
    teacher_encoder = model.teacher_encoder
    assert teacher_encoder is not None
    teacher_encoder.load_state_dict(encoder_state)
    teacher_target_projector = model.teacher_target_projector
    if teacher_target_projector is not None:
        projector_state = {
            key.removeprefix("target_projector."): value
            for key, value in state_dict.items()
            if key.startswith("target_projector.")
        }
        teacher_target_projector.load_state_dict(projector_state)
    teacher_encoder.requires_grad_(False)
    if teacher_target_projector is not None:
        teacher_target_projector.requires_grad_(False)


def latest_ckpt_path(directory: Path) -> str | None:
    checkpoint_dir = directory / "checkpoints"
    root = checkpoint_dir if checkpoint_dir.exists() else directory
    ckpts = sorted(
        [
            *root.rglob("*.ckpt"),
            *(p for p in root.rglob("*.pt") if is_training_checkpoint_path(p)),
        ],
        key=lambda p: p.stat().st_mtime,
    )
    return str(ckpts[-1]) if ckpts else None
