from pathlib import Path

import torch

from spectra_learning.models.model import PeakSetSIGReg


RUNTIME_PARAM_GROUP_KEYS = frozenset({"param_split_fn", "param_recombine_fn"})


def _strip_runtime_param_group_keys(state: dict) -> dict:
    if "param_groups" not in state:
        return state
    return {
        **state,
        "param_groups": [
            {
                key: value
                for key, value in group.items()
                if key not in RUNTIME_PARAM_GROUP_KEYS
            }
            for group in state["param_groups"]
        ],
    }


def optimizer_state_dict(optimizer: torch.optim.Optimizer) -> dict:
    state = _strip_runtime_param_group_keys(optimizer.state_dict())
    scalar_optimizer = getattr(optimizer, "scalar_optimizer", None)
    if scalar_optimizer is None:
        return state
    return {
        "state_dict": state,
        "scalar_optimizer_state": _strip_runtime_param_group_keys(
            scalar_optimizer.state_dict()
        ),
    }


def save_checkpoint(
    path: Path,
    model: PeakSetSIGReg,
    optimizers: list[torch.optim.Optimizer],
    schedulers: list[torch.optim.lr_scheduler.LRScheduler],
    global_step: int,
    epoch: int,
    loss: float,
    wandb_run_id: str | None = None,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizers": [optimizer_state_dict(opt) for opt in optimizers],
            "schedulers": [sched.state_dict() for sched in schedulers],
            "global_step": global_step,
            "epoch": epoch,
            "loss": loss,
            "wandb_run_id": wandb_run_id,
        },
        path,
    )


def prune_checkpoints(checkpoint_dir: Path, keep_top_k: int = 5) -> None:
    pts = sorted(checkpoint_dir.glob("step-*.pt"), key=lambda p: p.stat().st_mtime)
    if len(pts) <= keep_top_k:
        return
    losses = [(torch.load(p, map_location="cpu", weights_only=True).get("loss", float("inf")), p) for p in pts]
    losses.sort(key=lambda x: x[0])
    keep = {p for _, p in losses[:keep_top_k]}
    keep.add(pts[-1])
    for path in pts:
        if path not in keep:
            path.unlink()


def load_resume_model_state(
    model: PeakSetSIGReg,
    state_dict: dict[str, torch.Tensor],
) -> None:
    model.load_state_dict(state_dict)


def load_optimizer_state(optimizer: torch.optim.Optimizer, state: dict) -> None:
    scalar_optimizer = getattr(optimizer, "scalar_optimizer", None)
    if scalar_optimizer is None:
        optimizer.load_state_dict(state)
        return
    optimizer.load_state_dict(state["state_dict"])
    scalar_optimizer.load_state_dict(state["scalar_optimizer_state"])


def load_pretrained_weights(
    model: PeakSetSIGReg,
    checkpoint_path: str,
) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt["state_dict"]
    model.load_state_dict(state_dict)


def load_frozen_teacher_weights(
    model: PeakSetSIGReg,
    checkpoint_path: str,
) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt["state_dict"]
    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in state_dict.items()
        if key.startswith("encoder.")
    }
    model.teacher_encoder.load_state_dict(encoder_state)
    if model.teacher_target_projector is not None:
        projector_state = {
            key.removeprefix("target_projector."): value
            for key, value in state_dict.items()
            if key.startswith("target_projector.")
        }
        model.teacher_target_projector.load_state_dict(projector_state)
    model.teacher_encoder.requires_grad_(False)
    if model.teacher_target_projector is not None:
        model.teacher_target_projector.requires_grad_(False)


def latest_ckpt_path(directory: Path) -> str | None:
    checkpoint_dir = directory / "checkpoints"
    root = checkpoint_dir if checkpoint_dir.exists() else directory
    ckpts = sorted(
        [*root.rglob("*.ckpt"), *root.rglob("*.pt")],
        key=lambda p: p.stat().st_mtime,
    )
    return str(ckpts[-1]) if ckpts else None
