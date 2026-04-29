from pathlib import Path

import torch

from models.model import PeakSetSIGReg


def optimizer_state_dict(optimizer: torch.optim.Optimizer) -> dict:
    state = optimizer.state_dict()
    scalar_optimizer = getattr(optimizer, "scalar_optimizer", None)
    if scalar_optimizer is None:
        return state
    return {
        "state_dict": state,
        "scalar_optimizer_state": scalar_optimizer.state_dict(),
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
