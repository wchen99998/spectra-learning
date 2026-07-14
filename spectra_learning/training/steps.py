from contextlib import nullcontext
from typing import Any, Literal, cast, overload

import torch

from spectra_learning.models.diagnostics import _collapse_diagnostics
from spectra_learning.models.model import PeakSetJEPA
from spectra_learning.training.distributed import unwrap_model
from spectra_learning.training.schedules import LRSchedulerLike


@overload
def _forward_augmented_for_batch(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    return_collapse_data: Literal[True],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]: ...


@overload
def _forward_augmented_for_batch(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    return_collapse_data: Literal[False],
) -> dict[str, torch.Tensor]: ...


def _forward_augmented_for_batch(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    return_collapse_data: bool,
) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if return_collapse_data:
        return cast(
            tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
            model(batch, return_collapse_data=True),
        )
    return cast(dict[str, torch.Tensor], model(batch))


def train_step_impl(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    optimizers: list[torch.optim.Optimizer],
    schedulers: list[LRSchedulerLike],
    autocast_dtype: torch.dtype | None,
    grad_clip_norm: float | None,
    grad_scaler: torch.amp.GradScaler | None = None,
    compute_collapse_metrics: bool = False,
    global_step: int = 0,
    total_steps: int = 1,
    gradient_accumulation_steps: int = 1,
    accumulation_step: int = 0,
) -> dict[str, torch.Tensor]:
    device_type = next(model.parameters()).device.type
    autocast_ctx = (
        nullcontext()
        if autocast_dtype is None
        else torch.autocast(device_type=device_type, dtype=autocast_dtype)
    )
    optimizer_step = (accumulation_step + 1) % gradient_accumulation_steps == 0
    with _ddp_sync_context(model, optimizer_step):
        with autocast_ctx:
            if compute_collapse_metrics:
                metrics, collapse_data = _forward_augmented_for_batch(
                    model,
                    batch,
                    return_collapse_data=True,
                )
            else:
                metrics = _forward_augmented_for_batch(
                    model,
                    batch,
                    return_collapse_data=False,
                )
                collapse_data: dict[str, torch.Tensor] = {}
        if compute_collapse_metrics and collapse_data:
            with torch.no_grad():
                metrics.update(
                    _collapse_diagnostics(**cast(dict[str, Any], collapse_data))
                )
        step_skipped = _backward_and_step(
            metrics["loss"] / gradient_accumulation_steps,
            model,
            optimizers,
            grad_clip_norm,
            grad_scaler,
            optimizer_step=optimizer_step,
        )
    metrics = _clone_detached_metric_tensors(metrics)
    metrics["optimizer_step"] = metrics["loss"].new_tensor(float(optimizer_step))
    if gradient_accumulation_steps > 1:
        metrics["gradient_accumulation_steps"] = metrics["loss"].new_tensor(
            float(gradient_accumulation_steps)
        )
        metrics["micro_step"] = metrics["loss"].new_tensor(
            float(accumulation_step + 1)
        )
    if _grad_scaler_enabled(grad_scaler):
        metrics["grad_scale"] = metrics["loss"].new_tensor(float(grad_scaler.get_scale()))
        metrics["optimizer_step_skipped"] = metrics["loss"].new_tensor(
            float(step_skipped)
        )
    if not optimizer_step:
        return metrics
    base_model = cast(PeakSetJEPA, unwrap_model(model))
    ema_momentum = None
    if not step_skipped:
        ema_momentum = base_model.update_ema_teacher(global_step + 1, total_steps)
        for scheduler in schedulers:
            scheduler.step()
    if ema_momentum is not None:
        metrics["ema_teacher_momentum"] = metrics["loss"].new_tensor(ema_momentum)
    return metrics


def _backward_and_step(
    loss: torch.Tensor,
    model: torch.nn.Module,
    optimizers: list[torch.optim.Optimizer],
    grad_clip_norm: float | None,
    grad_scaler: torch.amp.GradScaler | None,
    *,
    optimizer_step: bool,
) -> bool:
    if not _grad_scaler_enabled(grad_scaler):
        loss.backward()
        if not optimizer_step:
            return False
        _clip_grad_norm(model, grad_clip_norm)
        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        return False

    grad_scaler.scale(loss).backward()
    if not optimizer_step:
        return False
    active_optimizers = [
        optimizer for optimizer in optimizers if _optimizer_has_grad(optimizer)
    ]
    if grad_clip_norm is not None and grad_clip_norm > 0:
        for optimizer in active_optimizers:
            grad_scaler.unscale_(optimizer)
        _clip_grad_norm(model, grad_clip_norm)
    previous_scale = float(grad_scaler.get_scale())
    for optimizer in active_optimizers:
        grad_scaler.step(optimizer)
    if active_optimizers:
        grad_scaler.update()
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    return bool(active_optimizers) and float(grad_scaler.get_scale()) < previous_scale


def _ddp_sync_context(
    model: torch.nn.Module,
    optimizer_step: bool,
):
    return (
        nullcontext()
        if optimizer_step or not hasattr(model, "no_sync")
        else model.no_sync()
    )


def _clone_detached_metric_tensors(
    metrics: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in metrics.items()}


def _grad_scaler_enabled(grad_scaler: torch.amp.GradScaler | None) -> bool:
    return grad_scaler is not None and grad_scaler.is_enabled()


def _optimizer_has_grad(optimizer: torch.optim.Optimizer) -> bool:
    return any(
        param.grad is not None
        for group in optimizer.param_groups
        for param in group["params"]
    )


def _clip_grad_norm(
    model: torch.nn.Module,
    grad_clip_norm: float | None,
) -> None:
    if grad_clip_norm is not None and grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=grad_clip_norm,
            foreach=True,
        )
