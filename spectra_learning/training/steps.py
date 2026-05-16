from contextlib import nullcontext
from typing import Any, Literal, cast, overload

import torch

from spectra_learning.models.diagnostics import _collapse_diagnostics
from spectra_learning.training.distributed import unwrap_model
from spectra_learning.training.modules import PretrainModule
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
    compute_collapse_metrics: bool = False,
    global_step: int = 0,
    total_steps: int = 1,
) -> dict[str, torch.Tensor]:
    device_type = next(model.parameters()).device.type
    autocast_ctx = (
        nullcontext()
        if autocast_dtype is None
        else torch.autocast(device_type=device_type, dtype=autocast_dtype)
    )
    torch.compiler.cudagraph_mark_step_begin()
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
            metrics.update(_collapse_diagnostics(**cast(dict[str, Any], collapse_data)))
    metrics["loss"].backward()
    if grad_clip_norm is not None and grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=grad_clip_norm,
            foreach=True,
        )
    for optimizer in optimizers:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    pretrain_module = cast(PretrainModule, unwrap_model(model))
    ema_momentum = pretrain_module.update_ema_teacher(global_step + 1, total_steps)
    for scheduler in schedulers:
        scheduler.step()
    if ema_momentum is not None:
        metrics["ema_teacher_momentum"] = metrics["loss"].new_tensor(ema_momentum)
    return metrics
