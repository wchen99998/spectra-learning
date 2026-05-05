from contextlib import nullcontext

import torch

from spectra_learning.models.diagnostics import _collapse_diagnostics
from spectra_learning.models.model import PeakSetSIGReg
from spectra_learning.training.distributed import unwrap_model


def _forward_augmented_for_batch(
    model: PeakSetSIGReg,
    batch: dict[str, torch.Tensor],
    *,
    return_collapse_data: bool,
):
    if return_collapse_data:
        return model(batch, return_collapse_data=True)
    return model(batch)


def train_step_impl(
    model: PeakSetSIGReg,
    batch: dict[str, torch.Tensor],
    optimizers: list[torch.optim.Optimizer],
    schedulers: list[torch.optim.lr_scheduler.LRScheduler],
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
            collapse_data = {}
    if compute_collapse_metrics and collapse_data:
        with torch.no_grad():
            metrics.update(_collapse_diagnostics(**collapse_data))
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
    ema_momentum = unwrap_model(model).update_ema_teacher(global_step + 1, total_steps)
    for scheduler in schedulers:
        scheduler.step()
    if ema_momentum is not None:
        metrics["ema_teacher_momentum"] = metrics["loss"].new_tensor(ema_momentum)
    return metrics
