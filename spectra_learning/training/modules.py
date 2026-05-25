import torch
from typing import cast

from spectra_learning.models.model import PeakSetJEPA


class PretrainModule(torch.nn.Module):
    def __init__(self, model: PeakSetJEPA) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        return_collapse_data: bool = False,
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        return self.model(
            batch,
            return_collapse_data=return_collapse_data,
        )

    def update_ema_teacher(self, global_step: int, total_steps: int) -> float | None:
        return self.model.update_ema_teacher(global_step, total_steps)


def split_pretrain_module(
    module: torch.nn.Module,
) -> tuple[PeakSetJEPA, torch.nn.Module | None]:
    if isinstance(module, PretrainModule):
        return module.model, None
    return cast(PeakSetJEPA, module), None
