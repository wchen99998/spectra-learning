from __future__ import annotations

import math
from collections.abc import Iterable, Iterator

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling

    def adapter_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.lora_a.parameters()
        yield from self.lora_b.parameters()


def apply_lora_to_linear_modules(
    module: nn.Module,
    *,
    target_suffixes: Iterable[str],
    rank: int,
    alpha: float,
    dropout: float,
) -> tuple[str, ...]:
    targets = tuple(target_suffixes)
    applied: list[str] = []

    def visit(parent: nn.Module, prefix: str) -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear) and full_name.endswith(targets):
                setattr(
                    parent,
                    child_name,
                    LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout),
                )
                applied.append(full_name)
            else:
                visit(child, full_name)

    visit(module, "")
    return tuple(applied)


def lora_parameters(module: nn.Module) -> Iterator[nn.Parameter]:
    for child in module.modules():
        if isinstance(child, LoRALinear):
            yield from child.adapter_parameters()


def lora_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
        if ".lora_a." in key
        or ".lora_b." in key
        or key.startswith("lora_a.")
        or key.startswith("lora_b.")
    }


def load_lora_state_dict(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> None:
    module.load_state_dict(state_dict, strict=False)
