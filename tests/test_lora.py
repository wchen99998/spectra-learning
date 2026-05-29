from __future__ import annotations

import torch
from torch import nn

from spectra_learning.models.lora import (
    LoRALinear,
    apply_lora_to_linear_modules,
    lora_parameters,
    lora_state_dict,
)


class _ToyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.single_attention = nn.Module()
        self.single_attention.wqkv = nn.Linear(4, 12, bias=False)
        self.single_attention.wo = nn.Linear(4, 4, bias=False)
        self.single_transition = nn.Module()
        self.single_transition.w1 = nn.Linear(4, 8, bias=False)
        self.single_transition.w2 = nn.Linear(8, 4, bias=False)
        self.embed = nn.Linear(4, 4, bias=False)


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_ToyBlock()])


def test_lora_linear_starts_as_base_linear() -> None:
    torch.manual_seed(0)
    base = nn.Linear(4, 3)
    wrapped = LoRALinear(base, rank=2, alpha=4.0, dropout=0.0)
    x = torch.randn(5, 4)

    assert torch.allclose(wrapped(x), base(x))
    assert not wrapped.base.weight.requires_grad
    assert not wrapped.base.bias.requires_grad
    assert wrapped.lora_a.weight.requires_grad
    assert wrapped.lora_b.weight.requires_grad

    wrapped(x).sum().backward()

    assert wrapped.base.weight.grad is None
    assert wrapped.lora_a.weight.grad is not None
    assert wrapped.lora_b.weight.grad is not None


def test_apply_lora_replaces_matching_linear_suffixes_only() -> None:
    model = _ToyModel()

    applied = apply_lora_to_linear_modules(
        model,
        target_suffixes=(
            "single_attention.wqkv",
            "single_transition.w1",
        ),
        rank=2,
        alpha=4.0,
        dropout=0.0,
    )

    assert applied == (
        "blocks.0.single_attention.wqkv",
        "blocks.0.single_transition.w1",
    )
    assert isinstance(model.blocks[0].single_attention.wqkv, LoRALinear)
    assert isinstance(model.blocks[0].single_transition.w1, LoRALinear)
    assert isinstance(model.blocks[0].single_attention.wo, nn.Linear)
    assert isinstance(model.blocks[0].single_transition.w2, nn.Linear)
    assert isinstance(model.blocks[0].embed, nn.Linear)

    params = list(lora_parameters(model))
    assert [id(param) for param in params] == [
        id(model.blocks[0].single_attention.wqkv.lora_a.weight),
        id(model.blocks[0].single_attention.wqkv.lora_b.weight),
        id(model.blocks[0].single_transition.w1.lora_a.weight),
        id(model.blocks[0].single_transition.w1.lora_b.weight),
    ]

    state = lora_state_dict(model)
    assert set(state) == {
        "blocks.0.single_attention.wqkv.lora_a.weight",
        "blocks.0.single_attention.wqkv.lora_b.weight",
        "blocks.0.single_transition.w1.lora_a.weight",
        "blocks.0.single_transition.w1.lora_b.weight",
    }
