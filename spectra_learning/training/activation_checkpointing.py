from __future__ import annotations

from functools import partial
from typing import Any

import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    checkpoint_wrapper,
)
from torch.utils.checkpoint import (
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)

from spectra_learning.models.model import PeakSetJEPA
from spectra_learning.training.modules import PretrainModule


AC_MODES = {"none", "full", "selective"}


def apply_activation_checkpointing(module: nn.Module, config: Any) -> None:
    mode = str(_config_get(config, "activation_checkpoint_mode", "none")).lower()
    if mode not in AC_MODES:
        raise ValueError(
            "activation_checkpoint_mode must be one of none, full, selective"
        )
    if mode == "none":
        return
    model = _base_model(module)
    every_n = int(_config_get(config, "activation_checkpoint_every_n_layers", 1))
    preserve_rng_state = bool(
        _config_get(config, "activation_checkpoint_preserve_rng_state", True)
    )
    targets = tuple(
        _config_get(config, "activation_checkpoint_modules", ("encoder", "predictor"))
    )
    if "encoder" in targets:
        _wrap_blocks(
            model.encoder.blocks,
            mode=mode,
            every_n=every_n,
            preserve_rng_state=preserve_rng_state,
        )
    if "predictor" in targets:
        _wrap_blocks(
            model.masked_latent_predictor,
            mode=mode,
            every_n=every_n,
            preserve_rng_state=preserve_rng_state,
        )


def _base_model(module: nn.Module) -> PeakSetJEPA:
    if isinstance(module, PretrainModule):
        return module.model
    return module  # type: ignore[return-value]


def _wrap_blocks(
    blocks: nn.ModuleList,
    *,
    mode: str,
    every_n: int,
    preserve_rng_state: bool,
) -> None:
    for block_idx, block in enumerate(blocks, start=1):
        if block_idx % every_n != 0:
            continue
        blocks[block_idx - 1] = _wrap_block(
            block,
            mode=mode,
            preserve_rng_state=preserve_rng_state,
        )


def _wrap_block(
    block: nn.Module,
    *,
    mode: str,
    preserve_rng_state: bool,
) -> nn.Module:
    if mode == "full":
        return checkpoint_wrapper(
            block,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            preserve_rng_state=preserve_rng_state,
        )
    return checkpoint_wrapper(
        block,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        context_fn=partial(create_selective_checkpoint_contexts, _selective_policy),
        preserve_rng_state=preserve_rng_state,
    )


def _selective_policy(
    ctx,
    op,
    *args,
    **kwargs,
) -> CheckpointPolicy:
    del ctx, args, kwargs
    if op in _ops_to_save():
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def _ops_to_save() -> set:
    aten = torch.ops.aten
    return {
        aten.mm.default,
        aten.bmm.default,
        aten.addmm.default,
        aten.linear.default,
        aten._scaled_dot_product_flash_attention.default,
        aten._scaled_dot_product_efficient_attention.default,
        aten._scaled_dot_product_attention_math.default,
    }


def _config_get(config: Any, key: str, default: Any) -> Any:
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)
