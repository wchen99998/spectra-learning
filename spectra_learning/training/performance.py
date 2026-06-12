from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import torch
from ml_collections import config_dict
from torch._functorch.partitioners import get_default_op_list
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    checkpoint_wrapper,
)
from torch.utils.checkpoint import (
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)

from spectra_learning.models.model import PeakSetJEPA
from spectra_learning.models.pairmixer import PairMixerBlock
from spectra_learning.training.modules import PretrainModule, split_pretrain_module


def _config_get(config: config_dict.ConfigDict, key: str, default: Any) -> Any:
    return config.get(key, default)


def activation_checkpoint_mode(config: config_dict.ConfigDict) -> str:
    mode = str(_config_get(config, "activation_checkpoint_mode", "none"))
    mode = mode.lower().replace("-", "_")
    assert mode in {"none", "selective", "full"}, (
        "activation_checkpoint_mode must be one of "
        "'none', 'selective', or 'full'."
    )
    return mode


def _pair_mixer_lists(model: PeakSetJEPA) -> Iterable[torch.nn.ModuleList]:
    yield model.encoder.blocks
    yield model.masked_latent_predictor


def _save_ops() -> dict[Any, CheckpointPolicy]:
    compute_ops = {
        torch.ops.aten._scaled_dot_product_cudnn_attention.default,
        torch.ops.aten._scaled_dot_product_attention_math.default,
        torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default,
        torch.ops.aten.linear.default,
    }
    save_ops = {
        op.default
        for op in get_default_op_list().compute_intensive_ops
    }
    save_ops.update(compute_ops)
    return {op: CheckpointPolicy.MUST_SAVE for op in save_ops}


def _selective_checkpoint_context_fn():
    save_ops = _save_ops()
    mm_ops = (torch.ops.aten.mm.default, torch.ops.aten.linear.default)

    def policy(ctx, func, *args, **kwargs) -> CheckpointPolicy:
        del kwargs
        if func in mm_ops and args[0].shape[-1] >= 128:
            return CheckpointPolicy.PREFER_RECOMPUTE
        return save_ops.get(func, CheckpointPolicy.PREFER_RECOMPUTE)

    return create_selective_checkpoint_contexts(policy)


def _checkpoint_block(
    block: torch.nn.Module,
    *,
    mode: str,
    preserve_rng_state: bool,
) -> torch.nn.Module:
    if mode == "full":
        return checkpoint_wrapper(
            block,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            preserve_rng_state=preserve_rng_state,
        )
    return checkpoint_wrapper(
        block,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        context_fn=_selective_checkpoint_context_fn,
        preserve_rng_state=preserve_rng_state,
    )


def apply_activation_checkpointing(
    module: torch.nn.Module,
    config: config_dict.ConfigDict,
) -> None:
    mode = activation_checkpoint_mode(config)
    if mode == "none":
        return
    preserve_rng_state = bool(
        _config_get(config, "activation_checkpoint_preserve_rng_state", True)
    )
    model, _ = split_pretrain_module(module)
    wrapped = 0
    for blocks in _pair_mixer_lists(model):
        for index, block in enumerate(blocks):
            blocks[index] = _checkpoint_block(
                block,
                mode=mode,
                preserve_rng_state=preserve_rng_state,
            )
            wrapped += 1
    logging.info("Applied %s activation checkpointing to %d PairMixer blocks.", mode, wrapped)


def compile_forward(
    module: torch.nn.Module,
    config: config_dict.ConfigDict,
    compile_mode: str | None = None,
) -> None:
    compile_mode = (
        str(_config_get(config, "compile_mode", "max-autotune"))
        if compile_mode is None
        else compile_mode
    )
    if compile_mode.lower() == "none":
        return
    compile_scope = str(_config_get(config, "compile_scope", "module")).lower()
    if compile_scope == "module":
        module.compile(
            mode=compile_mode,
            fullgraph=False,
        )
        logging.info("Compiled full training module with torch.compile mode=%s.", compile_mode)
        return

    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True
    model, _ = split_pretrain_module(module)
    compiled = 0
    for blocks in _pair_mixer_lists(model):
        for block in blocks:
            block.compile(mode=compile_mode, fullgraph=True)
            compiled += 1
    logging.info("Compiled %d PairMixer blocks with torch.compile mode=%s.", compiled, compile_mode)


def register_bf16_adamw_state_hook(optimizer: torch.optim.Optimizer) -> None:
    def _bf16_state_init_hook(
        optimizer: torch.optim.Optimizer,
        args: tuple,
        kwargs: dict,
    ) -> None:
        del args, kwargs
        for group in optimizer.param_groups:
            if not group.get("fused"):
                continue
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = optimizer.state[param]
                if len(state) == 0:
                    state["step"] = (
                        torch.zeros((), dtype=torch.float32, device=param.device)
                        if group.get("capturable") or group.get("fused")
                        else torch.tensor(0.0, dtype=torch.float32)
                    )
                    state["exp_avg"] = torch.zeros_like(
                        param,
                        dtype=torch.bfloat16,
                        memory_format=torch.preserve_format,
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        param,
                        dtype=torch.bfloat16,
                        memory_format=torch.preserve_format,
                    )

    optimizer.register_step_pre_hook(_bf16_state_init_hook)


def register_bf16_adamw_state_hooks(
    optimizers: Iterable[torch.optim.Optimizer],
    config: config_dict.ConfigDict,
) -> None:
    state_dtype = str(_config_get(config, "optimizer_state_dtype", "fp32")).lower()
    if state_dtype not in {"bf16", "bfloat16"}:
        return
    for optimizer in optimizers:
        register_bf16_adamw_state_hook(optimizer)
    logging.info("Registered bf16 AdamW optimizer-state hooks.")
