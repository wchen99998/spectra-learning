from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import torch
from ml_collections import config_dict

from spectra_learning.models.model import PeakSetJEPA
from spectra_learning.training.modules import PretrainModule, split_pretrain_module


def _config_get(config: config_dict.ConfigDict, key: str, default: Any) -> Any:
    return config.get(key, default)


def _pair_mixer_lists(model: PeakSetJEPA) -> Iterable[torch.nn.ModuleList]:
    yield model.encoder.blocks
    yield model.masked_latent_predictor


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
