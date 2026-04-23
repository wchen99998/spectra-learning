import importlib.util
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.loggers import CSVLogger
from ml_collections import config_dict

from models.model import PeakSetSIGReg
from utils.spectra_preprocessing import PEAK_MZ_MAX

_GNS_MUON_DYNAMIC_PATCHED = False


@torch.compile(fullgraph=True, mode="reduce-overhead", dynamic=True)
def _stacked_muon_update_pre_orthogonalize(
    gradients: torch.Tensor,
    momentums: torch.Tensor,
    momentum: torch.Tensor,
    nesterov: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    gradients = gradients.to(dtype=momentums.dtype)
    updated_momentums = momentums * momentum
    updated_momentums = updated_momentums + gradients
    if nesterov:
        updates = updated_momentums * momentum + gradients
    else:
        updates = updated_momentums
    return updates.to(dtype=torch.bfloat16), updated_momentums


def _mark_matrices_dynamic(x: torch.Tensor) -> torch.Tensor:
    dynamic_dims = {0}
    if x.ndim >= 2:
        dynamic_dims.add(x.ndim - 2)
    if x.ndim >= 1:
        dynamic_dims.add(x.ndim - 1)
    for dim in sorted(dynamic_dims):
        torch._dynamo.mark_dynamic(x, dim)
    return x


def _dynamic_muon_update_pre_orthogonalize(
    G: list[torch.Tensor],
    M: list[torch.Tensor],
    momentum: torch.Tensor,
    nesterov: bool,
) -> list[torch.Tensor]:
    if not G:
        return []
    gradients = _mark_matrices_dynamic(torch.stack(G, dim=0))
    momentums = _mark_matrices_dynamic(torch.stack(M, dim=0))
    updates, updated_momentums = _stacked_muon_update_pre_orthogonalize(
        gradients,
        momentums,
        momentum,
        nesterov,
    )
    for momentum_buffer, updated in zip(
        M,
        updated_momentums.unbind(0),
        strict=True,
    ):
        momentum_buffer.copy_(updated)
    return list(updates.unbind(0))


def _sorted_create_param_batches(
    params: list[torch.Tensor],
) -> list[list[torch.Tensor]]:
    groups: dict[tuple[torch.Size, torch.dtype], list[torch.Tensor]] = {}
    for param in params:
        groups.setdefault((param.shape, param.dtype), []).append(param)

    batches = list(groups.values())
    for batch in batches:
        batch.sort(key=lambda param: param.data_ptr())

    def _batch_order_key(batch: list[torch.Tensor]) -> tuple[int, int, int, int]:
        rows, cols = batch[0].shape[-2:]
        if rows == cols:
            branch_kind = 0
        elif rows > cols:
            branch_kind = 1
        else:
            branch_kind = 2
        return (-len(batch), branch_kind, -max(rows, cols), -min(rows, cols))

    batches.sort(key=_batch_order_key)
    return batches


def patch_gns_muon_compile_for_dynamic_shapes() -> None:
    global _GNS_MUON_DYNAMIC_PATCHED
    if _GNS_MUON_DYNAMIC_PATCHED:
        return

    from gram_newton_schulz.gram_newton_schulz import GramNewtonSchulz
    from gram_newton_schulz.muon import muon as muon_mod
    from gram_newton_schulz.muon.muon_utils import muon_opt_utils
    from gram_newton_schulz.standard_newton_schulz import StandardNewtonSchulz

    gram_call = getattr(
        GramNewtonSchulz.__call__,
        "_torchdynamo_orig_callable",
        GramNewtonSchulz.__call__,
    )
    compiled_gram_call = torch.compile(
        gram_call,
        fullgraph=True,
        mode="reduce-overhead",
        dynamic=True,
    )
    
    def _dynamic_gram_call(self, X):
        return compiled_gram_call(self, _mark_matrices_dynamic(X))

    _dynamic_gram_call.__wrapped__ = gram_call
    _dynamic_gram_call._torchdynamo_orig_callable = gram_call
    GramNewtonSchulz.__call__ = _dynamic_gram_call

    standard_call = getattr(
        StandardNewtonSchulz.__call__,
        "_torchdynamo_orig_callable",
        StandardNewtonSchulz.__call__,
    )
    compiled_standard_call = torch.compile(
        standard_call,
        fullgraph=True,
        mode="reduce-overhead",
        dynamic=True,
    )

    def _dynamic_standard_call(self, X):
        return compiled_standard_call(self, _mark_matrices_dynamic(X))

    _dynamic_standard_call.__wrapped__ = standard_call
    _dynamic_standard_call._torchdynamo_orig_callable = standard_call
    StandardNewtonSchulz.__call__ = _dynamic_standard_call

    muon_mod.muon_update_pre_orthogonalize = _dynamic_muon_update_pre_orthogonalize
    muon_opt_utils.muon_update_pre_orthogonalize = _dynamic_muon_update_pre_orthogonalize
    muon_mod.create_param_batches = _sorted_create_param_batches
    muon_opt_utils.create_param_batches = _sorted_create_param_batches
    _GNS_MUON_DYNAMIC_PATCHED = True


def load_config(path: str | Path) -> config_dict.ConfigDict:
    path = Path(path)
    spec = importlib.util.spec_from_file_location("experiment_config", path)
    assert spec is not None, f"Could not load module spec from {path}"
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.get_config()


def build_model_from_config(config: config_dict.ConfigDict) -> PeakSetSIGReg:
    encoder_num_layers = int(
        config.get("encoder_num_layers", config.get("num_layers"))
    )
    encoder_num_heads = int(
        config.get("encoder_num_heads", config.get("num_heads"))
    )
    encoder_num_kv_heads = config.get(
        "encoder_num_kv_heads",
        config.get("num_kv_heads", None),
    )
    return PeakSetSIGReg(
        model_dim=int(config.model_dim),
        encoder_num_layers=encoder_num_layers,
        encoder_num_heads=encoder_num_heads,
        encoder_num_kv_heads=encoder_num_kv_heads,
        attention_mlp_multiple=float(config.attention_mlp_multiple),
        feature_mlp_hidden_dim=int(config.get("feature_mlp_hidden_dim", 128)),
        encoder_fourier_strategy=str(
            config.get("encoder_fourier_strategy", "log_spaced")
        ),
        encoder_fourier_x_min=float(config.get("encoder_fourier_x_min", 3e-3)),
        encoder_fourier_x_max=float(config.get("encoder_fourier_x_max", 1000.0)),
        encoder_fourier_funcs=str(config.get("encoder_fourier_funcs", "both")),
        encoder_fourier_num_freqs=int(config.get("encoder_fourier_num_freqs", 256)),
        encoder_fourier_sigma=float(config.get("encoder_fourier_sigma", 10.0)),
        encoder_fourier_trainable=bool(
            config.get("encoder_fourier_trainable", False)
        ),
        encoder_fourier_input_scale=float(
            config.get(
                "encoder_fourier_input_scale",
                config.get("peak_mz_max", config.get("max_precursor_mz", PEAK_MZ_MAX)),
            )
        ),
        masked_token_loss_weight=float(config.get("masked_token_loss_weight", 0.0)),
        masked_token_loss_type=str(config.get("masked_token_loss_type", "l1")),
        jepa_target_normalization=str(
            config.get("jepa_target_normalization", "none")
        ),
        jepa_target_layers=config.get("jepa_target_layers", None),
        representation_regularizer=str(
            config.get("representation_regularizer", "none")
        ),
        masked_latent_predictor_num_layers=int(
            config.get("masked_latent_predictor_num_layers", 2)
        ),
        masked_latent_predictor_num_heads=int(
            config.get("masked_latent_predictor_num_heads", 8)
        ),
        sigreg_num_slices=int(config.get("sigreg_num_slices", 256)),
        sigreg_lambda=float(config.get("sigreg_lambda", 0.02)),
        sigreg_precursor_scale=float(config.get("sigreg_precursor_scale", 1.0)),
        vicreg_lambda=float(config.get("vicreg_lambda", 0.02)),
        vicreg_inv_coeff=float(config.get("vicreg_inv_coeff", 0.0)),
        vicreg_var_coeff=float(config.get("vicreg_var_coeff", 25.0)),
        vicreg_cov_coeff=float(config.get("vicreg_cov_coeff", 1.0)),
        vicreg_variance_target=float(config.get("vicreg_variance_target", 1.0)),
        vicreg_eps=float(config.get("vicreg_eps", 1e-4)),
        jepa_num_target_blocks=int(config.get("jepa_num_target_blocks", 2)),
        jepa_context_fraction=float(config.get("jepa_context_fraction", 0.5)),
        jepa_target_fraction=float(config.get("jepa_target_fraction", 0.25)),
        encoder_qk_norm=bool(config.get("encoder_qk_norm", False)),
        norm_type=str(config.get("norm_type", "rmsnorm")),
        encoder_use_position_embedding=bool(
            config.get("encoder_use_position_embedding", True)
        ),
        encoder_apply_final_norm=bool(config.get("encoder_apply_final_norm", True)),
        predictor_apply_final_norm=bool(
            config.get("predictor_apply_final_norm", True)
        ),
        use_precursor_token=bool(config.get("use_precursor_token", False)),
        num_peaks=int(config.get("num_peaks", 64)),
        encoder_num_register_tokens=int(
            config.get("encoder_num_register_tokens", 0)
        ),
        predictor_num_register_tokens=int(
            config.get("predictor_num_register_tokens", 0)
        ),
        temporal_predictor_num_layers=int(
            config.get("temporal_predictor_num_layers", 0)
        ),
        predictor_dim=config.get("predictor_dim", None),
        predictor_dropout=float(config.get("predictor_dropout", 0.0)),
    )


def collect_and_log_param_metrics(model: torch.nn.Module) -> dict[str, float]:
    by_module: dict[str, list[int]] = {}
    total = trainable = 0
    for name, param in model.named_parameters():
        numel = int(param.numel())
        module_name = name.split(".", 1)[0]
        counts = by_module.setdefault(module_name, [0, 0])
        total += numel
        counts[0] += numel
        if param.requires_grad:
            trainable += numel
            counts[1] += numel
    logging.info(
        "Model parameters: total=%s trainable=%s non_trainable=%s",
        f"{total:,}",
        f"{trainable:,}",
        f"{total - trainable:,}",
    )
    metrics: dict[str, float] = {
        "model/params_total": float(total),
        "model/params_trainable": float(trainable),
        "model/params_non_trainable": float(total - trainable),
    }
    for module_name in sorted(by_module):
        mod_total, mod_train = by_module[module_name]
        logging.info(
            "  [%s] total=%s trainable=%s",
            module_name,
            f"{mod_total:,}",
            f"{mod_train:,}",
        )
        metrics[f"model/params_total/{module_name}"] = float(mod_total)
        metrics[f"model/params_trainable/{module_name}"] = float(mod_train)
    return metrics


def auto_run_name(config: Any) -> str:
    """Generate a descriptive run name from key config hyperparameters."""
    dim = int(config.get("model_dim", 0))
    layers = int(config.get("encoder_num_layers", 0))
    heads = int(config.get("encoder_num_heads", 0))
    pred_layers = int(config.get("masked_latent_predictor_num_layers", 0))
    bs = int(config.get("batch_size", 0))
    opt = str(config.get("optimizer", "adamw")).lower()
    lr = float(config.get("learning_rate", 0))
    epochs = config.get("num_epochs", "?")

    # Estimate total params (build model transiently if needed)
    model = build_model_from_config(config)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    del model
    if total_params >= 1_000_000_000:
        param_str = f"{total_params / 1_000_000_000:.1f}B"
    else:
        param_str = f"{total_params / 1_000_000:.0f}M"

    wd = float(config.get("weight_decay", 0))
    min_lr = config.get("min_learning_rate", None)
    warmup = int(config.get("warmup_steps", 0))

    parts = [
        param_str,
        f"d{dim}",
        f"L{layers}",
        f"h{heads}",
        f"pred{pred_layers}",
        f"bs{bs}",
        opt,
        f"lr{lr:.0e}",
        f"wd{wd:.0e}",
        f"ep{epochs}",
    ]

    if min_lr is not None:
        parts.append(f"minlr{float(min_lr):.0e}")
    if warmup > 0:
        parts.append(f"wu{warmup // 1000}k")
    if config.get("use_precursor_token", False):
        parts.append("prec")
    norm = str(config.get("norm_type", "")).lower()
    if norm and norm != "rmsnorm":
        parts.append(norm)
    target_layers = config.get("jepa_target_layers", None)
    if target_layers:
        parts.append(f"tgt{'_'.join(str(x) for x in target_layers)}")
    regularizer = str(config.get("representation_regularizer", "none")).lower()
    if regularizer and regularizer != "none":
        parts.append(regularizer)
        lambda_key = "vicreg_lambda" if regularizer == "vicreg" else "sigreg_lambda"
        parts.append(f"lam{float(config.get(lambda_key, 0.0)):.0e}")
    run_name_suffix = str(config.get("run_name_suffix", "")).strip()
    if run_name_suffix:
        parts.append(run_name_suffix)

    return "_".join(parts)


def _build_wandb_init_kwargs(config: Any | None) -> dict[str, Any]:
    if config is None:
        return {}
    wandb_kwargs = dict(config.get("wandb_kwargs", {}) or {})
    resume_id = str(config.get("wandb_resume_id", "") or "")
    if not resume_id:
        resume_id = os.environ.get("WANDB_RESUME_ID", "")
    if resume_id:
        wandb_kwargs.setdefault("id", resume_id)
        wandb_kwargs.setdefault("resume", "must")
        wandb_kwargs.pop("name", None)
        return wandb_kwargs
    if "name" not in wandb_kwargs:
        wandb_kwargs["name"] = auto_run_name(config)
    return wandb_kwargs


def _to_serialisable_config(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _to_serialisable_config(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_serialisable_config(v) for v in value]
    return str(value)


def _config_to_wandb_dict(config: Any | None) -> dict[str, Any]:
    if config is None:
        return {}
    if callable(getattr(config, "to_dict", None)):
        return dict(_to_serialisable_config(config.to_dict()))
    if isinstance(config, Mapping):
        return dict(_to_serialisable_config(config))
    return dict(_to_serialisable_config(vars(config)))


def build_logger(config: config_dict.ConfigDict, workdir: Path) -> pl.loggers.Logger:
    if config.get("enable_wandb", False):
        from lightning.pytorch.loggers import WandbLogger

        wandb_kwargs = _build_wandb_init_kwargs(config)
        logger = WandbLogger(
            project=config.get("wandb_project", "md4"),
            save_dir=str(workdir),
            log_model=False,
            **wandb_kwargs,
        )
        logger.log_hyperparams(_config_to_wandb_dict(config))
        return logger
    return CSVLogger(save_dir=str(workdir), name="csv_logs")


def load_pretrained_weights(
    model: PeakSetSIGReg,
    checkpoint_path: str,
) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    sd = ckpt.get("state_dict") or ckpt.get("model") or ckpt
    prefixed = {
        k.removeprefix("model."): v for k, v in sd.items() if k.startswith("model.")
    }
    sd = prefixed or sd
    for key in tuple(sd):
        if key.endswith(
            (
                "position_embedding.weight",
                "predictor_position_embedding.weight",
            )
        ):
            sd.pop(key)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    allowed_missing_suffixes = (
        "position_embedding.weight",
        "predictor_position_embedding.weight",
        "cls_token",
        "register_tokens",
        "predictor_register_tokens",
        "temporal_query_token",
    )
    allowed_missing_prefixes = ("masked_latent_readout.", "sigreg.")
    unexpected = [
        key for key in unexpected
        if not key.endswith(
            (
                "temporal_query_token",
                "sigreg_lambda_target",
                "sigreg_lambda_current",
                "sigreg_lambda_step",
            )
        )
    ]
    missing = [
        key for key in missing
        if not key.endswith(allowed_missing_suffixes)
        and not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
    ]
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint load mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )


def latest_ckpt_path(directory: Path) -> str | None:
    checkpoint_dir = directory / "checkpoints"
    root = checkpoint_dir if checkpoint_dir.exists() else directory
    ckpts = sorted(
        [*root.rglob("*.ckpt"), *root.rglob("*.pt")],
        key=lambda p: p.stat().st_mtime,
    )
    return str(ckpts[-1]) if ckpts else None
