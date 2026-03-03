from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import lightning.pytorch.loggers
import torch
from lightning.pytorch.loggers import CSVLogger
from ml_collections import config_dict

from models.model import PeakSetSIGReg
from utils import wandb_writer


def load_config(path: str | Path) -> config_dict.ConfigDict:
    path = Path(path)
    spec = importlib.util.spec_from_file_location("experiment_config", path)
    assert spec is not None, f"Could not load module spec from {path}"
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.get_config()


def build_model_from_config(config: config_dict.ConfigDict) -> PeakSetSIGReg:
    return PeakSetSIGReg(
        num_peaks=int(config.num_peaks),
        model_dim=int(config.model_dim),
        encoder_num_layers=int(config.num_layers),
        encoder_num_heads=int(config.num_heads),
        encoder_num_kv_heads=config.get("num_kv_heads", None),
        attention_mlp_multiple=float(config.attention_mlp_multiple),
        feature_mlp_hidden_dim=int(config.get("feature_mlp_hidden_dim", 128)),
        encoder_use_rope=bool(config.get("encoder_use_rope", False)),
        rope_mz_max=float(config.get("rope_mz_max", 1000.0)),
        rope_mz_precision=float(config.get("rope_mz_precision", 0.1)),
        rope_modulo_2pi=bool(config.get("rope_modulo_2pi", True)),
        masked_token_loss_weight=float(config.get("masked_token_loss_weight", 0.0)),
        masked_token_loss_type=str(config.get("masked_token_loss_type", "l1")),
        representation_regularizer=str(config.get("representation_regularizer", "sigreg")),
        masked_latent_predictor_num_layers=int(
            config.get("masked_latent_predictor_num_layers", 2)
        ),
        encoder_block_type=str(config.get("encoder_block_type", "transformer")),
        isab_num_inducing_points=int(config.get("isab_num_inducing_points", 32)),
        pooling_type=str(config.get("pooling_type", "pma")),
        pma_fp16_high_precision=bool(config.get("pma_fp16_high_precision", False)),
        pma_num_heads=config.get("pma_num_heads", int(config.num_heads)),
        pma_num_seeds=int(config.get("pma_num_seeds", 1)),
        sigreg_num_slices=int(config.get("sigreg_num_slices", 256)),
        sigreg_lambda=float(config.get("sigreg_lambda", 0.1)),
        sigreg_lambda_warmup_steps=int(config.get("sigreg_lambda_warmup_steps", 0)),
        vicreg_beta=float(config.get("vicreg_beta", 1e-3)),
        vicreg_sim_coeff=float(config.get("vicreg_sim_coeff", 0.0)),
        vicreg_std_coeff=float(config.get("vicreg_std_coeff", 25.0)),
        vicreg_cov_coeff=float(config.get("vicreg_cov_coeff", 1.0)),
        multicrop_num_local_views=int(config.get("multicrop_num_local_views", 6)),
        multicrop_local_keep_fraction=float(
            config.get("multicrop_local_keep_fraction", 0.25)
        ),
        use_ema_teacher_target=bool(config.get("use_ema_teacher_target", False)),
        teacher_ema_decay=float(config.get("teacher_ema_decay", 0.996)),
        teacher_ema_decay_start=float(config.get("teacher_ema_decay_start", 0.0)),
        teacher_ema_decay_warmup_steps=int(config.get("teacher_ema_decay_warmup_steps", 0)),
        sigreg_mz_jitter_std=float(config.get("sigreg_mz_jitter_std", 0.0001)),
        sigreg_intensity_jitter_std=float(
            config.get("sigreg_intensity_jitter_std", 0.001)
        ),
        encoder_qk_norm=bool(config.get("encoder_qk_norm", False)),
        encoder_post_norm=bool(config.get("encoder_post_norm", False)),
        normalize_jepa_targets=bool(config.get("normalize_jepa_targets", False)),
        norm_type=str(config.get("norm_type", "rmsnorm")),
    )


def collect_model_param_summary(model: torch.nn.Module) -> dict[str, Any]:
    by_module: dict[str, dict[str, int]] = {}
    total_params = 0
    trainable_params = 0
    total_tensors = 0
    trainable_tensors = 0

    for name, param in model.named_parameters():
        numel = int(param.numel())
        is_trainable = bool(param.requires_grad)
        module_name = name.split(".", 1)[0]

        module_summary = by_module.setdefault(
            module_name,
            {
                "total_params": 0,
                "trainable_params": 0,
                "non_trainable_params": 0,
                "total_tensors": 0,
                "trainable_tensors": 0,
                "non_trainable_tensors": 0,
            },
        )

        total_params += numel
        total_tensors += 1
        module_summary["total_params"] += numel
        module_summary["total_tensors"] += 1

        if is_trainable:
            trainable_params += numel
            trainable_tensors += 1
            module_summary["trainable_params"] += numel
            module_summary["trainable_tensors"] += 1
        else:
            module_summary["non_trainable_params"] += numel
            module_summary["non_trainable_tensors"] += 1

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": total_params - trainable_params,
        "total_tensors": total_tensors,
        "trainable_tensors": trainable_tensors,
        "non_trainable_tensors": total_tensors - trainable_tensors,
        "by_module": by_module,
    }


def log_model_param_summary(summary: dict[str, Any]) -> None:
    logging.info(
        "Model parameters: total=%s trainable=%s non_trainable=%s",
        f"{summary['total_params']:,}",
        f"{summary['trainable_params']:,}",
        f"{summary['non_trainable_params']:,}",
    )
    logging.info(
        "Model parameter tensors: total=%s trainable=%s non_trainable=%s",
        f"{summary['total_tensors']:,}",
        f"{summary['trainable_tensors']:,}",
        f"{summary['non_trainable_tensors']:,}",
    )
    for module_name in sorted(summary["by_module"]):
        module_summary = summary["by_module"][module_name]
        logging.info(
            "Model parameters [%s]: total=%s trainable=%s non_trainable=%s",
            module_name,
            f"{module_summary['total_params']:,}",
            f"{module_summary['trainable_params']:,}",
            f"{module_summary['non_trainable_params']:,}",
        )


def model_param_summary_to_metrics(summary: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {
        "model/params_total": float(summary["total_params"]),
        "model/params_trainable": float(summary["trainable_params"]),
        "model/params_non_trainable": float(summary["non_trainable_params"]),
        "model/param_tensors_total": float(summary["total_tensors"]),
        "model/param_tensors_trainable": float(summary["trainable_tensors"]),
        "model/param_tensors_non_trainable": float(summary["non_trainable_tensors"]),
    }
    for module_name, module_summary in summary["by_module"].items():
        metrics[f"model/params_total/{module_name}"] = float(module_summary["total_params"])
        metrics[f"model/params_trainable/{module_name}"] = float(module_summary["trainable_params"])
        metrics[f"model/params_non_trainable/{module_name}"] = float(module_summary["non_trainable_params"])
    return metrics


def build_logger(config: config_dict.ConfigDict, workdir: Path) -> pl.loggers.Logger:
    if config.get("enable_wandb", False):
        from lightning.pytorch.loggers import WandbLogger

        wandb_kwargs = wandb_writer.build_wandb_init_kwargs(config)
        logger = WandbLogger(
            project=config.get("wandb_project", "md4"),
            save_dir=str(workdir),
            log_model=False,
            **wandb_kwargs,
        )
        logger.log_hyperparams(wandb_writer.config_to_wandb_dict(config))
        return logger

    return CSVLogger(save_dir=str(workdir), name="csv_logs")


def load_pretrained_weights(
    model: PeakSetSIGReg,
    checkpoint_path: str,
) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    model_state = {
        k.removeprefix("model."): v
        for k, v in state_dict.items()
        if k.startswith("model.")
    }
    if not model_state:
        model_state = state_dict
    model.load_state_dict(model_state, strict=True)


def latest_ckpt_path(directory: Path) -> str | None:
    checkpoint_dir = directory / "checkpoints"
    search_root = checkpoint_dir if checkpoint_dir.exists() else directory
    ckpts = sorted(
        [
            *search_root.rglob("*.ckpt"),
            *search_root.rglob("*.pt"),
        ],
        key=lambda p: p.stat().st_mtime,
    )
    if not ckpts:
        return None
    return str(ckpts[-1])
