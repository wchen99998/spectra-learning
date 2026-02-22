from __future__ import annotations

import importlib.util
from pathlib import Path

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
        mz_fourier_num_frequencies=int(config.get("mz_fourier_num_frequencies", 32)),
        mz_fourier_min_freq=float(config.get("mz_fourier_min_freq", 1.0)),
        mz_fourier_max_freq=float(config.get("mz_fourier_max_freq", 100.0)),
        mz_fourier_learnable=bool(config.get("mz_fourier_learnable", False)),
        encoder_use_rope=bool(config.get("encoder_use_rope", False)),
        encoder_block_type=str(config.get("encoder_block_type", "transformer")),
        isab_num_inducing_points=int(config.get("isab_num_inducing_points", 32)),
        encoder_fp16_high_precision_stem=bool(
            config.get("encoder_fp16_high_precision_stem", False)
        ),
        pooling_type=str(config.get("pooling_type", "pma")),
        pma_fp16_high_precision=bool(config.get("pma_fp16_high_precision", False)),
        pma_num_heads=config.get("pma_num_heads", int(config.num_heads)),
        pma_num_seeds=int(config.get("pma_num_seeds", 1)),
        sigreg_use_projector=bool(config.get("sigreg_use_projector", True)),
        sigreg_proj_hidden_dim=int(config.get("sigreg_proj_hidden_dim", 2048)),
        sigreg_proj_output_dim=int(config.get("sigreg_proj_output_dim", 128)),
        sigreg_proj_norm=str(config.get("sigreg_proj_norm", "rmsnorm")),
        sigreg_num_slices=int(config.get("sigreg_num_slices", 256)),
        sigreg_lambda=float(config.get("sigreg_lambda", 0.1)),
        multicrop_num_global_views=int(config.get("multicrop_num_global_views", 2)),
        multicrop_num_local_views=int(config.get("multicrop_num_local_views", 6)),
        multicrop_global_keep_fraction=float(
            config.get("multicrop_global_keep_fraction", 0.80)
        ),
        multicrop_local_keep_fraction=float(
            config.get("multicrop_local_keep_fraction", 0.25)
        ),
        sigreg_mz_jitter_std=float(config.get("sigreg_mz_jitter_std", 0.0001)),
        sigreg_intensity_jitter_std=float(
            config.get("sigreg_intensity_jitter_std", 0.001)
        ),
    )


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
