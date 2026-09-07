"""Frozen 1.8B MAE teacher and matching 1.8B JEPA student."""

from importlib import import_module

from ml_collections import config_dict

from spectra_learning.config import load_config


def get_config() -> config_dict.ConfigDict:
    cfg = import_module("configs.1b_mae_teacher_jepa").get_config()
    teacher = load_config(cfg.frozen_teacher_config_path)

    # Preserve the teacher's backbone width, attention geometry, peak inputs,
    # and acquisition conditioning; the student's objective is latent prediction.
    for key in (
        "model_dim",
        "predictor_dim",
        "encoder_num_heads",
        "masked_latent_predictor_num_heads",
        "encoder_fourier_mlp_hidden_dim",
        "feature_mlp_hidden_dim",
        "num_peaks",
        "encoder_use_cls_token",
        "peak_ordering",
        "encoder_metadata_schema",
        "encoder_metadata_conditioning",
        "encoder_metadata_condition_dim",
        "spectrum_metadata_dropout_probability",
    ):
        cfg[key] = teacher[key]

    # The 1.8B student plus frozen encoder exceeds v6e HBM with replicated
    # Adam moments, even at micro-batch 8 with full activation checkpointing.
    cfg.batch_size = 1024
    cfg.jax_zero1_optimizer = True
    # Avoid flat sharding's full-model gradient/update concatenation buffers.
    cfg.jax_zero1_mode = "per_leaf"
    cfg.run_name_suffix = (
        cfg.run_name_suffix.replace("-1b-student-", "-1p8b-student-")
        .replace("-bs2048-", "-bs1024-")
        .replace("-bf16state-", "-bf16state-zero1-perleaf-")
    )
    return cfg
