"""Absorbing m/z diffusion with the reference 1.8B JEPA student backbone."""

from importlib import import_module

from ml_collections import config_dict

from spectra_learning.data.spectra import PEAK_MZ_MAX


def get_config() -> config_dict.ConfigDict:
    cfg = import_module("configs.1p8b_mae_teacher_jepa").get_config()

    cfg.training_mode = "mae"
    cfg.mae_objective = "absorbing_diffusion"
    cfg.use_ema_teacher = False
    cfg.frozen_teacher_config_path = None
    cfg.frozen_teacher_checkpoint_path = None
    cfg.frozen_teacher_update_every_steps = 0

    cfg.mae_loss_weight = 1.0
    cfg.masked_token_loss_weight = 0.0
    cfg.jepa_mae_loss_weight = 0.0
    cfg.mae_intensity_loss_weight = 0.0
    cfg.contrastive_loss_weight = 0.0
    cfg.online_probe_loss_weight = 0.0
    cfg.distogram_loss_weight = 0.0
    cfg.latent_pair_loss_weight = 0.0
    cfg.mae_mz_objective = "exact_ce"
    # Reuse every output of the reference student's readout as a categorical
    # logit, preserving its exact trainable parameter count.
    cfg.mae_mz_head = "readout"
    cfg.jepa_mae_mz_bin_size = PEAK_MZ_MAX / cfg.model_dim
    cfg.mae_mz_multiscale_widths = ()
    cfg.mae_mz_multiscale_weight = 0.0

    cfg.jepa_mask_strategy = ["random"]
    cfg.jepa_num_target_blocks = 1
    # Diffusion samples K uniformly from 1..N, independently of these legacy
    # fixed-mask fractions. Keep one stage and the reference accumulation.
    cfg.jepa_context_fraction_schedule = (cfg.jepa_context_fraction,)
    cfg.jepa_target_fraction_schedule = (cfg.jepa_target_fraction,)
    cfg.jepa_mask_schedule_steps = ()

    cfg.run_name_suffix = (
        "jax-absorbing-mz-diffusion-1p8b-d2048-l25-pred5-h8-n128-cls-"
        "k-uniform-exactce-mzbins2048-readout-intensity-conditioned-"
        "bs1024-ga8-adam-bf16state-zero1-perleaf-lr3e-4-2m"
    )
    return cfg
