from importlib import import_module

from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    cfg = import_module("configs.1b_pairmixer_dense_adamw").get_config()

    cfg.device_backend = "jax"
    cfg.dataloader_output_format = "numpy"
    cfg.dataloader_pin_memory = False
    cfg.dataloader_persistent_workers = False
    cfg.training_mode = "mae_teacher_jepa"
    cfg.use_ema_teacher = False
    cfg.frozen_teacher_config_path = (
        "configs/1b_pairmixer_dense_adamw_legacy_encoder.py"
    )
    cfg.frozen_teacher_checkpoint_path = (
        "gs://metal-repeater-411410-spectra-checkpoints/skypilot/"
        "1b-pairmixer-dense-adamw-xla-noac-v6e4x8-east5-b2048-ga8-16-16-"
        "47h-20260716-112129/checkpoints/orbax/300000"
    )

    cfg.masked_token_loss_weight = 1.0
    cfg.contrastive_loss_weight = 0.0
    cfg.online_probe_loss_weight = 0.0
    cfg.mae_loss_weight = 0.0
    cfg.mae_intensity_loss_weight = 0.0
    cfg.jepa_mae_loss_weight = 0.0
    cfg.distogram_loss_weight = 0.0
    cfg.latent_pair_loss_weight = 0.0

    cfg.training_max_steps = 300_000
    cfg.batch_size = 1024
    cfg.learning_rate = 3e-4
    cfg.min_learning_rate = 3e-6
    cfg.warmup_steps = 10_000
    cfg.jepa_context_fraction_schedule = (0.35, 0.55, 0.75)
    cfg.jepa_target_fraction_schedule = (0.50, 0.30, 0.10)
    cfg.jepa_mask_schedule_steps = (100_000, 200_000)
    cfg.gradient_accumulation_steps = 4
    cfg.gradient_accumulation_steps_schedule = (4, 8, 8)
    cfg.activation_checkpoint_mode = "full"

    cfg.checkpoint_every_steps = 500
    cfg.jax_checkpoint_max_to_keep = 5
    cfg.val_every_n_steps = 10_000
    cfg.val_num_steps = 50
    cfg.max_duration_hours = None

    cfg.run_name_suffix = (
        "jax-mae-teacher-jepa-1b-teacher-t0hvg5o5-step300k-"
        "v6e4x8-bs1024-ga4-8-8-lr3e-4-mask100k-200k-target-only-compact-"
        "2d-recoverable-ckpt500"
    )
    return cfg
