from importlib import import_module

from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    cfg = import_module("configs.1b_pairmixer_dense_adamw").get_config()

    cfg.device_backend = "jax"
    cfg.dataloader_output_format = "numpy"
    cfg.dataloader_pin_memory = False
    cfg.dataloader_persistent_workers = False
    # The 65 training shards split across 8 hosts as 9/8/8/8/8/8/8/8.
    cfg.gems_hdf5_shuffle_seed = 67
    cfg.gems_hdf5_revision = "b098155fccb7335b752f2690bf16d2479d69bf5b"
    cfg.training_mode = "mae_teacher_jepa"
    cfg.use_ema_teacher = False
    cfg.encoder_metadata_schema = "massive_v2_acquisition_v1"
    cfg.encoder_metadata_conditioning = "adaln_zero"
    cfg.encoder_metadata_condition_dim = 256
    cfg.spectrum_metadata_dropout_probability = 0.10
    cfg.num_peaks = 127
    cfg.encoder_use_cls_token = True
    cfg.peak_ordering = "mz"
    cfg.frozen_teacher_config_path = (
        "configs/1p8b_singlemixer_dense_adam_metadata_adaln_"
        "multiscale_n128_b6144_666k_d2048.py"
    )
    cfg.frozen_teacher_checkpoint_path = (
        "gs://metal-repeater-411410-spectra-checkpoints/skypilot/"
        "1p8b-sm-hd256-adam-v6e64-b6144-ga1-666k-3d-mr6-"
        "20260828-024511/checkpoints/orbax/666000"
    )

    cfg.masked_token_loss_weight = 1.0
    cfg.contrastive_loss_weight = 0.0
    cfg.online_probe_loss_weight = 0.0
    cfg.mae_loss_weight = 0.0
    cfg.mae_intensity_loss_weight = 0.0
    cfg.jepa_mae_loss_weight = 0.0
    cfg.distogram_loss_weight = 0.0
    cfg.latent_pair_loss_weight = 0.0

    cfg.training_max_steps = 2_000_000
    cfg.batch_size = 2_048
    cfg.optimizer = "adam"
    cfg.b1 = 0.9
    cfg.weight_decay = 0.0
    # The 1.516B frozen encoder leaves insufficient HBM for replicated fp32
    # Adam moments. bf16 moments save about 4 GB per chip and are the setting
    # used by the completed 1.8B teacher run.
    cfg.optimizer_state_dtype = "bf16"
    cfg.learning_rate = 3e-4
    cfg.min_learning_rate = 3e-6
    cfg.warmup_steps = 10_000
    # Match the 1.8B MAE teacher's fixed random masking. A single stage keeps
    # the JAX student's compact token capacities without changing the task.
    cfg.jepa_mask_strategy = ["random"]
    cfg.jepa_context_fraction = 0.60
    cfg.jepa_target_fraction = 0.25
    cfg.jepa_context_fraction_schedule = (cfg.jepa_context_fraction,)
    cfg.jepa_target_fraction_schedule = (cfg.jepa_target_fraction,)
    cfg.jepa_mask_schedule_steps = ()
    cfg.gradient_accumulation_steps = 8
    cfg.gradient_accumulation_steps_schedule = (8,)
    cfg.activation_checkpoint_mode = "full"

    cfg.checkpoint_every_steps = 500
    cfg.jax_checkpoint_max_to_keep = 5
    cfg.val_every_n_steps = 10_000
    cfg.val_num_steps = 50
    cfg.max_duration_hours = None

    cfg.run_name_suffix = (
        "jax-mae-teacher-jepa-1b-student-1p8b-sm-hd256-adam-step666k-"
        "n128-cls-v6e4x8-bs2048-ga8-s9-lr3e-4-random-ctx60-target25-"
        "target-only-adam-bf16state-2m-7d-mr6-ckpt500"
    )
    return cfg
