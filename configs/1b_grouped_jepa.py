from importlib import import_module

from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    cfg = import_module("configs.1b_singlemixer_dense_muon").get_config()

    cfg.training_task = "grouped_jepa"
    cfg.training_mode = "grouped_jepa"
    cfg.device_backend = "jax"
    cfg.group_jepa_spectra_per_group = 8
    cfg.group_jepa_teacher_spectra_per_group = 3
    cfg.group_jepa_groups_per_batch = 256
    cfg.group_jepa_ema_momentum = 0.9992
    # Compute the next target before the EMA update in the current XLA step.
    # Targets are one EMA update old after step 1. On TPU v6e-1 this measured
    # 83.14 ms/step versus 88.94 ms/step for fused same-step targets.
    cfg.group_jepa_teacher_target_mode = "lookahead"
    cfg.group_jepa_init_checkpoint_path = (
        "gs://metal-repeater-411410-spectra-checkpoints/skypilot/"
        "1b-singlemixer-n64-muon-v6e64-b4096-ga2-ctx60-t25-use5a-"
        "datapart-20260811-045140/checkpoints/orbax/350000"
    )
    cfg.pairmixer_transition_type = "feedforward"
    cfg.attention_mlp_multiple = 4

    cfg.batch_size = cfg.group_jepa_groups_per_batch
    cfg.training_max_steps = 100_000
    cfg.num_epochs = 100
    cfg.optimizer = "muon"
    cfg.learning_rate = 4e-5
    cfg.min_learning_rate = 4e-7
    cfg.muon_adam_learning_rate = None
    cfg.warmup_steps = 5_000
    cfg.gradient_accumulation_steps = 2
    cfg.weight_decay = 0.1
    cfg.grad_clip_norm = 1.0

    cfg.checkpoint_every_steps = 5_000
    cfg.val_every_n_steps = 10_000
    cfg.val_num_steps = 50
    cfg.max_duration_hours = None
    cfg.msg_probe_every_n_steps = -1.0
    cfg.msg_probe_at_final_step = False
    cfg.jax_enable_async_checkpointing = True
    cfg.jax_checkpoint_max_to_keep = 5
    cfg.run_name_suffix = (
        "jax-grouped-jepa-massive-v2-1b-singlemixer-n64-g256-s8-t3-"
        "ema9992-muon-lr4e-5-100k"
    )
    return cfg
