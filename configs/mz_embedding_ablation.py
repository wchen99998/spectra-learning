from importlib import import_module

from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    cfg = import_module("configs.100m_pairmixer_dense_adamw").get_config()

    cfg.device_backend = "torch"
    cfg.optimizer = "adam"
    cfg.optimizer_fused = True
    cfg.optimizer_state_dtype = "fp32"
    cfg.weight_decay = 0.0
    cfg.learning_rate = 3e-4
    cfg.min_learning_rate = 3e-5
    cfg.warmup_steps = 100
    cfg.b2 = 0.95
    cfg.grad_clip_norm = 1.0

    cfg.batch_size = 512
    cfg.gradient_accumulation_steps = 4
    cfg.training_max_steps = 1_000
    cfg.num_epochs = 1
    cfg.activation_checkpoint_mode = "none"
    cfg.ddp_static_graph = False

    cfg.encoder_mz_embedding = "fourier"
    cfg.encoder_discrete_mz_bin_size = 0.02
    cfg.encoder_discrete_mz_coarse_bin_size = 1.0
    cfg.encoder_discrete_mz_embedding_dim = 70

    cfg.compile_mode = "default"
    cfg.compile_scope = "blocks"
    cfg.dataloader_num_workers = 8
    cfg.device_prefetch_size = 4
    cfg.throughput_warmup_steps = 25
    cfg.log_every_n_steps = 25
    cfg.disable_progress_bar = True
    cfg.collapse_metrics_every_n_steps = 0
    cfg.checkpoint_every_steps = 1_000_000

    cfg.val_every_n_steps = 250
    cfg.val_num_steps = 32
    cfg.msg_probe_every_n_steps = -1
    cfg.msg_probe_at_final_step = False

    cfg.enable_wandb = False
    cfg.max_duration_hours = None
    cfg.run_name_suffix = "mz-embedding-ablation-2xh100-adam"

    return cfg
