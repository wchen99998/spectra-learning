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
    cfg.warmup_steps = 250
    cfg.b2 = 0.95
    cfg.grad_clip_norm = 1.0

    cfg.batch_size = 512
    cfg.gradient_accumulation_steps = 2
    cfg.training_max_steps = 2_500
    cfg.num_epochs = 1
    cfg.activation_checkpoint_mode = "none"
    cfg.ddp_static_graph = False

    cfg.model_dim = 448
    cfg.encoder_num_layers = 10
    cfg.encoder_num_heads = 7
    cfg.feature_mlp_hidden_dim = 896
    cfg.encoder_fourier_mlp_hidden_dim = 896
    cfg.pairmixer_pair_dim = 224
    cfg.pairmixer_pair_feature_hidden_dim = 448
    cfg.predictor_dim = 448
    cfg.masked_latent_predictor_num_heads = 7

    cfg.encoder_mz_embedding = "fourier"
    cfg.encoder_mz_token_bin_size = 0.02
    cfg.encoder_mz_token_embedding_dim = 38
    cfg.jepa_mae_mz_bin_size = 0.5

    cfg.compile_mode = "default"
    cfg.compile_scope = "blocks"
    cfg.dataloader_num_workers = 8
    cfg.device_prefetch_size = 4
    cfg.throughput_warmup_steps = 25
    cfg.log_every_n_steps = 25
    cfg.disable_progress_bar = True
    cfg.collapse_metrics_every_n_steps = 0
    cfg.checkpoint_every_steps = 1_000_000

    cfg.val_every_n_steps = 500
    cfg.val_num_steps = 64
    cfg.msg_probe_every_n_steps = -1
    cfg.msg_probe_at_final_step = False

    cfg.enable_wandb = False
    cfg.max_duration_hours = None
    cfg.run_name_suffix = "mz-token-ablation-52m-2p5k-2xh100-adam"

    return cfg
