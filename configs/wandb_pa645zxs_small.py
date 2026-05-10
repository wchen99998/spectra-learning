from ml_collections import config_dict

from configs.wandb_pa645zxs import get_config as get_base_config


def get_config() -> config_dict.ConfigDict:
    cfg = get_base_config()

    cfg.num_epochs = 50

    cfg.batch_size = 256
    cfg.model_dim = 512
    cfg.predictor_learning_rate_ratio = 1.0
    cfg.encoder_num_layers = 14
    cfg.encoder_num_heads = 8
    cfg.encoder_num_kv_heads = 8
    cfg.encoder_use_position_embedding = True
    cfg.encoder_use_fourier_features = False
    cfg.encoder_fourier_mlp_hidden_dim = 1024
    cfg.encoder_fourier_mlp_num_layers = 4

    cfg.predictor_dim = 256
    cfg.masked_latent_predictor_num_layers = 6
    cfg.masked_latent_predictor_num_heads = 8

    cfg.training_mode = "mae"
    cfg.use_ema_teacher = False
    cfg.masked_token_loss_weight = 0
    cfg.jepa_mae_loss_weight = 0.0
    cfg.mae_loss_weight = 1.0
    cfg.collapse_metrics_every_n_steps = 0
    if "jepa_target_layers" in cfg:
        del cfg["jepa_target_layers"]

    cfg.msg_probe_every_n_steps = 0.3
    cfg.jepa_mask_strategy = "intensity_aware"
    cfg.jepa_num_target_blocks = 2
    cfg.jepa_context_fraction = 0.35
    cfg.jepa_target_fraction = 0.25
    cfg.jepa_block_min_len = 1
    cfg.jepa_mask_lengths = (2, 4, 8, 12)
    cfg.jepa_mask_round_from = 3

    cfg.optimizer = "adamw"

    cfg.run_name_suffix = "mae-all-masks-80m-200ep-pred20pct"

    return cfg
