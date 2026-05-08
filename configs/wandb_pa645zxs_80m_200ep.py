from ml_collections import config_dict

from configs.wandb_pa645zxs import get_config as get_base_config


def get_config() -> config_dict.ConfigDict:
    cfg = get_base_config()

    cfg.num_epochs = 200

    cfg.model_dim = 512
    cfg.predictor_learning_rate_ratio = 1.0
    cfg.encoder_num_layers = 25
    cfg.encoder_num_heads = 16
    cfg.encoder_num_kv_heads = 16
    cfg.encoder_use_position_embedding = True


    cfg.predictor_dim = 352
    cfg.masked_latent_predictor_num_layers = 10
    cfg.masked_latent_predictor_num_heads = 11

    cfg.training_mode = "mae"
    cfg.use_ema_teacher = False
    cfg.masked_token_loss_weight = 0
    cfg.jepa_mae_loss_weight = 0.0
    cfg.mae_loss_weight = 1.0
    cfg.collapse_metrics_every_n_steps = 0
    if "jepa_target_layers" in cfg:
        del cfg["jepa_target_layers"]

    cfg.jepa_mask_strategy = "all"
    cfg.jepa_context_fraction = 0.35
    cfg.jepa_target_fraction = 0.2
    cfg.jepa_block_min_len = 1
    cfg.jepa_mask_lengths = (2, 4, 8, 12)
    cfg.jepa_mask_round_from = 3

    cfg.run_name_suffix = "mae-all-masks-80m-200ep-pred20pct"

    return cfg
