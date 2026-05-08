from configs.gems_small_norm import get_config as get_base_config


def get_config():
    cfg = get_base_config()
    cfg.training_mode = "mae"
    cfg.use_ema_teacher = False
    cfg.masked_token_loss_weight = 0.0
    cfg.jepa_mae_loss_weight = 0.0
    cfg.mae_loss_weight = 1.0
    cfg.jepa_target_layers = [cfg.encoder_num_layers]
    cfg.representation_regularizer = "none"
    cfg.sigreg_lambda = 0.0
    cfg.collapse_metrics_every_n_steps = 0
    cfg.run_name_suffix = "mae-binned"
    return cfg
