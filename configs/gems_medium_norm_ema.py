from configs.gems_small_norm import get_config as get_base_config


def get_config():
    cfg = get_base_config()
    cfg.use_ema_teacher = True
    cfg.ema_teacher_momentum = 0.9995
    cfg.ema_teacher_momentum_mid = 0.99
    cfg.ema_teacher_momentum_final = 0.999
    cfg.ema_teacher_schedule_peak_fraction = 0.2
    cfg.ema_teacher_schedule = "slow-fast-slow"
    cfg.sigreg_lambda = 0
    cfg.run_name_suffix = "ema-teacher"

    cfg.model_dim = 384
    cfg.encoder_num_layers = 14
    cfg.encoder_num_heads = 12
    cfg.encoder_num_kv_heads = 12
    cfg.jepa_target_layers = [5, 8, 12, 14]
    return cfg
