from configs.gems_small_norm import get_config as get_base_config


def get_config():
    cfg = get_base_config()
    cfg.use_ema_teacher = True
    cfg.ema_teacher_momentum = 0.9995
    cfg.ema_teacher_momentum_mid = 0.99
    cfg.ema_teacher_momentum_final = 0.999
    cfg.ema_teacher_schedule_peak_fraction = 0.2
    cfg.ema_teacher_schedule = "slow-fast-slow"
    cfg.run_name_suffix = "ema-teacher"
    return cfg
