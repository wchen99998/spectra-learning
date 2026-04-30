from configs.gems_small_norm import get_config as get_base_config


def get_config():
    cfg = get_base_config()
    cfg.use_ema_teacher = True
    cfg.ema_teacher_momentum_start = 0.9995
    cfg.ema_teacher_momentum_mid = 0.95
    cfg.ema_teacher_momentum_final = 0.999
    cfg.ema_teacher_schedule_peak_fraction = 0.35
    cfg.ema_teacher_schedule = "slow-fast-slow"
    cfg.sigreg_lambda = 0

    cfg.jepa_context_fraction = 0.35
    cfg.jepa_target_fraction = 0.2
    cfg.jepa_block_min_len = 1
    cfg.jepa_mask_strategy = "all"
    cfg.jepa_mask_lengths = (2, 4, 8, 12,)
    cfg.jepa_mask_round_from = 3

    cfg.weight_decay = 1e-3

    cfg.target_projector_dim = -1


    cfg.run_name_suffix = "ema-teacher"
    return cfg
