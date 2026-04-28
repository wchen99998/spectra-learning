from configs.gems_small_norm import get_config as get_base_config


def get_config():
    cfg = get_base_config()
    cfg.use_ema_teacher = True
    cfg.ema_teacher_momentum = 0.9995
    cfg.ema_teacher_momentum_mid = 0.95
    cfg.ema_teacher_momentum_final = 0.999
    cfg.ema_teacher_schedule_peak_fraction = 0.35
    cfg.ema_teacher_schedule = "slow-fast-slow"
    cfg.sigreg_lambda = 0
    cfg.spectral_bias_relative_kind = "none"

    cfg.encoder_use_cls_token = False
    cfg.encoder_num_register_tokens = 0
    cfg.encoder_apply_final_norm = False
    cfg.predictor_apply_final_norm = False
    cfg.norm_type = "rmsnorm"

    cfg.jepa_context_fraction = 0.2
    cfg.jepa_context_fraction_range = (0.2, 0.3)
    cfg.jepa_target_fraction = 0.4
    cfg.jepa_target_fraction_range = (0.3, 0.45)
    cfg.jepa_block_min_len = 1
    cfg.jepa_mask_strategy = "all"
    cfg.jepa_mask_lengths = (2, 4, 8, 12,)
    cfg.jepa_mask_round_from = 3

    cfg.num_epochs = 5
    cfg.learning_rate = 3e-4
    cfg.warmup_steps = 10_000
    cfg.min_learning_rate = 1e-4
    cfg.b2 = 0.999
    cfg.weight_decay = 0.05
    cfg.optimizer = "adamw"

    cfg.use_target_projector = False


    cfg.run_name_suffix = "ema-teacher"
    return cfg
