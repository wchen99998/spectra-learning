from configs.gems_small_norm_ema import get_config as get_base_config


def get_config():
    cfg = get_base_config()
    cfg.encoder_use_cls_token = False
    cfg.encoder_num_register_tokens = 0
    cfg.predictor_num_register_tokens = 0
    return cfg
