from configs.gems_small_norm_ema import get_config as get_base_config


def get_config():
    cfg = get_base_config()
    cfg.target_projector_dim = -1
    return cfg
