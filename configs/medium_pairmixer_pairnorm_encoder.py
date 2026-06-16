from ml_collections import config_dict

from configs.medium_pairmixer_encoder import get_config as get_medium_config


def get_config() -> config_dict.ConfigDict:
    cfg = get_medium_config()

    cfg.encoder_apply_final_pair_norm = True
    cfg.run_name_suffix = "mae-medium-pairmixer-pairnorm-92m"

    return cfg
