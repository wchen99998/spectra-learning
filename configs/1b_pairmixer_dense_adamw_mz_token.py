from importlib import import_module

from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    cfg = import_module("configs.1b_pairmixer_dense_adamw").get_config()

    cfg.encoder_mz_embedding = "token"
    cfg.encoder_mz_token_bin_size = 0.1
    cfg.pairmixer_mz_embedding = "token"
    cfg.pairmixer_mz_token_bin_size = 0.1
    cfg.run_name_suffix += "-mz-token01-single-pair"

    return cfg
