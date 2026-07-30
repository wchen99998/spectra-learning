from importlib import import_module

from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    cfg = import_module("configs.1b_pairmixer_dense_adamw").get_config()
    cfg.pairmixer_use_pair_bias = False
    cfg.run_name_suffix += "-no-pair-bias"
    return cfg
