from importlib import import_module

from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    cfg = import_module(
        "configs."
        "1b_singlemixer_dense_muon_metadata_adaln_"
        "sparse_ordinal_n128_b2048_1m"
    ).get_config()

    cfg.training_max_steps = 500_000
    cfg.max_duration_hours = 95.0
    cfg.min_learning_rate = cfg.learning_rate / 10
    cfg.run_name_suffix = cfg.run_name_suffix.removesuffix("-1m") + "-500k"

    return cfg
