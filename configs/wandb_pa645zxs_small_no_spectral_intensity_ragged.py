from ml_collections import config_dict

from configs.wandb_pa645zxs_small_spectral import get_config as get_spectral_config


def get_config() -> config_dict.ConfigDict:
    cfg = get_spectral_config()

    cfg.spectral_bias_relative_kind = "none"
    cfg.spectral_bias_use_precursor = False
    cfg.spectral_bias_use_intensity = False

    del cfg["jepa_mask_strategy"]
    cfg.jepa_mask_strategy = ["intensity_aware", "ragged"]

    cfg.run_name_suffix = "mae-no-spectral-bias-intensity-aware-ragged"

    return cfg
