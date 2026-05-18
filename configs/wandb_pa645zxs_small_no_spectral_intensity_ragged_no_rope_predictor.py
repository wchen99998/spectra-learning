from ml_collections import config_dict

from configs.wandb_pa645zxs_small_no_spectral_intensity_ragged_rope_predictor import (
    get_config as get_rope_predictor_config,
)


def get_config() -> config_dict.ConfigDict:
    cfg = get_rope_predictor_config()

    cfg.predictor_use_rope = False
    cfg.run_name_suffix = "mae-no-spectral-bias-intensity-aware-ragged-no-rope-predictor"

    return cfg
