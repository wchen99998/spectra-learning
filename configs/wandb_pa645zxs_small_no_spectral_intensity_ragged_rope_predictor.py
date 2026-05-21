from ml_collections import config_dict

from configs.wandb_pa645zxs_small_no_spectral_intensity_ragged import (
    get_config as get_ragged_config,
)


def get_config() -> config_dict.ConfigDict:
    cfg = get_ragged_config()

    cfg.masked_token_input_mode = "latent_token"
    cfg.encoder_use_cls_token = True
    cfg.run_name_suffix = "mae-no-spectral-bias-intensity-aware-ragged-predictor"

    return cfg
