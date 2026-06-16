from ml_collections import config_dict

from configs.wandb_pa645zxs_small_no_spectral_intensity_ragged_rope_predictor import (
    get_config as get_rope_predictor_config,
)


def get_config() -> config_dict.ConfigDict:
    cfg = get_rope_predictor_config()
    cfg.jepa_num_target_blocks = 1
    cfg.run_name_suffix = (
        "mae-intensity-aware-ragged-no-rope-predictor-no-mz-warp-1view-ctx70-tgt30"
    )

    return cfg
