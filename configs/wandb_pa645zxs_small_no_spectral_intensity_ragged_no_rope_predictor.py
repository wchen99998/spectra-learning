from ml_collections import config_dict

from configs.wandb_pa645zxs_small_no_spectral_intensity_ragged_rope_predictor import (
    get_config as get_rope_predictor_config,
)


def get_config() -> config_dict.ConfigDict:
    cfg = get_rope_predictor_config()

    cfg.predictor_use_rope = False
    cfg.encoder_use_cls_token = True
    cfg.encoder_fourier_warp_enabled = False
    cfg.jepa_num_target_blocks = 1
    cfg.jepa_context_fraction = 0.70
    cfg.jepa_target_fraction = 0.30
    cfg.jepa_intensity_aware_context_fraction = 0.70
    cfg.jepa_intensity_aware_target_fraction = 0.30
    cfg.jepa_intensity_aware_tail_target_mix = 1.0
    cfg.jepa_intensity_aware_local_gap_probability = 0.0
    cfg.run_name_suffix = (
        "mae-no-spectral-bias-intensity-aware-ragged-no-rope-predictor-cls-no-mz-warp-1view-ctx70-tgt30"
    )

    return cfg
