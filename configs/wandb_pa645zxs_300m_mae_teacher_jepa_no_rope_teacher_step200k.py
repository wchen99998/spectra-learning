from ml_collections import config_dict

from configs.wandb_pa645zxs_small_no_spectral_intensity_ragged_no_rope_predictor import (
    get_config as get_teacher_base_config,
)


TEACHER_CONFIG_PATH = (
    "configs/wandb_pa645zxs_small_no_spectral_intensity_ragged_no_rope_predictor.py"
)
TEACHER_CHECKPOINT_PATH = (
    "/vol/checkpoints/mae_teachers/"
    "wandb_pa645zxs_small_no_spectral_intensity_ragged_no_rope_predictor_cls_no_warp_1gpu_20260521_080232/"
    "step-00200000.pt"
)


def get_config() -> config_dict.ConfigDict:
    cfg = get_teacher_base_config()

    cfg.training_mode = "mae_teacher_jepa"
    cfg.frozen_teacher_config_path = TEACHER_CONFIG_PATH
    cfg.frozen_teacher_checkpoint_path = TEACHER_CHECKPOINT_PATH
    cfg.use_ema_teacher = False

    cfg.model_dim = 1024
    cfg.encoder_num_layers = 20
    cfg.encoder_num_heads = 16
    cfg.encoder_num_kv_heads = 16
    cfg.feature_mlp_hidden_dim = 2048
    cfg.encoder_fourier_mlp_hidden_dim = 1024
    cfg.encoder_fourier_mlp_num_layers = 4

    cfg.predictor_dim = 512
    cfg.masked_latent_predictor_num_layers = 12
    cfg.masked_latent_predictor_num_heads = 16

    cfg.masked_token_loss_weight = 1
    cfg.jepa_mae_loss_weight = 0.0
    cfg.mae_loss_weight = 0.0
    cfg.predictor_learning_rate_ratio = 1.0
    cfg.collapse_metrics_every_n_steps = 250
    cfg.msg_probe_every_n_steps = 0.5
    cfg.max_duration_hours = 23.5

    cfg.jepa_target_layers = [5, 8, 12, 14]
    cfg.jepa_target_normalization = "none"

    cfg.run_name_suffix = "mae-teacher-jepa-300m-no-rope-teacher-step200k-23h"

    return cfg
