from ml_collections import config_dict

from configs.medium_pairformer_encoder import get_config as get_teacher_config


TEACHER_CONFIG_PATH = "configs/medium_pairformer_encoder.py"
TEACHER_CHECKPOINT_PATH = (
    "gs://metal-repeater-411410-spectra-checkpoints/"
    "pairformer_medium/checkpoints/step-00100000.pt"
)


def get_config() -> config_dict.ConfigDict:
    cfg = get_teacher_config()

    cfg.training_backend = "torchax"
    cfg.training_mode = "mae_teacher_jepa"
    cfg.frozen_teacher_config_path = TEACHER_CONFIG_PATH
    cfg.frozen_teacher_checkpoint_path = TEACHER_CHECKPOINT_PATH
    cfg.use_ema_teacher = False

    cfg.masked_token_loss_weight = 1
    cfg.mae_loss_weight = 0.0
    cfg.jepa_mae_loss_weight = 0.0
    cfg.pair_latent_loss_weight = 0.001
    cfg.distogram_loss_weight = 0.0
    cfg.predictor_dropout = 0.0
    cfg.msg_probe_every_n_steps = 0
    cfg.collapse_metrics_every_n_steps = 0
    cfg.compile_mode = "none"
    cfg.optimizer_fused = False

    cfg.run_name_suffix = "torchax-mae-teacher-jepa-medium-step100000"

    return cfg
