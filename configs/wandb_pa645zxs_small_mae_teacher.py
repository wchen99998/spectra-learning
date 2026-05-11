from ml_collections import config_dict

from configs.wandb_pa645zxs_small import get_config as get_mae_config


def get_config() -> config_dict.ConfigDict:
    cfg = get_mae_config()

    cfg.training_mode = "mae_teacher_jepa"
    cfg.frozen_teacher_checkpoint_path = (
        "checkpoints/modal/no_fourier_embed_sentinel/step-01250000.pt"
    )
    cfg.use_ema_teacher = False
    cfg.predictor_learning_rate_ratio = 1.0
    cfg.masked_token_loss_weight = 1
    cfg.jepa_mae_loss_weight = 0.0
    cfg.mae_loss_weight = 0.0
    cfg.collapse_metrics_every_n_steps = 250
    cfg.msg_probe_every_n_steps = 0.5

    cfg.run_name_suffix = "mae-teacher-jepa-all-masks-small-step1250000-pred20pct"

    return cfg
