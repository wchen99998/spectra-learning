from ml_collections import config_dict

from configs.wandb_pa645zxs_80m_200ep import get_config as get_mae_config


def get_config() -> config_dict.ConfigDict:
    cfg = get_mae_config()

    cfg.training_mode = "mae_teacher_jepa"
    cfg.frozen_teacher_checkpoint_path = (
        "experiments/wandb_pa645zxs_80m_200ep_mae_2gpu_10ep/checkpoints/last.pt"
    )
    cfg.use_ema_teacher = False
    cfg.masked_token_loss_weight = 1
    cfg.jepa_mae_loss_weight = 0.0
    cfg.mae_loss_weight = 0.0
    cfg.collapse_metrics_every_n_steps = 250
    cfg.msg_probe_every_n_steps = 0.5

    cfg.jepa_target_layers = [9, 14, 21, 25]
    cfg.jepa_mask_strategy = "all"
    cfg.jepa_context_fraction = 0.35
    cfg.jepa_target_fraction = 0.2
    cfg.jepa_block_min_len = 1
    cfg.jepa_mask_lengths = (2, 4, 8, 12)
    cfg.jepa_mask_round_from = 3

    cfg.run_name_suffix = "mae-teacher-jepa-all-masks-80m-200ep-pred20pct"

    return cfg
