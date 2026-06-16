from ml_collections import config_dict

from configs.wandb_pa645zxs_small import get_config as get_small_config


def get_config() -> config_dict.ConfigDict:
    cfg = get_small_config()

    cfg.num_epochs = 25
    cfg.encoder_use_position_embedding = False
    cfg.encoder_apply_final_norm = True
    cfg.predictor_apply_final_norm = True
    cfg.encoder_fourier_mlp_hidden_dim = 1024
    cfg.encoder_fourier_mlp_num_layers = 4
    cfg.encoder_fourier_num_freqs = 64
    cfg.precursor_peak_exclusion_window_da = 0
    cfg.masked_token_input_mode = "mz_sentinel"
    cfg.jepa_mae_mz_bin_size = 0.1
    cfg.msg_probe_every_n_steps = 0.25
    cfg.val_every_n_steps = cfg.msg_probe_every_n_steps
    cfg.val_num_steps = 64
    cfg.msg_probe_batch_size = 256
    cfg.learning_rate = 2e-4
    cfg.weight_decay = 0.01
    cfg.min_learning_rate = 1e-05
    cfg.warmup_steps = 10_000
    cfg.b2 = 0.95

    del cfg["jepa_mask_strategy"]
    cfg.jepa_mask_strategy = ["intensity_aware", "ragged"]

    cfg.run_name_suffix = "mae-intensity-aware-ragged"

    return cfg
