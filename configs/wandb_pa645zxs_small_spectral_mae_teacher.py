from ml_collections import config_dict

from configs.wandb_pa645zxs import get_config as get_base_config


def get_config() -> config_dict.ConfigDict:
    cfg = get_base_config()

    cfg.num_epochs = 25

    cfg.batch_size = 512
    cfg.model_dim = 512
    cfg.predictor_learning_rate_ratio = 1.0
    cfg.encoder_num_layers = 14
    cfg.encoder_num_heads = 8
    cfg.encoder_num_kv_heads = 8
    cfg.encoder_use_position_embedding = False
    cfg.encoder_use_fourier_features = True
    cfg.encoder_apply_final_norm = True
    cfg.predictor_apply_final_norm = True
    cfg.encoder_fourier_mlp_hidden_dim = 1024
    cfg.encoder_fourier_mlp_num_layers = 4
    cfg.encoder_fourier_num_freqs = 64
    cfg.precursor_peak_exclusion_window_da = 0
    cfg.masked_token_input_mode= "mz_sentinel"

    cfg.spectral_bias_relative_kind = "harmonic"
    cfg.spectral_bias_use_precursor = False  # first isolate it
    cfg.spectral_bias_use_intensity = False
    cfg.spectral_bias_num_freqs = 16
    cfg.spectral_bias_fourier_x_min = 0.1  # or 1.0
    cfg.spectral_bias_fourier_x_max = 1000.0
    cfg.spectral_bias_init_std = 0.0

    cfg.predictor_dim = 256
    cfg.masked_latent_predictor_num_layers = 6
    cfg.masked_latent_predictor_num_heads = 8

    cfg.training_mode = "mae_teacher_jepa"
    cfg.frozen_teacher_checkpoint_path = (
        "experiments/small_exp3_spectra_cotraincov2_adw/checkpoints/step-00875000.pt"
    )
    cfg.use_ema_teacher = False
    cfg.masked_token_loss_weight = 1
    cfg.jepa_mae_loss_weight = 0.0
    cfg.mae_loss_weight = 0.0
    cfg.collapse_metrics_every_n_steps = 0
    if "jepa_target_layers" in cfg:
        del cfg["jepa_target_layers"]

    cfg.msg_probe_every_n_steps = 0.3
    cfg.msg_probe_batch_size = 256
    cfg.msg_probe_backend = "modal"
    cfg.jepa_mask_strategy = "intensity_aware"
    cfg.jepa_num_target_blocks = 4
    cfg.jepa_context_fraction = 0.35
    cfg.jepa_target_fraction = 0.25
    cfg.jepa_block_min_len = 1
    cfg.jepa_mask_lengths = (2, 4, 8, 12)
    cfg.jepa_mask_round_from = 3
    cfg.train_covariance_pooling = False
    cfg.msg_probe_fingerprint = "morgan"
    cfg.msg_probe_tune_metric = "msg_probe/test/auc_morgan_mean"
    

    cfg.optimizer = "adamw"
    cfg.learning_rate = 2e-4
    cfg.weight_decay = 0.01
    cfg.min_learning_rate = 2e-05
    cfg.b2 = 0.95
    cfg.warmup_steps = 10_000

    cfg.run_name_suffix = "mae-teacher-jepa-small-spectral-step875000-morgan4096r2"

    return cfg
