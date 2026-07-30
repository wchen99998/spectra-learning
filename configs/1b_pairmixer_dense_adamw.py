from importlib import import_module

from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    cfg = import_module("configs.300m_pairmixer_dense_adamw").get_config()

    cfg.model_dim = 1536
    cfg.encoder_num_layers = 25
    cfg.encoder_num_heads = 12
    cfg.feature_mlp_hidden_dim = 3072
    cfg.encoder_fourier_mlp_hidden_dim = 3072
    cfg.learning_rate = 3e-4
    cfg.weight_decay = 0.1
    cfg.optimizer_state_dtype = "fp32"

    cfg.pairmixer_pair_dim = 640
    cfg.pairmixer_pair_feature_hidden_dim = 1280
    cfg.mae_loss_weight = 0.7
    cfg.mae_intensity_loss_weight = 0.0
    cfg.distogram_loss_weight = 0.3

    cfg.jepa_context_fraction_schedule = (0.35, 0.55, 0.75)
    cfg.jepa_target_fraction_schedule = (0.50, 0.30, 0.10)
    cfg.jepa_mask_schedule_steps = (250_000, 350_000)
    cfg.gradient_accumulation_steps = 8
    cfg.gradient_accumulation_steps_schedule = (8, 16, 16)
    cfg.activation_checkpoint_mode = "none"
    cfg.max_duration_hours = 95.0

    cfg.predictor_dim = 1536
    cfg.masked_latent_predictor_num_layers = 5
    cfg.masked_latent_predictor_num_heads = 12

    cfg.run_name_suffix = (
        "mae-massive1b-200m-v6e-d1536-p640-l25-h12-nomassprior-"
        "feat3072-fmlp3072-pred1536-l5-h12-bs2048-ga8-16-16-adamw-fp32state-"
        "lr3e-4-wd1e-1-"
        "random-mask65-45-25-target50-30-10-intensityorder-fullctx-"
        "noac-noprobe-val500x10k"
    )

    return cfg
