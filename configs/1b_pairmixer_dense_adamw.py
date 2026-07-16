from importlib import import_module

from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    cfg = import_module("configs.300m_pairmixer_dense_adamw").get_config()

    cfg.model_dim = 1536
    cfg.encoder_num_layers = 25
    cfg.encoder_num_heads = 12
    cfg.feature_mlp_hidden_dim = 3072
    cfg.encoder_fourier_mlp_hidden_dim = 3072
    cfg.optimizer_state_dtype = "bf16"

    cfg.pairmixer_pair_dim = 640
    cfg.pairmixer_pair_feature_hidden_dim = 1280
    cfg.pairmixer_encoder_projection_kernel = "xla"
    cfg.pairmixer_predictor_projection_kernel = "pallas"
    cfg.pairmixer_encoder_projection_kernel_schedule = (
        "xla",
        "xla",
        "pallas",
    )
    cfg.pairmixer_predictor_projection_kernel_schedule = (
        "pallas",
        "xla",
        "xla",
    )

    cfg.jepa_context_fraction_schedule = (0.35, 0.55, 0.75)
    cfg.jepa_target_fraction_schedule = (0.50, 0.30, 0.10)
    cfg.jepa_mask_schedule_steps = (666_667, 1_333_334)
    cfg.gradient_accumulation_steps = 8
    cfg.gradient_accumulation_steps_schedule = (8, 8, 8)

    cfg.predictor_dim = 1536
    cfg.masked_latent_predictor_num_layers = 5
    cfg.masked_latent_predictor_num_heads = 12

    cfg.run_name_suffix = (
        "mae-massive1b-200m-v6e-d1536-p640-l25-h12-nomassprior-"
        "feat3072-fmlp3072-pred1536-l5-h12-bs2048-ga8-adamw-bf16state-lr6e-4-"
        "random-mask65-45-25-target50-30-10-splitkernels-intensityorder-fullctx-"
        "noac-noprobe-val500x10k"
    )

    return cfg
