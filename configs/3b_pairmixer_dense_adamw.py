from importlib import import_module

from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    cfg = import_module("configs.1b_pairmixer_dense_adamw").get_config()

    cfg.model_dim = 2560
    cfg.encoder_num_layers = 30
    cfg.encoder_num_heads = 20
    cfg.feature_mlp_hidden_dim = 5120
    cfg.encoder_fourier_mlp_hidden_dim = 5120
    cfg.learning_rate = 6e-4
    cfg.weight_decay = 0.01
    cfg.optimizer_state_dtype = "bf16"

    cfg.pairmixer_pair_dim = 768
    cfg.pairmixer_pair_feature_hidden_dim = 1536
    cfg.mae_loss_weight = 1.0
    cfg.distogram_loss_weight = 1.0

    cfg.predictor_dim = 2560
    cfg.masked_latent_predictor_num_layers = 3
    cfg.masked_latent_predictor_num_heads = 20

    cfg.gradient_accumulation_steps = 16
    cfg.gradient_accumulation_steps_schedule = (16, 32, 32)
    cfg.max_duration_hours = 47.0
    cfg.jax_mesh_devices = "16"
    cfg.run_name_suffix = (
        "mae-massive3b-300m-v7x8-d2560-p768-l30-h20-nomassprior-"
        "feat5120-fmlp5120-pred2560-l3-h20-bs2048-ga16-32-32-adamw-bf16state-"
        "lr6e-4-random-mask65-45-25-target50-30-10-intensityorder-fullctx-"
        "noac-noprobe-val500x10k"
    )

    return cfg
