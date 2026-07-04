from importlib import import_module

from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    cfg = import_module("configs.100m_pairmixer_dense_adamw").get_config()

    cfg.model_dim = 1024
    cfg.training_max_steps = 2_000_000
    cfg.encoder_num_layers = 19
    cfg.encoder_num_heads = 8
    cfg.feature_mlp_hidden_dim = 1536
    cfg.encoder_fourier_mlp_hidden_dim = 2048

    cfg.pairmixer_pair_dim = 256
    cfg.pairmixer_pair_feature_hidden_dim = 512
    cfg.activation_checkpoint_mode = "none"

    cfg.predictor_dim = 1024
    cfg.masked_latent_predictor_num_layers = 3
    cfg.masked_latent_predictor_num_heads = 8

    cfg.run_name_suffix = (
        "mae-massive300m-49m-v6e-d1024-p256-l19-h8-nomassprior-"
        "feat1536-fmlp2048-pred1024-l3-h8-bs2048-ga4-adamw-lr6e-4-random-"
        "fullctx-noac-noprobe-val500x10k"
    )

    return cfg
