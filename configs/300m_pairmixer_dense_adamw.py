from importlib import import_module

from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    cfg = import_module("configs.100m_pairmixer_dense_adamw").get_config()

    cfg.model_dim = 896
    cfg.training_max_steps = 2_000_000
    cfg.encoder_num_layers = 18
    cfg.encoder_num_heads = 7
    cfg.feature_mlp_hidden_dim = 1536
    cfg.encoder_fourier_mlp_hidden_dim = 1792

    cfg.pairmixer_pair_dim = 384
    cfg.pairmixer_pair_feature_hidden_dim = 768
    cfg.activation_checkpoint_mode = "none"
    cfg.peak_ordering = "intensity"

    cfg.predictor_dim = 896
    cfg.masked_latent_predictor_num_layers = 3
    cfg.masked_latent_predictor_num_heads = 7

    cfg.run_name_suffix = (
        "mae-massive300m-44m-v6e-d896-p384-l18-h7-nomassprior-"
        "feat1536-fmlp1792-pred896-l3-h7-bs2048-ga4-adamw-lr6e-4-random-"
        "intensityorder-fullctx-noac-noprobe-val500x10k"
    )

    return cfg
