from ml_collections import config_dict
from configs.ar_spectra_coarse_to_fine import get_config as get_ar_config


def get_config() -> config_dict.ConfigDict:
    cfg = get_ar_config()

    cfg.ar_model_dim = 1792
    cfg.ar_num_layers = 8
    cfg.ar_num_heads = 14
    cfg.ar_mlp_multiple = 4.0
    cfg.wandb_kwargs = {
        "name": "ar-spectra-jax-rope-multilevel-300m-bs2048-bf16-250k",
    }

    return cfg
