from ml_collections import config_dict

from configs.medium_pairmixer_100m_20m_mae_beta_isoflops import (
    get_config as get_beta_config,
)


def get_config() -> config_dict.ConfigDict:
    cfg = get_beta_config()

    cfg.device_backend = "jax"
    cfg.optimizer = "muon"
    cfg.learning_rate = 3e-04
    cfg.min_learning_rate = 3e-05
    cfg.muon_beta = 0.95
    cfg.muon_ns_steps = 4
    cfg.muon_ns_coeffs = (3.4445, -4.7750, 2.0315)
    cfg.muon_eps = 1e-08
    cfg.muon_mu_dtype = "float32"
    cfg.muon_nesterov = True
    cfg.muon_adaptive = False
    cfg.muon_preconditioning = "schatten"
    cfg.muon_adjust_lr_fn = "match_rms_adamw"
    cfg.muon_adam_learning_rate = cfg.learning_rate
    cfg.muon_adam_min_learning_rate = cfg.min_learning_rate
    cfg.muon_adam_b1 = 0.9
    cfg.muon_adam_b2 = cfg.b2
    cfg.muon_adam_eps_root = 0.0
    cfg.muon_adam_weight_decay = 0.0
    cfg.muon_consistent_rms = 0.2
    cfg.run_name_suffix = "mae-100m-20m-beta-isoflops-100k-bs1024-ga2-muon-matchrms-lr3e-4"

    return cfg
