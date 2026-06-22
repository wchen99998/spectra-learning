from ml_collections import config_dict
import math

from configs.medium_pairmixer_100m_20m_mae_beta_isoflops import (
    get_config as get_beta_config,
)


def get_config() -> config_dict.ConfigDict:
    cfg = get_beta_config()

    cfg.device_backend = "jax"
    cfg.optimizer = "muon"
    lr_scale = math.sqrt(2.0)
    cfg.training_max_steps = 250_000
    cfg.learning_rate = 3e-04 * lr_scale
    cfg.min_learning_rate = 3e-05 * lr_scale
    cfg.muon_beta = 0.95
    cfg.muon_ns_steps = 5
    cfg.muon_ns_coeffs = (3.4445, -4.7750, 2.0315)
    cfg.muon_eps = 1e-08
    cfg.muon_mu_dtype = "float32"
    cfg.muon_nesterov = True
    cfg.muon_adaptive = False
    cfg.muon_preconditioning = "frobenius"
    cfg.muon_adjust_lr_fn = "match_rms_adamw"
    cfg.muon_adam_learning_rate = cfg.learning_rate
    cfg.muon_adam_min_learning_rate = cfg.min_learning_rate
    cfg.muon_adam_b1 = 0.9
    cfg.muon_adam_b2 = cfg.b2
    cfg.muon_adam_eps_root = 0.0
    cfg.muon_adam_weight_decay = 0.0
    cfg.muon_consistent_rms = 0.2
    cfg.batch_size = 2048
    cfg.gradient_accumulation_steps = 4
    cfg.jax_mesh_devices = "16"
    cfg.activation_checkpoint_mode = "selective"
    cfg.jax_precompile_variant = "all"
    cfg.jax_precompile_eval_steps = True
    cfg.jax_precompile_msg_probe = True
    cfg.mae_context_encoder_pack_tokens = 0
    cfg.mae_context_encoder_pack_token_choices = ()
    # cfg.mae_context_encoder_pack_tokens = 20
    # cfg.mae_context_encoder_pack_token_choices = (20,)
    cfg.pairmixer_pair_dim = 256
    cfg.pairmixer_pair_feature_hidden_dim = 512
    cfg.jepa_intensity_aware_context_fraction = 0.35
    cfg.peak_filtering = "grouped"
    cfg.pairmixer_block_type = "dense"
    cfg.num_peaks = 47
    cfg.num_epochs = 20
    cfg.dataloader_num_workers = 32
    cfg.grouped_peak_shoulder_da = 0.02
    cfg.grouped_peak_isotope_charges = (1, 2, 3)
    cfg.msg_probe_every_n_steps = 100_000
    cfg.msg_probe_at_final_step = True
    cfg.nist_murcko_probe_num_repeats = 1
    cfg.val_every_n_steps = 25_000
    cfg.val_num_steps = 1_000
    cfg.run_name_suffix = "mae-100m-20m-beta-isoflops-250k-bs4096-ga4-muon-matchrms-lr4p24e-4-pack20-fullcompile-selective-probe1x100k-final-val1k25k"

    return cfg
