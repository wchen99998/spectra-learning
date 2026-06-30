from ml_collections import config_dict
import math

from configs.medium_pairmixer_100m_20m_mae_beta_isoflops import (
    get_config as get_beta_config,
)


def get_config() -> config_dict.ConfigDict:
    cfg = get_beta_config()

    cfg.device_backend = "jax"
    cfg.optimizer = "adamw"
    lr_scale = math.sqrt(2.0)
    cfg.training_max_steps = 4_000_000
    cfg.learning_rate = 6e-04
    cfg.min_learning_rate = 6e-05
    cfg.batch_size = 2048
    cfg.gradient_accumulation_steps = 4
    cfg.jax_mesh_devices = "16"
    cfg.activation_checkpoint_mode = "selective"
    cfg.mae_context_encoder_pack_tokens = 0
    cfg.mae_context_encoder_pack_token_choices = ()
    # cfg.mae_context_encoder_pack_tokens = 20
    # cfg.mae_context_encoder_pack_token_choices = (20,)
    cfg.pairmixer_pair_dim = 256
    cfg.pairmixer_pair_feature_hidden_dim = 512
    cfg.jepa_mask_strategy = ["random"]
    cfg.jepa_context_fraction = 0.35
    cfg.jepa_target_fraction = 0.25
    cfg.peak_filtering = "grouped"
    cfg.pairmixer_block_type = "FastMixer"
    # round(31 * 0.35) context + round(31 * 0.25) target + 1 CLS.
    cfg.pairmixer_fast_max_visible_tokens = 20
    cfg.num_peaks = 31
    cfg.num_epochs = 98
    cfg.dataloader_num_workers = 32
    cfg.grouped_peak_shoulder_da = 0.02
    cfg.grouped_peak_isotope_charges = (1, 2, 3)
    cfg.msg_probe_every_n_steps = 100_000
    cfg.msg_probe_at_final_step = True
    cfg.nist_murcko_probe_num_repeats = 1
    cfg.val_every_n_steps = 10_000
    cfg.val_num_steps = 500
    cfg.run_name_suffix = (
        "mae-massive100m-20m-beta-isoflops-1e19-bs2048-ga4-muon-matchrms-"
        "lr4p24e-4-default-fullctx-selective-probe1x100k-final-val500x10k"
    )

    return cfg
