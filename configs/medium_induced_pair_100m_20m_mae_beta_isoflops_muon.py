from ml_collections import config_dict

from configs.medium_pairmixer_100m_20m_mae_beta_isoflops_muon import (
    get_config as get_pairmixer_config,
)


def get_config() -> config_dict.ConfigDict:
    cfg = get_pairmixer_config()

    cfg.pairmixer_block_type = "induced"
    cfg.induced_pair_num_inducing = 8
    cfg.run_name_suffix = (
        "mae-100m-20m-beta-isoflops-250k-bs4096-ga4-muon-matchrms-"
        "lr4p24e-4-pack20-fullcompile-selective-probe1x100k-final-"
        "val1k25k-inducedm8"
    )

    return cfg
