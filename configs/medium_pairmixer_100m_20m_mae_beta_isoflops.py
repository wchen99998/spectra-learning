from ml_collections import config_dict

from configs.medium_pairmixer_100m_20m_mae_alpha_isoflops import (
    get_config as get_alpha_config,
)


def get_config() -> config_dict.ConfigDict:
    cfg = get_alpha_config()

    cfg.artifact_dir = "data/massive_v1_ms2_100m_stratified_x16"
    cfg.num_epochs = 2
    cfg.activation_checkpoint_mode = "selective"
    cfg.run_name_suffix = "mae-massive100m-20m-beta-isoflops-100k-bs1024-ga2"

    return cfg
