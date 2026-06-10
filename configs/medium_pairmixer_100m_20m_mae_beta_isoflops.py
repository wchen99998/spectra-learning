from ml_collections import config_dict

from configs.medium_pairmixer_100m_20m_mae_alpha_isoflops import (
    get_config as get_alpha_config,
)


def get_config() -> config_dict.ConfigDict:
    cfg = get_alpha_config()

    cfg.artifact_dir = "data/gems_artifacts_beta"
    cfg.gems_native_repo_id = "cjim8889/gems-b-native"
    cfg.num_epochs = 2
    cfg.run_name_suffix = "mae-100m-20m-beta-isoflops-100k-bs1024-ga2"

    return cfg
