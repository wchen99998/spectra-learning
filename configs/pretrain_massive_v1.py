from ml_collections import config_dict

from configs.pretrain import get_config as get_pretrain_config


def get_config() -> config_dict.ConfigDict:
    """Temporary compatibility config for v1-bound training resumes."""
    config = get_pretrain_config()
    config.artifact_dir = "data/massive_v1_ms2_100m_stratified_x16"
    config.gems_hdf5_repo_id = (
        "novogaia/massive-v1-ms2-100m-stratified-x16"
    )
    config.gems_hdf5_revision = (
        "7ff47061cbde23e4cdd113378dcfb489e86b32c4"
    )
    config.gems_hdf5_manifest = "fdataloader_shards.json"
    return config
