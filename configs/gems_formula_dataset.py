from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    cfg = config_dict.ConfigDict()

    cfg.dataset = "gems_formula"
    cfg.seed = 42
    cfg.batch_size = 512
    cfg.shuffle_buffer = 200_000
    cfg.tfrecord_buffer_size = 250_000
    cfg.drop_remainder = False
    cfg.dataloader_num_workers = 1
    cfg.dataloader_prefetch_factor = 2
    cfg.dataloader_persistent_workers = True
    cfg.dataloader_pin_memory = True

    cfg.gcp_key_path = "/home/wuhao/md4/key.json"
    cfg.gems_formula_tfrecord_dir = "data/gems_formula_tfrecord"
    cfg.gems_formula_raw_csv_path = (
        "data/gems_formula/raw/GeMS_2m_combined_formula_identifications.csv"
    )
    cfg.gems_formula_gcs_uri = (
        "gs://main-novogaia-bucket/gems/GeMS_2m_combined_formula_identifications.csv"
    )
    cfg.gems_formula_column_name = "formula"
    cfg.gems_adduct_column_name = "adduct"
    cfg.gems_formula_split_seed = 42
    cfg.gems_formula_num_shards = 16
    cfg.gems_formula_drop_remainder = False

    return cfg
