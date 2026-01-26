from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    cfg = config_dict.ConfigDict()
    cfg.dataset = "gems_a"
    cfg.tfrecord_dir = "data/gems_peaklist_tfrecord"
    cfg.batch_size = 32
    cfg.validation_fraction = 0.05
    cfg.shuffle_buffer = 10_000
    cfg.split_seed = 42
    cfg.num_shards = 4
    cfg.drop_remainder = False
    cfg.max_precursor_mz = 1000.0
    cfg.seed = 42
    return cfg
