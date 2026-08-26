from importlib import import_module

from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    cfg = import_module(
        "configs.1b_singlemixer_dense_muon_metadata_adaln"
    ).get_config()

    cfg.batch_size = 2_048
    cfg.num_peaks = 127
    cfg.mae_mz_multiscale_widths = (2, 4, 8, 16, 32, 64)
    cfg.mae_mz_multiscale_weight = 1.0
    cfg.training_max_steps = 1_000_000
    cfg.max_duration_hours = 167.0
    cfg.run_name_suffix = (
        "jax-mae-massive-v2-10gb-1b-200m-v6e-singlemixer-metadata-adaln-"
        "multiscale-n128-d1536-l25-h12-swiglu-nomassprior-feat3072-"
        "fmlp3072-xattnrope1536-l5-h12-bs2048-ga2-mae1-muon-cls-"
        "fp32state-lr3e-4-wu20k-wd1e-1-random-ctx60-target25-mzorder-"
        "fullctx-noac-noprobe-val500x10k-1m"
    )

    return cfg
