"""1B pair-only model, 47 peaks plus CLS, 32 v6e chips, 1M optimizer steps.

Global batch 512 = 4 spectra/chip/microstep x 32 chips x GA4. Keeps the
validated selective recomputation and metadata AdaLN-Zero configuration.
The schedule targets 1M steps, with a 72-hour training limit. Launch with a
three-day allocation limit as well; checkpoint unfinished training at cutoff.
"""

from importlib import import_module

from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    cfg = import_module(
        "configs.1b_fastmixer_pair_distogram_multiscale_n48_v6e128_ga4"
    ).get_config()
    cfg.jax_mesh_devices = "32"
    cfg.batch_size = 512
    cfg.training_max_steps = 1_000_000
    cfg.max_duration_hours = 72.0
    cfg.checkpoint_every_steps = 10_000
    cfg.log_every_n_steps = 25
    cfg.run_name_suffix = (
        "-paironly-d1536-l15-pairpred5-n48-pairdisto-multiscale-"
        "metadata-adaln-adam-bf16state-bs512-ga4-v6e32-1m-selective"
    )
    return cfg
