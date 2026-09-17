"""1B pair-only model with 47 peaks plus CLS on 128 TPU v6e chips.

Keeps the 15 encoder and five predictor blocks of the 1.8B pair-only run,
with pair width reduced to 1,536. Global batch 2,048 at GA4 gives four
spectra per chip per microstep. Replicated bf16 Adam moments avoid ZeRO-1.
For the previous sentinel-input architecture, selective recomputation measured
0.859 s/update on four v6e chips with the same per-chip batch shape; 128-chip
compilation estimated 27.48 GiB/chip.
See docs/benchmarks/paironly1b_n48_v6e128_ga4.md for measurement limits.

The parent supplies metadata AdaLN-Zero, multiscale distogram supervision,
context-only encoding and masked predictor queries, the current data mixing,
and a 72-hour limit.
"""

from importlib import import_module

from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    cfg = import_module(
        "configs.1p8b_fastmixer_pair_distogram_multiscale_n64_v6e128_ga4"
    ).get_config()
    cfg.model_dim = 1536
    cfg.predictor_dim = 1536
    cfg.pairmixer_pair_dim = 1536
    cfg.num_peaks = 47
    cfg.activation_checkpoint_mode = "selective"
    cfg.run_name_suffix = (
        "-paironly-d1536-l15-pairpred5-n48-pairdisto-multiscale-"
        "metadata-adaln-adam-bf16state-bs2048-ga4-v6e128-666k-selective"
    )
    return cfg
