"""1.778B pair-only encoder/predictor on 128 TPU v6e chips at GA4.

Global batch 2,048 = 4 samples/chip/microstep x 128 chips x 4 microsteps.

Only pair tensors pass through 15 encoder and 5 predictor blocks at width
2,048. Context pair features retain intensity and precursor information; masked
peaks enter only as predictor queries. Encoder pair inputs and predictor
queries use 2D rotary position encoding. Metadata uses AdaLN-Zero.
A 2,000-bin distance head uses 0.5 Da bins and multi-scale CE.

The token budget is 63 peaks plus one CLS position, represented only in the
pair stream. Per-block recomputation and replicated bf16 Adam avoid ZeRO-1.
The previous sentinel-input architecture's AOT compilation estimated
30.68 GiB/chip, with 0.56 GiB reserved-budget headroom.
This is compiler buffer sizing, not a runtime throughput or stability benchmark.
See docs/benchmarks/paironly_v6e_ga4.md for results and reproduction commands.

Launch with the repository's distributed TPU setup; all 128 devices must
be visible to JAX before constructing the data mesh. Stage the data on every
host using the existing shard planner and windowed row shuffle.
"""

from importlib import import_module

from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    cfg = import_module(
        "configs.1p8b_singlemixer_dense_adam_metadata_adaln_"
        "multiscale_n128_b6144_666k_d2048"
    ).get_config()
    cfg.training_mode = "pair_distogram"
    cfg.pairmixer_block_type = "fastmixer"
    cfg.pairmixer_pair_dim = 2048
    cfg.encoder_num_layers = 15
    cfg.masked_latent_predictor_num_layers = 5
    cfg.num_peaks = 63
    cfg.encoder_use_cls_token = True
    cfg.encoder_apply_final_pair_norm = True
    cfg.masked_token_input_mode = "latent_token"
    cfg.mae_loss_weight = 0.0
    cfg.mae_mz_multiscale_widths = ()
    cfg.mae_mz_multiscale_weight = 0.0
    cfg.distogram_loss_weight = 1.0
    cfg.distogram_multiscale_widths = (2, 4, 8, 16, 32, 64)
    cfg.distogram_multiscale_weight = 1.0
    cfg.activation_checkpoint_mode = "full"
    cfg.activation_checkpoint_modules = ("encoder", "predictor", "distogram")
    cfg.optimizer_state_dtype = "bf16"
    cfg.jax_shard_optimizer = False
    cfg.jax_zero1_optimizer = False
    cfg.jax_distributed_initialize = True
    cfg.jax_mesh_devices = "128"
    cfg.gradient_accumulation_steps = 4
    cfg.batch_size = 2048
    cfg.max_duration_hours = 72.0

    # Current input contract: shuffled storage-block windows, independent
    # host order, and sample/epoch-derived masking and metadata dropout.
    cfg.gems_hdf5_shuffle_window_blocks = 2048
    cfg.gems_hdf5_shuffle_seed = 67
    cfg.seed = 66

    # The existing MSG probe consumes single-token encoder outputs.
    cfg.msg_probe_every_n_steps = -1
    cfg.msg_probe_at_final_step = False
    cfg.run_name_suffix = (
        "-paironly-d2048-l15-pairpred5-n64-pairdisto-multiscale-"
        "metadata-adaln-adam-bf16state-bs2048-ga4-v6e128-666k"
    )
    return cfg
