from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    cfg = config_dict.ConfigDict()

    # Runtime
    cfg.device = "auto"
    cfg.device_backend = "jax"
    cfg.jax_checkpoint_max_to_keep = 15
    cfg.jax_compilation_cache_dir = ""
    cfg.jax_compile_stall_threshold_seconds = 0.0
    cfg.jax_distributed_initialize = False
    cfg.jax_enable_async_checkpointing = True
    cfg.jax_enable_compilation_cache = True
    cfg.jax_explain_cache_misses = False
    cfg.jax_log_compiles = False
    cfg.jax_log_update_stats = False
    cfg.jax_mesh_devices = "64"
    cfg.jax_persistent_cache_min_compile_time_secs = None
    cfg.jax_persistent_cache_min_entry_size_bytes = None
    cfg.jax_profile_dir = ""
    cfg.jax_profile_start_step = 5_000
    cfg.jax_profile_steps = 0
    cfg.jax_timing_barriers = False
    cfg.max_duration_hours = 47.0
    cfg.training_flops_per_optimizer_step = None
    cfg.training_flops_per_parameter = 6.0
    cfg.training_flops_per_sample = None

    # Dataset
    cfg.training_task = "pretrain"
    cfg.artifact_dir = "data/massive_v2_ms2_t095_l080_sharded_10gb"
    cfg.gems_hdf5_repo_id = (
        "novogaia/massive-v2-ms2-t095-l080-sharded-10gb"
    )
    cfg.gems_hdf5_revision = "b098155fccb7335b752f2690bf16d2479d69bf5b"
    cfg.gems_hdf5_manifest = "manifest.json"
    cfg.gems_hdf5_spectrum_dataset = "spectrum"
    cfg.gems_hdf5_precursor_dataset = "precursor_mz"
    cfg.gems_hdf5_retention_time_dataset = "RT"
    cfg.gems_hdf5_ms_level_dataset = "MS level"
    cfg.gems_hdf5_rows_per_block = 0
    cfg.batch_size = 4_096
    cfg.gradient_accumulation_steps = 2
    cfg.shuffle_buffer = 1_000_000
    cfg.drop_remainder = True
    cfg.max_precursor_mz = 1_000
    cfg.min_peak_intensity = 0.0001
    cfg.peak_drop_min_intensity = 0.0001
    cfg.precursor_peak_exclusion_window_da = 0
    cfg.peak_ordering = "mz"
    cfg.seed = 66

    # Single-stream encoder
    cfg.num_peaks = 63
    cfg.model_dim = 1_536
    cfg.encoder_num_layers = 25
    cfg.encoder_num_heads = 12
    cfg.encoder_use_cls_token = True
    cfg.encoder_metadata_schema = "massive_v2_acquisition_v1"
    cfg.encoder_metadata_conditioning = "adaln_zero"
    cfg.encoder_metadata_condition_dim = 256
    cfg.spectrum_metadata_dropout_probability = 0.10
    cfg.encoder_use_position_embedding = True
    cfg.encoder_apply_final_norm = True
    cfg.encoder_apply_final_pair_norm = True
    cfg.encoder_mz_embedding = "fourier"
    cfg.encoder_mz_token_bin_size = 0.02
    cfg.encoder_mz_token_embedding_dim = 77
    cfg.encoder_fourier_mlp_hidden_dim = 3_072
    cfg.encoder_fourier_mlp_num_layers = 4
    cfg.encoder_fourier_num_freqs = 64
    cfg.encoder_fourier_x_max = 1_000
    cfg.encoder_fourier_x_min = 0.003
    cfg.feature_mlp_hidden_dim = 3_072
    cfg.pairmixer_block_type = "fastmixer-dense"
    cfg.pairmixer_transition_type = "feedforward"
    cfg.pairmixer_pair_dim = 640
    cfg.pairmixer_pair_feature_hidden_dim = 1_280
    cfg.pairmixer_dropout = 0.0
    cfg.pairmixer_use_pair_bias = True
    cfg.pairmixer_mz_embedding = "fourier"
    cfg.pairmixer_mz_token_bin_size = 0.1
    cfg.pairmixer_mz_token_embedding_dim = 128
    cfg.pairmixer_fourier_num_freqs = 16
    cfg.pairmixer_fourier_x_min = 0.01
    cfg.pairmixer_relative_fourier_x_min = 0.001
    cfg.pairmixer_relative_fourier_x_max = 1.0
    cfg.attention_mlp_multiple = 4
    cfg.norm_eps = 1e-5

    # Cross-attention predictor
    cfg.predictor_dim = 1_536
    cfg.predictor_dropout = 0.1
    cfg.predictor_apply_final_norm = True
    cfg.masked_latent_predictor_num_layers = 5
    cfg.masked_latent_predictor_num_heads = 12
    cfg.target_projector_dim = -1
    cfg.masked_token_input_mode = "latent_token"
    cfg.masked_mz_sentinel = -1.0

    # MAE objective and masking
    cfg.training_mode = "mae"
    cfg.jepa_num_target_blocks = 1
    cfg.jepa_mask_strategy = ["random"]
    cfg.jepa_context_fraction = 0.60
    cfg.jepa_target_fraction = 0.25
    cfg.mae_loss_weight = 1.0
    cfg.jepa_mae_mz_bin_size = 0.5
    cfg.jepa_mae_intensity_bin_size = 0.1
    cfg.jepa_mae_intensity_max = 1.0
    cfg.mae_intensity_loss_weight = 0.0

    # Training
    cfg.num_epochs = 98
    cfg.training_max_steps = 300_000
    cfg.autocast_dtype = "bf16"
    cfg.compile_mode = "max-autotune-no-cudagraphs"
    cfg.activation_checkpoint_mode = "none"
    cfg.activation_checkpoint_every_n_layers = 1
    cfg.activation_checkpoint_modules = ("encoder", "predictor")
    cfg.activation_checkpoint_preserve_rng_state = True
    cfg.device_prefetch_size = 8
    cfg.throughput_warmup_steps = 25
    cfg.log_every_n_steps = 250
    cfg.collapse_metrics_every_n_steps = 0
    cfg.checkpoint_every_steps = 50_000
    cfg.dataloader_num_workers = 32
    cfg.dataloader_prefetch_factor = 2
    cfg.dataloader_persistent_workers = True
    cfg.dataloader_pin_memory = False
    cfg.dataloader_multiprocessing_context = "spawn"
    cfg.dataloader_output_format = "numpy"

    # Validation; MSG probe disabled
    cfg.msg_probe_every_n_steps = -1.0
    cfg.msg_probe_at_final_step = False
    cfg.val_every_n_steps = 10_000.0
    cfg.val_num_steps = 500

    # Optimizer
    cfg.optimizer = "adamw"
    cfg.optimizer_fused = True
    cfg.optimizer_state_dtype = "fp32"
    cfg.learning_rate = 3e-4
    cfg.min_learning_rate = 6e-6
    cfg.warmup_steps = 5_000
    cfg.weight_decay = 0.01
    cfg.b2 = 0.95
    cfg.grad_clip_norm = 1.0

    # Logging
    cfg.enable_wandb = True
    cfg.wandb_project = "jepa-finalrun"
    cfg.wandb_kwargs = {}
    cfg.wandb_resume_from_env = False
    cfg.wandb_resume_id = ""
    cfg.wandb_shared_label = "train"
    cfg.wandb_shared_mode = False
    cfg.wandb_shared_primary = True
    cfg.wandb_shared_update_finish_state = False
    cfg.run_name_suffix = (
        "jax-mae-massive-v2-10gb-1b-200m-v6e-singlemixer-metadata-adaln-n64-d1536-l25-h12-"
        "nomassprior-feat3072-fmlp3072-xattnrope1536-l5-h12-bs4096-"
        "ga2-mae1-adamw-cls-fp32state-lr3e-4-wd1e-1-"
        "random-ctx60-target25-mzorder-fullctx-"
        "noac-noprobe-val500x10k"
    )

    return cfg
