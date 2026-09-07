"""SJEPA run using the one-billion-parameter config's encoder settings.

Each spectrum produces two global views retaining 75% of its peaks and three
local views retaining 20%. The mean global CLS is a stop-gradient teacher target
for global-to-local CLS MSE, with per-step resampled SIGReg weighted at 0.1.
"""

from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    cfg = config_dict.ConfigDict()

    # Runtime
    cfg.device = "auto"
    cfg.device_backend = "jax"
    cfg.jax_checkpoint_max_to_keep = 5
    cfg.jax_compilation_cache_dir = ""
    cfg.jax_compile_stall_threshold_seconds = 0.0
    cfg.jax_distributed_initialize = False
    cfg.jax_enable_async_checkpointing = True
    cfg.jax_enable_compilation_cache = True
    cfg.jax_explain_cache_misses = False
    cfg.jax_grad_all_reduce_dtype = "bfloat16"
    cfg.jax_log_compiles = False
    cfg.jax_log_update_stats = False
    cfg.jax_mesh_devices = "32"
    cfg.jax_persistent_cache_min_compile_time_secs = None
    cfg.jax_persistent_cache_min_entry_size_bytes = None
    cfg.jax_profile_dir = ""
    cfg.jax_profile_start_step = 5_000
    cfg.jax_profile_steps = 0
    cfg.jax_shard_optimizer = False
    cfg.jax_timing_barriers = False
    cfg.max_duration_hours = 95.0
    cfg.training_flops_per_optimizer_step = None
    cfg.training_flops_per_parameter = 6.0
    cfg.training_flops_per_sample = None

    # Dataset
    cfg.training_task = "sjepa"
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
    # The 65 training shards split across 8 hosts as 9/8/8/8/8/8/8/8.
    cfg.gems_hdf5_shuffle_seed = 67
    cfg.nist_murcko_probe_num_repeats = 1
    cfg.batch_size = 2_048
    cfg.gradient_accumulation_steps = 2
    cfg.drop_remainder = True
    cfg.max_precursor_mz = 1_000
    cfg.min_peak_intensity = 0.0001
    cfg.peak_drop_min_intensity = 0.0001
    cfg.precursor_peak_exclusion_window_da = 0
    cfg.peak_ordering = "mz"
    cfg.seed = 66

    # Single-stream encoder
    cfg.num_peaks = 127
    cfg.model_dim = 1_536
    cfg.encoder_num_layers = 25
    cfg.encoder_num_heads = 6
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
    cfg.pairmixer_transition_type = "swiglu"
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
    cfg.attention_mlp_multiple = 8 / 3
    cfg.norm_eps = 1e-5

    # Unused legacy predictor settings retained from the copied model config.
    cfg.predictor_dim = 1_536
    cfg.predictor_mlp_multiple = 8 / 3
    cfg.predictor_dropout = 0.0
    cfg.predictor_apply_final_norm = True
    cfg.masked_latent_predictor_num_layers = 5
    cfg.masked_latent_predictor_num_heads = 6
    cfg.target_projector_dim = -1
    cfg.masked_token_input_mode = "latent_token"
    cfg.masked_mz_sentinel = -1.0

    # SJEPA objective and views
    cfg.training_mode = "sjepa"
    cfg.sjepa_num_global_views = 2
    cfg.sjepa_num_local_views = 3
    cfg.sjepa_global_view_fraction = 0.75
    cfg.sjepa_local_view_fraction = 0.20
    cfg.sjepa_invariance_loss_weight = 1.0
    cfg.sjepa_teacher_stop_gradient = True
    cfg.sjepa_sigreg_loss_weight = 0.1
    cfg.sjepa_sigreg_num_slices = 256
    cfg.sjepa_sigreg_gather_embeddings = True

    # Legacy JEPA/MAE fields used by shared data and model settings.
    cfg.use_ema_teacher = False
    cfg.jepa_num_target_blocks = 0
    cfg.jepa_mask_strategy = ["random"]
    cfg.jepa_context_fraction = 0.60
    cfg.jepa_target_fraction = 0.25
    cfg.jepa_block_min_len = 1
    cfg.jepa_mask_lengths = (2, 4, 8, 12)
    cfg.jepa_mask_round_from = 3
    cfg.jepa_target_normalization = "none"
    cfg.masked_token_loss_weight = 0.0
    cfg.jepa_mae_loss_weight = 0.0
    cfg.mae_loss_weight = 0.0
    cfg.distogram_loss_weight = 0.0
    cfg.latent_pair_loss_weight = 0.0
    cfg.latent_pair_target_normalization = "none"
    cfg.jepa_mae_mz_bin_size = 0.5
    cfg.jepa_mae_intensity_bin_size = 0.1
    cfg.jepa_mae_intensity_max = 1.0
    cfg.mae_mz_multiscale_widths = (2, 4, 8, 16, 32, 64)
    cfg.mae_mz_multiscale_weight = 1.0
    cfg.mae_intensity_loss_weight = 0.0
    cfg.jepa_allow_target_overlap = False
    cfg.jepa_intensity_aware_tau = 0.5
    cfg.jepa_intensity_aware_alpha = 0.75
    cfg.jepa_intensity_aware_beta_context = 0.85
    cfg.jepa_intensity_aware_beta_target = 0.85
    cfg.jepa_intensity_aware_eps_context = 0.04
    cfg.jepa_intensity_aware_eps_target = 0.08
    cfg.jepa_intensity_aware_anchor_keep = 0.60
    cfg.jepa_intensity_aware_min_eff_context = 4.0
    cfg.jepa_intensity_aware_min_eff_target = 1.8
    cfg.jepa_intensity_aware_tail_target_mix = 0.10
    cfg.jepa_intensity_aware_local_gap_da = 1.0
    cfg.jepa_intensity_aware_local_gap_probability = 0.50
    cfg.jepa_intensity_aware_min_unused_mass = 0.09

    # Pooling and contrastive defaults retained for config compatibility.
    cfg.covariance_pooling_dim = 64
    cfg.contrastive_temperature = 0.1
    cfg.contrastive_loss_weight = 1.0
    cfg.contrastive_triplet_hard_fraction = None
    cfg.online_probe_loss_weight = 1.0
    cfg.contrastive_batch_size = 128
    cfg.contrastive_covariance_dim = 64
    cfg.contrastive_online_probe_hidden_dim = 1_024
    cfg.contrastive_pairs_per_epoch = 30_000
    cfg.contrastive_val_pairs_per_epoch = 10_000
    cfg.contrastive_val_every_n_steps = 1_000
    cfg.contrastive_single_pair_include_diagonal = False
    cfg.contrastive_compile_mode = "none"

    # Teacher defaults are inactive in SJEPA.
    cfg.ema_teacher_momentum_start = 0.996
    cfg.ema_teacher_momentum_mid = None
    cfg.ema_teacher_momentum_final = 0.99925
    cfg.ema_teacher_schedule_peak_fraction = 0.35
    cfg.ema_teacher_schedule = "cosine"
    cfg.frozen_teacher_config_path = None
    cfg.frozen_teacher_checkpoint_path = None

    # Training
    cfg.num_epochs = 98
    cfg.training_max_steps = 2_000_000
    cfg.autocast_dtype = "bf16"
    cfg.compile_mode = "max-autotune-no-cudagraphs"
    cfg.activation_checkpoint_mode = "none"
    cfg.activation_checkpoint_every_n_layers = 1
    cfg.activation_checkpoint_modules = ("encoder", "predictor")
    cfg.activation_checkpoint_preserve_rng_state = True
    cfg.throughput_warmup_steps = 25
    cfg.log_every_n_steps = 250
    cfg.collapse_metrics_every_n_steps = 0
    cfg.checkpoint_every_steps = 50_000
    cfg.dataloader_num_workers = 4
    cfg.dataloader_prefetch_factor = 2
    cfg.dataloader_persistent_workers = True
    cfg.dataloader_pin_memory = False
    cfg.dataloader_multiprocessing_context = "spawn"
    cfg.dataloader_output_format = "numpy"

    # MSG probe
    cfg.msg_probe_early_stopping = True
    cfg.msg_probe_early_stopping_min_delta = 0.0001
    cfg.msg_probe_early_stopping_min_epochs = 20
    cfg.msg_probe_early_stopping_patience = 5
    cfg.msg_probe_every_n_steps = -1.0
    cfg.msg_probe_at_final_step = False
    cfg.val_every_n_steps = 10_000.0
    cfg.val_num_steps = 500
    cfg.msg_probe_learning_rate = 0.0003
    cfg.msg_probe_mlp_hidden_dim = 256
    cfg.msg_probe_num_epochs = 100
    cfg.msg_probe_pma_num_heads = 8
    cfg.msg_probe_pma_num_seeds = 32
    cfg.msg_probe_select_metric = "msg_probe/val/auc_fluorine"
    cfg.msg_probe_variants = ["single_pair_covariance"]
    cfg.msg_probe_warmup_epochs = 0.5
    cfg.msg_probe_grad_clip_norm = 1.0
    cfg.msg_probe_warmup_steps = 0
    cfg.msg_probe_weight_decay = 0
    cfg.msg_probe_batch_size = 512
    cfg.msg_probe_fingerprint = "maccs"
    cfg.msg_probe_num_repeats = None
    cfg.msg_probe_pairwise_alignment_num_pairs = 0

    # Pure Adam: no decoupled or coupled weight decay.
    cfg.optimizer = "adam"
    cfg.optimizer_fused = True
    cfg.optimizer_state_dtype = "fp32"
    cfg.learning_rate = 3e-4
    cfg.min_learning_rate = 3e-5
    cfg.warmup_steps = 20_000
    cfg.weight_decay = 0.0
    cfg.b1 = 0.9
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
        "jax-sjepa-massive-v2-10gb-1b-v6e-singlemixer-metadata-adaln-"
        "n128-d1536-l25-h6-hd256-swiglu-bs2048-ga2-v2g75-v3l20-"
        "cls-mse-stopgrad-sigreg256x1e-1-adam-fp32state-lr3e-4-wu20k-wd0-2m"
    )

    return cfg
