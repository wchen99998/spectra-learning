"""Standalone 290.4M pair-only JAX run: width 640, 25 encoder + 5 predictor blocks.

All experiment and runtime values are specified here, without config inheritance.
The current model has 290,429,168 trainable parameters. Context peaks alone enter
its encoder; masked peaks enter as predictor queries. Encoder pair inputs and
predictor queries use 2D rotary position encoding at the original peak positions.

Initial phase: 31 peaks plus CLS; 20 encoder and 28 predictor positions,
with up to eight target peaks.
Global batch 2048 = 16 spectra/device/microstep x 32 v6e devices x GA4.
Adam uses bf16 moments, metadata uses AdaLN-Zero, and distance supervision uses
2,000 bins of width 0.5 Da plus six coarse marginal cross-entropy scales.
The schedule targets 1M optimizer steps with a 48-hour duration limit.
"""

from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    cfg = config_dict.ConfigDict()

    # Training task and pair-only architecture
    cfg.training_task = 'pretrain'
    cfg.training_mode = 'pair_distogram'
    cfg.model_dim = 640
    cfg.pairmixer_pair_dim = 640
    cfg.encoder_num_layers = 25
    cfg.predictor_dim = 640
    cfg.masked_latent_predictor_num_layers = 5
    cfg.num_peaks = 31
    cfg.encoder_use_cls_token = True
    cfg.encoder_apply_final_pair_norm = True
    cfg.predictor_apply_final_norm = True
    cfg.pairmixer_block_type = 'fastmixer'
    cfg.pairmixer_transition_type = 'swiglu'
    cfg.attention_mlp_multiple = 8 / 3
    cfg.predictor_mlp_multiple = 8 / 3
    cfg.norm_eps = 1e-05
    cfg.masked_token_input_mode = 'latent_token'
    cfg.encoder_use_position_embedding = True

    # Pair features and metadata conditioning
    cfg.encoder_metadata_condition_dim = 256
    cfg.encoder_metadata_conditioning = 'adaln_zero'
    cfg.encoder_metadata_schema = 'massive_v2_acquisition_v1'
    cfg.pairmixer_dropout = 0.0
    cfg.pairmixer_fourier_num_freqs = 16
    cfg.pairmixer_fourier_x_min = 0.01
    cfg.pairmixer_mz_embedding = 'fourier'
    cfg.pairmixer_mz_token_bin_size = 0.1
    cfg.pairmixer_mz_token_embedding_dim = 128
    cfg.pairmixer_pair_feature_hidden_dim = 1280
    cfg.pairmixer_relative_fourier_x_max = 1.0
    cfg.pairmixer_relative_fourier_x_min = 0.001
    cfg.pairmixer_use_pair_bias = True
    cfg.spectrum_metadata_dropout_probability = 0.1

    # Masking and distance supervision
    cfg.distogram_loss_weight = 1.0
    cfg.distogram_multiscale_weight = 1.0
    cfg.distogram_multiscale_widths = (2, 4, 8, 16, 32, 64)
    cfg.jepa_allow_target_overlap = False
    cfg.jepa_block_min_len = 1
    cfg.jepa_context_fraction = 0.6
    cfg.jepa_intensity_aware_alpha = 0.75
    cfg.jepa_intensity_aware_anchor_keep = 0.6
    cfg.jepa_intensity_aware_beta_context = 0.85
    cfg.jepa_intensity_aware_beta_target = 0.85
    cfg.jepa_intensity_aware_eps_context = 0.04
    cfg.jepa_intensity_aware_eps_target = 0.08
    cfg.jepa_intensity_aware_local_gap_da = 1.0
    cfg.jepa_intensity_aware_local_gap_probability = 0.5
    cfg.jepa_intensity_aware_min_eff_context = 4.0
    cfg.jepa_intensity_aware_min_eff_target = 1.8
    cfg.jepa_intensity_aware_min_unused_mass = 0.09
    cfg.jepa_intensity_aware_tail_target_mix = 0.1
    cfg.jepa_intensity_aware_tau = 0.5
    cfg.jepa_mae_intensity_bin_size = 0.1
    cfg.jepa_mae_intensity_max = 1.0
    cfg.jepa_mae_loss_weight = 0.0
    cfg.jepa_mae_mz_bin_size = 0.5
    cfg.jepa_mask_lengths = (2, 4, 8, 12)
    cfg.jepa_mask_round_from = 3
    cfg.jepa_mask_strategy = ['random']
    cfg.jepa_num_target_blocks = 1
    cfg.jepa_target_fraction = 0.25
    cfg.jepa_target_normalization = 'none'
    cfg.latent_pair_loss_weight = 0.0
    cfg.latent_pair_target_normalization = 'none'
    cfg.mae_intensity_loss_weight = 0.0
    cfg.mae_loss_weight = 0.0
    cfg.mae_mz_multiscale_weight = 0.0
    cfg.mae_mz_multiscale_widths = ()
    cfg.masked_token_loss_weight = 0.0
    cfg.distogram_pair_dim = None

    # Dataset and input pipeline
    cfg.artifact_dir = 'data/massive_v2_ms2_t095_l080_sharded_10gb'
    cfg.gems_hdf5_manifest = 'manifest.json'
    cfg.gems_hdf5_ms_level_dataset = 'MS level'
    cfg.gems_hdf5_precursor_dataset = 'precursor_mz'
    cfg.gems_hdf5_repo_id = 'novogaia/massive-v2-ms2-t095-l080-sharded-10gb'
    cfg.gems_hdf5_retention_time_dataset = 'RT'
    cfg.gems_hdf5_revision = 'b098155fccb7335b752f2690bf16d2479d69bf5b'
    cfg.gems_hdf5_rows_per_block = 0
    cfg.gems_hdf5_shuffle_seed = 67
    cfg.gems_hdf5_shuffle_window_blocks = 2048
    cfg.gems_hdf5_spectrum_dataset = 'spectrum'
    cfg.seed = 66
    cfg.batch_size = 2048
    cfg.drop_remainder = True
    cfg.max_precursor_mz = 1000
    cfg.min_peak_intensity = 0.0001
    cfg.peak_drop_min_intensity = 0.0001
    cfg.precursor_peak_exclusion_window_da = 0
    cfg.peak_ordering = 'mz'
    cfg.dataloader_multiprocessing_context = 'spawn'
    cfg.dataloader_num_workers = 4
    cfg.dataloader_output_format = 'numpy'
    cfg.dataloader_persistent_workers = True
    cfg.dataloader_pin_memory = False
    cfg.dataloader_prefetch_factor = 2

    # Optimizer and training schedule
    cfg.optimizer = 'adam'
    cfg.optimizer_fused = True
    cfg.optimizer_state_dtype = 'bf16'
    cfg.b1 = 0.9
    cfg.b2 = 0.95
    cfg.learning_rate = 0.0006
    cfg.min_learning_rate = 6e-05
    cfg.warmup_steps = 20000
    cfg.weight_decay = 0.0
    cfg.grad_clip_norm = 1.0
    cfg.gradient_accumulation_steps = 4
    cfg.num_epochs = 98
    cfg.training_max_steps = 1000000
    cfg.max_duration_hours = 48.0
    cfg.autocast_dtype = 'bf16'

    # JAX runtime and activation checkpointing
    cfg.device = 'auto'
    cfg.device_backend = 'jax'
    cfg.jax_checkpoint_max_to_keep = 5
    cfg.jax_compilation_cache_dir = ''
    cfg.jax_compile_stall_threshold_seconds = 0.0
    cfg.jax_distributed_initialize = True
    cfg.jax_enable_async_checkpointing = True
    cfg.jax_enable_compilation_cache = True
    cfg.jax_explain_cache_misses = False
    cfg.jax_grad_all_reduce_dtype = 'bfloat16'
    cfg.jax_log_compiles = False
    cfg.jax_log_update_stats = False
    cfg.jax_mesh_devices = '32'
    cfg.jax_persistent_cache_min_compile_time_secs = None
    cfg.jax_persistent_cache_min_entry_size_bytes = None
    cfg.jax_profile_dir = ''
    cfg.jax_profile_start_step = 5000
    cfg.jax_profile_steps = 0
    cfg.jax_shard_optimizer = False
    cfg.jax_timing_barriers = False
    cfg.jax_zero1_optimizer = False
    cfg.compile_mode = 'max-autotune-no-cudagraphs'
    cfg.activation_checkpoint_every_n_layers = 1
    cfg.activation_checkpoint_mode = 'selective'
    cfg.activation_checkpoint_modules = ('encoder', 'predictor', 'distogram')
    cfg.activation_checkpoint_preserve_rng_state = True

    # Validation, checkpoint cadence, and logging
    cfg.val_every_n_steps = 10000.0
    cfg.val_num_steps = 500
    cfg.checkpoint_every_steps = 10000
    cfg.log_every_n_steps = 25
    cfg.collapse_metrics_every_n_steps = 0
    cfg.throughput_warmup_steps = 25
    cfg.training_flops_per_optimizer_step = None
    cfg.training_flops_per_parameter = 6.0
    cfg.training_flops_per_sample = None
    cfg.enable_wandb = True
    cfg.wandb_kwargs = {}
    cfg.wandb_project = 'jepa-finalrun'
    cfg.wandb_resume_from_env = False
    cfg.wandb_resume_id = ''
    cfg.wandb_shared_label = 'train'
    cfg.wandb_shared_mode = False
    cfg.wandb_shared_primary = True
    cfg.wandb_shared_update_finish_state = False
    cfg.run_name_suffix = (
        "-paironly-d640-l25-pairpred5-n32-pairdisto-multiscale-"
        "rope2d-metadata-adaln-adam-bf16state-bs2048-ga4-v6e32-1m-2d-selective"
    )

    # Inactive single-stream, teacher, contrastive, and probe settings
    cfg.contrastive_batch_size = 128
    cfg.contrastive_compile_mode = 'none'
    cfg.contrastive_covariance_dim = 64
    cfg.contrastive_loss_weight = 1.0
    cfg.contrastive_online_probe_hidden_dim = 1024
    cfg.contrastive_pairs_per_epoch = 30000
    cfg.contrastive_single_pair_include_diagonal = False
    cfg.contrastive_temperature = 0.1
    cfg.contrastive_triplet_hard_fraction = None
    cfg.contrastive_val_every_n_steps = 1000
    cfg.contrastive_val_pairs_per_epoch = 10000
    cfg.covariance_pooling_dim = 64
    cfg.ema_teacher_momentum_final = 0.99925
    cfg.ema_teacher_momentum_mid = None
    cfg.ema_teacher_momentum_start = 0.996
    cfg.ema_teacher_schedule = 'cosine'
    cfg.ema_teacher_schedule_peak_fraction = 0.35
    cfg.encoder_apply_final_norm = True
    cfg.encoder_fourier_mlp_hidden_dim = 4096
    cfg.encoder_fourier_mlp_num_layers = 4
    cfg.encoder_fourier_num_freqs = 64
    cfg.encoder_fourier_x_max = 1000
    cfg.encoder_fourier_x_min = 0.003
    cfg.encoder_mz_embedding = 'fourier'
    cfg.encoder_mz_token_bin_size = 0.02
    cfg.encoder_mz_token_embedding_dim = 77
    cfg.encoder_num_heads = 8
    cfg.feature_mlp_hidden_dim = 4096
    cfg.frozen_teacher_checkpoint_path = None
    cfg.frozen_teacher_config_path = None
    cfg.masked_latent_predictor_num_heads = 8
    cfg.masked_mz_sentinel = -1.0
    cfg.msg_probe_at_final_step = False
    cfg.msg_probe_batch_size = 512
    cfg.msg_probe_early_stopping = True
    cfg.msg_probe_early_stopping_min_delta = 0.0001
    cfg.msg_probe_early_stopping_min_epochs = 20
    cfg.msg_probe_early_stopping_patience = 5
    cfg.msg_probe_every_n_steps = -1.0
    cfg.msg_probe_fingerprint = 'maccs'
    cfg.msg_probe_grad_clip_norm = 1.0
    cfg.msg_probe_learning_rate = 0.0003
    cfg.msg_probe_mlp_hidden_dim = 256
    cfg.msg_probe_num_epochs = 100
    cfg.msg_probe_num_repeats = None
    cfg.msg_probe_pairwise_alignment_num_pairs = 0
    cfg.msg_probe_pma_num_heads = 8
    cfg.msg_probe_pma_num_seeds = 32
    cfg.msg_probe_select_metric = 'msg_probe/val/auc_fluorine'
    cfg.msg_probe_variants = ['single_pair_covariance']
    cfg.msg_probe_warmup_epochs = 0.5
    cfg.msg_probe_warmup_steps = 0
    cfg.msg_probe_weight_decay = 0
    cfg.nist_murcko_probe_num_repeats = 1
    cfg.online_probe_loss_weight = 1.0
    cfg.predictor_dropout = 0.0
    cfg.target_projector_dim = -1
    cfg.use_ema_teacher = False

    return cfg
