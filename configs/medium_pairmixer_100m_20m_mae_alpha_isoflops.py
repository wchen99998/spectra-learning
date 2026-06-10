from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    cfg = config_dict.ConfigDict()

    # Dataset
    cfg.artifact_dir = "data/gems_artifacts_alpha"
    cfg.gems_native_repo_id = "cjim8889/gems-a10-native"
    cfg.nist_murcko_probe_repo_id = "cjim8889/hr_msms_nist_mcebio_murcko_20260529"
    cfg.nist_murcko_probe_revision = "main"
    cfg.nist_murcko_probe_train_samples = 100_000
    cfg.nist_murcko_probe_val_samples = 20_000
    cfg.nist_murcko_probe_test_samples = 20_000
    cfg.nist_murcko_probe_num_repeats = 1
    cfg.batch_size = 512
    cfg.gradient_accumulation_steps = 4
    cfg.shuffle_buffer = 1_000_000
    cfg.drop_remainder = True
    cfg.max_precursor_mz = 1000
    cfg.min_peak_intensity = 0.0001
    cfg.peak_drop_min_intensity = 0.0001
    cfg.precursor_peak_exclusion_window_da = 0
    cfg.peak_ordering = "mz"
    cfg.seed = 66

    # Encoder
    cfg.num_peaks = 31
    cfg.model_dim = 640
    cfg.encoder_num_layers = 15
    cfg.encoder_num_heads = 10
    cfg.encoder_use_position_embedding = False
    cfg.encoder_apply_final_norm = True
    cfg.encoder_apply_final_pair_norm = True
    cfg.encoder_use_fourier_features = True
    cfg.encoder_fourier_input_scale = 1000
    cfg.encoder_fourier_mlp_hidden_dim = 1280
    cfg.encoder_fourier_mlp_num_layers = 4
    cfg.encoder_fourier_num_freqs = 64
    cfg.encoder_fourier_x_max = 1000
    cfg.encoder_fourier_x_min = 0.003
    cfg.feature_mlp_hidden_dim = 1280
    cfg.pairmixer_pair_dim = 256
    cfg.pairmixer_pair_feature_hidden_dim = 512
    cfg.pairmixer_use_pair_bias_attention = True
    cfg.pairmixer_use_fourier_features = True
    cfg.pairmixer_fourier_num_freqs = 16
    cfg.pairmixer_fourier_x_min = 0.01
    cfg.pairmixer_fourier_x_max = 1000
    cfg.pairmixer_relative_fourier_x_min = 0.001
    cfg.pairmixer_relative_fourier_x_max = 1.0
    cfg.attention_mlp_multiple = 4

    # Masked latent predictor
    cfg.predictor_dim = 640
    cfg.predictor_dropout = 0.1
    cfg.predictor_apply_final_norm = True
    cfg.predictor_use_rope = False
    cfg.mae_context_encoder_pack_tokens = 28
    cfg.masked_latent_predictor_num_layers = 3
    cfg.masked_latent_predictor_num_heads = 10
    cfg.target_projector_dim = -1
    cfg.masked_token_input_mode = "latent_token"

    # MAE masking and reconstruction objectives
    cfg.training_mode = "mae"
    cfg.use_ema_teacher = False
    cfg.jepa_num_target_blocks = 1
    cfg.jepa_mask_strategy = ["intensity_aware", "ragged"]
    cfg.jepa_context_fraction = 0.35
    cfg.jepa_target_fraction = 0.25
    cfg.jepa_block_min_len = 1
    cfg.jepa_mask_lengths = (2, 4, 8, 12)
    cfg.jepa_mask_round_from = 3
    cfg.jepa_target_normalization = "none"
    cfg.masked_token_loss_weight = 0.0
    cfg.jepa_mae_loss_weight = 0.0
    cfg.mae_loss_weight = 1.0
    cfg.distogram_loss_weight = 1.0
    cfg.latent_pair_loss_weight = 0.0
    cfg.latent_pair_target_normalization = "none"
    cfg.jepa_mae_mz_bin_size = 0.5
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

    # Pooling
    cfg.covariance_pooling_dim = 64
    cfg.train_covariance_pooling = False

    # Contrastive NIST Murcko training
    cfg.contrastive_temperature = 0.1
    cfg.contrastive_loss_weight = 1.0
    cfg.contrastive_triplet_hard_fraction = None
    cfg.online_probe_loss_weight = 1.0
    cfg.contrastive_batch_size = 128
    cfg.contrastive_covariance_dim = 64
    cfg.contrastive_projection_dim = 256
    cfg.contrastive_projection_hidden_dim = 1024
    cfg.contrastive_online_probe_hidden_dim = 1024
    cfg.contrastive_pairs_per_epoch = 30_000
    cfg.contrastive_val_pairs_per_epoch = 10_000
    cfg.contrastive_val_every_n_steps = 1_000
    cfg.contrastive_single_pair_include_diagonal = False
    cfg.contrastive_compile_mode = "none"

    # EMA teacher
    cfg.ema_teacher_momentum_start = 0.996
    cfg.ema_teacher_momentum_final = 0.99925
    cfg.ema_teacher_schedule = "cosine"

    # Training
    cfg.num_epochs = 8
    cfg.training_max_steps = 300_000
    cfg.autocast_dtype = "bf16"
    cfg.compile_mode = "reduce-overhead"
    cfg.activation_checkpoint_mode = "none"
    cfg.activation_checkpoint_every_n_layers = 1
    cfg.activation_checkpoint_modules = ("encoder", "predictor")
    cfg.activation_checkpoint_preserve_rng_state = True
    cfg.jax_scan_accumulation = True
    cfg.jax_pure_optax_step = True
    cfg.jax_scan_zero_init = True
    cfg.device_prefetch_size = 8
    cfg.throughput_warmup_steps = 25
    cfg.torchax_mesh_devices = 1
    cfg.torchax_mesh_axis = "data"
    cfg.torchax_distributed_initialize = False
    cfg.torchax_coordinator_address = ""
    cfg.torchax_cluster_detection_method = ""
    cfg.torchax_initialization_timeout = 300
    cfg.torchax_lazy_opt_state = True
    cfg.torchax_offload_opt_state_during_accumulation = True
    cfg.log_every_n_steps = 250
    cfg.collapse_metrics_every_n_steps = 0
    cfg.checkpoint_every_steps = 15_000
    cfg.dataloader_num_workers = 8
    cfg.dataloader_prefetch_factor = 2
    cfg.dataloader_persistent_workers = True
    cfg.dataloader_pin_memory = True

    # MSG probe
    cfg.probe_dataset = "nist-murcko"
    cfg.msg_probe_early_stopping = True
    cfg.msg_probe_early_stopping_min_delta = 0.0001
    cfg.msg_probe_early_stopping_min_epochs = 20
    cfg.msg_probe_early_stopping_patience = 5
    cfg.msg_probe_every_n_steps = 50_000.
    cfg.msg_probe_learning_rate = 0.0003
    cfg.msg_probe_max_test_samples = None
    cfg.msg_probe_max_train_samples = None
    cfg.msg_probe_max_val_samples = None
    cfg.msg_probe_mlp_hidden_dim = 256
    cfg.msg_probe_num_epochs = 100
    cfg.msg_probe_pma_num_heads = 8
    cfg.msg_probe_pma_num_seeds = 32
    cfg.msg_probe_tune_metric = "msg_probe/test/auc_maccs_mean"
    cfg.msg_probe_variants = ["single_pair_covariance"]
    cfg.msg_probe_warmup_epochs = 0.5
    cfg.msg_probe_grad_clip_norm = 1.0
    cfg.msg_probe_warmup_steps = 0
    cfg.msg_probe_weight_decay = 0
    cfg.msg_probe_batch_size = 512

    # Optimizer
    cfg.learning_rate = 4e-04
    cfg.min_learning_rate = 4e-05
    cfg.warmup_steps = 3_000
    cfg.weight_decay = 0.05
    cfg.b2 = 0.95
    cfg.grad_clip_norm = 0.
    cfg.optimizer = "adamw"
    cfg.optimizer_fused = True
    cfg.adamw_lr = None

    # Logging
    cfg.enable_wandb = True
    cfg.wandb_project = "jepa-debugging"
    cfg.run_name_suffix = "mae-100m-20m-alpha-isoflops-100k-bs1024-ga2"

    return cfg
