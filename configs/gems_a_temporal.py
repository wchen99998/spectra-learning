from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    cfg = config_dict.ConfigDict()

    # Dataset — temporal experiment-grouped pipeline
    cfg.temporal_repo_id = "cjim8889/gems-a10-grouped"
    cfg.temporal_revision = "main"
    cfg.temporal_data_dir = "data/gems_grouped"
    cfg.nist_full_probe_repo_id = "cjim8889/hr_msms_nist_probe_prepared"
    cfg.nist_full_probe_revision = "main"
    cfg.nist_full_probe_train_samples = 4_000
    cfg.nist_full_probe_val_samples = 1_000
    cfg.nist_full_probe_test_samples = 1_000
    cfg.nist_full_probe_num_repeats = 3
    cfg.batch_size = 256
    cfg.max_precursor_mz = 1000.0
    cfg.min_peak_intensity = 0.0001
    cfg.peak_ordering = "mz"
    cfg.seed = 42

    # Model
    cfg.num_peaks = 64
    cfg.model_dim = 512
    cfg.encoder_num_layers = 12
    cfg.encoder_num_heads = 8
    cfg.encoder_num_kv_heads = 8
    cfg.encoder_use_cls_token = True
    cfg.encoder_num_register_tokens = 0
    cfg.predictor_num_register_tokens = 0
    cfg.encoder_use_position_embedding = True
    cfg.encoder_qk_norm = False
    cfg.encoder_fourier_strategy = "log_spaced"
    cfg.encoder_fourier_x_min = 3e-3
    cfg.encoder_fourier_x_max = 1000.0
    cfg.encoder_fourier_funcs = "both"
    cfg.encoder_fourier_num_freqs = 256
    cfg.encoder_fourier_sigma = 10.0
    cfg.encoder_fourier_trainable = False
    cfg.encoder_fourier_input_scale = 1000.0
    cfg.spectral_bias_relative_kind = "harmonic"
    cfg.spectral_bias_use_precursor = True
    cfg.spectral_bias_use_intensity = False
    cfg.spectral_bias_num_freqs = 128
    cfg.spectral_bias_fourier_strategy = cfg.encoder_fourier_strategy
    cfg.spectral_bias_fourier_x_min = cfg.encoder_fourier_x_min
    cfg.spectral_bias_fourier_x_max = cfg.encoder_fourier_x_max
    cfg.spectral_bias_fourier_sigma = cfg.encoder_fourier_sigma
    cfg.spectral_bias_fourier_trainable = False
    cfg.spectral_bias_mass_scale = cfg.encoder_fourier_input_scale
    cfg.spectral_bias_precursor_scale = cfg.max_precursor_mz
    cfg.spectral_bias_rbf_num_basis = 64
    cfg.spectral_bias_rbf_delta_min = -cfg.max_precursor_mz
    cfg.spectral_bias_rbf_delta_max = cfg.max_precursor_mz
    cfg.spectral_bias_rbf_use_absolute_delta = False
    cfg.spectral_bias_intensity_hidden_dim = 16
    cfg.spectral_bias_init_std = 0.0
    cfg.spectral_bias_clip = None
    cfg.attention_mlp_multiple = 4.0
    cfg.feature_mlp_hidden_dim = 128
    cfg.encoder_fourier_mlp_hidden_dim = 1024
    cfg.encoder_fourier_mlp_num_layers = 4
    cfg.sigreg_num_slices = 256
    cfg.sigreg_lambda = 0.02
    cfg.jepa_num_target_blocks = 2
    cfg.jepa_context_fraction = 0.5
    cfg.jepa_context_fraction_range = (0.5, 0.5)
    cfg.jepa_target_fraction = 0.25
    cfg.jepa_target_fraction_range = (0.25, 0.25)
    cfg.jepa_block_min_len = 1
    cfg.jepa_mask_strategy = "contiguous"
    cfg.norm_type = "layernorm"

    # Training
    cfg.num_epochs = 100
    cfg.learning_rate = 3e-4
    cfg.warmup_steps = 10_000
    cfg.min_learning_rate = None
    cfg.b2 = 0.98
    cfg.weight_decay = 1e-4
    cfg.device_prefetch_size = 8
    cfg.optimizer_capturable = True
    cfg.optimizer_fused = True
    cfg.log_every_n_steps = 100
    cfg.collapse_metrics_every_n_steps = 100
    cfg.checkpoint_every_steps = 25_000

    # DataLoader — native PyTorch, multiple workers OK (no TF)
    cfg.dataloader_num_workers = 8
    cfg.dataloader_persistent_workers = True
    cfg.dataloader_prefetch_factor = 2
    cfg.dataloader_pin_memory = True

    cfg.masked_token_loss_weight = 1.0
    cfg.jepa_mae_loss_weight = 1.0
    cfg.jepa_mae_mz_bin_size = 2.5
    cfg.jepa_mae_intensity_bin_size = 0.1
    cfg.jepa_mae_mz_max = 1000.0
    cfg.jepa_mae_intensity_max = 1.0
    cfg.jepa_target_normalization = "none"
    cfg.grad_clip_norm = 1.0
    cfg.masked_latent_predictor_num_layers = 4
    cfg.autocast_dtype = "bf16"
    cfg.compile_mode = "max-autotune"
    cfg.representation_regularizer = "none"
    cfg.train_covariance_pooling = True
    cfg.covariance_pooling_dim = 32
    cfg.msg_probe_every_n_steps = 0.25
    cfg.msg_probe_num_epochs = 20
    cfg.msg_probe_learning_rate = 3e-4
    cfg.msg_probe_weight_decay = 0.0
    cfg.msg_probe_warmup_steps = 0
    cfg.msg_probe_sample_size = None
    cfg.msg_probe_max_train_samples = None
    cfg.msg_probe_max_val_samples = None
    cfg.msg_probe_max_test_samples = None
    cfg.msg_probe_variants = ("mean", "covariance", "pma")
    cfg.msg_probe_mlp_hidden_dim = cfg.model_dim
    cfg.msg_probe_covariance_dim = cfg.covariance_pooling_dim
    cfg.msg_probe_pma_num_seeds = 4
    cfg.msg_probe_pma_num_heads = cfg.encoder_num_heads
    cfg.probe_dataset = "nist20"
    cfg.use_precursor_token = False

    # Temporal predictor
    cfg.temporal_predictor_num_layers = 4
    cfg.encoder_learning_rate = 3e-5
    cfg.pretrained_checkpoint = None  # set via CLI

    # Tune search space
    cfg.tune_param_space = [
    ]

    # System / logging
    cfg.enable_wandb = True
    cfg.wandb_project = "jepa-pretraining"
    cfg.wandb_run_name_prefix = "jepa_temporal"

    return cfg
