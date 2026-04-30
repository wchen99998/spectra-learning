from ml_collections import config_dict


aware_mixed = dict(
    tau=0.5,
    alpha=0.75,
    beta_context=0.85,
    beta_target=0.85,
    eps_context=0.04,
    eps_target=0.08,
    anchor_keep=0.60,
    min_eff_context=4.0,
    min_eff_target=1.8,
    tail_target_mix=0.10,
    local_gap_da=1.0,
    local_gap_probability=0.50,
    min_unused_mass=0.09,
)


def get_config() -> config_dict.ConfigDict:
    cfg = config_dict.ConfigDict()

    # Dataset
    cfg.artifact_dir = "data/gems_artifacts_alpha"
    cfg.gems_native_repo_id = "cjim8889/gems-a10-native"
    cfg.nist_full_probe_repo_id = "cjim8889/hr_msms_nist_probe_prepared"
    cfg.nist_full_probe_revision = "main"
    cfg.nist_full_probe_train_samples = 20_000
    cfg.nist_full_probe_val_samples = 5000
    cfg.nist_full_probe_test_samples = 5000
    cfg.nist_full_probe_num_repeats = 1
    cfg.batch_size = 256
    cfg.shuffle_buffer = 1_000_000
    cfg.drop_remainder = True
    cfg.max_precursor_mz = 1000
    cfg.min_peak_intensity = 0.0001
    cfg.peak_drop_min_intensity = 0.0001
    cfg.precursor_peak_exclusion_window_da = 5
    cfg.peak_ordering = "mz"
    cfg.seed = 66

    # Encoder
    cfg.num_peaks = 64
    cfg.model_dim = 384
    cfg.encoder_num_layers = 14
    cfg.encoder_num_heads = 12
    cfg.encoder_num_kv_heads = 12
    cfg.encoder_num_register_tokens = 2
    cfg.encoder_use_position_embedding = False
    cfg.encoder_apply_final_norm = False
    cfg.encoder_qk_norm = False
    cfg.encoder_fourier_funcs = "both"
    cfg.encoder_fourier_input_scale = 1000
    cfg.encoder_fourier_num_freqs = 256
    cfg.encoder_fourier_sigma = 10
    cfg.encoder_fourier_strategy = "log_spaced"
    cfg.encoder_fourier_trainable = False
    cfg.encoder_fourier_x_max = 1000
    cfg.encoder_fourier_x_min = 0.003
    cfg.feature_mlp_hidden_dim = 512
    cfg.attention_mlp_multiple = 4
    cfg.norm_type = "layernorm"
    cfg.use_precursor_token = True

    # Masked latent predictor
    cfg.predictor_dim = 256
    cfg.predictor_dropout = 0.15
    cfg.predictor_num_register_tokens = 2
    cfg.predictor_apply_final_norm = False
    cfg.masked_latent_predictor_num_layers = 6
    cfg.masked_latent_predictor_num_heads = 8
    cfg.temporal_predictor_num_layers = 0
    cfg.target_projector_dim = 256

    # JEPA masking and targets
    cfg.jepa_num_target_blocks = 2
    cfg.jepa_mask_strategy = "intensity_aware"
    cfg.jepa_target_layers = [5, 8, 12, 14]
    cfg.jepa_target_normalization = "zscore"
    cfg.masked_token_loss_type = "l2"
    cfg.masked_token_loss_weight = 1
    for key, value in aware_mixed.items():
        setattr(cfg, f"jepa_intensity_aware_{key}", value)

    # Regularization and pooling
    cfg.representation_regularizer = "none"
    cfg.sigreg_lambda = 0
    cfg.sigreg_num_slices = 1024
    cfg.sigreg_precursor_scale = 1
    cfg.covariance_pooling_dim = 64

    # EMA teacher
    cfg.use_ema_teacher = True
    cfg.ema_teacher_momentum_start = 0.99925
    cfg.ema_teacher_momentum_final = 0.99925
    cfg.ema_teacher_schedule = "cosine"

    # Training
    cfg.num_epochs = 30
    cfg.autocast_dtype = "bf16"
    cfg.compile_mode = "reduce-overhead"
    cfg.device_prefetch_size = 8
    cfg.log_every_n_steps = 250
    cfg.collapse_metrics_every_n_steps = 250
    cfg.checkpoint_every_steps = 25_000
    cfg.dataloader_num_workers = 8
    cfg.dataloader_prefetch_factor = 2
    cfg.dataloader_persistent_workers = True
    cfg.dataloader_pin_memory = True

    # MSG probe
    cfg.probe_dataset = "nist-full"
    cfg.msg_probe_covariance_dim = 64
    cfg.msg_probe_early_stopping = True
    cfg.msg_probe_early_stopping_min_delta = 0.0001
    cfg.msg_probe_early_stopping_min_epochs = 20
    cfg.msg_probe_early_stopping_patience = 20
    cfg.msg_probe_every_n_steps = 0.25
    cfg.msg_probe_learning_rate = 0.001
    cfg.msg_probe_max_test_samples = None
    cfg.msg_probe_max_train_samples = None
    cfg.msg_probe_max_val_samples = None
    cfg.msg_probe_mlp_hidden_dim = 256
    cfg.msg_probe_num_epochs = 100
    cfg.msg_probe_pma_num_heads = 8
    cfg.msg_probe_pma_num_seeds = 32
    cfg.msg_probe_tune_metric = "msg_probe/test/auc_maccs_mean"
    cfg.msg_probe_tune_param_space = [
        {
            "args": [0.0001, 0.0003, 0.001],
            "dist": "grid",
            "param": "msg_probe_learning_rate",
        },
        {
            "args": [0, 0.01, 0.1],
            "dist": "grid",
            "param": "msg_probe_weight_decay",
        },
    ]
    cfg.msg_probe_variants = ["covariance"]
    cfg.msg_probe_warmup_steps = 0
    cfg.msg_probe_weight_decay = 0

    # Optimizer
    cfg.learning_rate = 0.0001
    cfg.predictor_learning_rate_ratio = 2.
    cfg.min_learning_rate = 0.0001
    cfg.warmup_steps = 50_000
    cfg.weight_decay = 0.01
    cfg.b2 = 0.999
    cfg.grad_clip_norm = 1
    cfg.optimizer = "muon"
    cfg.optimizer_capturable = True
    cfg.optimizer_fused = True
    cfg.adamw_lr = None
    cfg.muon_lr = None
    cfg.muon_adjust_lr_fn = "match_rms_adamw"
    cfg.muon_momentum = 0.95
    cfg.muon_nesterov = True
    cfg.muon_ns_steps = 5
    cfg.muon_weight_decay = None

    # Logging
    cfg.enable_wandb = True
    cfg.wandb_project = "jepa-debugging"
    cfg.wandb_run_name_prefix = "jepa_masked_latent_index"
    cfg.run_name_suffix = "ema-teacher-intensity-aware-mixed"

    return cfg
