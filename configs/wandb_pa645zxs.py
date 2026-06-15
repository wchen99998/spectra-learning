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
    cfg.gems_native_repo_id = "cjim8889/gems_native_20260615"
    cfg.gems_native_hf_subdir = "gems_a10_native"
    cfg.nist_murcko_probe_repo_id = "cjim8889/msms_evaluation_100ktrain_20260615"
    cfg.nist_murcko_probe_revision = "main"
    cfg.nist_murcko_probe_num_repeats = 1
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
    cfg.encoder_use_position_embedding = False
    cfg.encoder_apply_final_norm = False
    cfg.encoder_fourier_input_scale = 1000
    cfg.encoder_fourier_num_freqs = 256
    cfg.encoder_fourier_x_max = 1000
    cfg.encoder_fourier_x_min = 0.003
    cfg.feature_mlp_hidden_dim = 1024
    cfg.pairmixer_pair_dim = 384
    cfg.pairmixer_pair_feature_hidden_dim = 512
    cfg.attention_mlp_multiple = 4

    # Masked latent predictor
    cfg.predictor_dim = 256
    cfg.predictor_dropout = 0.1
    cfg.predictor_apply_final_norm = False
    cfg.masked_latent_predictor_num_layers = 6
    cfg.masked_latent_predictor_num_heads = 8
    cfg.target_projector_dim = -1

    # JEPA masking and targets
    cfg.jepa_num_target_blocks = 3
    cfg.jepa_mask_strategy = "intensity_aware"
    cfg.jepa_target_layers = [5, 8, 12, 14]
    cfg.jepa_target_normalization = "none"
    cfg.masked_token_loss_weight = 1
    for key, value in aware_mixed.items():
        setattr(cfg, f"jepa_intensity_aware_{key}", value)

    # Pooling
    cfg.covariance_pooling_dim = 64

    # EMA teacher
    cfg.use_ema_teacher = True
    cfg.ema_teacher_momentum_start = 0.996
    cfg.ema_teacher_momentum_final = 0.99925
    cfg.ema_teacher_schedule = "cosine"

    # Training
    cfg.num_epochs = 50
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
    cfg.probe_dataset = "nist-murcko"
    cfg.covariance_pooling_dim = 64
    cfg.train_covariance_pooling = False
    cfg.msg_probe_early_stopping = True
    cfg.msg_probe_early_stopping_min_delta = 0.0001
    cfg.msg_probe_early_stopping_min_epochs = 20
    cfg.msg_probe_early_stopping_patience = 20
    cfg.msg_probe_every_n_steps = 1.0
    cfg.msg_probe_learning_rate = 0.001
    cfg.msg_probe_mlp_hidden_dim = 256
    cfg.msg_probe_num_epochs = 100
    cfg.msg_probe_pma_num_heads = 8
    cfg.msg_probe_pma_num_seeds = 32
    cfg.msg_probe_select_metric = "msg_probe/test/auc_fluorine"
    cfg.msg_probe_variants = ["covariance"]
    cfg.msg_probe_warmup_steps = 0
    cfg.msg_probe_weight_decay = 0

    # Optimizer
    cfg.learning_rate = 0.0003
    cfg.min_learning_rate = 0.0001
    cfg.warmup_steps = 50_000
    cfg.weight_decay = 0.05
    cfg.b2 = 0.999
    cfg.grad_clip_norm = 1
    cfg.optimizer = "adamw"
    cfg.optimizer_fused = True
    cfg.adamw_lr = None

    # Logging
    cfg.enable_wandb = True
    cfg.wandb_project = "jepa-debugging"
    cfg.run_name_suffix = "ema-teacher-intensity-aware-mixed"

    return cfg
