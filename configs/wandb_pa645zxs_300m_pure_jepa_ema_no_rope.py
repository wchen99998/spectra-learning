from ml_collections import config_dict


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
    cfg.precursor_peak_exclusion_window_da = 0
    cfg.peak_ordering = "mz"
    cfg.seed = 66

    # Encoder
    cfg.num_peaks = 64
    cfg.model_dim = 1024
    cfg.encoder_num_layers = 20
    cfg.encoder_num_heads = 16
    cfg.encoder_use_position_embedding = False
    cfg.encoder_apply_final_norm = True
    cfg.encoder_use_fourier_features = True
    cfg.encoder_fourier_input_scale = 1000
    cfg.encoder_fourier_mlp_hidden_dim = 1024
    cfg.encoder_fourier_mlp_num_layers = 4
    cfg.encoder_fourier_num_freqs = 64
    cfg.encoder_fourier_x_max = 1000
    cfg.encoder_fourier_x_min = 0.003
    cfg.feature_mlp_hidden_dim = 2048
    cfg.attention_mlp_multiple = 4

    # Masked latent predictor
    cfg.predictor_dim = 512
    cfg.predictor_dropout = 0.1
    cfg.predictor_apply_final_norm = True
    cfg.predictor_use_rope = False
    cfg.masked_latent_predictor_num_layers = 12
    cfg.masked_latent_predictor_num_heads = 16
    cfg.target_projector_dim = -1
    cfg.masked_token_input_mode = "latent_token"

    # Pure JEPA targets and masking
    cfg.training_mode = "jepa"
    cfg.masked_token_loss_weight = 1
    cfg.jepa_mae_loss_weight = 0.0
    cfg.mae_loss_weight = 0.0
    cfg.jepa_target_layers = [20]
    cfg.jepa_target_normalization = "none"
    cfg.jepa_mask_strategy = ["intensity_aware", "ragged"]
    cfg.jepa_num_target_blocks = 1
    cfg.jepa_context_fraction = 0.70
    cfg.jepa_target_fraction = 0.30
    cfg.jepa_block_min_len = 1
    cfg.jepa_mask_lengths = (2, 4, 8, 12)
    cfg.jepa_mask_round_from = 3
    cfg.jepa_mae_mz_bin_size = 0.1

    # Intensity-aware masking
    cfg.jepa_intensity_aware_tau = 0.5
    cfg.jepa_intensity_aware_alpha = 0.75
    cfg.jepa_intensity_aware_beta_context = 0.85
    cfg.jepa_intensity_aware_beta_target = 0.85
    cfg.jepa_intensity_aware_eps_context = 0.04
    cfg.jepa_intensity_aware_eps_target = 0.08
    cfg.jepa_intensity_aware_anchor_keep = 0.60
    cfg.jepa_intensity_aware_min_eff_context = 4.0
    cfg.jepa_intensity_aware_min_eff_target = 1.8
    cfg.jepa_intensity_aware_tail_target_mix = 1.0
    cfg.jepa_intensity_aware_local_gap_da = 1.0
    cfg.jepa_intensity_aware_local_gap_probability = 0.0
    cfg.jepa_intensity_aware_min_unused_mass = 0.09
    cfg.jepa_intensity_aware_context_fraction = 0.70
    cfg.jepa_intensity_aware_target_fraction = 0.30

    # Pooling
    cfg.covariance_pooling_dim = 64
    cfg.train_covariance_pooling = False

    # EMA teacher
    cfg.use_ema_teacher = True
    cfg.frozen_teacher_config_path = None
    cfg.frozen_teacher_checkpoint_path = None
    cfg.ema_teacher_momentum_start = 0.99
    cfg.ema_teacher_momentum_final = 0.999
    cfg.ema_teacher_schedule = "cosine"

    # Training
    cfg.num_epochs = 25
    cfg.autocast_dtype = "bf16"
    cfg.compile_mode = "reduce-overhead"
    cfg.device_prefetch_size = 8
    cfg.log_every_n_steps = 250
    cfg.collapse_metrics_every_n_steps = 250
    cfg.checkpoint_every_steps = 25_000
    cfg.max_duration_hours = 23.5
    cfg.dataloader_num_workers = 8
    cfg.dataloader_prefetch_factor = 2
    cfg.dataloader_persistent_workers = True
    cfg.dataloader_pin_memory = True

    # MSG probe
    cfg.probe_dataset = "nist-murcko"
    cfg.msg_probe_batch_size = 256
    cfg.msg_probe_early_stopping = True
    cfg.msg_probe_early_stopping_min_delta = 0.0001
    cfg.msg_probe_early_stopping_min_epochs = 20
    cfg.msg_probe_early_stopping_patience = 20
    cfg.msg_probe_every_n_steps = 0.5
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
    cfg.learning_rate = 0.0002
    cfg.min_learning_rate = 0.00001
    cfg.warmup_steps = 10_000
    cfg.weight_decay = 0.01
    cfg.b2 = 0.95
    cfg.grad_clip_norm = 1
    cfg.optimizer = "adamw"
    cfg.optimizer_fused = True
    cfg.adamw_lr = None

    # Logging
    cfg.enable_wandb = True
    cfg.wandb_project = "jepa-debugging"
    cfg.run_name_suffix = "pure-jepa-ema-300m-no-rope-m099-0999-23h"

    return cfg
