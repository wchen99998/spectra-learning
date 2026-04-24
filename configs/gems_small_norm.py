from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    cfg = config_dict.ConfigDict()

    # Dataset
    cfg.artifact_dir = "data/gems_artifacts_alpha"
    cfg.gems_native_repo_id = "cjim8889/gems-a10-native"
    cfg.nist_full_probe_repo_id = "cjim8889/hr_msms_nist_probe_prepared"
    cfg.nist_full_probe_revision = "main"
    cfg.nist_full_probe_train_samples = 4_000
    cfg.nist_full_probe_test_samples = 1_000
    cfg.nist_full_probe_num_repeats = 3
    cfg.batch_size = 256
    cfg.shuffle_buffer = 1_000_000
    cfg.drop_remainder = True
    cfg.max_precursor_mz = 1000.0
    cfg.min_peak_intensity = 0.0001
    cfg.peak_drop_min_intensity = 0.0001
    cfg.precursor_peak_exclusion_window_da = 5.0
    cfg.peak_ordering = "mz"
    cfg.seed = 66

    # Model


    cfg.num_peaks = 64
    cfg.model_dim = 256
    cfg.encoder_num_layers = 12
    cfg.encoder_num_heads = 8
    cfg.encoder_num_kv_heads = 8
    cfg.encoder_num_register_tokens = 2
    cfg.encoder_apply_final_norm = True
    cfg.encoder_qk_norm = False
    cfg.encoder_fourier_strategy = "log_spaced"
    cfg.encoder_fourier_x_min = 3e-3
    cfg.encoder_fourier_x_max = 1000.0
    cfg.encoder_fourier_funcs = "both"
    cfg.encoder_fourier_num_freqs = 256
    cfg.encoder_fourier_sigma = 10.0
    cfg.encoder_fourier_trainable = False
    cfg.encoder_fourier_input_scale = 1000.0
    cfg.attention_mlp_multiple = 4.0
    cfg.feature_mlp_hidden_dim = 512
    cfg.sigreg_num_slices = 256
    cfg.sigreg_lambda = 0.003
    cfg.sigreg_precursor_scale = 1.0

    cfg.jepa_num_target_blocks = 3
    cfg.encoder_use_position_embedding = False
    cfg.jepa_context_fraction = 0.15
    cfg.jepa_context_fraction_range = (0.1, 0.25)
    cfg.jepa_target_fraction = 0.3
    cfg.jepa_target_fraction_range = (0.15, 0.35)
    cfg.jepa_block_min_len = 1
    cfg.jepa_mask_strategy = "all"
    cfg.jepa_mask_lengths = (2, 4, 8, 12, 16)
    cfg.jepa_mask_round_from = 3
    cfg.norm_type = "layernorm"

    # Predictor
    cfg.predictor_num_register_tokens = 4
    cfg.predictor_apply_final_norm = True
    cfg.masked_latent_predictor_num_layers = 8
    cfg.masked_latent_predictor_num_heads = 8
    cfg.temporal_predictor_num_layers = 0
    cfg.predictor_dim = 256
    cfg.target_projector_dim = cfg.model_dim
    cfg.predictor_dropout = 0.15

    # Training
    cfg.num_epochs = 10
    cfg.learning_rate = 1e-4
    cfg.warmup_steps = 10_000
    cfg.min_learning_rate = 3e-5
    cfg.b2 = 0.999
    cfg.weight_decay = 0.05
    cfg.optimizer = "muon"
    cfg.device_prefetch_size = 8
    cfg.optimizer_capturable = True
    cfg.optimizer_fused = True
    cfg.log_every_n_steps = 250
    cfg.checkpoint_every_steps = 25_000
    cfg.dataloader_num_workers = 8
    cfg.dataloader_prefetch_factor = 2
    cfg.dataloader_persistent_workers = True
    cfg.dataloader_pin_memory = True

    cfg.masked_token_loss_weight = 1.0
    cfg.masked_token_loss_type = "l2"
    cfg.jepa_target_normalization = "zscore"
    cfg.jepa_target_layers = [3, 5, 8, 12]
    cfg.grad_clip_norm = 1.0
    cfg.autocast_dtype = "bf16"
    cfg.compile_mode = "default"
    cfg.representation_regularizer = "slot-sigreg-enc"
    cfg.msg_probe_every_n_steps = 0.25
    cfg.msg_probe_num_epochs = 20
    cfg.msg_probe_learning_rate = 1e-3
    cfg.msg_probe_weight_decay = 0.0
    cfg.msg_probe_warmup_steps = 0
    cfg.msg_probe_max_train_samples = None
    cfg.msg_probe_max_test_samples = None
    cfg.msg_probe_variants = ("mean", "covariance", "pma")
    cfg.msg_probe_mlp_hidden_dim = cfg.model_dim
    cfg.msg_probe_covariance_dim = 32
    cfg.msg_probe_pma_num_seeds = 32
    cfg.msg_probe_pma_num_heads = cfg.encoder_num_heads
    cfg.probe_dataset = "nist-full"
    cfg.msg_probe_tune_metric = "msg_probe/test/auc_maccs_mean"
    cfg.msg_probe_tune_param_space = [
        {
            "param": "msg_probe_learning_rate",
            "dist": "grid",
            "args": [1e-4, 3e-4, 1e-3],
        },
        {
            "param": "msg_probe_weight_decay",
            "dist": "grid",
            "args": [0.0, 1e-2, 0.1],
        },
    ]
    cfg.use_precursor_token = True
    cfg.muon_lr = None
    cfg.adamw_lr = None
    cfg.muon_momentum = 0.95
    cfg.muon_nesterov = True
    cfg.muon_ns_steps = 5
    cfg.muon_weight_decay = None
    cfg.muon_adjust_lr_fn = "match_rms_adamw"

    # System / logging
    cfg.enable_wandb = True
    cfg.wandb_project = "jepa-debugging"
    cfg.wandb_run_name_prefix = "jepa_masked_latent_index"

    return cfg
