from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    cfg = config_dict.ConfigDict()

    # Dataset
    cfg.dataset = "gems_a"
    cfg.tfrecord_dir = "data/gems_peaklist_tfrecord_alpha"
    cfg.gems_tfrecord_repo_id = "cjim8889/gems-a10-tfrecords"
    cfg.batch_size = 256
    cfg.shuffle_buffer = 1_000_000
    cfg.tfrecord_buffer_size = 250_000
    cfg.drop_remainder = True
    cfg.max_precursor_mz = 1000.0
    cfg.min_peak_intensity = 0.0001
    cfg.peak_ordering = "mz"
    cfg.seed = 42

    # Model
    cfg.model_type = "sigreg_peak_set"
    cfg.num_peaks = 64
    cfg.model_dim = 512
    cfg.num_layers = 12
    cfg.num_heads = 8
    cfg.num_kv_heads = 8
    cfg.encoder_num_register_tokens = 0
    cfg.predictor_num_register_tokens = 0
    cfg.encoder_qk_norm = False
    cfg.encoder_fourier_strategy = "lin_float_int"
    cfg.encoder_fourier_x_min = 1e-4
    cfg.encoder_fourier_x_max = 1000.0
    cfg.encoder_fourier_funcs = "sin"
    cfg.encoder_fourier_num_freqs = 512
    cfg.encoder_fourier_sigma = 10.0
    cfg.encoder_fourier_trainable = True
    cfg.attention_mlp_multiple = 4.0
    cfg.feature_mlp_hidden_dim = 1024
    cfg.sigreg_num_slices = 256
    cfg.sigreg_lambda = 0.1
    cfg.jepa_num_target_blocks = 2
    cfg.jepa_context_fraction = 0.5
    cfg.jepa_target_fraction = 0.1
    cfg.jepa_block_min_len = 1
    cfg.sigreg_mz_jitter_std = 0.001
    cfg.sigreg_intensity_jitter_std = 0.05
    cfg.norm_type = "layernorm"

    # Training
    cfg.num_epochs = 100
    cfg.learning_rate = 1e-4
    cfg.warmup_steps = 0
    cfg.learning_rate_schedule = "cosine"
    cfg.min_learning_rate = None
    cfg.b2 = 0.98
    cfg.weight_decay = 1e-4
    cfg.optimizer = "muon"
    cfg.device_prefetch_size = 8
    cfg.non_blocking_device_transfer = True
    cfg.optimizer_capturable = True
    cfg.optimizer_fused = True
    cfg.log_every_n_steps = 100
    cfg.val_check_interval = 1.0
    cfg.checkpoint_every_steps = 25_000
    cfg.limit_train_batches = 1.0
    cfg.limit_val_batches = 0.1
    cfg.limit_test_batches = 1.0
    cfg.num_sanity_val_steps = 0
    cfg.dataloader_num_workers = 1
    cfg.dataloader_prefetch_factor = 2
    cfg.dataloader_persistent_workers = True
    cfg.dataloader_pin_memory = True

    cfg.masked_token_loss_weight = 1.0
    cfg.masked_token_loss_type = "l2"
    cfg.normalize_jepa_targets = False
    cfg.jepa_target_normalization = "none"
    cfg.use_ema_teacher_target = True
    cfg.teacher_ema_decay = 0.999
    cfg.teacher_ema_decay_start = 0.999
    cfg.teacher_ema_decay_warmup_steps = 0
    cfg.teacher_ema_update_every = 2
    cfg.grad_clip_norm = 1.0
    cfg.masked_latent_predictor_num_layers = 4
    cfg.predictor_num_heads = 8
    cfg.autocast_dtype = "bf16"
    cfg.compile_mode = "reduce-overhead"
    cfg.representation_regularizer = "none"
    cfg.sigreg_lambda_warmup_steps = 50_000
    cfg.msg_probe_every_n_steps = 0.25
    cfg.msg_probe_cache_dir = None
    cfg.msg_probe_pooling_type = "pma"
    cfg.msg_probe_pma_num_heads = cfg.num_heads
    cfg.msg_probe_pma_num_seeds = 32
    cfg.msg_probe_num_epochs = 10
    cfg.msg_probe_learning_rate = 4e-4
    cfg.msg_probe_weight_decay = 1e-4
    cfg.msg_probe_warmup_steps = 0
    cfg.msg_probe_max_train_samples = None
    cfg.msg_probe_max_test_samples = None
    cfg.probe_dataset = "nist20"
    cfg.use_precursor_token = False
    cfg.temporal_predictor_num_layers = 0
    cfg.encoder_finetune_lr = None
    cfg.muon_lr = None
    cfg.adamw_lr = None
    cfg.muon_momentum = 0.95
    cfg.muon_nesterov = True
    cfg.muon_ns_steps = 5
    cfg.muon_weight_decay = None
    cfg.muon_adjust_lr_fn = "match_rms_adamw"

    # Tune search space
    cfg.tune_param_space = [
        {"param": "sigreg_lambda", "dist": "grid", "args": [100.0]},
    ]

    # System / logging
    cfg.enable_wandb = True
    cfg.wandb_project = "jepa-debugging"
    cfg.wandb_run_name_prefix = "jepa_masked_latent_index"

    return cfg
