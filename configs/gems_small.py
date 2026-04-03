from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    cfg = config_dict.ConfigDict()

    # Dataset
    cfg.tfrecord_dir = "data/gems_peaklist_tfrecord_alpha"
    cfg.gems_tfrecord_repo_id = "cjim8889/gems-a10-tfrecords"
    cfg.batch_size = 256
    cfg.shuffle_buffer = 1_000_000
    cfg.tfrecord_buffer_size = 250_000
    cfg.drop_remainder = True
    cfg.max_precursor_mz = 1000.0
    cfg.min_peak_intensity = 0.0001
    cfg.peak_ordering = "mz"
    cfg.seed = 66

    # Model
    cfg.num_peaks = 64
    cfg.model_dim = 768
    cfg.encoder_num_layers = 12
    cfg.encoder_num_heads = 12
    cfg.encoder_num_kv_heads = 12
    cfg.encoder_num_register_tokens = 4
    cfg.encoder_apply_final_norm = False
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
    cfg.jepa_num_target_blocks = 2
    cfg.jepa_context_fraction = 0.35
    cfg.jepa_target_fraction = 0.2
    cfg.jepa_block_min_len = 1
    cfg.augmentation_mz_jitter_std = 0.0002
    cfg.augmentation_intensity_jitter_std = 0.001
    cfg.norm_type = "layernorm"

    # Predictor
    cfg.predictor_num_register_tokens = 4
    cfg.predictor_apply_final_norm = False
    cfg.masked_latent_predictor_num_layers = 10
    cfg.masked_latent_predictor_num_heads = 16
    cfg.temporal_predictor_num_layers = 0
    cfg.predictor_dim = 384
    cfg.predictor_dropout = 0.1

    # Training
    cfg.num_epochs = 20
    cfg.learning_rate = 5e-4
    cfg.warmup_steps = 20_000
    cfg.min_learning_rate = 3e-5
    cfg.b2 = 0.95
    cfg.weight_decay = 0.05
    cfg.optimizer = "muon"
    cfg.device_prefetch_size = 8
    cfg.optimizer_capturable = True
    cfg.optimizer_fused = True
    cfg.log_every_n_steps = 100
    cfg.checkpoint_every_steps = 25_000
    cfg.dataloader_num_workers = 1
    cfg.dataloader_prefetch_factor = 2
    cfg.dataloader_persistent_workers = True
    cfg.dataloader_pin_memory = True

    cfg.masked_token_loss_weight = 1.0
    cfg.masked_token_loss_type = "l2"
    cfg.jepa_target_normalization = "none"
    cfg.jepa_target_layers = [1, 4, 8, 12]
    cfg.use_ema_teacher_target = True
    cfg.teacher_ema_decay = 0.995
    cfg.teacher_ema_decay_start = 0.99
    cfg.teacher_ema_decay_warmup_steps = 500_000
    cfg.teacher_ema_update_every = 2
    cfg.grad_clip_norm = 1.0
    cfg.autocast_dtype = "bf16"
    cfg.compile_mode = "reduce-overhead"
    cfg.representation_regularizer = "none"
    cfg.msg_probe_every_n_steps = 0.2
    cfg.msg_probe_pooling_type = "pma"
    cfg.msg_probe_pma_num_heads = cfg.encoder_num_heads
    cfg.msg_probe_pma_num_seeds = 64
    cfg.msg_probe_num_epochs = 20
    cfg.msg_probe_learning_rate = 3e-4
    cfg.msg_probe_weight_decay = 0.0
    cfg.msg_probe_warmup_steps = 0
    cfg.msg_probe_hidden_dim = 1024
    cfg.msg_probe_num_layers = 2
    cfg.msg_probe_dropout = 0.0
    cfg.msg_probe_activation = "gelu"
    cfg.msg_probe_init = "default"
    cfg.msg_probe_max_train_samples = None
    cfg.msg_probe_max_test_samples = None
    cfg.probe_dataset = "nist20"
    cfg.msg_probe_tune_metric = "msg_probe/test/auc_fg_mean"
    cfg.msg_probe_tune_param_space = [
        {"param": "msg_probe_hidden_dim", "dist": "grid", "args": [0, 512, 1024]},
        {"param": "msg_probe_num_layers", "dist": "grid", "args": [1, 2]},
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
        {
            "param": "msg_probe_dropout",
            "dist": "grid",
            "args": [0.0, 0.1, 0.3],
        },
    ]
    cfg.use_precursor_token = False
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
