from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    cfg = config_dict.ConfigDict()

    # Dataset — temporal experiment-grouped pipeline
    cfg.temporal_repo_id = "cjim8889/gems-a10-grouped"
    cfg.temporal_revision = "main"
    cfg.temporal_data_dir = "data/gems_grouped"
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
    cfg.feature_mlp_hidden_dim = 128
    cfg.sigreg_num_slices = 256
    cfg.sigreg_lambda = 0.1
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
    cfg.checkpoint_every_steps = 25_000

    # DataLoader — native PyTorch, multiple workers OK (no TF)
    cfg.dataloader_num_workers = 4
    cfg.dataloader_persistent_workers = True
    cfg.dataloader_prefetch_factor = 2
    cfg.dataloader_pin_memory = True

    cfg.masked_token_loss_weight = 1.0
    cfg.mae_loss_weight = 1.0
    cfg.masked_token_loss_type = "l2"
    cfg.jepa_target_normalization = "none"
    cfg.grad_clip_norm = 1.0
    cfg.masked_latent_predictor_num_layers = 4
    cfg.autocast_dtype = "bf16"
    cfg.compile_mode = "max-autotune"
    cfg.representation_regularizer = "none"
    cfg.msg_probe_every_n_steps = 0.25
    cfg.msg_probe_pooling_type = "pma"
    cfg.msg_probe_pma_num_heads = cfg.encoder_num_heads
    cfg.msg_probe_pma_num_seeds = 32
    cfg.msg_probe_num_epochs = 10
    cfg.msg_probe_learning_rate = 4e-4
    cfg.msg_probe_weight_decay = 1e-4
    cfg.msg_probe_warmup_steps = 0
    cfg.msg_probe_max_train_samples = None
    cfg.msg_probe_max_test_samples = None
    cfg.probe_dataset = "nist20"
    cfg.use_precursor_token = False

    # Temporal predictor
    cfg.temporal_predictor_num_layers = 4
    cfg.encoder_learning_rate = 3e-5
    cfg.pretrained_checkpoint = None  # set via CLI

    # Tune search space
    cfg.tune_param_space = [
        {"param": "sigreg_lambda", "dist": "grid", "args": [100.0]},
    ]

    # System / logging
    cfg.enable_wandb = True
    cfg.wandb_project = "jepa-pretraining"
    cfg.wandb_run_name_prefix = "jepa_temporal"

    return cfg
