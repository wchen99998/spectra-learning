from ml_collections import config_dict


def apply_training_defaults(cfg: config_dict.ConfigDict) -> None:
    """Apply shared training infrastructure defaults.

    Call this first in each get_config().
    Per-experiment configs should override values that differ afterward.
    """
    # Dataloader
    cfg.non_blocking_device_transfer = True
    cfg.optimizer_capturable = True
    cfg.optimizer_fused = True
    cfg.autocast_dtype = "fp32"
    cfg.compile_mode = "max-autotune"
    cfg.encoder_qk_norm = False
    cfg.norm_type = "rmsnorm"
    cfg.norm_position = "prenorm"
    cfg.masked_token_loss_weight = 0.0
    cfg.representation_regularizer = "sigreg"
    cfg.masked_latent_predictor_num_layers = 2
    cfg.jepa_num_target_blocks = 2
    cfg.jepa_context_fraction = 0.5
    cfg.jepa_target_fraction = 0.25
    cfg.jepa_block_min_len = 1
    cfg.use_precursor_token = False
    cfg.grad_clip_norm = None
    # num_workers=0: TF's internal thread pool handles parallelism; forking
    # a subprocess duplicates the TF runtime and can deadlock during shuffle
    # buffer fill.  The batch-first pipeline (batched_parse_and_transform)
    # already achieves >270 b/s in-process.
    cfg.device_prefetch_size = 8
    cfg.dataloader_num_workers = 1
    cfg.dataloader_prefetch_factor = 2
    cfg.dataloader_persistent_workers = True
    cfg.dataloader_pin_memory = True
    cfg.gems_tfrecord_repo_id = None
    cfg.gems_tfrecord_revision = "main"
    # cfg.msg_probe_every_n_steps = 0
    cfg.msg_probe_cache_dir = None
    cfg.msg_probe_num_epochs = 10
    cfg.msg_probe_learning_rate = 1e-3
    cfg.msg_probe_weight_decay = 1e-4
    cfg.msg_probe_warmup_steps = 0
    cfg.msg_probe_max_train_samples = None
    cfg.msg_probe_max_test_samples = None
    cfg.msg_probe_type = "linear"  # "linear" or "mlp"
    cfg.msg_probe_mlp_hidden_dim = None  # defaults to model_dim in run_msg_probe
    cfg.msg_probe_mlp_num_layers = 2
    cfg.msg_probe_mlp_activation = "silu"
    cfg.probe_dataset = "nist20"  # "massspec" or "nist20"

    # Temporal predictor defaults
    cfg.temporal_predictor_num_layers = 0
    cfg.encoder_finetune_lr = None  # falls back to learning_rate if None

    # Muon optimizer defaults (only used when cfg.optimizer == "muon")
    cfg.muon_lr = None  # Falls back to cfg.learning_rate
    cfg.adamw_lr = None  # Falls back to cfg.learning_rate
    cfg.muon_momentum = 0.95
    cfg.muon_nesterov = True
    cfg.muon_ns_steps = 5
    cfg.muon_weight_decay = None  # Falls back to cfg.weight_decay
    cfg.muon_adjust_lr_fn = "match_rms_adamw"


def apply_tune_defaults(cfg: config_dict.ConfigDict) -> None:
    """Apply default tune search space definition.

    Each entry is {"param": <config key>, "dist": <distribution>, "args": [...]}.
    Supported distributions: grid, loguniform, uniform, choice, randint, quniform.
    """
    cfg.tune_param_space = [
        # {"param": "learning_rate", "dist": "grid", "args": [1e-4, 4e-4]},
        # {"param": "jepa_target_fraction", "dist": "grid", "args": [0.125, 0.25]},
        # {"param": "weight_decay", "dist": "grid", "args": [1e-4, 1e-3, 1e-2]},
        {"param": "sigreg_lambda", "dist": "grid", "args": [100.0]},
    ]
