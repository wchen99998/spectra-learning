from ml_collections import config_dict

from configs._defaults import apply_training_defaults, apply_tune_defaults


def get_config() -> config_dict.ConfigDict:
    cfg = config_dict.ConfigDict()
    apply_training_defaults(cfg)

    # Dataset
    cfg.dataset = "gems_a"
    cfg.tfrecord_dir = "data/gems_peaklist_tfrecord"
    cfg.gems_tfrecord_repo_id = "cjim8889/gems-a-tfrecords"
    cfg.batch_size = 128
    cfg.shuffle_buffer = 1_000_000
    cfg.tfrecord_buffer_size = 250_000
    cfg.drop_remainder = True
    cfg.max_precursor_mz = 1000.0
    cfg.min_peak_intensity = 0.001
    cfg.peak_ordering = "mz"
    cfg.seed = 42


    # Model (strict SIGReg peak set)
    cfg.model_type = "sigreg_peak_set"
    cfg.num_peaks = 60
    cfg.model_dim = 256
    cfg.num_layers = 10
    cfg.num_heads = 8
    cfg.num_kv_heads = 8
    cfg.encoder_use_rope = True
    cfg.attention_mlp_multiple = 4.0
    cfg.feature_mlp_hidden_dim = 128
    cfg.pooling_type = "mean"
    cfg.pma_num_heads = cfg.num_heads
    cfg.pma_num_seeds = 32
    cfg.sigreg_num_slices = 256
    cfg.sigreg_lambda = 0.1
    cfg.jepa_num_target_blocks = 5
    cfg.jepa_context_fraction = 0.25
    cfg.jepa_target_fraction = 0.15
    cfg.jepa_block_min_len = 1
    cfg.sigreg_mz_jitter_std = 0.0001
    cfg.sigreg_intensity_jitter_std = 0.001
    # Training (short smoke run)
    cfg.num_epochs = 5
    cfg.learning_rate = 3e-4
    cfg.warmup_steps = 70_000
    cfg.learning_rate_schedule = "cosine"
    cfg.min_learning_rate = None
    cfg.b2 = 0.98
    cfg.weight_decay = 1e-4
    cfg.optimizer = "muon"
    cfg.device_prefetch_size = 8
    cfg.log_every_n_steps = 100
    cfg.val_check_interval = 1.0
    cfg.checkpoint_every_steps = 25000
    cfg.limit_train_batches = 1.0
    cfg.limit_val_batches = 0.1
    cfg.limit_test_batches = 1.0
    cfg.num_sanity_val_steps = 0
    cfg.dataloader_num_workers = 1
    cfg.dataloader_persistent_workers = True

    # Tune search space
    apply_tune_defaults(cfg)

    # System / logging
    cfg.enable_wandb = True
    cfg.wandb_project = "token-mass-spec-pretraining"
    cfg.wandb_run_name_prefix = "jepa"

    return cfg
