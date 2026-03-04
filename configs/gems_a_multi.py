from ml_collections import config_dict

from configs._defaults import apply_final_probe_defaults, apply_training_defaults, apply_tune_defaults


def get_config() -> config_dict.ConfigDict:
    cfg = config_dict.ConfigDict()
    
    apply_training_defaults(cfg)
    # Dataset
    cfg.dataset = "gems_a"
    cfg.tfrecord_dir = "data/gems_peaklist_tfrecord"
    cfg.batch_size = 256
    cfg.validation_fraction = 0.05
    cfg.shuffle_buffer = 1_000_000
    cfg.tfrecord_buffer_size = 250_000
    cfg.split_seed = 42
    cfg.num_shards = 4
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
    cfg.rope_mz_max = 1000.0
    cfg.rope_mz_precision = 0.1
    cfg.rope_modulo_2pi = True
    cfg.encoder_qk_norm = True
    cfg.encoder_post_norm = True
    cfg.attention_mlp_multiple = 4.0
    cfg.feature_mlp_hidden_dim = 128
    cfg.pooling_type = "pma"
    cfg.pma_num_heads = cfg.num_heads
    cfg.pma_num_seeds = 32
    cfg.sigreg_num_slices = 256
    cfg.sigreg_lambda = 0.1
    cfg.multicrop_num_local_views = 2
    cfg.multicrop_local_keep_fraction = 0.25
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

    # Post-fit attentive probe (overrides from shared defaults)
    apply_final_probe_defaults(cfg)
    cfg.final_probe_num_epochs = 5
    cfg.final_probe_warmup_steps = 50

    # Tune search space
    apply_tune_defaults(cfg)

    # System / logging
    cfg.enable_wandb = True
    cfg.wandb_project = "token-mass-spec-pretraining"
    cfg.wandb_run_name_prefix = "jepa"

    return cfg
