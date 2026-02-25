from ml_collections import config_dict

from configs._defaults import apply_final_probe_defaults, apply_training_defaults, apply_tune_defaults


def get_config() -> config_dict.ConfigDict:
    cfg = config_dict.ConfigDict()

    # Dataset
    cfg.dataset = "gems_a"
    cfg.tfrecord_dir = "data/gems_peaklist_tfrecord"
    cfg.batch_size = 512
    cfg.validation_fraction = 0.05
    cfg.shuffle_buffer = 200_000
    cfg.tfrecord_buffer_size = 250_000
    cfg.split_seed = 42
    cfg.num_shards = 4
    cfg.drop_remainder = True
    cfg.max_precursor_mz = 1000.0
    cfg.peak_ordering = "intensity"
    cfg.seed = 42

    # Model (strict SIGReg peak set)
    cfg.model_type = "sigreg_peak_set"
    cfg.num_peaks = 60
    cfg.model_dim = 768
    cfg.num_layers = 20
    cfg.num_heads = 12
    cfg.num_kv_heads = 6
    cfg.encoder_use_rope = True
    cfg.rope_mz_max = 1000.0
    cfg.rope_mz_precision = 0.1
    cfg.rope_complement_heads = cfg.num_heads // 2
    cfg.rope_modulo_2pi = True
    cfg.attention_mlp_multiple = 4.0
    cfg.feature_mlp_hidden_dim = 128
    cfg.pooling_type = "pma"
    cfg.pma_num_heads = cfg.num_heads
    cfg.pma_num_seeds = 1
    cfg.sigreg_use_projector = True
    cfg.sigreg_proj_hidden_dim = 2048
    cfg.sigreg_proj_output_dim = 128
    cfg.sigreg_lambda = 0.1
    cfg.multicrop_num_global_views = 2
    cfg.multicrop_num_local_views = 6
    cfg.multicrop_global_keep_fraction = 0.80
    cfg.multicrop_local_keep_fraction = 0.25
    cfg.sigreg_mz_jitter_std = 0.0001
    cfg.sigreg_intensity_jitter_std = 0.001

    # Training (short smoke run)
    cfg.num_epochs = 5
    cfg.learning_rate = 3e-4
    cfg.warmup_steps = 50_000
    cfg.learning_rate_schedule = "cosine"
    cfg.min_learning_rate = None
    cfg.b2 = 0.98
    cfg.weight_decay = 1e-4
    cfg.optimizer = "adamw"
    cfg.device_prefetch_size = 1
    cfg.log_every_n_steps = 500
    cfg.val_check_interval = 0.25
    cfg.checkpoint_every_steps = 10000
    cfg.limit_train_batches = 1.0
    cfg.limit_val_batches = 1.0
    cfg.limit_test_batches = 1.0
    cfg.num_sanity_val_steps = 0
    apply_training_defaults(cfg)

    # Post-fit attentive probe (overrides from shared defaults)
    apply_final_probe_defaults(cfg)
    cfg.final_probe_warmup_steps = 500

    # Tune search space
    apply_tune_defaults(cfg)

    # System / logging
    cfg.enable_wandb = True
    cfg.wandb_project = "token-mass-spec-pretraining"

    return cfg
