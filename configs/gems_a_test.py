from ml_collections import config_dict


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
    cfg.pair_sequence_length = 128
    cfg.intensity_scaling = "linear"
    cfg.mz_representation = "neutral_loss"
    cfg.peak_ordering = "intensity"
    cfg.seed = 42

    # Model (BERT)
    cfg.model_type = "bert"
    cfg.model_dim = 768
    cfg.num_layers = 20
    cfg.num_heads = 12
    cfg.num_kv_heads = 6
    cfg.attention_mlp_multiple = 4.0
    cfg.num_segments = 2
    cfg.pad_token_id = 0
    cfg.cls_token_id = 1
    cfg.sep_token_id = 2
    # Filled by input_pipeline.create_datasets
    cfg.vocab_size = 0
    cfg.max_length = 0
    cfg.precursor_bins = 0
    cfg.precursor_offset = 0

    # Training (short smoke run)
    cfg.num_epochs = 5
    cfg.learning_rate = 3e-4
    cfg.warmup_steps = 50_000
    cfg.learning_rate_schedule = "cosine"
    cfg.min_learning_rate = None
    cfg.b2 = 0.98
    cfg.weight_decay = 1e-4
    cfg.optimizer = "adamw"
    cfg.clip = 0.
    cfg.device_prefetch_size = 1
    cfg.log_every_n_steps = 500
    cfg.train_step_log_interval = 500
    cfg.val_check_interval = 0.25
    cfg.checkpoint_every_steps = 10000
    cfg.init_seed = 0
    cfg.enable_linear_probe = True
    cfg.probe_bits = 1024
    cfg.probe_fit_bias = True
    cfg.probe_peak_ordering = "intensity"
    cfg.limit_train_batches = 1.0
    cfg.limit_val_batches = 1.0
    cfg.limit_test_batches = 1.0
    cfg.num_sanity_val_steps = 0
    cfg.profile_enabled = False
    cfg.profile_wait_steps = 20
    cfg.profile_warmup_steps = 20
    cfg.profile_active_steps = 40
    cfg.profile_repeat = 1
    cfg.profile_record_shapes = True
    cfg.profile_with_stack = True
    cfg.profile_profile_memory = True
    cfg.profile_trace_dir = "profiler"
    cfg.non_blocking_device_transfer = True
    cfg.optimizer_capturable = True
    cfg.optimizer_fused = True
    cfg.dataloader_num_workers = 1
    cfg.dataloader_prefetch_factor = 2
    cfg.dataloader_persistent_workers = True
    cfg.dataloader_pin_memory = True
    cfg.train_log_extra_metrics_on_step = False
    cfg.cache_rope_frequencies = True

    # System / logging
    cfg.enable_wandb = True
    cfg.wandb_project = "token-mass-spec-pretraining"

    return cfg
