from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    cfg = config_dict.ConfigDict()
    # Dataset
    cfg.dataset = "gems_a"
    cfg.tfrecord_dir = "data/gems_peaklist_tfrecord"
    cfg.batch_size = 32
    cfg.validation_fraction = 0.05
    cfg.shuffle_buffer = 10_000
    cfg.tfrecord_buffer_size = 250_000
    cfg.split_seed = 42
    cfg.num_shards = 4
    cfg.drop_remainder = False
    cfg.max_precursor_mz = 1000.0
    cfg.pair_sequence_length = 128
    cfg.intensity_scaling = "linear"
    cfg.mz_representation = "mz"
    cfg.peak_ordering = "intensity"
    cfg.seed = 42
    cfg.gcp_key_path = "/home/wuhao/md4/key.json"
    cfg.gems_formula_tfrecord_dir = "data/gems_formula_tfrecord"
    cfg.gems_formula_raw_csv_path = (
        "data/gems_formula/raw/GeMS_2m_combined_formula_identifications.csv"
    )
    cfg.gems_formula_gcs_uri = (
        "gs://main-novogaia-bucket/gems/GeMS_2m_combined_formula_identifications.csv"
    )
    cfg.gems_formula_column_name = "formula"
    cfg.gems_adduct_column_name = "adduct"
    cfg.gems_formula_split_seed = 42
    cfg.gems_formula_num_shards = 16
    cfg.gems_formula_drop_remainder = False

    # Model (strict SIGReg peak set)
    cfg.model_type = "sigreg_peak_set"
    cfg.num_peaks = 60
    cfg.model_dim = 256
    cfg.num_layers = 6
    cfg.num_heads = 8
    cfg.num_kv_heads = None
    cfg.attention_mlp_multiple = 4.0
    cfg.feature_mlp_hidden_dim = 128
    cfg.pooling_type = "pma"
    cfg.pma_num_heads = cfg.num_heads
    cfg.pma_num_seeds = 1
    cfg.sigreg_use_projector = True
    cfg.sigreg_proj_hidden_dim = 2048
    cfg.sigreg_proj_output_dim = 128
    cfg.sigreg_lambda = 10.0
    cfg.sigreg_drop_prob = 0.20
    cfg.sigreg_mz_jitter_std = 0.005
    cfg.sigreg_intensity_jitter_std = 0.05

    # Training
    cfg.num_epochs = 10
    cfg.learning_rate = 1e-4
    cfg.warmup_steps = 100
    cfg.learning_rate_schedule = "cosine"
    cfg.min_learning_rate = None
    cfg.b2 = 0.98
    cfg.weight_decay = 0.01
    cfg.optimizer = "adamw"
    cfg.clip = 1.0
    cfg.device_prefetch_size = 8
    cfg.log_every_n_steps = 50
    cfg.train_step_log_interval = 50
    cfg.val_check_interval = 200.0
    cfg.num_eval_steps = 50
    cfg.checkpoint_every_steps = 10_000
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

    # System / logging
    cfg.enable_wandb = False
    cfg.wandb_project = "md4"
    cfg.num_transformer_blocks = None

    return cfg
