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
    cfg.pair_sequence_length = 128
    cfg.intensity_scaling = "linear"
    cfg.mz_representation = "neutral_loss"
    cfg.peak_ordering = "intensity"
    cfg.seed = 42

    # Model (strict SIGReg peak set)
    cfg.model_type = "sigreg_peak_set"
    cfg.num_peaks = 60
    cfg.model_dim = 256
    cfg.num_layers = 10
    cfg.num_heads = 8
    cfg.num_kv_heads = 4
    cfg.attention_mlp_multiple = 4.0
    cfg.feature_mlp_hidden_dim = 128
    cfg.pooling_type = "pma"
    cfg.pma_num_heads = cfg.num_heads
    cfg.pma_num_seeds = 32
    cfg.mz_fourier_num_frequencies = 64
    cfg.mz_fourier_min_freq = 1.0
    cfg.mz_fourier_max_freq = 50_000.0
    cfg.mz_fourier_learnable = False
    cfg.sigreg_use_projector = True
    cfg.sigreg_proj_hidden_dim = 2048
    cfg.sigreg_proj_output_dim = 128
    cfg.sigreg_num_slices = 512
    cfg.sigreg_lambda = 0.1
    cfg.sigreg_drop_prob = 0.25
    cfg.sigreg_mz_jitter_std = 0.01
    cfg.sigreg_intensity_jitter_std = 0.1

    # Training (short smoke run)
    cfg.num_epochs = 5
    cfg.learning_rate = 2e-4
    cfg.warmup_steps = 20_000
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
    cfg.checkpoint_every_steps = 25000
    cfg.init_seed = 0
    cfg.probe_peak_ordering = "intensity"
    cfg.limit_train_batches = 0.5
    cfg.limit_val_batches = 1.0
    cfg.limit_test_batches = 1.0
    cfg.num_sanity_val_steps = 0
    apply_training_defaults(cfg)

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
