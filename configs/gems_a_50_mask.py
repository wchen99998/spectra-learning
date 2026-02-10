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
    cfg.sigreg_lambda = 1.0
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
    cfg.enable_linear_probe = True
    cfg.probe_bits = 1024
    cfg.probe_fit_bias = True
    cfg.probe_peak_ordering = "intensity"
    cfg.limit_train_batches = 0.5
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

    # Epoch-end MSG eval fine-tuning
    cfg.eval_msg_finetune_num_epochs = 5
    cfg.eval_msg_finetune_feature_source = "projector"
    cfg.eval_msg_finetune_trainable_scope = "full"
    cfg.eval_msg_finetune_head_hidden_dim = 512
    cfg.eval_msg_finetune_learning_rate = 1e-4
    cfg.eval_msg_finetune_weight_decay = 1e-4
    cfg.eval_msg_finetune_warmup_steps = 50
    cfg.eval_msg_finetune_peak_ordering = "intensity"

    # System / logging
    cfg.enable_wandb = True
    cfg.wandb_project = "token-mass-spec-pretraining"
    cfg.wandb_run_name_prefix = "jepa"

    return cfg
