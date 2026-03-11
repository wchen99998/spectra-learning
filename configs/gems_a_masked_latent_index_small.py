from ml_collections import config_dict

from configs._defaults import apply_training_defaults, apply_tune_defaults


def get_config() -> config_dict.ConfigDict:
    cfg = config_dict.ConfigDict()
    apply_training_defaults(cfg)

    # Dataset
    cfg.dataset = "gems_a"
    cfg.tfrecord_dir = "data/gems_peaklist_tfrecord_alpha"
    cfg.gems_tfrecord_repo_id = "cjim8889/gems-a10-tfrecords"
    cfg.batch_size = 256
    cfg.shuffle_buffer = 1_000_000
    cfg.tfrecord_buffer_size = 250_000
    cfg.drop_remainder = True
    cfg.max_precursor_mz = 1000.0
    cfg.min_peak_intensity = 0.001
    cfg.peak_ordering = "mz"
    cfg.seed = 42

    # Model
    cfg.model_type = "sigreg_peak_set"
    cfg.num_peaks = 63
    cfg.model_dim = 256
    cfg.num_layers = 10
    cfg.num_heads = 8
    cfg.num_kv_heads = 8
    cfg.encoder_use_rope = True
    cfg.encoder_qk_norm = False
    cfg.attention_mlp_multiple = 4.0
    cfg.feature_mlp_hidden_dim = 128
    cfg.pooling_type = "pma"
    cfg.pma_num_heads = cfg.num_heads
    cfg.pma_num_seeds = 32
    cfg.sigreg_num_slices = 256
    cfg.sigreg_lambda = 0.1
    cfg.jepa_num_target_blocks = 2
    cfg.jepa_context_fraction = 0.3
    cfg.jepa_target_fraction = 0.25
    cfg.jepa_block_min_len = 1
    cfg.sigreg_mz_jitter_std = 0.005
    cfg.sigreg_intensity_jitter_std = 0.05
    cfg.norm_type = "layernorm"
    cfg.normalize_jepa_targets = False

    # Training
    cfg.num_epochs = 100
    cfg.learning_rate = 3e-4
    cfg.warmup_steps = 10_000
    cfg.learning_rate_schedule = "l2_sum"
    cfg.min_learning_rate = None
    cfg.b2 = 0.98
    cfg.weight_decay = 1e-5
    cfg.optimizer = "muon"
    cfg.device_prefetch_size = 8
    cfg.log_every_n_steps = 100
    cfg.val_check_interval = 1.0
    cfg.checkpoint_every_steps = 25_000
    cfg.limit_train_batches = 1.0
    cfg.limit_val_batches = 0.1
    cfg.limit_test_batches = 1.0
    cfg.num_sanity_val_steps = 0
    cfg.dataloader_num_workers = 1
    cfg.dataloader_persistent_workers = True

    cfg.masked_token_loss_weight = 1.0
    cfg.masked_token_loss_type = "l2"
    cfg.use_ema_teacher_target = False
    cfg.teacher_ema_decay = 0.999
    cfg.teacher_ema_decay_start = 0.98
    cfg.teacher_ema_decay_warmup_steps = 100_000
    cfg.grad_clip_norm = 1.0
    cfg.masked_latent_predictor_num_layers = 2
    cfg.predictor_num_heads = 4
    cfg.autocast_dtype = "bf16"
    cfg.representation_regularizer = "sigreg"
    cfg.gco_var_floor_target = 1.0
    cfg.gco_corr_target = 0.60
    cfg.gco_log_lambda_min = -2.0
    cfg.sigreg_lambda_warmup_steps = 50_000
    cfg.msg_probe_every_n_steps = 0.5
    cfg.use_precursor_token = True

    # Tune search space
    apply_tune_defaults(cfg)

    # System / logging
    cfg.enable_wandb = True
    cfg.wandb_project = "jepa-pretraining"
    cfg.wandb_run_name_prefix = "jepa_masked_latent_index"

    return cfg
