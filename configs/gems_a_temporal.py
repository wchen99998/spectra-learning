from ml_collections import config_dict

from configs._defaults import apply_training_defaults, apply_tune_defaults


def get_config() -> config_dict.ConfigDict:
    cfg = config_dict.ConfigDict()
    apply_training_defaults(cfg)

    # Dataset — temporal experiment-grouped pipeline
    cfg.dataset = "gems_a_temporal"
    cfg.pipeline = "temporal"
    cfg.temporal_repo_id = "cjim8889/gems-a10-grouped"
    cfg.temporal_revision = "main"
    cfg.temporal_data_dir = "data/gems_grouped"
    cfg.batch_size = 256
    cfg.drop_remainder = True
    cfg.max_precursor_mz = 1000.0
    cfg.min_peak_intensity = 0.0001
    cfg.peak_ordering = "mz"
    cfg.seed = 42

    # Model
    cfg.model_type = "sigreg_peak_set"
    cfg.num_peaks = 64
    cfg.model_dim = 512
    cfg.num_layers = 12
    cfg.num_heads = 8
    cfg.num_kv_heads = 8
    cfg.encoder_use_rope = True
    cfg.encoder_qk_norm = False
    cfg.attention_mlp_multiple = 4.0
    cfg.feature_mlp_hidden_dim = 128
    cfg.sigreg_num_slices = 256
    cfg.sigreg_lambda = 0.1
    cfg.sigreg_mz_jitter_std = 0.001
    cfg.sigreg_intensity_jitter_std = 0.05
    cfg.norm_type = "rmsnorm"

    # Training
    cfg.num_epochs = 100
    cfg.learning_rate = 3e-4
    cfg.warmup_steps = 10_000
    cfg.learning_rate_schedule = "cosine"
    cfg.min_learning_rate = None
    cfg.b2 = 0.98
    cfg.weight_decay = 1e-4
    cfg.optimizer = "muon"
    cfg.device_prefetch_size = 8
    cfg.log_every_n_steps = 100
    cfg.val_check_interval = 1.0
    cfg.checkpoint_every_steps = 25_000
    cfg.limit_train_batches = 1.0
    cfg.limit_val_batches = 0.1
    cfg.limit_test_batches = 1.0
    cfg.num_sanity_val_steps = 0

    # DataLoader — native PyTorch, multiple workers OK (no TF)
    cfg.dataloader_num_workers = 4
    cfg.dataloader_persistent_workers = True
    cfg.dataloader_prefetch_factor = 2
    cfg.dataloader_pin_memory = True

    cfg.masked_token_loss_weight = 1.0
    cfg.grad_clip_norm = 1.0
    cfg.masked_latent_predictor_num_layers = 4
    cfg.predictor_num_heads = 8
    cfg.autocast_dtype = "bf16"
    cfg.compile_mode = "max-autotune"
    cfg.representation_regularizer = "none"
    cfg.msg_probe_every_n_steps = 0.25
    cfg.msg_probe_pooling_type = "pma"
    cfg.msg_probe_pma_num_heads = cfg.num_heads
    cfg.msg_probe_pma_num_seeds = 32
    cfg.use_precursor_token = False

    # Temporal predictor
    cfg.temporal_predictor_num_layers = 4
    cfg.encoder_finetune_lr = 3e-5
    cfg.pretrained_checkpoint = None  # set via CLI

    # Tune search space
    apply_tune_defaults(cfg)

    # System / logging
    cfg.enable_wandb = True
    cfg.wandb_project = "jepa-pretraining"
    cfg.wandb_run_name_prefix = "jepa_temporal"

    return cfg
