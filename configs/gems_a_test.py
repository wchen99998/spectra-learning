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
    cfg.seed = 42

    # Model (BERT)
    cfg.model_type = "bert"
    cfg.model_dim = 768
    cfg.num_layers = 20
    cfg.num_heads = 12
    cfg.num_kv_heads = 6
    cfg.attention_mlp_multiple = 4.0
    cfg.num_segments = 2
    cfg.mask_ratio = 0.3
    cfg.mask_token_id = 3
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
    cfg.val_check_interval = 0.25
    cfg.checkpoint_dir = ""
    cfg.checkpoint_every_steps = 10000
    cfg.init_seed = 0

    # System / logging
    cfg.enable_wandb = True
    cfg.wandb_project = "token-mass-spec-pretraining"

    return cfg
