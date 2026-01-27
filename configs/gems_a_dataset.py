import jax.numpy as jnp
from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    cfg = config_dict.ConfigDict()
    # Dataset
    cfg.dataset = "gems_a"
    cfg.tfrecord_dir = "data/gems_peaklist_tfrecord"
    cfg.batch_size = 32
    cfg.validation_fraction = 0.05
    cfg.shuffle_buffer = 10_000
    cfg.split_seed = 42
    cfg.num_shards = 4
    cfg.drop_remainder = False
    cfg.max_precursor_mz = 1000.0
    cfg.pair_sequence_length = 256
    cfg.seed = 42

    # Model (BERT)
    cfg.model_type = "bert"
    cfg.model_dim = 256
    cfg.num_layers = 6
    cfg.num_heads = 8
    cfg.num_kv_heads = None
    cfg.attention_mlp_multiple = 4.0
    cfg.num_segments = 2
    cfg.mask_ratio = 1.0
    cfg.mask_token_id = 3
    cfg.pad_token_id = 0
    cfg.cls_token_id = 1
    cfg.sep_token_id = 2
    cfg.dtype = jnp.float32
    cfg.param_dtype = jnp.float32

    # Filled by input_pipeline.create_datasets
    cfg.vocab_size = 0
    cfg.max_length = 0
    cfg.precursor_bins = 0
    cfg.precursor_offset = 0

    # Training
    cfg.num_train_steps = 1000
    cfg.num_epochs = 0
    cfg.learning_rate = 1e-4
    cfg.warmup_steps = 100
    cfg.learning_rate_schedule = "cosine"
    cfg.min_learning_rate = None
    cfg.b2 = 0.98
    cfg.weight_decay = 0.01
    cfg.optimizer = "adamw"
    cfg.clip = 1.0
    cfg.device_prefetch_size = 8
    cfg.log_loss_every_steps = 50
    cfg.eval_every_steps = 200
    cfg.num_eval_steps = 50
    cfg.checkpoint_dir = ""
    cfg.checkpoint_every_steps = 10_000
    cfg.init_seed = 0

    # System / logging
    cfg.enable_wandb = False
    cfg.wandb_project = "md4"
    cfg.start_profiler = False
    cfg.initialize_multihost = False
    cfg.num_transformer_blocks = None

    # Mesh / sharding
    cfg.mesh_config = config_dict.ConfigDict()
    cfg.mesh_config.mesh_shape = (1, 1)
    cfg.mesh_config.mesh_axis_names = ("data", "model")

    return cfg
