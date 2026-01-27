import jax.numpy as jnp
from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    cfg = config_dict.ConfigDict()

    # Dataset
    cfg.dataset = "gems_a"
    cfg.tfrecord_dir = "data/gems_peaklist_tfrecord"
    cfg.batch_size = 128
    cfg.validation_fraction = 0.05
    cfg.shuffle_buffer = 10_000
    cfg.split_seed = 42
    cfg.num_shards = 4
    cfg.drop_remainder = True
    cfg.max_precursor_mz = 1000.0
    cfg.pair_sequence_length = 256
    cfg.seed = 42

    # Model (BERT)
    cfg.model_type = "bert"
    cfg.model_dim = 512
    cfg.num_layers = 16
    cfg.num_heads = 8
    cfg.num_kv_heads = 4
    cfg.attention_mlp_multiple = 4.0
    cfg.num_segments = 2
    cfg.mask_ratio = 0.3
    cfg.mask_token_id = 3
    cfg.pad_token_id = 0
    cfg.cls_token_id = 1
    cfg.sep_token_id = 2
    cfg.dtype = jnp.bfloat16
    cfg.param_dtype = jnp.float32

    # Filled by input_pipeline.create_datasets
    cfg.vocab_size = 0
    cfg.max_length = 0
    cfg.precursor_bins = 0
    cfg.precursor_offset = 0

    # Training (short smoke run)
    cfg.num_train_steps = 200_000
    cfg.num_epochs = 0
    cfg.learning_rate = 3e-4
    cfg.warmup_steps = 20000
    cfg.learning_rate_schedule = "cosine"
    cfg.min_learning_rate = None
    cfg.b2 = 0.99
    cfg.weight_decay = 0.01
    cfg.optimizer = "adamw"
    cfg.clip = 0.
    cfg.device_prefetch_size = 8
    cfg.log_loss_every_steps = 2000
    cfg.eval_every_steps = 10000
    cfg.num_eval_steps = 500
    cfg.checkpoint_dir = ""
    cfg.checkpoint_every_steps = 10000
    cfg.init_seed = 0

    # System / logging
    cfg.enable_wandb = True
    cfg.wandb_project = "md4"
    cfg.start_profiler = False
    cfg.initialize_multihost = False

    # Mesh / sharding
    cfg.mesh_config = config_dict.ConfigDict()
    cfg.mesh_config.mesh_shape = (1,)
    cfg.mesh_config.mesh_axis_names = ("data",)
    cfg.logical_axis_rules = [
        ("batch", "data"),
        ("hidden", None),
        ("attn_qkv", None),
        ("attn_o", None),
        ("ff_mlp", None),
        ("embed_vocab", None),
        ("input_embed", None),
        ("cross_attn", None),
        ("cond", None),
        ("cond_input", None),
        ("cond_hidden", None),
        ("cond_output", None),
        ("vocab", None),
    ]

    return cfg
