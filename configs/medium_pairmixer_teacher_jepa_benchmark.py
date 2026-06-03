from ml_collections import config_dict

from configs.medium_pairformer_encoder_torchax_mae_teacher_jepa import (
    get_config as get_teacher_jepa_config,
)


def get_config() -> config_dict.ConfigDict:
    cfg = get_teacher_jepa_config()

    cfg.batch_size = 128
    cfg.masked_latent_predictor_block_type = "pairmixer"
    cfg.predictor_pairmixer_use_pair_bias_attention = True
    cfg.predictor_dim = 384
    cfg.masked_latent_predictor_num_layers = 4
    cfg.masked_latent_predictor_num_heads = 12
    cfg.pairformer_pair_num_heads = 12
    cfg.pairformer_refresh_pair = False
    cfg.pairformer_refresh_pair_layers = []
    cfg.predictor_dropout = 0.0

    cfg.enable_wandb = False
    cfg.log_every_n_steps = 0
    cfg.checkpoint_every_steps = 0
    cfg.msg_probe_every_n_steps = 0
    cfg.collapse_metrics_every_n_steps = 0
    cfg.run_name_suffix = "torchax-pairmixer-teacher-jepa-medium-benchmark"

    return cfg
