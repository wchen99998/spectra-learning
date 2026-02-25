from ml_collections import config_dict

from configs.gems_a_multi import get_config as get_base_config


def get_config() -> config_dict.ConfigDict:
    cfg = get_base_config()
    cfg.use_masked_token_input = True
    cfg.rope_complement_heads = 0
    cfg.masked_token_position_mode = "index"
    cfg.masked_token_attention_mode = "masked_query_to_unmasked_kv"
    cfg.masked_token_loss_weight = 1.0
    cfg.use_ema_teacher_target = True
    cfg.teacher_ema_decay = 0.996
    cfg.grad_clip_norm = 1.0
    cfg.masked_latent_predictor_num_layers = 3
    cfg.wandb_run_name_prefix = "jepa_masked_latent_index"
    return cfg
