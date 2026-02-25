from ml_collections import config_dict

from configs.gems_a_multi import get_config as get_base_config


def get_config() -> config_dict.ConfigDict:
    cfg = get_base_config()
    cfg.multicrop_keep_masked_tokens = True
    cfg.use_masked_token_input = True
    cfg.masked_token_position_mode = "index"
    cfg.masked_token_attention_mode = "masked_query_to_unmasked_kv"
    cfg.masked_token_loss_weight = 1.0
    cfg.masked_latent_normalize = True
    cfg.grad_clip_norm = 1.0
    cfg.dataloader_num_workers = 0
    cfg.dataloader_persistent_workers = False
    cfg.wandb_run_name_prefix = "jepa_masked_latent_index"
    return cfg
