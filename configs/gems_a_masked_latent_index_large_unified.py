from ml_collections import config_dict

from configs.gems_a_multi import get_config as get_base_config


def get_config() -> config_dict.ConfigDict:
    cfg = get_base_config()
    cfg.batch_size = 128
    cfg.masked_token_loss_weight = 1.0
    cfg.use_ema_teacher_target = True
    cfg.teacher_ema_decay = 0.99925
    cfg.grad_clip_norm = 0.
    cfg.masked_latent_predictor_num_layers = 4
    cfg.multicrop_local_keep_fraction = 0.5

    cfg.masked_token_loss_type = "l2"
    cfg.num_epochs = 50
    cfg.autocast_dtype = "bf16"
    cfg.learning_rate = 3e-4
    cfg.representation_regularizer = "sigreg"
    cfg.model_dim = 512
    cfg.num_layers = 14
    cfg.num_heads = 16
    cfg.num_kv_heads = 16

    cfg.sigreg_lambda_warmup_steps = 50_000
    cfg.wandb_run_name_prefix = "jepa_masked_latent_index"
    return cfg
