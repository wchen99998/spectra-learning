from ml_collections import config_dict

from configs.gems_a_multi import get_config as get_base_config


def get_config() -> config_dict.ConfigDict:
    cfg = get_base_config()
    cfg.batch_size = 128
    cfg.masked_token_loss_weight = 1.0
    cfg.use_ema_teacher_target = True
    cfg.teacher_ema_decay = 0.99925
    cfg.teacher_ema_decay_start = 0.98
    cfg.teacher_ema_decay_warmup_steps = 100_000
    cfg.grad_clip_norm = 0.
    cfg.masked_latent_predictor_num_layers = 2
    cfg.multicrop_local_keep_fraction = 0.25

    cfg.encoder_qk_norm = False
    cfg.masked_token_loss_type = "cosine"
    cfg.norm_type = "layernorm"
    cfg.normalize_jepa_targets = True
    cfg.num_epochs = 5
    cfg.autocast_dtype = "bf16"
    cfg.learning_rate = 3e-4
    cfg.representation_regularizer = "gco"
    cfg.gco_std_target = 0.9
    cfg.gco_log_lambda_min = -7.0
    cfg.sigreg_lambda_warmup_steps = 0
    cfg.model_dim = 1024
    cfg.num_layers = 16
    cfg.num_heads = 16
    cfg.num_kv_heads = 16

    cfg.msg_linear_probe_every_n_steps = 100_000
    cfg.sigreg_mz_jitter_std = 0.005
    cfg.sigreg_intensity_jitter_std = 0.05

    cfg.warmup_steps = 10_000
    cfg.optimizer = "muon"

    cfg.wandb_run_name_prefix = "jepa_masked_latent_index"
    return cfg
