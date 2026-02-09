from configs.gems_a_50_mask import get_config as _get_pretrain_config


def get_config():
    cfg = _get_pretrain_config()

    # Finetune-specific task settings
    cfg.fingerprint_bits = 1024
    cfg.use_massspec_metadata = True
    cfg.metadata_hidden_dim = 128
    cfg.metadata_dropout = 0.0

    # Finetune optimizer defaults (used by finetune.py when present)
    cfg.finetune_learning_rate = 1e-4
    cfg.finetune_weight_decay = 1e-4

    # Finetune schedule / eval cadence
    cfg.num_epochs = 20
    cfg.val_check_interval = 1.0
    cfg.checkpoint_every_steps = 2000

    # Logging
    cfg.wandb_project = "token-mass-spec-finetuning"
    cfg.wandb_run_name_prefix = "gems_a_50_mask_finetune"

    return cfg
