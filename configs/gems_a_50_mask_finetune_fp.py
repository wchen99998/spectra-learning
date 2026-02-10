from configs.gems_a_50_mask_finetune import get_config as _get_finetune_config


def get_config():
    cfg = _get_finetune_config()

    # Pretrained checkpoint path (override via --checkpoint CLI arg)
    cfg.finetune_checkpoint = ""

    # Feature extraction point: "encoder" (768-d) or "projector" (128-d)
    cfg.finetune_feature_source = "encoder"

    # Backbone freezing
    cfg.finetune_freeze_backbone = True

    # Prediction head
    cfg.finetune_head_hidden_dim = 512

    # Optimizer
    cfg.finetune_learning_rate = 1e-4
    cfg.finetune_weight_decay = 1e-4

    # Schedule
    cfg.finetune_num_epochs = 10
    cfg.warmup_steps = 500

    # Target
    cfg.fingerprint_bits = 1024

    # Logging
    cfg.wandb_project = "token-mass-spec-finetuning"
    cfg.wandb_run_name_prefix = "finetune_fp"

    return cfg
