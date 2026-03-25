"""Temporal finetuning config — inherits from the small spatial pretraining config.

Usage:
    python train_temporal.py \
        --config configs/gems_a_temporal_small.py \
        --workdir experiments/temporal_small \
        --pretrained_checkpoint experiments/spatial_run/checkpoints/last.pt
"""

from configs.gems_a_masked_latent_index_small import get_config as _base_config


def get_config():
    cfg = _base_config()

    # ── Dataset: switch to temporal experiment-grouped pipeline ──
    cfg.dataset = "gems_a_temporal"
    cfg.pipeline = "temporal"
    cfg.temporal_repo_id = "cjim8889/gems-a10-grouped"
    cfg.temporal_revision = "main"
    cfg.temporal_data_dir = "data/gems_grouped"

    # DataLoader — native PyTorch, multiple workers OK (no TF)
    cfg.dataloader_num_workers = 4
    cfg.dataloader_prefetch_factor = 2
    cfg.dataloader_pin_memory = True

    # ── Temporal predictor ──
    cfg.temporal_predictor_num_layers = 4
    cfg.encoder_finetune_lr = 3e-5
    cfg.pretrained_checkpoint = None  # set via CLI

    cfg.msg_probe_every_n_steps = 25000
    cfg.num_train_steps = 1_000_000
    # ── Logging ──
    cfg.wandb_run_name_prefix = "jepa_temporal_small"

    return cfg
