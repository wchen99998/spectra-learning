"""Inline experiment: run msg_probe with a randomly initialized model."""

from __future__ import annotations

import logging
import sys

import torch
from ml_collections import config_dict

from models.model import PeakSetSIGReg
from utils.msg_probe import run_msg_probe

logging.basicConfig(level=logging.INFO, stream=sys.stdout)


def make_config() -> config_dict.ConfigDict:
    cfg = config_dict.ConfigDict()
    cfg.seed = 42

    # Data
    cfg.tfrecord_dir = "data/gems_peaklist_tfrecord_alpha"
    cfg.batch_size = 256
    cfg.shuffle_buffer = 10_000
    cfg.tfrecord_buffer_size = 250_000
    cfg.max_precursor_mz = 1000.0
    cfg.min_peak_intensity = 0.001
    cfg.peak_ordering = "mz"

    # Model
    cfg.num_peaks = 63
    cfg.model_dim = 256
    cfg.num_heads = 8
    cfg.num_layers = 10
    cfg.num_kv_heads = 8
    cfg.encoder_use_rope = True
    cfg.encoder_qk_norm = False
    cfg.attention_mlp_multiple = 4.0
    cfg.feature_mlp_hidden_dim = 128
    cfg.sigreg_num_slices = 256
    cfg.sigreg_lambda = 0.1
    cfg.representation_regularizer = "sigreg"
    cfg.norm_type = "layernorm"
    cfg.use_precursor_token = True
    cfg.masked_token_loss_weight = 0.0
    cfg.masked_token_loss_type = "l1"
    cfg.masked_latent_predictor_num_layers = 2

    # Probe — small/fast for inline test
    cfg.msg_probe_pooling_type = "pma"
    cfg.msg_probe_pma_num_heads = cfg.num_heads
    cfg.msg_probe_pma_num_seeds = 4
    cfg.msg_probe_num_epochs = 2
    cfg.msg_probe_learning_rate = 1e-3
    cfg.msg_probe_weight_decay = 1e-2
    cfg.msg_probe_warmup_steps = 20
    cfg.msg_probe_max_train_samples = 1024
    cfg.msg_probe_max_test_samples = 512
    return cfg


def main() -> None:
    cfg = make_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PeakSetSIGReg(
        model_dim=cfg.model_dim,
        encoder_num_layers=cfg.num_layers,
        encoder_num_heads=cfg.num_heads,
        encoder_num_kv_heads=cfg.num_kv_heads,
        attention_mlp_multiple=cfg.attention_mlp_multiple,
        feature_mlp_hidden_dim=cfg.feature_mlp_hidden_dim,
        encoder_use_rope=cfg.encoder_use_rope,
        encoder_qk_norm=cfg.encoder_qk_norm,
        representation_regularizer=cfg.representation_regularizer,
        sigreg_num_slices=cfg.sigreg_num_slices,
        sigreg_lambda=cfg.sigreg_lambda,
        norm_type=cfg.norm_type,
        use_precursor_token=cfg.use_precursor_token,
        masked_token_loss_weight=cfg.masked_token_loss_weight,
        masked_token_loss_type=cfg.masked_token_loss_type,
        masked_latent_predictor_num_layers=cfg.masked_latent_predictor_num_layers,
    ).to(device)
    model.eval()

    print(f"Model on {device}, params={sum(p.numel() for p in model.parameters()):,}")
    metrics = run_msg_probe(
        config=cfg,
        model=model,
        device=device,
    )
    print("\n=== MSG Probe Results ===")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
