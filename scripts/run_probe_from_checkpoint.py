"""Run MSG probe from a saved checkpoint."""

from __future__ import annotations

import logging
import sys

import torch

logging.basicConfig(level=logging.INFO, stream=sys.stdout)


def main() -> None:
    from configs.gems_a_masked_latent_index_small import get_config
    from models.model import PeakSetSIGReg
    from utils.msg_probe import run_msg_probe

    cfg = get_config()
    # Override probe pooling
    cfg.msg_probe_pma_num_seeds = 64
    cfg.msg_probe_pma_num_heads = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PeakSetSIGReg(
        model_dim=cfg.model_dim,
        encoder_num_layers=cfg.num_layers,
        encoder_num_heads=cfg.num_heads,
        encoder_num_kv_heads=cfg.num_kv_heads,
        attention_mlp_multiple=cfg.attention_mlp_multiple,
        feature_mlp_hidden_dim=cfg.feature_mlp_hidden_dim,
        encoder_fourier_strategy=cfg.encoder_fourier_strategy,
        encoder_fourier_x_min=cfg.encoder_fourier_x_min,
        encoder_fourier_x_max=cfg.encoder_fourier_x_max,
        encoder_fourier_funcs=cfg.encoder_fourier_funcs,
        encoder_fourier_num_freqs=cfg.encoder_fourier_num_freqs,
        encoder_fourier_sigma=cfg.encoder_fourier_sigma,
        encoder_fourier_trainable=cfg.encoder_fourier_trainable,
        encoder_qk_norm=cfg.encoder_qk_norm,
        representation_regularizer=cfg.representation_regularizer,
        sigreg_num_slices=cfg.sigreg_num_slices,
        sigreg_lambda=cfg.sigreg_lambda,
        norm_type=cfg.norm_type,
        use_precursor_token=cfg.use_precursor_token,
        masked_token_loss_weight=cfg.masked_token_loss_weight,
        masked_token_loss_type=cfg.masked_token_loss_type,
        jepa_target_normalization=cfg.get("jepa_target_normalization", "none"),
        masked_latent_predictor_num_layers=cfg.masked_latent_predictor_num_layers,
    )

    ckpt_path = "experiments/TEST_MASKEDJEPA_small_sigreg_nist20_correct_bounded/trial_000/checkpoints/step-00525000.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = {
        k: v for k, v in ckpt["model"].items()
        if not k.startswith("teacher_encoder")
    }
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    encoder = sum(p.numel() for p in model.encoder.parameters())
    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path} (step {ckpt['global_step']})")
    print(f"Params: {total:,} total, {encoder:,} encoder")
    print(f"Probe: PMA seeds={cfg.msg_probe_pma_num_seeds}, heads={cfg.msg_probe_pma_num_heads}")
    print(f"Probe epochs: {cfg.msg_probe_num_epochs}")

    metrics = run_msg_probe(config=cfg, model=model, device=device)

    print("\n=== MSG Probe Results ===")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
