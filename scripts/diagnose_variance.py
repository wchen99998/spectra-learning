"""Diagnose per-layer activation variance from a checkpoint.

Usage:
    python scripts/diagnose_variance.py \
        --config configs/gems_a_multi.py \
        --workdir experiments/TEST_LATEST_BLOCK_JEPA
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.model import PeakSetSIGReg


def load_config(config_path: str):
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()


def build_model(cfg) -> PeakSetSIGReg:
    return PeakSetSIGReg(
        num_peaks=cfg.num_peaks,
        model_dim=cfg.model_dim,
        encoder_num_layers=cfg.num_layers,
        encoder_num_heads=cfg.num_heads,
        encoder_num_kv_heads=getattr(cfg, "num_kv_heads", None),
        attention_mlp_multiple=cfg.attention_mlp_multiple,
        feature_mlp_hidden_dim=cfg.feature_mlp_hidden_dim,
        encoder_fourier_strategy=cfg.encoder_fourier_strategy,
        encoder_fourier_x_min=cfg.encoder_fourier_x_min,
        encoder_fourier_x_max=cfg.encoder_fourier_x_max,
        encoder_fourier_funcs=cfg.encoder_fourier_funcs,
        encoder_fourier_num_freqs=cfg.encoder_fourier_num_freqs,
        encoder_fourier_sigma=cfg.encoder_fourier_sigma,
        encoder_fourier_trainable=cfg.encoder_fourier_trainable,
        encoder_use_rope=cfg.encoder_use_rope,
        sigreg_num_slices=cfg.sigreg_num_slices,
        sigreg_lambda=cfg.sigreg_lambda,
        jepa_num_target_blocks=cfg.jepa_num_target_blocks,
        jepa_context_fraction=cfg.jepa_context_fraction,
        jepa_target_fraction=cfg.jepa_target_fraction,
        jepa_block_min_len=cfg.jepa_block_min_len,
        sigreg_mz_jitter_std=cfg.sigreg_mz_jitter_std,
        sigreg_intensity_jitter_std=cfg.sigreg_intensity_jitter_std,
    )


def make_synthetic_batch(
    cfg, batch_size: int = 32, device: str = "cpu"
) -> dict[str, torch.Tensor]:
    """Create a synthetic block-masked JEPA batch for probing activations."""
    N = cfg.num_peaks
    K = cfg.jepa_num_target_blocks
    peak_mz = torch.rand(batch_size, N, device=device).sort(dim=-1).values
    peak_intensity = torch.rand(batch_size, N, device=device)
    valid_mask = torch.rand(batch_size, N, device=device) < 0.8
    context_len = max(1, round(N * float(cfg.jepa_context_fraction)))
    target_len = max(1, round(N * float(cfg.jepa_target_fraction)))
    context_mask = torch.zeros(batch_size, N, device=device, dtype=torch.bool)
    context_mask[:, :context_len] = valid_mask[:, :context_len]
    target_masks = torch.zeros(batch_size, K, N, device=device, dtype=torch.bool)
    for target_idx in range(K):
        start = context_len + (target_idx * target_len)
        end = min(start + target_len, N)
        target_masks[:, target_idx, start:end] = valid_mask[:, start:end]

    return {
        "peak_mz": peak_mz,
        "peak_intensity": peak_intensity,
        "peak_valid_mask": valid_mask,
        "context_mask": context_mask,
        "target_masks": target_masks,
    }


@torch.no_grad()
def diagnose_encoder(model: PeakSetSIGReg, batch: dict[str, torch.Tensor]):
    """Hook into each transformer block and measure residual stream statistics."""
    encoder = model.encoder

    mz = batch["peak_mz"]
    intensity = batch["peak_intensity"]
    valid_mask = batch["peak_valid_mask"]
    visible_mask = batch["context_mask"]

    # Step 1: Embedder output
    x = encoder.embedder(mz, intensity)
    _report("embedder_out", x, visible_mask)

    # Step 2: Per-block activations
    block_mask = None
    if visible_mask is not None:
        from networks.transformer_torch import create_visible_block_mask

        block_mask = create_visible_block_mask(visible_mask)

    for i, block in enumerate(encoder.blocks):
        # Pre-norm values (what the attention/FFN sees)
        attn_normed = block.attention_norm(x)
        _report(f"block_{i:02d}/attn_pre_norm", attn_normed, visible_mask)

        # Attention output (residual contribution)
        attn_out = block.attention(
            attn_normed, freqs_cos=None, freqs_sin=None, block_mask=block_mask
        )
        _report(f"block_{i:02d}/attn_out", attn_out, visible_mask)

        h = x + attn_out
        _report(f"block_{i:02d}/post_attn_residual", h, visible_mask)

        ffn_normed = block.ffn_norm(h)
        _report(f"block_{i:02d}/ffn_pre_norm", ffn_normed, visible_mask)

        ffn_out = block.feed_forward(ffn_normed)
        _report(f"block_{i:02d}/ffn_out", ffn_out, visible_mask)

        x = h + ffn_out
        _report(f"block_{i:02d}/post_ffn_residual", x, visible_mask)

    # Pooling (mean)
    pooled = model.pool(x, visible_mask)
    _report_1d("pool_mean", pooled)


def _report(name: str, x: torch.Tensor, valid_mask: torch.Tensor):
    """Report statistics for [B, N, D] tensor, masking invalid positions."""
    xf = x.float()
    mask = valid_mask.unsqueeze(-1)  # [B, N, 1]
    valid_vals = xf[mask.expand_as(xf)]

    rms = xf.pow(2).mean(dim=-1).sqrt()  # [B, N]
    rms_valid = rms[valid_mask].mean().item()

    var_per_dim = xf[mask.expand_as(xf)].view(-1, xf.shape[-1]).var(dim=0)
    mean_var = var_per_dim.mean().item()

    abs_max = valid_vals.abs().max().item()
    mean_abs = valid_vals.abs().mean().item()

    print(
        f"  {name:40s}  RMS={rms_valid:10.4f}  Var={mean_var:10.4f}  AbsMax={abs_max:10.4f}  MeanAbs={mean_abs:10.4f}"
    )


def _report_1d(name: str, x: torch.Tensor):
    """Report statistics for [B, D] tensor."""
    xf = x.float()
    rms = xf.pow(2).mean(dim=-1).sqrt().mean().item()
    var = xf.var(dim=0).mean().item()
    abs_max = xf.abs().max().item()
    mean_abs = xf.abs().mean().item()
    print(
        f"  {name:40s}  RMS={rms:10.4f}  Var={var:10.4f}  AbsMax={abs_max:10.4f}  MeanAbs={mean_abs:10.4f}"
    )


@torch.no_grad()
def diagnose_norm_weights(model: PeakSetSIGReg):
    """Report RMSNorm weight statistics."""
    print("\n=== RMSNorm weight statistics ===")
    encoder = model.encoder
    for i, block in enumerate(encoder.blocks):
        attn_w = block.attention_norm.weight
        ffn_w = block.ffn_norm.weight
        print(
            f"  block_{i:02d}/attn_norm  mean={attn_w.mean():.4f}  std={attn_w.std():.4f}  min={attn_w.min():.4f}  max={attn_w.max():.4f}"
        )
        print(
            f"  block_{i:02d}/ffn_norm   mean={ffn_w.mean():.4f}  std={ffn_w.std():.4f}  min={ffn_w.min():.4f}  max={ffn_w.max():.4f}"
        )


@torch.no_grad()
def diagnose_attention_weights(model: PeakSetSIGReg):
    """Report attention and FFN weight norms per layer."""
    print("\n=== Weight norms per layer ===")
    for i, block in enumerate(model.encoder.blocks):
        wqkv_norm = block.attention.wqkv.weight.norm().item()
        wo_norm = block.attention.wo.weight.norm().item()
        if hasattr(block.feed_forward, "w12"):
            w12_norm = block.feed_forward.w12.weight.norm().item()
            w_label = f"w12={w12_norm:.4f}"
        else:
            w1_norm = block.feed_forward.w1.weight.norm().item()
            w_label = f"w1={w1_norm:.4f}"
        w2_norm = block.feed_forward.w2.weight.norm().item()
        print(
            f"  block_{i:02d}  wqkv={wqkv_norm:.4f}  wo={wo_norm:.4f}  {w_label}  w2={w2_norm:.4f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--checkpoint", default="last.pt", help="Checkpoint filename")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = build_model(cfg)

    ckpt_path = Path(args.workdir) / "trial_000" / "checkpoints" / args.checkpoint
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.to(args.device)

    print(
        f"\n=== Activation diagnosis (batch_size={args.batch_size}, device={args.device}) ==="
    )
    batch = make_synthetic_batch(cfg, batch_size=args.batch_size, device=args.device)
    diagnose_encoder(model, batch)

    diagnose_norm_weights(model)
    diagnose_attention_weights(model)


if __name__ == "__main__":
    main()
