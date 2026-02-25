"""Diagnose per-layer activation variance from a checkpoint.

Usage:
    python scripts/diagnose_variance.py \
        --config configs/gems_a_multi.py \
        --workdir experiments/TEST_ISAB_LATEST_MULTIVIEWS_5_mha_pma
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
        encoder_use_rope=cfg.encoder_use_rope,
        rope_mz_max=getattr(cfg, "rope_mz_max", 1000.0),
        rope_mz_precision=getattr(cfg, "rope_mz_precision", 0.1),
        rope_complement_heads=getattr(cfg, "rope_complement_heads", None),
        rope_modulo_2pi=getattr(cfg, "rope_modulo_2pi", True),
        encoder_fp16_high_precision_stem=getattr(cfg, "encoder_fp16_high_precision_stem", False),
        sigreg_use_projector=cfg.sigreg_use_projector,
        sigreg_proj_hidden_dim=cfg.sigreg_proj_hidden_dim,
        sigreg_proj_output_dim=cfg.sigreg_proj_output_dim,
        sigreg_proj_norm=cfg.sigreg_proj_norm,
        sigreg_num_slices=cfg.sigreg_num_slices,
        sigreg_lambda=cfg.sigreg_lambda,
        multicrop_num_local_views=cfg.multicrop_num_local_views,
        multicrop_local_keep_fraction=cfg.multicrop_local_keep_fraction,
        sigreg_mz_jitter_std=cfg.sigreg_mz_jitter_std,
        sigreg_intensity_jitter_std=cfg.sigreg_intensity_jitter_std,
        pooling_type=cfg.pooling_type,
        pma_fp16_high_precision=getattr(cfg, "pma_fp16_high_precision", False),
        pma_num_heads=getattr(cfg, "pma_num_heads", None),
        pma_num_seeds=cfg.pma_num_seeds,
        encoder_block_type=cfg.encoder_block_type,
        isab_num_inducing_points=getattr(cfg, "isab_num_inducing_points", 32),
    )


def make_synthetic_batch(cfg, batch_size: int = 32, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Create a realistic synthetic batch for probing activations."""
    N = cfg.num_peaks
    V = 1 + cfg.multicrop_num_local_views
    total_B = V * batch_size

    # Realistic mz values (0-1000 range, normalized)
    peak_mz = torch.rand(total_B, N, device=device).sort(dim=-1).values
    peak_intensity = torch.rand(total_B, N, device=device)
    precursor_mz = torch.rand(total_B, device=device)

    # ~80% valid peaks
    valid_mask = torch.rand(total_B, N, device=device) < 0.8

    return {
        "fused_mz": peak_mz,
        "fused_intensity": peak_intensity,
        "fused_precursor_mz": precursor_mz,
        "fused_valid_mask": valid_mask,
        "fused_masked_positions": torch.zeros(total_B, N, device=device, dtype=torch.bool),
    }


@torch.no_grad()
def diagnose_encoder(model: PeakSetSIGReg, batch: dict[str, torch.Tensor]):
    """Hook into each transformer block and measure residual stream statistics."""
    encoder = model.encoder

    mz = batch["fused_mz"]
    intensity = batch["fused_intensity"]
    valid_mask = batch["fused_valid_mask"]
    precursor_mz = batch["fused_precursor_mz"]

    # Step 1: Embedder output
    x = encoder.embedder(mz, intensity, precursor_mz)
    _report("embedder_out", x, valid_mask)

    # Step 2: Per-block activations
    block_mask = None
    if valid_mask is not None:
        from networks.transformer_torch import create_padding_block_mask
        block_mask = create_padding_block_mask(valid_mask)

    for i, block in enumerate(encoder.blocks):
        # Pre-norm values (what the attention/FFN sees)
        attn_normed = block.attention_norm(x)
        _report(f"block_{i:02d}/attn_pre_norm", attn_normed, valid_mask)

        # Attention output (residual contribution)
        attn_out = block.attention(attn_normed, freqs_cos=None, freqs_sin=None, block_mask=block_mask)
        _report(f"block_{i:02d}/attn_out", attn_out, valid_mask)

        h = x + attn_out
        _report(f"block_{i:02d}/post_attn_residual", h, valid_mask)

        ffn_normed = block.ffn_norm(h)
        _report(f"block_{i:02d}/ffn_pre_norm", ffn_normed, valid_mask)

        ffn_out = block.feed_forward(ffn_normed)
        _report(f"block_{i:02d}/ffn_out", ffn_out, valid_mask)

        x = h + ffn_out
        _report(f"block_{i:02d}/post_ffn_residual", x, valid_mask)

    # Final norm
    final_out = encoder.final_norm(x)
    _report("final_norm_out", final_out, valid_mask)

    # Pooling
    pooled, pooled_raw = model.pool_with_raw(final_out, valid_mask)
    _report_1d("pool_raw", pooled_raw)
    _report_1d("pool_normed", pooled)

    # Projector
    proj = model.projector(pooled)
    _report_1d("projector_out", proj)


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

    print(f"  {name:40s}  RMS={rms_valid:10.4f}  Var={mean_var:10.4f}  AbsMax={abs_max:10.4f}  MeanAbs={mean_abs:10.4f}")


def _report_1d(name: str, x: torch.Tensor):
    """Report statistics for [B, D] tensor."""
    xf = x.float()
    rms = xf.pow(2).mean(dim=-1).sqrt().mean().item()
    var = xf.var(dim=0).mean().item()
    abs_max = xf.abs().max().item()
    mean_abs = xf.abs().mean().item()
    print(f"  {name:40s}  RMS={rms:10.4f}  Var={var:10.4f}  AbsMax={abs_max:10.4f}  MeanAbs={mean_abs:10.4f}")


@torch.no_grad()
def diagnose_norm_weights(model: PeakSetSIGReg):
    """Report RMSNorm weight statistics."""
    print("\n=== RMSNorm weight statistics ===")
    encoder = model.encoder
    for i, block in enumerate(encoder.blocks):
        attn_w = block.attention_norm.weight
        ffn_w = block.ffn_norm.weight
        print(f"  block_{i:02d}/attn_norm  mean={attn_w.mean():.4f}  std={attn_w.std():.4f}  min={attn_w.min():.4f}  max={attn_w.max():.4f}")
        print(f"  block_{i:02d}/ffn_norm   mean={ffn_w.mean():.4f}  std={ffn_w.std():.4f}  min={ffn_w.min():.4f}  max={ffn_w.max():.4f}")
    final_w = encoder.final_norm.weight
    print(f"  final_norm            mean={final_w.mean():.4f}  std={final_w.std():.4f}  min={final_w.min():.4f}  max={final_w.max():.4f}")
    pool_w = model.pool_norm.weight
    print(f"  pool_norm             mean={pool_w.mean():.4f}  std={pool_w.std():.4f}  min={pool_w.min():.4f}  max={pool_w.max():.4f}")


@torch.no_grad()
def diagnose_attention_weights(model: PeakSetSIGReg):
    """Report attention and FFN weight norms per layer."""
    print("\n=== Weight norms per layer ===")
    for i, block in enumerate(model.encoder.blocks):
        wqkv_norm = block.attention.wqkv.weight.norm().item()
        wo_norm = block.attention.wo.weight.norm().item()
        if hasattr(block.feed_forward, 'w12'):
            w12_norm = block.feed_forward.w12.weight.norm().item()
            w_label = f"w12={w12_norm:.4f}"
        else:
            w1_norm = block.feed_forward.w1.weight.norm().item()
            w_label = f"w1={w1_norm:.4f}"
        w2_norm = block.feed_forward.w2.weight.norm().item()
        print(f"  block_{i:02d}  wqkv={wqkv_norm:.4f}  wo={wo_norm:.4f}  {w_label}  w2={w2_norm:.4f}")


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

    print(f"\n=== Activation diagnosis (batch_size={args.batch_size}, device={args.device}) ===")
    batch = make_synthetic_batch(cfg, batch_size=args.batch_size, device=args.device)
    diagnose_encoder(model, batch)

    diagnose_norm_weights(model)
    diagnose_attention_weights(model)


if __name__ == "__main__":
    main()
