"""Evaluate SigReg projection-count sensitivity on a fixed checkpoint."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from spectra_learning.models.losses import SlotwiseSIGReg
from spectra_learning.models.encoder import PeakSetEncoder
from spectra_learning.probes.massspec.data import MassSpecProbeData
from spectra_learning.probes.massspec.msg_probe import iter_massspec_probe
from spectra_learning.training.api import build_model_from_config, latest_ckpt_path, load_config, load_pretrained_weights


def _parse_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def _device(name: str) -> torch.device:
    if name == "cuda":
        return torch.device("cuda")
    return torch.device("cpu")


def _sample_directions(
    dim: int,
    slices: int,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    directions = torch.randn(dim, slices, generator=generator, device=device, dtype=dtype)
    return directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)


def _trapezoid_quadrature(knots: int, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.linspace(0, 3, knots, device=device, dtype=dtype)
    dt = 3 / (knots - 1)
    weights = torch.full((knots,), 2 * dt, device=device, dtype=dtype)
    weights[[0, -1]] = dt
    window = torch.exp(-t.square() / 2.0)
    return t, weights * window


def _simpson_quadrature(knots: int, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.linspace(0, 3, knots, device=device, dtype=dtype)
    dt = 3 / (knots - 1)
    coeff = torch.ones(knots, device=device, dtype=dtype)
    coeff[1:-1:2] = 4
    coeff[2:-1:2] = 2
    window = torch.exp(-t.square() / 2.0)
    return t, (2 * dt / 3) * coeff * window


def _gauss_legendre_quadrature(knots: int, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    nodes, weights = np.polynomial.legendre.leggauss(knots)
    t_np = 1.5 * (nodes + 1.0)
    w_np = 3.0 * weights
    t = torch.as_tensor(t_np, device=device, dtype=dtype)
    w = torch.as_tensor(w_np, device=device, dtype=dtype)
    window = torch.exp(-t.square() / 2.0)
    return t, w * window


def _quadrature(rule: str, knots: int, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    if rule == "trapezoid":
        return _trapezoid_quadrature(knots, device=device, dtype=dtype)
    if rule == "simpson":
        return _simpson_quadrature(knots, device=device, dtype=dtype)
    if rule == "gauss-legendre":
        return _gauss_legendre_quadrature(knots, device=device, dtype=dtype)
    raise ValueError(f"Unknown quadrature rule: {rule}")


def _sigreg_loss(
    reps: torch.Tensor,
    mask: torch.Tensor,
    directions: torch.Tensor,
    *,
    t: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if t is None or weights is None:
        sigreg = SlotwiseSIGReg(num_slices=int(directions.shape[1])).to(reps.device)
        return sigreg(reps, valid_mask=mask, directions=directions)
    return _sigreg_loss_with_quadrature(reps, mask, directions, t=t, weights=weights)


def _sigreg_loss_with_quadrature(
    reps: torch.Tensor,
    mask: torch.Tensor,
    directions: torch.Tensor,
    *,
    t: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    batch_size, num_views, num_slots, dim = reps.unsqueeze(1).shape
    slot_proj = reps.reshape(batch_size * num_views, num_slots, dim)
    projected = torch.einsum("bld,ds->bls", slot_proj, directions)
    x_t = projected.unsqueeze(-1) * t
    slot_mask = mask.reshape(batch_size * num_views, num_slots).to(
        dtype=reps.dtype,
        device=reps.device,
    )
    sample_count = slot_mask.sum(0)
    safe_count = sample_count.clamp_min(1.0)
    weight_view = slot_mask.unsqueeze(-1).unsqueeze(-1)
    cos_mean = (x_t.cos() * weight_view).sum(0) / safe_count[:, None, None]
    sin_mean = (x_t.sin() * weight_view).sum(0) / safe_count[:, None, None]
    phi = torch.exp(-t.square() / 2.0)
    err = (cos_mean - phi).square() + sin_mean.square()
    statistic = err @ weights
    return (statistic.mean(-1) * sample_count).sum()


def _sigreg_loss_batched(
    reps: torch.Tensor,
    mask: torch.Tensor,
    directions: torch.Tensor,
    *,
    batch_size: int,
    t: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if t is None or weights is None:
        sigreg = SlotwiseSIGReg(num_slices=int(directions.shape[1])).to(reps.device)
        t = sigreg.t.to(device=reps.device, dtype=reps.dtype)
        weights = sigreg.weights.to(device=reps.device, dtype=reps.dtype)
    else:
        t = t.to(device=reps.device, dtype=reps.dtype)
        weights = weights.to(device=reps.device, dtype=reps.dtype)
    phi = torch.exp(-t.square() / 2.0)
    slots = int(reps.shape[1])
    slices = int(directions.shape[1])
    cos_sum = torch.zeros(slots, slices, t.numel(), device=reps.device, dtype=reps.dtype)
    sin_sum = torch.zeros_like(cos_sum)
    sample_count = torch.zeros(slots, device=reps.device, dtype=reps.dtype)
    for start in range(0, int(reps.shape[0]), batch_size):
        x = reps[start : start + batch_size]
        m = mask[start : start + batch_size].to(dtype=reps.dtype)
        projected = torch.einsum("bld,ds->bls", x, directions)
        x_t = projected.unsqueeze(-1) * t
        weight_view = m.unsqueeze(-1).unsqueeze(-1)
        cos_sum += (x_t.cos() * weight_view).sum(0)
        sin_sum += (x_t.sin() * weight_view).sum(0)
        sample_count += m.sum(0)
    safe_count = sample_count.clamp_min(1.0)
    cos_mean = cos_sum / safe_count[:, None, None]
    sin_mean = sin_sum / safe_count[:, None, None]
    err = (cos_mean - phi).square() + sin_mean.square()
    statistic = err @ weights
    return (statistic.mean(-1) * sample_count).sum()


def _grad_for_directions(
    reps: torch.Tensor,
    mask: torch.Tensor,
    directions: torch.Tensor,
    *,
    t: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    x = reps.detach().clone().requires_grad_(True)
    loss = _sigreg_loss(x, mask, directions, t=t, weights=weights)
    loss.backward()
    return x.grad.detach().flatten()


def _extract_representations(
    *,
    config_path: Path,
    checkpoint_path: Path,
    split: str,
    max_samples: int,
    device: torch.device,
    encoder_precision: str = "fp32",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    config = load_config(config_path)
    data = MassSpecProbeData.from_config(config)
    model = build_model_from_config(config)
    load_pretrained_weights(model, str(checkpoint_path))
    model.to(device).eval()

    reps: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    seen = 0
    started = time.time()
    use_bf16_autocast = encoder_precision in ("bf16", "bf16_embedder_fp32")
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if device.type == "cuda" and use_bf16_autocast
        else nullcontext()
    )
    with torch.no_grad():
        for batch in iter_massspec_probe(
            data,
            split,
            seed=int(config.seed),
            peak_ordering=str(config.get("peak_ordering", "intensity")),
            drop_remainder=False,
            max_samples=max_samples,
        ):
            peak_mz = batch["peak_mz"].to(device)
            peak_intensity = batch["peak_intensity"].to(device)
            peak_valid_mask = batch["peak_valid_mask"].to(device)
            precursor_mz = batch.get("precursor_mz", None)
            if precursor_mz is not None:
                precursor_mz = precursor_mz.to(device)
            if encoder_precision == "bf16_embedder_fp32":
                with autocast_ctx:
                    embeddings = _encoder_forward_embedder_fp32(
                        model.encoder,
                        peak_mz,
                        peak_intensity,
                        peak_valid_mask,
                        precursor_mz=precursor_mz,
                    )
            else:
                with autocast_ctx:
                    embeddings = model.encoder(
                        peak_mz,
                        peak_intensity,
                        valid_mask=peak_valid_mask,
                        precursor_mz=precursor_mz,
                    )
            peak_embeddings, _ = model.encoder.split_peak_and_cls(embeddings)
            take = min(int(peak_embeddings.shape[0]), max_samples - seen)
            reps.append(peak_embeddings[:take].detach().cpu().float())
            masks.append(peak_valid_mask[:take].detach().cpu())
            seen += take
            if seen >= max_samples:
                break
    rep = torch.cat(reps, dim=0).float()
    mask = torch.cat(masks, dim=0).bool()
    meta = {
        "samples": int(rep.shape[0]),
        "slots": int(rep.shape[1]),
        "dim": int(rep.shape[2]),
        "valid_tokens": int(mask.sum().item()),
        "extract_seconds": time.time() - started,
        "checkpoint_sigreg_num_slices": int(config.get("sigreg_num_slices", 256)),
        "checkpoint_sigreg_lambda": float(config.get("sigreg_lambda", 0.0)),
        "representation_regularizer": str(config.get("representation_regularizer", "none")),
        "split": split,
        "encoder_precision": encoder_precision,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return rep, mask, meta


def _encoder_forward_embedder_fp32(
    encoder: PeakSetEncoder,
    peak_mz: torch.Tensor,
    peak_intensity: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    block_indices: tuple[int, ...] = (),
    precursor_mz: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    with torch.autocast(device_type=peak_mz.device.type, enabled=False):
        x = encoder._add_positions(
            encoder.embedder(peak_mz.float(), peak_intensity.float())
        )
    seq_len = peak_mz.shape[1]
    selected = set(block_indices)
    selected_peak_outputs: dict[int, torch.Tensor] = {}
    special_len = int(encoder.use_cls_token) + encoder.num_register_tokens
    x, attn_mask = encoder._append_special_tokens(x, valid_mask)
    from spectra_learning.models.transformer import create_visible_attention_mask

    attn_mask = create_visible_attention_mask(attn_mask)
    for block_idx, block in enumerate(encoder.blocks, start=1):
        attn_bias = encoder._spectral_attn_bias(
            block_idx - 1,
            peak_mz,
            peak_intensity,
            precursor_mz,
            num_special_tokens=special_len,
        )
        x = block(x, attn_mask=attn_mask, attn_bias=attn_bias)
        if block_idx in selected and block_idx != encoder.num_layers:
            selected_peak_outputs[block_idx] = x[:, :seq_len]
    x = encoder.final_norm(x)
    if encoder.num_layers in selected:
        selected_peak_outputs[encoder.num_layers] = x[:, :seq_len]
    peak_x = x[:, :seq_len]
    if encoder.use_cls_token:
        cls_x = x[:, seq_len]
        output = torch.cat([peak_x, cls_x.unsqueeze(1)], dim=1)
    else:
        output = peak_x
    if block_indices:
        return output, [selected_peak_outputs[idx] for idx in block_indices]
    return output


def _geometry(reps: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    x = reps[mask]
    norms = x.norm(dim=-1)
    x = x - x.mean(dim=0, keepdim=True)
    cov = x.T @ x / float(x.shape[0])
    eig = torch.linalg.eigvalsh(cov).clamp_min(0).flip(0)
    eig_sum = eig.sum()
    participation = eig_sum.square() / eig.square().sum().clamp_min(1e-12)
    return {
        "mean_token_norm": float(norms.mean()),
        "std_token_norm": float(norms.std()),
        "cov_trace": float(eig_sum),
        "participation_ratio": float(participation),
        "effective_rank_fraction": float(participation / eig.numel()),
        "top1_variance_fraction": float(eig[0] / eig_sum),
        "top8_variance_fraction": float(eig[:8].sum() / eig_sum),
        "top32_variance_fraction": float(eig[:32].sum() / eig_sum),
        "condition_top_to_median": float(eig[0] / eig[eig.numel() // 2].clamp_min(1e-12)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot(results_dir: Path, rows: list[dict[str, Any]], checkpoint_slices: int) -> None:
    slices = np.asarray([row["num_slices"] for row in rows], dtype=np.float64)
    rel_error = np.asarray([row["prefix_loss_rel_error_vs_ref"] for row in rows], dtype=np.float64)
    cv = np.asarray([row["loss_cv"] for row in rows], dtype=np.float64)
    grad_cos = np.asarray([row["grad_cosine_vs_ref"] for row in rows], dtype=np.float64)
    ms = np.asarray([row["mean_eval_ms"] for row in rows], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax in axes.flat:
        ax.set_xscale("log", base=2)
        ax.axvline(checkpoint_slices, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax.grid(alpha=0.3)
        ax.set_xlabel("SigReg projections")
    axes[0, 0].plot(slices, rel_error, marker="o")
    axes[0, 0].set_ylabel("abs relative loss error vs reference")
    axes[0, 0].set_title("High-slice convergence")
    axes[0, 1].plot(slices, cv, marker="o", color="#b45309")
    axes[0, 1].set_ylabel("loss coefficient of variation")
    axes[0, 1].set_title("Monte Carlo estimator variance")
    axes[1, 0].plot(slices, grad_cos, marker="o", color="#047857")
    axes[1, 0].set_ylim(0, 1.02)
    axes[1, 0].set_ylabel("gradient cosine vs reference")
    axes[1, 0].set_title("Optimization signal alignment")
    axes[1, 1].plot(slices, ms, marker="o", color="#7c3aed")
    axes[1, 1].set_ylabel("mean eval time (ms)")
    axes[1, 1].set_title("Cost")
    fig.suptitle("SigReg projection-count sensitivity on fixed encoder representations")
    fig.savefig(results_dir / "sigreg_projection_sweep.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    ax.set_xscale("log", base=2)
    ax.plot(slices, grad_cos, marker="o", label="gradient cosine")
    ax.plot(slices, 1.0 - rel_error, marker="o", label="1 - relative loss error")
    ax.axvline(checkpoint_slices, color="black", linestyle="--", linewidth=1, alpha=0.5, label="checkpoint K")
    ax.set_xlabel("SigReg projections")
    ax.set_ylabel("agreement with 4096-projection reference")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Projection count controls gradient fidelity faster than loss scale")
    fig.savefig(results_dir / "sigreg_projection_alignment.png", dpi=180)
    plt.close(fig)


def _plot_quadrature(results_dir: Path, rows: list[dict[str, Any]]) -> None:
    labels = [row["name"] for row in rows]
    rel_error = np.asarray([row["loss_rel_error_vs_ref"] for row in rows], dtype=np.float64)
    grad_cos = np.asarray([row["grad_cosine_vs_ref"] for row in rows], dtype=np.float64)
    ms = np.asarray([row["mean_eval_ms"] for row in rows], dtype=np.float64)
    x = np.arange(len(rows))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    axes[0].bar(x, rel_error, color="#2563eb")
    axes[0].set_ylabel("abs relative loss error")
    axes[0].set_title("Loss scale vs GL-129")
    axes[1].bar(x, grad_cos, color="#047857")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_ylabel("gradient cosine")
    axes[1].set_title("Gradient alignment")
    axes[2].bar(x, ms, color="#7c3aed")
    axes[2].set_ylabel("mean eval time (ms)")
    axes[2].set_title("Cost")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("SigReg quadrature-rule sensitivity at fixed projection directions")
    fig.savefig(results_dir / "sigreg_quadrature_sweep.png", dpi=180)
    plt.close(fig)


def _plot_dtype(results_dir: Path, rows: list[dict[str, Any]], rep_metrics: dict[str, float]) -> None:
    labels = [row["name"] for row in rows]
    loss_error = np.asarray([row["loss_rel_error_vs_fp32"] for row in rows], dtype=np.float64)
    grad_cos = np.asarray([row["grad_cosine_vs_fp32"] for row in rows], dtype=np.float64)
    ms = np.asarray([row["mean_eval_ms"] for row in rows], dtype=np.float64)
    x = np.arange(len(rows))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    axes[0].bar(x, loss_error, color="#2563eb")
    axes[0].set_ylabel("abs relative loss error")
    axes[0].set_title("Loss drift vs fp32")
    axes[1].bar(x, grad_cos, color="#047857")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_ylabel("gradient cosine")
    axes[1].set_title("Gradient alignment")
    axes[2].bar(x, ms, color="#7c3aed")
    axes[2].set_ylabel("mean eval time (ms)")
    axes[2].set_title("SigReg cost")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(
        "bf16 vs fp32: "
        f"mixed token cosine mean={rep_metrics['mixed_token_cosine_mean']:.6f}, "
        f"mixed relative RMSE={rep_metrics['mixed_relative_rmse']:.6f}"
    )
    fig.savefig(results_dir / "sigreg_dtype_sweep.png", dpi=180)
    plt.close(fig)


def _plot_layerwise_dtype(results_dir: Path, rows: list[dict[str, Any]]) -> None:
    labels = [row["stage"] for row in rows]
    rel = np.asarray([row["relative_rmse"] for row in rows], dtype=np.float64)
    cos = np.asarray([row["token_cosine_mean"] for row in rows], dtype=np.float64)
    x = np.arange(len(rows))
    fig, ax1 = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    ax2 = ax1.twinx()
    ax1.plot(x, rel, marker="o", color="#b45309", label="relative RMSE")
    ax2.plot(x, cos, marker="o", color="#047857", label="mean token cosine")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right")
    ax1.set_ylabel("relative RMSE")
    ax2.set_ylabel("mean token cosine")
    ax2.set_ylim(0, 1.02)
    ax1.grid(alpha=0.3)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="center left")
    ax1.set_title("Layerwise encoder drift under bf16 autocast")
    fig.savefig(results_dir / "encoder_dtype_layerwise.png", dpi=180)
    plt.close(fig)


def _compare_representations(
    fp32_reps: torch.Tensor,
    bf16_reps: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    x = fp32_reps[mask]
    y = bf16_reps[mask]
    diff = y - x
    token_cos = torch.nn.functional.cosine_similarity(x, y, dim=-1)
    return {
        "mean_abs_error": float(diff.abs().mean()),
        "max_abs_error": float(diff.abs().max()),
        "rmse": float(diff.square().mean().sqrt()),
        "relative_rmse": float(diff.norm() / x.norm()),
        "token_cosine_mean": float(token_cos.mean()),
        "token_cosine_std": float(token_cos.std()),
        "token_cosine_p01": float(torch.quantile(token_cos, 0.01)),
        "token_cosine_min": float(token_cos.min()),
    }


def _layer_metric_row(
    name: str,
    fp32: torch.Tensor,
    bf16: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, Any]:
    x = fp32[mask].float()
    y = bf16[mask].float()
    diff = y - x
    token_cos = torch.nn.functional.cosine_similarity(x, y, dim=-1)
    return {
        "stage": name,
        "relative_rmse": float(diff.norm() / x.norm()),
        "rmse": float(diff.square().mean().sqrt()),
        "token_cosine_mean": float(token_cos.mean()),
        "token_cosine_p01": float(torch.quantile(token_cos, 0.01)),
        "token_cosine_min": float(token_cos.min()),
    }


def _evaluate_layerwise_dtype(
    *,
    config_path: Path,
    checkpoint_path: Path,
    split: str,
    max_samples: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    config = load_config(config_path)
    data = MassSpecProbeData.from_config(config)
    model = build_model_from_config(config)
    load_pretrained_weights(model, str(checkpoint_path))
    model.to(device).eval()
    batches: list[dict[str, torch.Tensor]] = []
    seen = 0
    for batch in iter_massspec_probe(
        data,
        split,
        seed=int(config.seed),
        peak_ordering=str(config.get("peak_ordering", "intensity")),
        drop_remainder=False,
        max_samples=max_samples,
    ):
        take = min(int(batch["peak_mz"].shape[0]), max_samples - seen)
        entry = {
            "peak_mz": batch["peak_mz"][:take].to(device),
            "peak_intensity": batch["peak_intensity"][:take].to(device),
            "peak_valid_mask": batch["peak_valid_mask"][:take].to(device),
        }
        if "precursor_mz" in batch:
            entry["precursor_mz"] = batch["precursor_mz"][:take].to(device)
        batches.append(entry)
        seen += take
        if seen >= max_samples:
            break
    peak_mz = torch.cat([batch["peak_mz"] for batch in batches], dim=0)
    peak_intensity = torch.cat([batch["peak_intensity"] for batch in batches], dim=0)
    valid_mask = torch.cat([batch["peak_valid_mask"] for batch in batches], dim=0).bool()
    precursor_mz = (
        torch.cat([batch["precursor_mz"] for batch in batches], dim=0)
        if "precursor_mz" in batches[0]
        else None
    )
    encoder = model.encoder
    block_indices = tuple(
        idx
        for idx in (1, 3, 6, 9, int(config.encoder_num_layers))
        if idx <= int(config.encoder_num_layers)
    )
    with torch.no_grad():
        fp32_embed = encoder._add_positions(encoder.embedder(peak_mz, peak_intensity))
        fp32_out, fp32_blocks = encoder.forward_with_block_outputs(
            peak_mz,
            peak_intensity,
            valid_mask=valid_mask,
            block_indices=block_indices,
            precursor_mz=precursor_mz,
        )
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            bf16_embed = encoder._add_positions(encoder.embedder(peak_mz, peak_intensity))
            bf16_out, bf16_blocks = encoder.forward_with_block_outputs(
                peak_mz,
                peak_intensity,
                valid_mask=valid_mask,
                block_indices=block_indices,
                precursor_mz=precursor_mz,
            )
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            mixed_out, mixed_blocks = _encoder_forward_embedder_fp32(
                encoder,
                peak_mz,
                peak_intensity,
                valid_mask,
                precursor_mz=precursor_mz,
                block_indices=block_indices,
            )
    rows = [_layer_metric_row("embedder", fp32_embed, bf16_embed, valid_mask)]
    for idx, fp32_block, bf16_block in zip(block_indices, fp32_blocks, bf16_blocks):
        rows.append(_layer_metric_row(f"block-{idx}", fp32_block, bf16_block, valid_mask))
    rows.append(_layer_metric_row("final", fp32_out[:, :-1], bf16_out[:, :-1], valid_mask))
    rows.append(_layer_metric_row("mixed/block-1", fp32_blocks[0], mixed_blocks[0], valid_mask))
    rows.append(_layer_metric_row("mixed/final", fp32_out[:, :-1], mixed_out[:, :-1], valid_mask))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return rows


def _evaluate_dtype(
    *,
    fp32_reps: torch.Tensor,
    bf16_encoder_reps: torch.Tensor,
    mixed_encoder_reps: torch.Tensor,
    mask: torch.Tensor,
    grad_count: int,
    directions: torch.Tensor,
    eval_batch_size: int,
    repeats: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rep_metrics = _compare_representations(fp32_reps, bf16_encoder_reps, mask)
    mixed_rep_metrics = _compare_representations(fp32_reps, mixed_encoder_reps, mask)
    grad_mask = mask[:grad_count]
    t_fp32, w_fp32 = _trapezoid_quadrature(
        17,
        device=fp32_reps.device,
        dtype=torch.float32,
    )
    t_bf16, w_bf16 = _trapezoid_quadrature(
        17,
        device=fp32_reps.device,
        dtype=torch.bfloat16,
    )
    baseline_loss = float(
        _sigreg_loss_batched(
            fp32_reps,
            mask,
            directions,
            batch_size=eval_batch_size,
            t=t_fp32,
            weights=w_fp32,
        )
        .detach()
        .cpu()
    )
    baseline_grad = _grad_for_directions(
        fp32_reps[:grad_count],
        grad_mask,
        directions,
        t=t_fp32,
        weights=w_fp32,
    ).float()

    variants = [
        ("fp32-encoder fp32-sigreg", fp32_reps, directions.float(), t_fp32, w_fp32),
        ("bf16-encoder fp32-sigreg", bf16_encoder_reps, directions.float(), t_fp32, w_fp32),
        ("mixed-encoder fp32-sigreg", mixed_encoder_reps, directions.float(), t_fp32, w_fp32),
        (
            "fp32-encoder bf16-sigreg",
            fp32_reps.to(torch.bfloat16),
            directions.to(torch.bfloat16),
            t_bf16,
            w_bf16,
        ),
        (
            "bf16-encoder bf16-sigreg",
            bf16_encoder_reps.to(torch.bfloat16),
            directions.to(torch.bfloat16),
            t_bf16,
            w_bf16,
        ),
        (
            "mixed-encoder bf16-sigreg",
            mixed_encoder_reps.to(torch.bfloat16),
            directions.to(torch.bfloat16),
            t_bf16,
            w_bf16,
        ),
    ]
    rows: list[dict[str, Any]] = []
    for name, reps, dirs, t, w in variants:
        loss = float(
            _sigreg_loss_batched(
                reps,
                mask,
                dirs,
                batch_size=eval_batch_size,
                t=t,
                weights=w,
            )
            .detach()
            .float()
            .cpu()
        )
        grad = _grad_for_directions(
            reps[:grad_count],
            grad_mask,
            dirs,
            t=t,
            weights=w,
        ).float()
        grad_cosine = torch.nn.functional.cosine_similarity(grad, baseline_grad, dim=0).item()
        times: list[float] = []
        for _ in range(repeats):
            if fp32_reps.device.type == "cuda":
                torch.cuda.synchronize()
            started = time.time()
            _ = _sigreg_loss_batched(
                reps,
                mask,
                dirs,
                batch_size=eval_batch_size,
                t=t,
                weights=w,
            )
            if fp32_reps.device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.time() - started) * 1000.0)
        row = {
            "name": name,
            "loss": loss,
            "baseline_loss": baseline_loss,
            "loss_rel_error_vs_fp32": abs(loss - baseline_loss) / abs(baseline_loss),
            "grad_cosine_vs_fp32": grad_cosine,
            "mean_eval_ms": float(np.mean(times)),
            "std_eval_ms": float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
        }
        rows.append(row)
        print(
            f"D={name:<27s} rel_err={row['loss_rel_error_vs_fp32']:.5f} "
            f"grad_cos={grad_cosine:.5f} ms={row['mean_eval_ms']:.1f}",
            flush=True,
        )
    rep_metrics = {
        **{f"bf16_{key}": value for key, value in rep_metrics.items()},
        **{f"mixed_{key}": value for key, value in mixed_rep_metrics.items()},
    }
    return rep_metrics, rows


def _evaluate_quadrature(
    *,
    reps: torch.Tensor,
    mask: torch.Tensor,
    grad_reps: torch.Tensor,
    grad_mask: torch.Tensor,
    directions: torch.Tensor,
    reference_rule: str,
    reference_knots: int,
    candidates: list[tuple[str, int]],
    eval_batch_size: int,
    repeats: int,
) -> list[dict[str, Any]]:
    ref_t, ref_w = _quadrature(
        reference_rule,
        reference_knots,
        device=reps.device,
        dtype=reps.dtype,
    )
    ref_loss = float(
        _sigreg_loss_batched(
            reps,
            mask,
            directions,
            batch_size=eval_batch_size,
            t=ref_t,
            weights=ref_w,
        )
        .detach()
        .cpu()
    )
    ref_grad = _grad_for_directions(
        grad_reps,
        grad_mask,
        directions,
        t=ref_t,
        weights=ref_w,
    )
    rows: list[dict[str, Any]] = []
    for rule, knots in candidates:
        t, w = _quadrature(rule, knots, device=reps.device, dtype=reps.dtype)
        loss = float(
            _sigreg_loss_batched(
                reps,
                mask,
                directions,
                batch_size=eval_batch_size,
                t=t,
                weights=w,
            )
            .detach()
            .cpu()
        )
        grad = _grad_for_directions(
            grad_reps,
            grad_mask,
            directions,
            t=t,
            weights=w,
        )
        grad_cosine = torch.nn.functional.cosine_similarity(grad, ref_grad, dim=0).item()
        times: list[float] = []
        for _ in range(repeats):
            if reps.device.type == "cuda":
                torch.cuda.synchronize()
            started = time.time()
            _ = _sigreg_loss_batched(
                reps,
                mask,
                directions,
                batch_size=eval_batch_size,
                t=t,
                weights=w,
            )
            if reps.device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.time() - started) * 1000.0)
        rows.append(
            {
                "name": f"{rule}-{knots}",
                "rule": rule,
                "knots": knots,
                "loss": loss,
                "reference_loss": ref_loss,
                "loss_rel_error_vs_ref": abs(loss - ref_loss) / abs(ref_loss),
                "grad_cosine_vs_ref": grad_cosine,
                "mean_eval_ms": float(np.mean(times)),
                "std_eval_ms": float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
            }
        )
        print(
            f"Q={rule}-{knots:<3d} rel_err={rows[-1]['loss_rel_error_vs_ref']:.5f} "
            f"grad_cos={grad_cosine:.5f} ms={rows[-1]['mean_eval_ms']:.1f}",
            flush=True,
        )
    return rows


def _write_markdown(
    path: Path,
    *,
    meta: dict[str, Any],
    geometry: dict[str, float],
    rows: list[dict[str, Any]],
    quadrature_rows: list[dict[str, Any]],
    dtype_rep_metrics: dict[str, float],
    dtype_rows: list[dict[str, Any]],
    dtype_layer_rows: list[dict[str, Any]],
    reference_slices: int,
    quadrature_reference: str,
) -> None:
    checkpoint_row = next(row for row in rows if row["num_slices"] == meta["checkpoint_sigreg_num_slices"])
    stable_rows = [
        row
        for row in rows
        if row["prefix_loss_rel_error_vs_ref"] <= 0.02 and row["grad_cosine_vs_ref"] >= 0.95
    ]
    stable = stable_rows[0] if stable_rows else rows[-1]
    lines = [
        "# SigReg Projection Sweep",
        "",
        f"- Checkpoint SigReg projections: `{meta['checkpoint_sigreg_num_slices']}`",
        f"- Reference projections: `{reference_slices}`",
        f"- Split/samples: `{meta['split']}` / `{meta['samples']}` spectra, `{meta['valid_tokens']}` valid peak tokens",
        f"- Regularizer: `{meta['representation_regularizer']}`, lambda `{meta['checkpoint_sigreg_lambda']}`",
        "",
        "## Representation Geometry",
        "",
        f"- Participation ratio: `{geometry['participation_ratio']:.1f}` / `{meta['dim']}` dims (`{geometry['effective_rank_fraction']:.3f}` fraction)",
        f"- Top-1 / top-8 / top-32 variance: `{geometry['top1_variance_fraction']:.3f}` / `{geometry['top8_variance_fraction']:.3f}` / `{geometry['top32_variance_fraction']:.3f}`",
        f"- Token norm: `{geometry['mean_token_norm']:.3f} +/- {geometry['std_token_norm']:.3f}`",
        "",
        "## Main Result",
        "",
        f"- At the checkpoint setting (`K={checkpoint_row['num_slices']}`), loss CV is `{checkpoint_row['loss_cv']:.3f}`, relative loss error vs {reference_slices} is `{checkpoint_row['prefix_loss_rel_error_vs_ref']:.3f}`, and gradient cosine is `{checkpoint_row['grad_cosine_vs_ref']:.3f}`.",
        f"- The first setting meeting `<=2%` loss error and `>=0.95` gradient cosine is `K={stable['num_slices']}`.",
        "",
        "## Sweep Table",
        "",
        "| K | mean loss | loss CV | rel error vs ref | grad cosine | eval ms |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['num_slices']} | {row['loss_mean']:.2f} | {row['loss_cv']:.3f} | "
            f"{row['prefix_loss_rel_error_vs_ref']:.3f} | {row['grad_cosine_vs_ref']:.3f} | "
            f"{row['mean_eval_ms']:.2f} |"
        )
    if quadrature_rows:
        current_quad = next(
            row for row in quadrature_rows
            if row["rule"] == "trapezoid" and row["knots"] == 17
        )
        best_alt = min(
            [row for row in quadrature_rows if not (row["rule"] == "trapezoid" and row["knots"] == 17)],
            key=lambda row: row["loss_rel_error_vs_ref"],
        )
        lines.extend(
            [
                "",
                "## Quadrature",
                "",
                f"- Reference: `{quadrature_reference}` at fixed `K={meta['checkpoint_sigreg_num_slices']}` projection directions.",
                f"- Current rule (`trapezoid-17`) has relative loss error `{current_quad['loss_rel_error_vs_ref']:.5f}` and gradient cosine `{current_quad['grad_cosine_vs_ref']:.5f}`.",
                f"- Best tested alternative by loss error is `{best_alt['name']}` with relative loss error `{best_alt['loss_rel_error_vs_ref']:.5f}` and gradient cosine `{best_alt['grad_cosine_vs_ref']:.5f}`.",
                "",
                "| Rule | Knots | rel error vs ref | grad cosine | eval ms |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in quadrature_rows:
            lines.append(
                f"| {row['rule']} | {row['knots']} | {row['loss_rel_error_vs_ref']:.5f} | "
                f"{row['grad_cosine_vs_ref']:.5f} | {row['mean_eval_ms']:.2f} |"
            )
    if dtype_rows:
        full_bf16 = next(row for row in dtype_rows if row["name"] == "bf16-encoder bf16-sigreg")
        full_mixed = next(row for row in dtype_rows if row["name"] == "mixed-encoder bf16-sigreg")
        encoder_bf16 = next(row for row in dtype_rows if row["name"] == "bf16-encoder fp32-sigreg")
        encoder_mixed = next(row for row in dtype_rows if row["name"] == "mixed-encoder fp32-sigreg")
        math_bf16 = next(row for row in dtype_rows if row["name"] == "fp32-encoder bf16-sigreg")
        lines.extend(
            [
                "",
                "## bf16 vs fp32",
                "",
                f"- Encoder bf16 autocast representation drift: relative RMSE `{dtype_rep_metrics['bf16_relative_rmse']:.6f}`, mean token cosine `{dtype_rep_metrics['bf16_token_cosine_mean']:.6f}`, p01 token cosine `{dtype_rep_metrics['bf16_token_cosine_p01']:.6f}`.",
                f"- bf16 transformer with fp32 embedder drift: relative RMSE `{dtype_rep_metrics['mixed_relative_rmse']:.6f}`, mean token cosine `{dtype_rep_metrics['mixed_token_cosine_mean']:.6f}`, p01 token cosine `{dtype_rep_metrics['mixed_token_cosine_p01']:.6f}`.",
                f"- Encoder-only bf16 drift changes fp32 SigReg loss by `{encoder_bf16['loss_rel_error_vs_fp32']:.5f}` with gradient cosine `{encoder_bf16['grad_cosine_vs_fp32']:.5f}`.",
                f"- fp32-embedder mixed encoder changes fp32 SigReg loss by `{encoder_mixed['loss_rel_error_vs_fp32']:.5f}` with gradient cosine `{encoder_mixed['grad_cosine_vs_fp32']:.5f}`.",
                f"- SigReg arithmetic bf16 changes loss by `{math_bf16['loss_rel_error_vs_fp32']:.5f}` with gradient cosine `{math_bf16['grad_cosine_vs_fp32']:.5f}`.",
                f"- Full bf16 path changes loss by `{full_bf16['loss_rel_error_vs_fp32']:.5f}` with gradient cosine `{full_bf16['grad_cosine_vs_fp32']:.5f}`.",
                f"- fp32-embedder mixed path with bf16 SigReg changes loss by `{full_mixed['loss_rel_error_vs_fp32']:.5f}` with gradient cosine `{full_mixed['grad_cosine_vs_fp32']:.5f}`.",
                "",
                "| Variant | rel error vs fp32 | grad cosine | eval ms |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for row in dtype_rows:
            lines.append(
                f"| {row['name']} | {row['loss_rel_error_vs_fp32']:.5f} | "
                f"{row['grad_cosine_vs_fp32']:.5f} | {row['mean_eval_ms']:.2f} |"
            )
        if dtype_layer_rows:
            lines.extend(
                [
                    "",
                    "### Layerwise Encoder Drift",
                    "",
                    "| Stage | relative RMSE | mean token cosine | p01 token cosine |",
                    "| --- | ---: | ---: | ---: |",
                ]
            )
            for row in dtype_layer_rows:
                lines.append(
                    f"| {row['stage']} | {row['relative_rmse']:.6f} | "
                    f"{row['token_cosine_mean']:.6f} | {row['token_cosine_p01']:.6f} |"
                )
    lines.extend(
        [
            "",
            "## Diagram",
            "",
            "```mermaid",
            "flowchart LR",
            "  A[\"NIST spectra\"] --> B[\"Frozen checkpoint encoder\"]",
            "  B --> C[\"Peak-token representations\"]",
            "  C --> D[\"Slotwise SigReg with K random projections\"]",
            "  D --> E[\"Loss estimate\"]",
            "  D --> F[\"Representation gradient\"]",
            f"  E --> G[\"Compare to K={reference_slices} reference\"]",
            f"  F --> G",
            "  C --> H[\"Covariance spectrum and token norms\"]",
            "```",
            "",
            "Plots: `sigreg_projection_sweep.png`, `sigreg_projection_alignment.png`, `sigreg_quadrature_sweep.png`, `sigreg_dtype_sweep.png`, `encoder_dtype_layerwise.png`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gems_small_norm.py")
    parser.add_argument("--workdir", default="experiments/nistfull-robustprobing-sigreg-noema_enc_covariancepooling")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--outdir", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--split", default="massspec_test")
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--grad-samples", type=int, default=256)
    parser.add_argument("--slices", default="8,16,32,64,128,256,512,1024,2048")
    parser.add_argument("--reference-slices", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--quadrature-reference-rule", default="gauss-legendre")
    parser.add_argument("--quadrature-reference-knots", type=int, default=129)
    parser.add_argument("--quadrature-candidates", default="trapezoid:9,trapezoid:17,trapezoid:33,simpson:17,simpson:33,gauss-legendre:8,gauss-legendre:16,gauss-legendre:32")
    parser.add_argument("--skip-dtype-eval", action="store_true")
    parser.add_argument("--dtype-layer-samples", type=int, default=512)
    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else Path(latest_ckpt_path(workdir)).resolve()
    outdir = Path(args.outdir).resolve() if args.outdir else workdir / "sigreg_projection_eval"
    outdir.mkdir(parents=True, exist_ok=True)
    device = _device(args.device)

    reps_cpu, mask_cpu, meta = _extract_representations(
        config_path=Path(args.config),
        checkpoint_path=checkpoint,
        split=args.split,
        max_samples=int(args.max_samples),
        device=device,
        encoder_precision="fp32",
    )
    geom = _geometry(reps_cpu, mask_cpu)

    reps = reps_cpu.to(device)
    mask = mask_cpu.to(device)
    grad_reps = reps[: int(args.grad_samples)]
    grad_mask = mask[: int(args.grad_samples)]
    max_slices = max([*_parse_ints(args.slices), int(args.reference_slices)])
    ref_directions = _sample_directions(
        int(reps.shape[-1]),
        max_slices,
        seed=12345,
        device=device,
        dtype=reps.dtype,
    )
    ref_loss = float(
        _sigreg_loss_batched(
            reps,
            mask,
            ref_directions[:, : int(args.reference_slices)],
            batch_size=int(args.eval_batch_size),
        )
        .detach()
        .cpu()
    )
    ref_grad = _grad_for_directions(grad_reps, grad_mask, ref_directions[:, : int(args.reference_slices)])

    rows: list[dict[str, Any]] = []
    for slices in _parse_ints(args.slices):
        prefix_dirs = ref_directions[:, :slices]
        prefix_loss = float(
            _sigreg_loss_batched(
                reps,
                mask,
                prefix_dirs,
                batch_size=int(args.eval_batch_size),
            )
            .detach()
            .cpu()
        )
        grad = _grad_for_directions(grad_reps, grad_mask, prefix_dirs)
        grad_cosine = torch.nn.functional.cosine_similarity(grad, ref_grad, dim=0).item()
        losses: list[float] = []
        times: list[float] = []
        for repeat in range(int(args.repeats)):
            directions = _sample_directions(
                int(reps.shape[-1]),
                slices,
                seed=10_000 + repeat,
                device=device,
                dtype=reps.dtype,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.time()
            loss = _sigreg_loss_batched(
                reps,
                mask,
                directions,
                batch_size=int(args.eval_batch_size),
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.time() - started) * 1000.0)
            losses.append(float(loss.detach().cpu()))
        loss_mean = float(np.mean(losses))
        loss_std = float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0
        rows.append(
            {
                "num_slices": slices,
                "prefix_loss": prefix_loss,
                "reference_loss": ref_loss,
                "prefix_loss_rel_error_vs_ref": abs(prefix_loss - ref_loss) / abs(ref_loss),
                "grad_cosine_vs_ref": grad_cosine,
                "loss_mean": loss_mean,
                "loss_std": loss_std,
                "loss_cv": loss_std / abs(loss_mean),
                "mean_eval_ms": float(np.mean(times)),
                "std_eval_ms": float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
            }
        )
        print(
            f"K={slices:4d} loss_cv={rows[-1]['loss_cv']:.3f} "
            f"rel_err={rows[-1]['prefix_loss_rel_error_vs_ref']:.3f} "
            f"grad_cos={grad_cosine:.3f} ms={rows[-1]['mean_eval_ms']:.1f}",
            flush=True,
        )

    quadrature_candidates = [
        (rule, int(knots))
        for item in str(args.quadrature_candidates).split(",")
        for rule, knots in [item.split(":", 1)]
    ]
    quadrature_dirs = ref_directions[:, : int(meta["checkpoint_sigreg_num_slices"])]
    quadrature_rows = _evaluate_quadrature(
        reps=reps,
        mask=mask,
        grad_reps=grad_reps,
        grad_mask=grad_mask,
        directions=quadrature_dirs,
        reference_rule=str(args.quadrature_reference_rule),
        reference_knots=int(args.quadrature_reference_knots),
        candidates=quadrature_candidates,
        eval_batch_size=int(args.eval_batch_size),
        repeats=int(args.repeats),
    )
    dtype_rep_metrics: dict[str, float] = {}
    dtype_rows: list[dict[str, Any]] = []
    dtype_layer_rows: list[dict[str, Any]] = []
    if not args.skip_dtype_eval:
        bf16_reps_cpu, bf16_mask_cpu, _ = _extract_representations(
            config_path=Path(args.config),
            checkpoint_path=checkpoint,
            split=args.split,
            max_samples=int(args.max_samples),
            device=device,
            encoder_precision="bf16",
        )
        mixed_reps_cpu, _, _ = _extract_representations(
            config_path=Path(args.config),
            checkpoint_path=checkpoint,
            split=args.split,
            max_samples=int(args.max_samples),
            device=device,
            encoder_precision="bf16_embedder_fp32",
        )
        bf16_reps = bf16_reps_cpu.to(device)
        mixed_reps = mixed_reps_cpu.to(device)
        dtype_dirs = ref_directions[:, : int(meta["checkpoint_sigreg_num_slices"])]
        dtype_rep_metrics, dtype_rows = _evaluate_dtype(
            fp32_reps=reps,
            bf16_encoder_reps=bf16_reps,
            mixed_encoder_reps=mixed_reps,
            mask=mask,
            grad_count=int(args.grad_samples),
            directions=dtype_dirs,
            eval_batch_size=int(args.eval_batch_size),
            repeats=int(args.repeats),
        )
        dtype_layer_rows = _evaluate_layerwise_dtype(
            config_path=Path(args.config),
            checkpoint_path=checkpoint,
            split=args.split,
            max_samples=int(args.dtype_layer_samples),
            device=device,
        )

    payload = {
        "checkpoint": str(checkpoint),
        "config": str(Path(args.config).resolve()),
        "meta": meta,
        "geometry": geom,
        "reference_slices": int(args.reference_slices),
        "results": rows,
        "quadrature_reference": {
            "rule": str(args.quadrature_reference_rule),
            "knots": int(args.quadrature_reference_knots),
            "num_slices": int(meta["checkpoint_sigreg_num_slices"]),
        },
        "quadrature_results": quadrature_rows,
        "dtype_representation_metrics": dtype_rep_metrics,
        "dtype_results": dtype_rows,
        "dtype_layer_results": dtype_layer_rows,
    }
    (outdir / "sigreg_projection_eval.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    _write_csv(outdir / "sigreg_projection_eval.csv", rows)
    _write_csv(outdir / "sigreg_quadrature_eval.csv", quadrature_rows)
    if dtype_rows:
        _write_csv(outdir / "sigreg_dtype_eval.csv", dtype_rows)
        _write_csv(outdir / "encoder_dtype_layerwise.csv", dtype_layer_rows)
    _plot(outdir, rows, int(meta["checkpoint_sigreg_num_slices"]))
    _plot_quadrature(outdir, quadrature_rows)
    if dtype_rows:
        _plot_dtype(outdir, dtype_rows, dtype_rep_metrics)
        _plot_layerwise_dtype(outdir, dtype_layer_rows)
    _write_markdown(
        outdir / "analysis.md",
        meta=meta,
        geometry=geom,
        rows=rows,
        quadrature_rows=quadrature_rows,
        dtype_rep_metrics=dtype_rep_metrics,
        dtype_rows=dtype_rows,
        dtype_layer_rows=dtype_layer_rows,
        reference_slices=int(args.reference_slices),
        quadrature_reference=(
            f"{args.quadrature_reference_rule}-{int(args.quadrature_reference_knots)}"
        ),
    )
    print(f"Wrote {outdir}")


if __name__ == "__main__":
    main()
