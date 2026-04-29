from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from ml_collections import config_dict

import input_pipeline
from networks.transformer_torch import _build_norm
from utils.spectra_preprocessing import (
    DEFAULT_MIN_PEAK_INTENSITY,
    DEFAULT_PRECURSOR_PEAK_EXCLUSION_WINDOW_DA,
    preprocess_peak_batch_torch,
)
from utils.training import build_model_from_config


JEPA_MASK_STRATEGIES = ("contiguous", "ragged", "random")


def load_wandb_config(path: Path) -> config_dict.ConfigDict:
    raw = yaml.safe_load(path.read_text())
    return config_dict.ConfigDict(
        {
            key: value["value"]
            for key, value in raw.items()
            if isinstance(value, dict) and "value" in value
        }
    )


def patch_checkpoint_final_norms(model: torch.nn.Module) -> None:
    model.encoder.final_norm = _build_norm(
        model.model_dim,
        eps=model.norm_eps,
        norm_type=model.norm_type,
    )
    if model.teacher_encoder is not None:
        model.teacher_encoder.final_norm = _build_norm(
            model.model_dim,
            eps=model.norm_eps,
            norm_type=model.norm_type,
        )
    model.predictor_final_norm = _build_norm(
        model.predictor_dim,
        eps=model.norm_eps,
        norm_type=model.norm_type,
    )


def sample_mask_fraction(
    fraction_range: tuple[float, float],
    *,
    device: torch.device,
) -> float:
    low, high = (float(value) for value in fraction_range)
    return float(torch.empty((), device=device).uniform_(low, high).item())


def sample_mask_strategy(mask_strategy: str, *, device: torch.device) -> str:
    strategy = str(mask_strategy).lower()
    if strategy == "ragged_blocks":
        strategy = "ragged"
    if strategy == "all":
        strategy = JEPA_MASK_STRATEGIES[
            int(torch.randint(len(JEPA_MASK_STRATEGIES), (), device=device).item())
        ]
    return strategy


def sample_ragged_block_mask(
    active_positions: torch.Tensor,
    *,
    masked_fraction: float,
    lengths: tuple[int, ...],
    round_from: int,
) -> torch.Tensor:
    active_count = int(active_positions.sum().item())
    if active_count == 0:
        return torch.zeros_like(active_positions)
    compressed = torch.cumsum(active_positions.to(torch.int64), dim=0)
    compressed = compressed - active_positions.to(torch.int64)
    weights = torch.rand(len(lengths), device=active_positions.device)
    weights = weights / weights.sum()
    masks = []
    for length_idx, length in enumerate(lengths):
        block_len = int(length)
        max_elem = int(
            math.ceil(float(masked_fraction) * float(active_count) / float(block_len))
        )
        coeff_float = float(weights[length_idx].item()) * float(max_elem)
        coeff = (
            int(math.ceil(coeff_float))
            if length_idx < int(round_from)
            else int(round(coeff_float))
        )
        if coeff == 0:
            masks.append(torch.zeros_like(active_positions))
            continue
        effective_len = min(block_len, active_count)
        starts = torch.randint(
            0,
            active_count - effective_len + 1,
            (coeff,),
            device=active_positions.device,
        )
        block_mask = (compressed.unsqueeze(0) >= starts.unsqueeze(1)) & (
            compressed.unsqueeze(0) < (starts + effective_len).unsqueeze(1)
        )
        masks.append((block_mask & active_positions.unsqueeze(0)).any(dim=0))
    return torch.stack(masks, dim=0).any(dim=0)


def sample_contiguous_mask(
    active_positions: torch.Tensor,
    *,
    mask_count: int,
) -> torch.Tensor:
    active_count = int(active_positions.sum().item())
    if active_count == 0:
        return torch.zeros_like(active_positions)
    count = min(int(mask_count), active_count)
    start = int(
        torch.randint(
            active_count - count + 1,
            (),
            device=active_positions.device,
        ).item()
    )
    compressed = torch.cumsum(active_positions.to(torch.int64), dim=0)
    compressed = compressed - active_positions.to(torch.int64)
    return ((compressed >= start) & (compressed < start + count)) & active_positions


def sample_random_mask(
    active_positions: torch.Tensor,
    *,
    masked_fraction: float,
) -> torch.Tensor:
    active_count = int(active_positions.sum().item())
    if active_count == 0:
        return torch.zeros_like(active_positions)
    mask_count = min(
        int(round(float(masked_fraction) * float(active_count))),
        active_count,
    )
    if mask_count == 0:
        return torch.zeros_like(active_positions)
    active_indices = torch.where(active_positions)[0]
    selected = active_indices[
        torch.randperm(active_count, device=active_positions.device)[:mask_count]
    ]
    mask = torch.zeros_like(active_positions)
    mask[selected] = True
    return mask


def sample_block_masks(
    peak_valid_mask: torch.Tensor,
    *,
    num_target_blocks: int,
    context_fraction: float,
    target_fraction: float,
    block_min_len: int,
    mask_strategy: str,
    context_fraction_range: tuple[float, float],
    target_fraction_range: tuple[float, float],
    mask_lengths: tuple[int, ...],
    mask_round_from: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    strategy = str(mask_strategy).lower()
    if strategy == "ragged_blocks":
        strategy = "ragged"
    lengths = tuple(int(length) for length in mask_lengths)
    device = peak_valid_mask.device
    batch_size, num_peaks = peak_valid_mask.shape
    context_mask = torch.zeros(batch_size, num_peaks, dtype=torch.bool, device=device)
    target_masks = torch.zeros(
        batch_size,
        int(num_target_blocks),
        num_peaks,
        dtype=torch.bool,
        device=device,
    )
    for row_idx in range(batch_size):
        row_strategy = sample_mask_strategy(strategy, device=device)
        row_valid = peak_valid_mask[row_idx]
        valid_count = int(row_valid.sum().item())
        if valid_count == 0:
            continue
        row_context_fraction = float(context_fraction)
        row_target_fraction = float(target_fraction)
        if row_strategy == "random":
            row_context_fraction = sample_mask_fraction(
                context_fraction_range,
                device=device,
            )
            row_target_fraction = sample_mask_fraction(
                target_fraction_range,
                device=device,
            )
        desired_context = max(
            int(round(valid_count * row_context_fraction)),
            int(block_min_len),
        )
        reserve_for_targets = min(
            valid_count,
            int(num_target_blocks) * int(block_min_len),
        )
        context_len = min(desired_context, max(valid_count - reserve_for_targets, 1))
        target_len = min(
            max(int(round(valid_count * row_target_fraction)), int(block_min_len)),
            max(valid_count - context_len, 0),
        )

        if row_strategy == "contiguous":
            row_context = sample_contiguous_mask(row_valid, mask_count=context_len)
        elif row_strategy == "ragged":
            row_context = sample_ragged_block_mask(
                row_valid,
                masked_fraction=float(context_len) / float(valid_count),
                lengths=lengths,
                round_from=int(mask_round_from),
            )
        else:
            row_context = sample_random_mask(
                row_valid,
                masked_fraction=float(context_len) / float(valid_count),
            )
        context_mask[row_idx] = row_context

        valid_targets = row_valid & ~row_context
        if target_len == 0:
            continue
        if row_strategy == "contiguous":
            for block_idx in range(int(num_target_blocks)):
                target_masks[row_idx, block_idx] = sample_contiguous_mask(
                    valid_targets,
                    mask_count=target_len,
                )
            continue
        available = int(valid_targets.sum().item())
        if available == 0:
            continue
        target_fraction_on_available = float(target_len) / float(available)
        for block_idx in range(int(num_target_blocks)):
            if row_strategy == "ragged":
                target_masks[row_idx, block_idx] = sample_ragged_block_mask(
                    valid_targets,
                    masked_fraction=target_fraction_on_available,
                    lengths=lengths,
                    round_from=int(mask_round_from),
                )
            else:
                target_masks[row_idx, block_idx] = sample_random_mask(
                    valid_targets,
                    masked_fraction=target_fraction_on_available,
                )
    return context_mask, target_masks


def sample_rows(
    *,
    gems_dir: Path,
    split: str,
    shard_names: list[str],
    lengths: list[int],
    num_samples: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    starts = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(np.asarray(lengths, dtype=np.int64), out=starts[1:])
    global_indices = rng.integers(0, int(starts[-1]), size=num_samples, endpoint=False)
    spectra_out = np.empty((num_samples, 2, 128), dtype=np.float32)
    precursor_out = np.empty((num_samples,), dtype=np.float32)
    for shard_idx, shard_name in enumerate(shard_names):
        lo, hi = int(starts[shard_idx]), int(starts[shard_idx + 1])
        output_rows = np.flatnonzero((global_indices >= lo) & (global_indices < hi))
        if len(output_rows) == 0:
            continue
        local_rows = global_indices[output_rows] - lo
        order = np.argsort(local_rows)
        shard_dir = gems_dir / split / shard_name
        spectra = np.load(shard_dir / "spectra.npy", mmap_mode="r")
        precursor = np.load(shard_dir / "precursor_mz_raw.npy", mmap_mode="r")
        spectra_out[output_rows[order]] = np.asarray(
            spectra[local_rows[order]],
            dtype=np.float32,
        )
        precursor_out[output_rows[order]] = np.asarray(
            precursor[local_rows[order]],
            dtype=np.float32,
        )
    return torch.from_numpy(spectra_out), torch.from_numpy(precursor_out)


def quantiles(values: list[float] | np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "p05": float(np.quantile(x, 0.05)),
        "p25": float(np.quantile(x, 0.25)),
        "p50": float(np.quantile(x, 0.50)),
        "p75": float(np.quantile(x, 0.75)),
        "p95": float(np.quantile(x, 0.95)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def preprocess_samples(
    cfg: Any,
    spectra: torch.Tensor,
    precursor_raw: torch.Tensor,
) -> dict[str, torch.Tensor]:
    pre = preprocess_peak_batch_torch(
        spectra[:, 0, :],
        spectra[:, 1, :],
        precursor_raw,
        num_peaks=int(cfg.get("num_peaks", 64)),
        peak_drop_min_intensity=float(
            cfg.get(
                "peak_drop_min_intensity",
                cfg.get("min_peak_intensity", DEFAULT_MIN_PEAK_INTENSITY),
            )
        ),
        peak_ordering=str(cfg.get("peak_ordering", "mz")),
        max_precursor_mz=float(cfg.get("max_precursor_mz", 1000.0)),
        precursor_peak_exclusion_window_da=float(
            cfg.get(
                "precursor_peak_exclusion_window_da",
                DEFAULT_PRECURSOR_PEAK_EXCLUSION_WINDOW_DA,
            )
        ),
        min_peak_intensity=float(
            cfg.get("min_peak_intensity", DEFAULT_MIN_PEAK_INTENSITY)
        ),
    )
    nonempty = pre["peak_valid_mask"].any(dim=1)
    return {key: value[nonempty] for key, value in pre.items()}


def evaluate_strategy(
    *,
    cfg: Any,
    model: torch.nn.Module,
    pre: dict[str, torch.Tensor],
    strategy: str,
    seed: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    target_dim = int(model.target_projector_dim)
    num_ranks = int(cfg.num_peaks) + int(cfg.use_precursor_token)
    total_count = 0
    sum_vec = torch.zeros(target_dim, dtype=torch.float64)
    sum_norm2 = torch.zeros((), dtype=torch.float64)
    residual_sse = torch.zeros((), dtype=torch.float64)
    rank_count = torch.zeros(num_ranks, dtype=torch.float64)
    rank_sum = torch.zeros(num_ranks, target_dim, dtype=torch.float64)
    rank_sum_norm2 = torch.zeros(num_ranks, dtype=torch.float64)
    rank_sse = torch.zeros(num_ranks, dtype=torch.float64)
    context_fracs: list[float] = []
    target_fracs: list[float] = []
    valid_counts: list[float] = []
    losses: list[float] = []
    started = time.time()

    for start in range(0, int(pre["peak_valid_mask"].shape[0]), batch_size):
        end = min(start + batch_size, int(pre["peak_valid_mask"].shape[0]))
        batch = {key: value[start:end].clone() for key, value in pre.items()}
        original_valid_count = batch["peak_valid_mask"].sum(dim=1).float()
        context, targets = sample_block_masks(
            batch["peak_valid_mask"],
            num_target_blocks=int(cfg.jepa_num_target_blocks),
            context_fraction=float(cfg.jepa_context_fraction),
            target_fraction=float(cfg.jepa_target_fraction),
            block_min_len=int(cfg.jepa_block_min_len),
            mask_strategy=strategy,
            context_fraction_range=tuple(float(v) for v in cfg.jepa_context_fraction_range),
            target_fraction_range=tuple(float(v) for v in cfg.jepa_target_fraction_range),
            mask_lengths=tuple(int(v) for v in cfg.jepa_mask_lengths),
            mask_round_from=int(cfg.jepa_mask_round_from),
        )
        batch["context_mask"] = context
        batch["target_masks"] = targets
        if bool(cfg.use_precursor_token):
            batch = input_pipeline._prepend_precursor_token_torch(batch)
        batch = {key: value.to(device) for key, value in batch.items()}

        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            metrics, collapse_data = model.forward_augmented(
                batch,
                return_collapse_data=True,
            )

        teacher = collapse_data["teacher_targets"].float()
        prediction = collapse_data["predictor_output"].float()
        target_mask = collapse_data["target_masks"]
        teacher_expanded = teacher.unsqueeze(1).expand_as(prediction)
        diff = prediction - teacher_expanded
        weights = target_mask.float()
        masked_teacher = teacher_expanded * target_mask.unsqueeze(-1).float()

        batch_count = int(target_mask.sum().item())
        total_count += batch_count
        sum_vec += masked_teacher.sum(dim=(0, 1, 2)).double().cpu()
        sum_norm2 += (
            teacher_expanded.square().sum(dim=-1) * weights
        ).sum().double().cpu()
        residual_sse += (diff.square().sum(dim=-1) * weights).sum().double().cpu()
        rank_count += target_mask.sum(dim=(0, 1)).double().cpu()
        rank_sum += masked_teacher.sum(dim=(0, 1)).double().cpu()
        rank_sum_norm2 += (
            teacher_expanded.square().sum(dim=-1) * weights
        ).sum(dim=(0, 1)).double().cpu()
        rank_sse += (diff.square().sum(dim=-1) * weights).sum(dim=(0, 1)).double().cpu()

        if bool(cfg.use_precursor_token):
            context_count = collapse_data["context_mask"][:, 1:].sum(dim=1).float()
            target_count = collapse_data["target_masks"][:, :, 1:].sum(dim=2).float()
        else:
            context_count = collapse_data["context_mask"].sum(dim=1).float()
            target_count = collapse_data["target_masks"].sum(dim=2).float()
        context_fracs.extend((context_count.cpu() / original_valid_count).tolist())
        target_fracs.extend((target_count.cpu() / original_valid_count[:, None]).flatten().tolist())
        valid_counts.extend(original_valid_count.cpu().tolist())
        losses.append(float(metrics["masked_prediction_loss"].detach().float().cpu()))

    mean = sum_vec / total_count
    total_trace = float(sum_norm2 / total_count - mean.square().sum())
    residual_trace = float(residual_sse / total_count)
    rank_eta = []
    for rank_idx in range(num_ranks):
        if rank_count[rank_idx] == 0:
            continue
        rank_mean = rank_sum[rank_idx] / rank_count[rank_idx]
        rank_total = float(
            rank_sum_norm2[rank_idx] / rank_count[rank_idx]
            - rank_mean.square().sum()
        )
        if rank_total > 0.0:
            rank_eta.append(1.0 - float(rank_sse[rank_idx] / rank_count[rank_idx]) / rank_total)

    return {
        "num_target_instances": int(total_count),
        "target_dim": target_dim,
        "total_trace_cov": total_trace,
        "residual_trace": residual_trace,
        "eta2_explained_variance": 1.0 - residual_trace / total_trace,
        "masked_prediction_loss_mean_over_batches": float(np.mean(losses)),
        "valid_peak_count": quantiles(valid_counts),
        "context_fraction": quantiles(context_fracs),
        "target_fraction_per_view": quantiles(target_fracs),
        "rank_eta2_quantiles": quantiles(np.asarray(rank_eta, dtype=np.float64)),
        "elapsed_sec": float(time.time() - started),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-yaml",
        type=Path,
        default=Path(
            "experiments/gems_latest_impl/wandb/run-20260428_113925-m2zpcfa9/files/config.yaml"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("experiments/gems_latest_impl/checkpoints/step-00850000.pt"),
    )
    parser.add_argument("--num-samples", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["all", "contiguous", "ragged", "random"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/gems_jepa_predictability_step850k.json"),
    )
    args = parser.parse_args()

    cfg = load_wandb_config(args.config_yaml)
    gems_dir = Path(str(cfg.artifact_dir)) / "gems"
    metadata = json.loads((gems_dir / "metadata.json").read_text())
    spectra, precursor = sample_rows(
        gems_dir=gems_dir,
        split="validation",
        shard_names=list(metadata["validation_shards"]),
        lengths=[int(v) for v in metadata["validation_lengths"]],
        num_samples=int(args.num_samples),
        seed=int(args.seed),
    )
    pre = preprocess_samples(cfg, spectra, precursor)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(cfg)
    patch_checkpoint_final_norms(model)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    model.eval().to(device)

    result = {
        "estimator": (
            "trained JEPA predictor R2 lower-bound proxy for conditional-mean eta2"
        ),
        "formula_used": (
            "eta2 = 1 - E||prediction - teacher_target||^2 / TrCov(teacher_target)"
        ),
        "config_yaml": str(args.config_yaml),
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint["global_step"]),
        "checkpoint_loss": float(checkpoint["loss"]),
        "ignored_checkpoint_keys": list(unexpected),
        "missing_checkpoint_keys": list(missing),
        "artifact": str(gems_dir),
        "split": "validation",
        "sampled_spectra": int(args.num_samples),
        "nonempty_spectra": int(pre["peak_valid_mask"].shape[0]),
        "mask_policy": {
            "configured": str(cfg.jepa_mask_strategy),
            "context_fraction": float(cfg.jepa_context_fraction),
            "context_fraction_range": list(cfg.jepa_context_fraction_range),
            "target_fraction": float(cfg.jepa_target_fraction),
            "target_fraction_range": list(cfg.jepa_target_fraction_range),
            "num_target_blocks": int(cfg.jepa_num_target_blocks),
            "mask_lengths": list(cfg.jepa_mask_lengths),
        },
        "model": {
            "model_dim": int(cfg.model_dim),
            "encoder_layers": int(cfg.encoder_num_layers),
            "target_layers": list(cfg.jepa_target_layers),
            "target_projector_dim": int(model.target_projector_dim),
            "use_ema_teacher": bool(cfg.use_ema_teacher),
            "use_precursor_token": bool(cfg.use_precursor_token),
        },
        "strategies": {},
    }
    for idx, strategy in enumerate(args.strategies):
        result["strategies"][strategy] = evaluate_strategy(
            cfg=cfg,
            model=model,
            pre=pre,
            strategy=strategy,
            seed=int(args.seed) + 1009 * idx,
            batch_size=int(args.batch_size),
            device=device,
        )
        print(
            strategy,
            f"eta2={result['strategies'][strategy]['eta2_explained_variance']:.4f}",
            f"targets={result['strategies'][strategy]['num_target_instances']}",
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
