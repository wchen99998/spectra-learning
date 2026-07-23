from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from ml_collections import config_dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectra_learning.models.factory import build_model_from_config
from spectra_learning.models.peak_features import DiscretizedMzFeatures
from spectra_learning.training.checkpointing import (
    load_resume_model_state,
    load_torch_checkpoint,
)


VARIANTS = ("fourier", "discrete")
VALIDATION_METRICS = (
    "val/loss",
    "val/mae_mz_loss",
    "val/mae_intensity_loss",
    "val/distogram_loss",
)
PAIRED_VALIDATION_METRICS = (
    "loss",
    "mae_mz_loss",
    "mae_intensity_loss",
    "distogram_loss",
    "mae_mz_accuracy",
    "mae_intensity_accuracy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate paired m/z embedding ablation runs."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("experiments/mz_embedding_ablation"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[66, 67, 68])
    parser.add_argument("--suffix", default="1k")
    parser.add_argument("--geometry", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _run_dir(root: Path, seed: int, variant: str, suffix: str) -> Path:
    return root / f"seed{seed}_{variant}_{suffix}"


def _validation_curve(run_dir: Path) -> dict[str, dict[str, float]]:
    with (run_dir / "csv_logs" / "metrics.csv").open(newline="") as handle:
        rows = csv.DictReader(handle)
        return {
            row["step"]: {
                metric: float(row[metric])
                for metric in VALIDATION_METRICS
            }
            for row in rows
            if row["val/loss"]
        }


def _paired_validation(root: Path, seed: int) -> dict[str, Any]:
    return json.loads((root / f"seed{seed}_paired_validation.json").read_text())


def _effective_rank(embeddings: torch.Tensor) -> tuple[float, float]:
    centered = embeddings.float() - embeddings.float().mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / float(centered.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    probabilities = eigenvalues / eigenvalues.sum()
    entropy_rank = torch.exp(
        -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
    )
    participation_rank = eigenvalues.sum().square() / eigenvalues.square().sum()
    return float(entropy_rank), float(participation_rank)


@torch.no_grad()
def _embedding_geometry(run_dir: Path, device: torch.device) -> dict[str, float]:
    config = config_dict.ConfigDict(
        json.loads((run_dir / "config.json").read_text())
    )
    model = build_model_from_config(config)
    checkpoint = load_torch_checkpoint(
        run_dir / "checkpoints" / "last.pt",
        map_location="cpu",
        weights_only=True,
    )
    load_resume_model_state(model, checkpoint["model"])
    embedder = model.encoder.embedder.to(device).eval()

    bin_size = float(config.encoder_discrete_mz_bin_size)
    coarse_bin_size = float(config.encoder_discrete_mz_coarse_bin_size)
    mz_scale = float(config.encoder_mz_scale)
    bin_indices = torch.arange(
        round(mz_scale / bin_size),
        device=device,
    )
    mz_da = bin_indices.float() * bin_size
    peak_mz = mz_da / mz_scale
    branch_chunks = []
    full_chunks = []
    for chunk in peak_mz.split(4096):
        if isinstance(embedder.mz_features, DiscretizedMzFeatures):
            features = embedder.mz_features(chunk)
        else:
            features = embedder.mz_features(
                embedder._prepare_fourier_mz(chunk.unsqueeze(0)).squeeze(0)
            )
        branch_chunks.append(embedder.mz_ffn(features).float())
        full_chunks.append(
            embedder(
                chunk.unsqueeze(0),
                torch.ones_like(chunk).unsqueeze(0),
            )
            .squeeze(0)
            .float()
        )
    branch = torch.cat(branch_chunks)
    full = torch.cat(full_chunks)
    normalized = F.normalize(branch, dim=-1)

    result: dict[str, float] = {}
    for delta_da in (0.02, 0.1, 1.0, 10.0):
        offset = round(delta_da / bin_size)
        cosine = (normalized[:-offset] * normalized[offset:]).sum(dim=-1)
        result[f"branch_cosine_delta_{delta_da:g}da"] = float(cosine.mean())
        result[f"branch_l2_delta_{delta_da:g}da"] = float(
            (branch[:-offset] - branch[offset:]).norm(dim=-1).mean()
        )

    adjacent_cosine = (normalized[:-1] * normalized[1:]).sum(dim=-1)
    bins_per_coarse = round(coarse_bin_size / bin_size)
    crosses_da_boundary = (bin_indices[:-1] + 1).remainder(bins_per_coarse) == 0
    result["branch_cosine_adjacent_within_1da"] = float(
        adjacent_cosine[~crosses_da_boundary].mean()
    )
    result["branch_cosine_adjacent_across_1da"] = float(
        adjacent_cosine[crosses_da_boundary].mean()
    )
    branch_entropy_rank, branch_participation_rank = _effective_rank(branch[::5])
    full_entropy_rank, full_participation_rank = _effective_rank(full[::5])
    result.update(
        {
            "branch_effective_rank_entropy": branch_entropy_rank,
            "branch_effective_rank_participation": branch_participation_rank,
            "full_effective_rank_entropy": full_entropy_rank,
            "full_effective_rank_participation": full_participation_rank,
        }
    )
    return result


def _paired_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    sample_std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    sem = sample_std / math.sqrt(len(array))
    t_critical_95 = {2: 12.706, 3: 4.303}.get(len(array), 1.96)
    return {
        "mean": mean,
        "sample_std": sample_std,
        "sem": sem,
        "ci95_low": mean - t_critical_95 * sem,
        "ci95_high": mean + t_critical_95 * sem,
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    runs: dict[str, Any] = {}
    paired: dict[str, list[float]] = {
        metric: [] for metric in VALIDATION_METRICS
    }
    paired_curve_means: dict[str, list[float]] = {
        metric: [] for metric in VALIDATION_METRICS
    }
    paired_holdout: dict[str, list[float]] = {
        metric: [] for metric in PAIRED_VALIDATION_METRICS
    }
    paired_geometry: dict[str, list[float]] = {}
    for seed in args.seeds:
        seed_payload: dict[str, Any] = {}
        for variant in VARIANTS:
            run_dir = _run_dir(args.root, seed, variant, args.suffix)
            final_metrics = json.loads((run_dir / "metrics.json").read_text())
            payload: dict[str, Any] = {
                "validation_curve": _validation_curve(run_dir),
                "final": {
                    key: float(final_metrics[key])
                    for key in (
                        *VALIDATION_METRICS,
                        "model/params_trainable",
                        "model/flops_per_optimizer_step_estimate",
                        "run/measured_training_steps_per_second",
                        "run/peak_cuda_memory_allocated_bytes",
                    )
                },
            }
            if args.geometry:
                payload["geometry"] = _embedding_geometry(run_dir, device)
            seed_payload[variant] = payload
        holdout = _paired_validation(args.root, seed)
        seed_payload["paired_holdout"] = holdout
        for metric in PAIRED_VALIDATION_METRICS:
            paired_holdout[metric].append(
                float(holdout["metrics"][metric]["discrete_minus_fourier"])
            )
        for metric in VALIDATION_METRICS:
            paired[metric].append(
                seed_payload["discrete"]["final"][metric]
                - seed_payload["fourier"]["final"][metric]
            )
            steps = sorted(
                set(seed_payload["fourier"]["validation_curve"])
                & set(seed_payload["discrete"]["validation_curve"]),
                key=int,
            )
            paired_curve_means[metric].append(
                float(
                    np.mean(
                        [
                            seed_payload["discrete"]["validation_curve"][step][metric]
                            - seed_payload["fourier"]["validation_curve"][step][metric]
                            for step in steps
                        ]
                    )
                )
            )
        if args.geometry:
            for metric in seed_payload["fourier"]["geometry"]:
                paired_geometry.setdefault(metric, []).append(
                    seed_payload["discrete"]["geometry"][metric]
                    - seed_payload["fourier"]["geometry"][metric]
                )
        runs[str(seed)] = seed_payload

    return {
        "delta_definition": "discrete - fourier; negative favors discrete",
        "seeds": args.seeds,
        "runs": runs,
        "paired_final_deltas": {
            metric: {
                "per_seed": values,
                **_paired_summary(values),
            }
            for metric, values in paired.items()
        },
        "paired_curve_mean_deltas": {
            metric: {
                "per_seed": values,
                **_paired_summary(values),
            }
            for metric, values in paired_curve_means.items()
        },
        "paired_holdout_deltas": {
            metric: {
                "per_seed": values,
                **_paired_summary(values),
            }
            for metric, values in paired_holdout.items()
        },
        "paired_geometry_deltas": {
            metric: {
                "per_seed": values,
                **_paired_summary(values),
            }
            for metric, values in paired_geometry.items()
        },
    }


def main() -> None:
    args = parse_args()
    result = analyze(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
