from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as torch_dist
from ml_collections import config_dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectra_learning.data.gems.datamodule import GemsDataModule
from spectra_learning.models.factory import build_model_from_config
from spectra_learning.training.batch import BatchPrefetcher
from spectra_learning.training.checkpointing import (
    load_resume_model_state,
    load_torch_checkpoint,
)
from spectra_learning.training.distributed import (
    cleanup_distributed,
    init_distributed_from_env,
)
from spectra_learning.training.pretrain import compile_forward, seed_all
from spectra_learning.training.runtime import parse_autocast_dtype


DEFAULT_METRICS = (
    "loss",
    "mae_mz_loss",
    "mae_intensity_loss",
    "distogram_loss",
    "mae_mz_accuracy",
    "mae_intensity_accuracy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired validation comparison for two MAE checkpoints."
    )
    parser.add_argument("--fourier-run", type=Path, required=True)
    parser.add_argument("--token-run", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--prefetch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_config(run_dir: Path) -> config_dict.ConfigDict:
    return config_dict.ConfigDict(json.loads((run_dir / "config.json").read_text()))


def _load_model(
    run_dir: Path,
    config: config_dict.ConfigDict,
    device: torch.device,
) -> torch.nn.Module:
    model = build_model_from_config(config)
    checkpoint = load_torch_checkpoint(
        run_dir / "checkpoints" / "last.pt",
        map_location="cpu",
        weights_only=True,
    )
    load_resume_model_state(model, checkpoint["model"])
    model.to(device).eval()
    compile_forward(model, config)
    return model


def _autocast_context(device: torch.device, dtype: torch.dtype | None):
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def _paired_statistics(
    totals: torch.Tensor,
    batches: int,
) -> dict[str, dict[str, float]]:
    result = {}
    for index, metric in enumerate(DEFAULT_METRICS):
        fourier_sum, token_sum, delta_sum, delta_square_sum = totals[index]
        delta_mean = delta_sum / batches
        delta_variance = (
            (delta_square_sum - delta_sum.square() / batches)
            / (batches - 1)
        ).clamp_min(0.0)
        delta_sem = torch.sqrt(delta_variance / batches)
        result[metric] = {
            "fourier_mean": float(fourier_sum / batches),
            "token_mean": float(token_sum / batches),
            "token_minus_fourier": float(delta_mean),
            "paired_sem": float(delta_sem),
            "paired_ci95_low": float(delta_mean - 1.96 * delta_sem),
            "paired_ci95_high": float(delta_mean + 1.96 * delta_sem),
        }
    return result


def compare(args: argparse.Namespace) -> dict[str, Any]:
    distributed = init_distributed_from_env()
    fourier_config = _load_config(args.fourier_run)
    token_config = _load_config(args.token_run)
    assert fourier_config.jepa_mae_mz_bin_size == token_config.jepa_mae_mz_bin_size
    assert fourier_config.jepa_mae_mz_max == token_config.jepa_mae_mz_max
    seed_all(int(fourier_config.seed))
    datamodule = GemsDataModule(
        fourier_config,
        seed=int(fourier_config.seed),
        distributed_world_size=distributed.world_size,
        distributed_rank=distributed.rank,
        distributed_local_rank=distributed.local_rank,
    )
    fourier = _load_model(args.fourier_run, fourier_config, distributed.device)
    token = _load_model(args.token_run, token_config, distributed.device)
    autocast_dtype = parse_autocast_dtype(
        fourier_config.get("autocast_dtype", "bf16")
    )
    prefetcher = BatchPrefetcher(
        iter(datamodule.val_loader),
        distributed.device,
        prefetch_size=args.prefetch_size,
    )
    totals = torch.zeros(
        (len(DEFAULT_METRICS), 4),
        dtype=torch.float64,
        device=distributed.device,
    )
    local_steps = 0
    local_spectra = 0
    with torch.no_grad():
        while (
            local_steps < args.max_steps
            and (batch := prefetcher.next()) is not None
        ):
            with _autocast_context(distributed.device, autocast_dtype):
                fourier_metrics = fourier(batch)
                token_metrics = token(batch)
            for index, metric in enumerate(DEFAULT_METRICS):
                fourier_value = fourier_metrics[metric].double()
                token_value = token_metrics[metric].double()
                delta = token_value - fourier_value
                totals[index, 0] += fourier_value
                totals[index, 1] += token_value
                totals[index, 2] += delta
                totals[index, 3] += delta.square()
            local_steps += 1
            local_spectra += batch["peak_mz"].shape[0]

    counts = torch.tensor(
        [local_steps, local_spectra],
        dtype=torch.int64,
        device=distributed.device,
    )
    if distributed.is_distributed:
        torch_dist.all_reduce(totals, op=torch_dist.ReduceOp.SUM)
        torch_dist.all_reduce(counts, op=torch_dist.ReduceOp.SUM)
    batches, spectra = (int(value) for value in counts)
    result = {
        "delta_definition": "token - fourier",
        "metric_directions": {
            "losses": "negative favors token",
            "accuracies": "positive favors token",
        },
        "mz_target_bin_size_da": float(fourier_config.jepa_mae_mz_bin_size),
        "world_size": distributed.world_size,
        "local_steps": local_steps,
        "batches": batches,
        "spectra": spectra,
        "metrics": _paired_statistics(totals, batches),
    }
    if distributed.is_main:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
    cleanup_distributed(distributed)
    return result


def main() -> None:
    compare(parse_args())


if __name__ == "__main__":
    main()
