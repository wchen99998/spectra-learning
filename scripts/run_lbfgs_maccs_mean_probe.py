"""Run an offline LBFGS MACCS linear probe on frozen mean-pooled embeddings."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spectra_learning.config.loading import load_config
from spectra_learning.models.factory import build_model_from_config
from spectra_learning.probes.massspec.data import MassSpecProbeData
from spectra_learning.probes.massspec.msg_probe import iter_massspec_probe
from spectra_learning.probes.massspec.msg_settings import (
    MACCS_TASK,
    build_msg_probe_inputs,
    resolve_msg_probe_sample_limits,
)
from spectra_learning.training.checkpointing import latest_ckpt_path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("lbfgs_maccs_mean_probe")


def _resolve_checkpoint(workdir: Path, checkpoint: str) -> Path:
    if checkpoint:
        return Path(checkpoint).expanduser().resolve()
    last_path = workdir / "checkpoints" / "last.pt"
    if last_path.exists():
        return last_path
    return Path(latest_ckpt_path(workdir)).resolve()


def _move_batch(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _split_seed_config(
    config,
) -> dict[str, tuple[int, int | None, bool]]:
    max_train, max_val, max_test, randomize_test = resolve_msg_probe_sample_limits(
        config
    )
    train_seed = int(config.seed) + 1_100_000
    test_seed = int(config.seed) + 1_200_000
    return {
        "massspec_train": (train_seed, max_train, False),
        "massspec_val": (train_seed + 10_000, max_val, True),
        "massspec_test": (test_seed, max_test, randomize_test),
    }


def _extract_split_embeddings(
    *,
    model: torch.nn.Module,
    probe_data: MassSpecProbeData,
    split: str,
    seed: int,
    peak_ordering: str,
    max_samples: int | None,
    sample_randomly: bool,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    embeddings, targets, valid = [], [], []
    started = time.time()
    with torch.inference_mode():
        for batch in iter_massspec_probe(
            probe_data=probe_data,
            split=split,
            seed=seed,
            peak_ordering=peak_ordering,
            drop_remainder=False,
            max_samples=max_samples,
            sample_randomly=sample_randomly,
        ):
            batch = _move_batch(batch, device)
            encoder_output = model.encoder(
                batch["peak_mz"],
                batch["peak_intensity"],
                valid_mask=batch["peak_valid_mask"],
                precursor_mz=batch.get("precursor_mz", None),
            )
            peak_embeddings, _ = model.encoder.split_peak_and_cls(encoder_output)
            pooled = build_msg_probe_inputs(
                peak_embeddings,
                batch["peak_valid_mask"].to(dtype=torch.bool),
            )
            embeddings.append(pooled.detach().cpu().float())
            targets.append(batch["probe_maccs"].detach().cpu().to(torch.float32))
            valid.append(batch["probe_valid_mol"].detach().cpu().to(torch.bool))
    split_cache = {
        "embedding": torch.cat(embeddings, dim=0),
        "maccs": torch.cat(targets, dim=0),
        "valid_mol": torch.cat(valid, dim=0),
    }
    log.info(
        "Extracted %s: %d rows, %d valid molecules, dim=%d in %.1fs",
        split,
        int(split_cache["embedding"].shape[0]),
        int(split_cache["valid_mol"].sum()),
        int(split_cache["embedding"].shape[1]),
        time.time() - started,
    )
    return split_cache


def _extract_embeddings(
    *,
    config,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    probe_data = MassSpecProbeData.from_config(config)
    model = build_model_from_config(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in checkpoint["model"].items()
        if key.startswith("encoder.")
    }
    model.encoder.load_state_dict(encoder_state)
    model.to(device).eval()
    peak_ordering = str(config.get("peak_ordering", "intensity"))
    split_seeds = _split_seed_config(config)
    return {
        split: _extract_split_embeddings(
            model=model,
            probe_data=probe_data,
            split=split,
            seed=seed,
            peak_ordering=peak_ordering,
            max_samples=max_samples,
            sample_randomly=sample_randomly,
            device=device,
        )
        for split, (seed, max_samples, sample_randomly) in split_seeds.items()
    }


def _valid_xy(
    split_cache: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = split_cache["valid_mol"]
    return (
        split_cache["embedding"][valid].to(device=device, dtype=torch.float32),
        split_cache["maccs"][valid].to(device=device, dtype=torch.float32),
    )


def _score_maccs(
    *,
    prefix: str,
    logits: np.ndarray,
    target: np.ndarray,
    loss: float,
) -> dict[str, float]:
    pred = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
    positives = target.sum(axis=0)
    valid_metric_mask = (positives > 0) & (positives < target.shape[0])
    valid_target = target[:, valid_metric_mask].astype(np.float64)
    valid_pred = pred[:, valid_metric_mask].astype(np.float64)
    valid_positives = positives[valid_metric_mask].astype(np.float64)
    valid_negatives = float(target.shape[0]) - valid_positives
    ascending = np.argsort(valid_pred, axis=0)
    target_ascending = np.take_along_axis(valid_target, ascending, axis=0)
    negatives_before = np.cumsum(1.0 - target_ascending, axis=0)
    auc_values = (
        (target_ascending * negatives_before).sum(axis=0)
        / (valid_positives * valid_negatives)
    )
    target_descending = target_ascending[::-1]
    true_positives_at_rank = np.cumsum(target_descending, axis=0)
    ranks = np.arange(1, target_descending.shape[0] + 1, dtype=np.float64)[:, None]
    average_precision_values = (
        (target_descending * true_positives_at_rank / ranks).sum(axis=0)
        / valid_positives
    )

    positive_mask = positives > 0
    bit_pred = pred >= 0.5
    true_positives = (bit_pred & (target > 0)).sum(axis=0)
    predicted_positives = bit_pred.sum(axis=0)
    recall_values = true_positives[positive_mask] / positives[positive_mask]
    precision_values = np.divide(
        true_positives[positive_mask],
        predicted_positives[positive_mask],
        out=np.zeros_like(true_positives[positive_mask], dtype=np.float64),
        where=predicted_positives[positive_mask] > 0,
    )
    return {
        f"{prefix}/loss_bce": float(loss),
        f"{prefix}/samples": float(target.shape[0]),
        f"{prefix}/num_maccs_auc_bits": float(len(auc_values)),
        f"{prefix}/num_maccs_average_precision_bits": float(
            len(average_precision_values)
        ),
        f"{prefix}/num_maccs_recall_bits": float(len(recall_values)),
        f"{prefix}/num_maccs_precision_bits": float(len(precision_values)),
        f"{prefix}/auc_maccs_mean": float(np.mean(auc_values)),
        f"{prefix}/average_precision_maccs_mean": float(
            np.mean(average_precision_values)
        ),
        f"{prefix}/recall_maccs_mean": float(np.mean(recall_values)),
        f"{prefix}/precision_maccs_mean": float(np.mean(precision_values)),
    }


def _evaluate_split(
    *,
    linear: torch.nn.Linear,
    split_cache: dict[str, torch.Tensor],
    prefix: str,
    device: torch.device,
) -> dict[str, float]:
    x, y = _valid_xy(split_cache, device)
    with torch.inference_mode():
        logits = linear(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
    return _score_maccs(
        prefix=prefix,
        logits=logits.detach().cpu().numpy(),
        target=y.detach().cpu().numpy(),
        loss=float(loss.detach().cpu()),
    )


def _fit_lbfgs(
    *,
    cache: dict[str, dict[str, torch.Tensor]],
    device: torch.device,
    max_iter: int,
    history_size: int,
    tolerance_grad: float,
    tolerance_change: float,
    weight_decay: float,
) -> tuple[torch.nn.Linear, float, float]:
    x_train, y_train = _valid_xy(cache["massspec_train"], device)
    linear = torch.nn.Linear(
        int(x_train.shape[1]),
        int(y_train.shape[1]),
    ).to(device)
    optimizer = torch.optim.LBFGS(
        linear.parameters(),
        lr=1.0,
        max_iter=int(max_iter),
        history_size=int(history_size),
        tolerance_grad=float(tolerance_grad),
        tolerance_change=float(tolerance_change),
        line_search_fn="strong_wolfe",
    )

    losses: list[float] = []

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        logits = linear(x_train)
        loss = F.binary_cross_entropy_with_logits(logits, y_train)
        if weight_decay > 0:
            loss = loss + 0.5 * weight_decay * linear.weight.square().sum()
        loss.backward()
        losses.append(float(loss.detach().cpu()))
        return loss

    started = time.time()
    optimizer.step(closure)
    elapsed = time.time() - started
    return linear, losses[-1], elapsed


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--history-size", type=int, default=20)
    parser.add_argument("--tolerance-grad", type=float, default=1e-7)
    parser.add_argument("--tolerance-change", type=float, default=1e-9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    args = parser.parse_args()

    workdir = Path(args.workdir).expanduser().resolve()
    checkpoint_path = _resolve_checkpoint(workdir, args.checkpoint)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else workdir / "lbfgs_maccs_mean_probe" / checkpoint_path.stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(Path(args.config).expanduser().resolve())
    config.encoder_use_position_embedding = False
    config.msg_probe_fingerprint = MACCS_TASK

    device = torch.device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    global_step = int(checkpoint.get("global_step", -1))
    log.info("Checkpoint: %s (global_step=%d)", checkpoint_path, global_step)
    log.info("Encoder positional embedding enabled for probe: %s", False)

    cache_path = output_dir / "mean_pool_embeddings.pt"
    if cache_path.exists() and not args.force_extract:
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        log.info("Loaded cached embeddings from %s", cache_path)
    else:
        cache = _extract_embeddings(
            config=config,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        torch.save(cache, cache_path)
        log.info("Saved cached embeddings to %s", cache_path)

    weight_decay = (
        float(args.weight_decay)
        if args.weight_decay != 0.0
        else float(config.get("msg_probe_weight_decay", 0.0))
    )
    linear, train_objective, fit_seconds = _fit_lbfgs(
        cache=cache,
        device=device,
        max_iter=args.max_iter,
        history_size=args.history_size,
        tolerance_grad=args.tolerance_grad,
        tolerance_change=args.tolerance_change,
        weight_decay=weight_decay,
    )
    log.info(
        "LBFGS finished in %.1fs with final objective %.6f",
        fit_seconds,
        train_objective,
    )

    metrics: dict[str, float] = {
        "checkpoint/global_step": float(global_step),
        "lbfgs/max_iter": float(args.max_iter),
        "lbfgs/history_size": float(args.history_size),
        "lbfgs/weight_decay": float(weight_decay),
        "lbfgs/fit_seconds": float(fit_seconds),
        "lbfgs/train_objective": float(train_objective),
        "msg_probe/mean/num_maccs_bits": float(
            cache["massspec_train"]["maccs"].shape[1]
        ),
    }
    split_names = {
        "massspec_train": "train",
        "massspec_val": "val",
        "massspec_test": "test",
    }
    for cache_key, split_name in split_names.items():
        metrics.update(
            _evaluate_split(
                linear=linear,
                split_cache=cache[cache_key],
                prefix=f"msg_probe/mean/{split_name}",
                device=device,
            )
        )

    _write_json(output_dir / "metrics.json", metrics)
    torch.save(
        {
            "state_dict": linear.state_dict(),
            "input_dim": int(linear.in_features),
            "output_dim": int(linear.out_features),
            "checkpoint": str(checkpoint_path),
            "global_step": global_step,
            "encoder_use_position_embedding": False,
            "metrics": metrics,
        },
        output_dir / "lbfgs_linear_head.pt",
    )

    log.info(
        "Test MACCS: auc=%.4f ap=%.4f recall=%.4f precision=%.4f",
        metrics["msg_probe/mean/test/auc_maccs_mean"],
        metrics["msg_probe/mean/test/average_precision_maccs_mean"],
        metrics["msg_probe/mean/test/recall_maccs_mean"],
        metrics["msg_probe/mean/test/precision_maccs_mean"],
    )
    log.info("Wrote outputs to %s", output_dir)


if __name__ == "__main__":
    main()
