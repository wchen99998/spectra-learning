"""Train an offline attentive MACCS probe on frozen encoder token embeddings."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import random
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
from spectra_learning.probes.massspec.msg_modules import MsgPmaPool
from spectra_learning.probes.massspec.msg_probe import iter_massspec_probe
from spectra_learning.probes.massspec.msg_settings import (
    MACCS_TASK,
    resolve_msg_probe_sample_limits,
)
from spectra_learning.training.checkpointing import latest_ckpt_path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("attentive_maccs_probe")


class AttentiveMaccsProbe(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_seeds: int,
        num_heads: int,
        qk_norm: bool,
        norm_type: str,
    ) -> None:
        super().__init__()
        self.pooler = MsgPmaPool(
            input_dim=input_dim,
            num_seeds=num_seeds,
            num_heads=num_heads,
            qk_norm=qk_norm,
            norm_type=norm_type,
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        peak_embeddings: torch.Tensor,
        peak_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.head(self.pooler(peak_embeddings, peak_valid_mask))


def _resolve_checkpoint(workdir: Path, checkpoint: str) -> Path:
    if checkpoint:
        return Path(checkpoint).expanduser().resolve()
    last_path = workdir / "checkpoints" / "last.pt"
    if last_path.exists():
        return last_path
    return Path(latest_ckpt_path(workdir)).resolve()


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _split_seed_config(config) -> dict[str, tuple[int, int | None, bool]]:
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


def _extract_split_cache(
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
    embeddings, masks, targets, valid = [], [], [], []
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
            embeddings.append(peak_embeddings.detach().cpu().to(torch.float16))
            masks.append(batch["peak_valid_mask"].detach().cpu().to(torch.bool))
            targets.append(batch["probe_maccs"].detach().cpu().to(torch.float32))
            valid.append(batch["probe_valid_mol"].detach().cpu().to(torch.bool))
    split_cache = {
        "peak_embeddings": torch.cat(embeddings, dim=0),
        "peak_valid_mask": torch.cat(masks, dim=0),
        "maccs": torch.cat(targets, dim=0),
        "valid_mol": torch.cat(valid, dim=0),
    }
    log.info(
        "Extracted %s: %d rows, %d valid molecules, tokens=%d, dim=%d in %.1fs",
        split,
        int(split_cache["peak_embeddings"].shape[0]),
        int(split_cache["valid_mol"].sum()),
        int(split_cache["peak_embeddings"].shape[1]),
        int(split_cache["peak_embeddings"].shape[2]),
        time.time() - started,
    )
    return split_cache


def _extract_cache(
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
        split: _extract_split_cache(
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


def _valid_cache(split_cache: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    valid = split_cache["valid_mol"]
    return {
        "peak_embeddings": split_cache["peak_embeddings"][valid],
        "peak_valid_mask": split_cache["peak_valid_mask"][valid],
        "maccs": split_cache["maccs"][valid],
    }


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


def _iter_batches(
    split_cache: dict[str, torch.Tensor],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
):
    size = int(split_cache["maccs"].shape[0])
    if shuffle:
        generator = torch.Generator().manual_seed(int(seed))
        order = torch.randperm(size, generator=generator)
    else:
        order = torch.arange(size)
    for start in range(0, size, int(batch_size)):
        idx = order[start : start + int(batch_size)]
        yield (
            split_cache["peak_embeddings"][idx],
            split_cache["peak_valid_mask"][idx],
            split_cache["maccs"][idx],
        )


def _evaluate(
    *,
    probe: torch.nn.Module,
    split_cache: dict[str, torch.Tensor],
    prefix: str,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    logits, targets = [], []
    total_loss, total_count = 0.0, 0
    probe.eval()
    with torch.inference_mode():
        for peak_embeddings, peak_valid_mask, maccs in _iter_batches(
            split_cache,
            batch_size=batch_size,
            shuffle=False,
            seed=0,
        ):
            peak_embeddings = peak_embeddings.to(device=device, dtype=torch.float32)
            peak_valid_mask = peak_valid_mask.to(device=device)
            maccs = maccs.to(device=device)
            batch_logits = probe(peak_embeddings, peak_valid_mask)
            loss = F.binary_cross_entropy_with_logits(batch_logits, maccs)
            total_loss += float(loss.detach().cpu()) * int(maccs.shape[0])
            total_count += int(maccs.shape[0])
            logits.append(batch_logits.detach().cpu().numpy())
            targets.append(maccs.detach().cpu().numpy())
    return _score_maccs(
        prefix=prefix,
        logits=np.concatenate(logits, axis=0),
        target=np.concatenate(targets, axis=0),
        loss=total_loss / total_count,
    )


def _train_once(
    *,
    config,
    cache: dict[str, dict[str, torch.Tensor]],
    device: torch.device,
    seed: int,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    min_epochs: int,
    min_delta: float,
) -> tuple[torch.nn.Module, dict[str, float], list[dict[str, float]]]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    random.seed(int(seed))
    train_cache = _valid_cache(cache["massspec_train"])
    val_cache = _valid_cache(cache["massspec_val"])
    probe = AttentiveMaccsProbe(
        input_dim=int(train_cache["peak_embeddings"].shape[-1]),
        output_dim=int(train_cache["maccs"].shape[-1]),
        hidden_dim=int(config.get("msg_probe_mlp_hidden_dim", config.model_dim)),
        num_seeds=int(config.get("msg_probe_pma_num_seeds", 32)),
        num_heads=int(config.get("msg_probe_pma_num_heads", config.encoder_num_heads)),
        qk_norm=bool(config.get("encoder_qk_norm", False)),
        norm_type=str(config.get("norm_type", "layernorm")),
    ).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    best_auc = -float("inf")
    best_state: dict[str, torch.Tensor] = {}
    best_metrics: dict[str, float] = {}
    curve: list[dict[str, float]] = []
    bad_epochs = 0
    for epoch_idx in range(int(num_epochs)):
        probe.train()
        for peak_embeddings, peak_valid_mask, maccs in _iter_batches(
            train_cache,
            batch_size=batch_size,
            shuffle=True,
            seed=int(seed) + epoch_idx,
        ):
            peak_embeddings = peak_embeddings.to(device=device, dtype=torch.float32)
            peak_valid_mask = peak_valid_mask.to(device=device)
            maccs = maccs.to(device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = probe(peak_embeddings, peak_valid_mask)
            loss = F.binary_cross_entropy_with_logits(logits, maccs)
            loss.backward()
            optimizer.step()
        metrics = {
            **_evaluate(
                probe=probe,
                split_cache=train_cache,
                prefix="attentive_maccs/train",
                batch_size=batch_size,
                device=device,
            ),
            **_evaluate(
                probe=probe,
                split_cache=val_cache,
                prefix="attentive_maccs/val",
                batch_size=batch_size,
                device=device,
            ),
            "attentive_maccs/epoch": float(epoch_idx + 1),
        }
        curve.append(metrics)
        val_auc = float(metrics["attentive_maccs/val/auc_maccs_mean"])
        if val_auc > best_auc + float(min_delta):
            best_auc = val_auc
            best_state = copy.deepcopy(probe.state_dict())
            best_metrics = dict(metrics)
            bad_epochs = 0
        else:
            bad_epochs += 1
        log.info(
            "seed=%d epoch %d/%d train_auc=%.4f val_auc=%.4f val_ap=%.4f val_bce=%.4f",
            seed,
            epoch_idx + 1,
            num_epochs,
            metrics["attentive_maccs/train/auc_maccs_mean"],
            metrics["attentive_maccs/val/auc_maccs_mean"],
            metrics["attentive_maccs/val/average_precision_maccs_mean"],
            metrics["attentive_maccs/val/loss_bce"],
        )
        if epoch_idx + 1 >= int(min_epochs) and bad_epochs >= int(patience):
            break
    probe.load_state_dict(best_state)
    return probe, best_metrics, curve


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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=0.0001)
    parser.add_argument("--seeds", type=int, default=1)
    args = parser.parse_args()

    workdir = Path(args.workdir).expanduser().resolve()
    checkpoint_path = _resolve_checkpoint(workdir, args.checkpoint)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else workdir / "attentive_maccs_probe" / checkpoint_path.stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(Path(args.config).expanduser().resolve())
    config.encoder_use_position_embedding = False
    config.msg_probe_fingerprint = MACCS_TASK
    batch_size = int(args.batch_size or config.get("msg_probe_batch_size", config.batch_size))
    device = torch.device(args.device)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    global_step = int(checkpoint.get("global_step", -1))
    log.info("Checkpoint: %s (global_step=%d)", checkpoint_path, global_step)
    log.info("Encoder positional embedding enabled for probe: %s", False)

    cache_path = output_dir / "sequence_embeddings.pt"
    if cache_path.exists() and not args.force_extract:
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        log.info("Loaded cached sequence embeddings from %s", cache_path)
    else:
        cache = _extract_cache(
            config=config,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        torch.save(cache, cache_path)
        log.info("Saved cached sequence embeddings to %s", cache_path)

    trials = []
    best_trial: dict[str, Any] | None = None
    best_probe: torch.nn.Module | None = None
    started = time.time()
    for seed_idx in range(int(args.seeds)):
        seed = int(config.seed) + 10_000 * seed_idx
        probe, best_metrics, curve = _train_once(
            config=config,
            cache=cache,
            device=device,
            seed=seed,
            num_epochs=args.epochs,
            batch_size=batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            min_epochs=args.min_epochs,
            min_delta=args.min_delta,
        )
        test_metrics = _evaluate(
            probe=probe,
            split_cache=_valid_cache(cache["massspec_test"]),
            prefix="attentive_maccs/test",
            batch_size=batch_size,
            device=device,
        )
        trial = {
            "seed": seed,
            "best_metrics": {**best_metrics, **test_metrics},
            "curve": curve,
            "best_val_auc": best_metrics["attentive_maccs/val/auc_maccs_mean"],
            "best_epoch": int(best_metrics["attentive_maccs/epoch"]),
        }
        trials.append(trial)
        if best_trial is None or trial["best_val_auc"] > best_trial["best_val_auc"]:
            best_trial = trial
            best_probe = probe
        log.info(
            "seed=%d best_epoch=%d val_auc=%.4f test_auc=%.4f test_ap=%.4f",
            seed,
            trial["best_epoch"],
            trial["best_metrics"]["attentive_maccs/val/auc_maccs_mean"],
            trial["best_metrics"]["attentive_maccs/test/auc_maccs_mean"],
            trial["best_metrics"]["attentive_maccs/test/average_precision_maccs_mean"],
        )

    assert best_trial is not None and best_probe is not None
    payload = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_global_step": global_step,
        "encoder_use_position_embedding": False,
        "fit_seconds": time.time() - started,
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "batch_size": int(batch_size),
        "num_seeds": int(args.seeds),
        "best_seed": int(best_trial["seed"]),
        "best_epoch": int(best_trial["best_epoch"]),
        "metrics": best_trial["best_metrics"],
        "trials": trials,
    }
    _write_json(output_dir / "metrics.json", payload)
    torch.save(
        {
            "state_dict": best_probe.state_dict(),
            "checkpoint": str(checkpoint_path),
            "global_step": global_step,
            "encoder_use_position_embedding": False,
            "metrics": best_trial["best_metrics"],
        },
        output_dir / "attentive_maccs_probe.pt",
    )
    log.info(
        "Best attentive test MACCS: auc=%.4f ap=%.4f recall=%.4f precision=%.4f",
        best_trial["best_metrics"]["attentive_maccs/test/auc_maccs_mean"],
        best_trial["best_metrics"][
            "attentive_maccs/test/average_precision_maccs_mean"
        ],
        best_trial["best_metrics"]["attentive_maccs/test/recall_maccs_mean"],
        best_trial["best_metrics"]["attentive_maccs/test/precision_maccs_mean"],
    )
    log.info("Wrote outputs to %s", output_dir)


if __name__ == "__main__":
    main()
