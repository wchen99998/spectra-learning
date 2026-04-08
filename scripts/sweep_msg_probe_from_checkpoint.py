"""Exhaustive MSG probe sweep from a fixed checkpoint.

This script:
1. Extracts frozen encoder token embeddings once for a checkpoint.
2. Exhaustively sweeps probe-head capacity and optimizer hyperparameters.
3. Ranks trials by the best probe epoch on a chosen validation metric.
4. Reevaluates saved checkpoints with the default probe config and the best tuned config.
5. Writes JSON/CSV/Markdown summaries plus PNG plots.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from ml_collections import config_dict

from input_pipeline import numpy_batch_to_torch
from models.model import PeakSetEncoder
from utils.massspec_probe_data import MassSpecProbeData
from utils.msg_probe import (
    FG_SMARTS,
    REGRESSION_TARGET_KEYS,
    MsgLinearProbe,
    MsgProbePooler,
    MsgProbeSplitTargets,
    _build_task_spec,
    _probe_task_names,
    _probe_task_output_dims,
    msg_probe_metric_higher_is_better,
    _new_epoch_state,
    _probe_step,
    _score_epoch_state,
    _update_epoch_state,
)
from utils.schedulers import learning_rate_at_step
from utils.training import (
    build_model_from_config,
    latest_ckpt_path,
    load_config,
    load_pretrained_weights,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("sweep_msg_probe_from_checkpoint")


def _clone_config(config: config_dict.ConfigDict) -> config_dict.ConfigDict:
    return config.copy_and_resolve_references()


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _load_param_space(
    config: config_dict.ConfigDict,
    override_json: str,
) -> list[dict[str, Any]]:
    if override_json:
        payload = json.loads(override_json)
    else:
        payload = list(config.get("msg_probe_tune_param_space", []))
    if isinstance(payload, dict):
        return [
            {"param": name, "dist": "grid", "args": values}
            for name, values in payload.items()
        ]
    return payload


def _expand_grid_param_space(
    param_space: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if param_space and "param" not in param_space[0]:
        return [dict(entry) for entry in param_space]
    names: list[str] = []
    values: list[list[Any]] = []
    for entry in param_space:
        dist = str(entry.get("dist", "grid"))
        if dist != "grid":
            raise ValueError(f"Only grid search is supported here, got {dist!r}")
        names.append(str(entry["param"]))
        values.append(list(entry.get("args", entry.get("values", []))))
    if not names:
        return [{}]
    return [dict(zip(names, combo)) for combo in itertools.product(*values)]


def _format_param_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2g}"
    return str(value)


def _trial_name(idx: int, params: dict[str, Any]) -> str:
    if not params:
        return f"trial_{idx:03d}"
    parts = [f"{key}={_format_param_value(value)}" for key, value in params.items()]
    return f"trial_{idx:03d}_" + "_".join(parts)


def _checkpoint_step(path: Path) -> int | None:
    match = re.search(r"step-(\d+)", path.stem)
    return int(match.group(1)) if match else None


def _iter_probe_batches(
    probe_data: MassSpecProbeData,
    split: str,
    *,
    peak_ordering: str,
    max_samples: int | None,
):
    dataset = probe_data.build_dataset(
        split,
        seed=0,
        peak_ordering=peak_ordering,
        shuffle=False,
        drop_remainder=False,
    )
    size = int(probe_data.info[f"{split}_size"])
    if max_samples is not None:
        size = min(size, int(max_samples))
    seen = 0
    for batch in dataset.as_numpy_iterator():
        if seen >= size:
            break
        take = min(int(batch["peak_mz"].shape[0]), size - seen)
        if take != batch["peak_mz"].shape[0]:
            batch = {key: value[:take] for key, value in batch.items()}
        seen += take
        yield numpy_batch_to_torch(batch)


def _allocate_split_cache(
    size: int,
    token_shape: tuple[int, int],
) -> dict[str, torch.Tensor]:
    num_tokens, model_dim = token_shape
    cache: dict[str, torch.Tensor] = {
        "token_embeddings": torch.empty(
            (size, num_tokens, model_dim),
            dtype=torch.float16,
        ),
        "peak_valid_mask": torch.empty((size, num_tokens), dtype=torch.bool),
        "probe_valid_mol": torch.empty(size, dtype=torch.bool),
    }
    for name in REGRESSION_TARGET_KEYS:
        cache[f"probe_{name}"] = torch.empty(size, dtype=torch.float32)
    for name in FG_SMARTS:
        cache[f"probe_fg_{name}"] = torch.empty(size, dtype=torch.int32)
    return cache


def _extract_split_cache(
    *,
    model: torch.nn.Module,
    device: torch.device,
    probe_data: MassSpecProbeData,
    split: str,
    peak_ordering: str,
    max_samples: int | None,
) -> dict[str, torch.Tensor]:
    size = int(probe_data.info[f"{split}_size"])
    if max_samples is not None:
        size = min(size, int(max_samples))
    cache: dict[str, torch.Tensor] | None = None
    offset = 0
    started = time.time()
    with torch.no_grad():
        for batch in _iter_probe_batches(
            probe_data,
            split,
            peak_ordering=peak_ordering,
            max_samples=max_samples,
        ):
            peak_mz = batch["peak_mz"].to(device)
            peak_intensity = batch["peak_intensity"].to(device)
            peak_valid_mask = batch["peak_valid_mask"].to(device)
            token_embeddings = model.encoder(
                peak_mz,
                peak_intensity,
                valid_mask=peak_valid_mask,
            )
            token_embeddings, _ = PeakSetEncoder.split_peak_and_cls(token_embeddings)
            if cache is None:
                cache = _allocate_split_cache(
                    size=size,
                    token_shape=(
                        int(token_embeddings.shape[1]),
                        int(token_embeddings.shape[2]),
                    ),
                )
            take = int(token_embeddings.shape[0])
            sl = slice(offset, offset + take)
            cache["token_embeddings"][sl] = token_embeddings.cpu().to(torch.float16)
            cache["peak_valid_mask"][sl] = batch["peak_valid_mask"].cpu()
            cache["probe_valid_mol"][sl] = batch["probe_valid_mol"].to(torch.bool).cpu()
            for name in REGRESSION_TARGET_KEYS:
                cache[f"probe_{name}"][sl] = batch[f"probe_{name}"].to(torch.float32).cpu()
            for name in FG_SMARTS:
                cache[f"probe_fg_{name}"][sl] = batch[f"probe_fg_{name}"].to(torch.int32).cpu()
            offset += take
    assert cache is not None
    log.info(
        "Extracted %s cache: %d samples, tokens=%d, dim=%d in %.1fs",
        split,
        size,
        int(cache["token_embeddings"].shape[1]),
        int(cache["token_embeddings"].shape[2]),
        time.time() - started,
    )
    return cache


def extract_checkpoint_cache(
    *,
    config: config_dict.ConfigDict,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    probe_data = MassSpecProbeData.from_config(config)
    model = build_model_from_config(config)
    load_pretrained_weights(model, str(checkpoint_path))
    model.to(device).eval()
    peak_ordering = str(config.get("peak_ordering", "intensity"))
    _mts = config.get("msg_probe_max_train_samples", None)
    _mte = config.get("msg_probe_max_test_samples", None)
    max_train_samples = int(_mts) if _mts is not None else None
    max_test_samples = int(_mte) if _mte is not None else None
    log.info("Extracting frozen embeddings from %s on %s", checkpoint_path, device)
    train_cache = _extract_split_cache(
        model=model,
        device=device,
        probe_data=probe_data,
        split="massspec_train",
        peak_ordering=peak_ordering,
        max_samples=max_train_samples,
    )
    test_cache = _extract_split_cache(
        model=model,
        device=device,
        probe_data=probe_data,
        split="massspec_test",
        peak_ordering=peak_ordering,
        max_samples=max_test_samples,
    )
    return {"train": train_cache, "test": test_cache}


def _build_split_targets_from_cache(
    cache: dict[str, torch.Tensor],
) -> MsgProbeSplitTargets:
    valid = cache["probe_valid_mol"].numpy().astype(bool, copy=False)
    regression = {
        name: cache[f"probe_{name}"].numpy()[valid].astype(np.float32, copy=False)
        for name in REGRESSION_TARGET_KEYS
    }
    classification = {
        name: cache[f"probe_fg_{name}"].numpy()[valid].astype(np.int32, copy=False)
        for name in FG_SMARTS
    }
    return MsgProbeSplitTargets(regression=regression, classification=classification)


def _iter_cached_batches(
    cache: dict[str, torch.Tensor],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
):
    size = int(cache["probe_valid_mol"].shape[0])
    if shuffle:
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(size, generator=generator)
    else:
        order = torch.arange(size)
    for start in range(0, size, batch_size):
        idx = order[start : start + batch_size]
        yield {key: value[idx] for key, value in cache.items()}


def _move_cached_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    moved: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if key == "token_embeddings":
            moved[key] = value.to(device=device, dtype=torch.float32)
        else:
            moved[key] = value.to(device=device)
    return moved


def run_cached_msg_probe(
    *,
    config: config_dict.ConfigDict,
    train_cache: dict[str, torch.Tensor],
    test_cache: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    task_spec = _build_task_spec(
        train_targets=_build_split_targets_from_cache(train_cache),
        test_targets=_build_split_targets_from_cache(test_cache),
    )
    pooler = MsgProbePooler(
        model_dim=int(train_cache["token_embeddings"].shape[-1]),
        pooling_type=str(config.get("msg_probe_pooling_type", "pma")),
        pma_num_heads=int(
            config.get(
                "msg_probe_pma_num_heads",
                config.get("encoder_num_heads", config.get("num_heads")),
            )
        ),
        pma_num_seeds=int(config.get("msg_probe_pma_num_seeds", 1)),
        norm_type=str(config.get("norm_type", "rmsnorm")),
    )
    probe = MsgLinearProbe(
        input_dim=int(train_cache["token_embeddings"].shape[-1]),
        task_names=_probe_task_names(task_spec),
        pooler=pooler,
        task_output_dims=_probe_task_output_dims(task_spec),
        hidden_dim=int(config.get("msg_probe_hidden_dim", 0)),
        num_layers=int(config.get("msg_probe_num_layers", 1)),
        dropout=float(config.get("msg_probe_dropout", 0.0)),
        activation=str(config.get("msg_probe_activation", "gelu")),
        init_method=str(config.get("msg_probe_init", "default")),
    ).to(device)
    num_probe_epochs = int(config.get("msg_probe_num_epochs", 20))
    probe_lr = float(config.get("msg_probe_learning_rate", 1e-3))
    probe_weight_decay = float(config.get("msg_probe_weight_decay", 1e-2))
    probe_warmup_steps = int(config.get("msg_probe_warmup_steps", 0))
    batch_size = int(config.batch_size)
    train_size = int(train_cache["probe_valid_mol"].shape[0])
    steps_per_epoch = math.ceil(train_size / batch_size)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=probe_lr,
        weight_decay=probe_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step_idx: (
            learning_rate_at_step(
                step_idx + 1,
                base_lr=probe_lr,
                total_steps=max(1, num_probe_epochs * steps_per_epoch),
                warmup_steps=probe_warmup_steps,
            )
            / probe_lr
        ),
    )

    def feature_extractor(
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return batch["token_embeddings"], batch["peak_valid_mask"]

    final_metrics: dict[str, float] = {}
    curve: list[dict[str, float]] = []
    for epoch_idx in range(num_probe_epochs):
        probe.train()
        train_state = _new_epoch_state(task_spec)
        for batch in _iter_cached_batches(
            train_cache,
            batch_size=batch_size,
            shuffle=True,
            seed=int(config.seed) + epoch_idx,
        ):
            batch = _move_cached_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            result = _probe_step(
                probe,
                batch,
                task_spec=task_spec,
                device=device,
                feature_extractor=feature_extractor,
            )
            if result is None:
                continue
            result["loss_total"].backward()
            optimizer.step()
            scheduler.step()
            _update_epoch_state(train_state, result, task_spec)

        probe.eval()
        test_state = _new_epoch_state(task_spec)
        with torch.no_grad():
            for batch in _iter_cached_batches(
                test_cache,
                batch_size=batch_size,
                shuffle=False,
                seed=0,
            ):
                batch = _move_cached_batch(batch, device)
                result = _probe_step(
                    probe,
                    batch,
                    task_spec=task_spec,
                    device=device,
                    feature_extractor=feature_extractor,
                )
                if result is None:
                    continue
                _update_epoch_state(test_state, result, task_spec)
        final_metrics = {
            **_score_epoch_state(
                prefix="msg_probe/train",
                epoch_state=train_state,
                task_spec=task_spec,
            ),
            **_score_epoch_state(
                prefix="msg_probe/test",
                epoch_state=test_state,
                task_spec=task_spec,
            ),
            "msg_probe/num_fg_tasks": float(len(task_spec.classification_tasks)),
            "msg_probe_epoch": float(epoch_idx + 1),
        }
        curve.append(dict(final_metrics))
    return {"final_metrics": final_metrics, "curve": curve}


def _select_best_epoch(
    curve: list[dict[str, float]],
    metric_key: str,
) -> dict[str, float]:
    if not curve:
        raise ValueError("Probe curve is empty")
    if msg_probe_metric_higher_is_better(metric_key):
        return max(curve, key=lambda metrics: float(metrics[metric_key]))
    return min(curve, key=lambda metrics: float(metrics[metric_key]))


def _evaluate_trial(
    *,
    base_config: config_dict.ConfigDict,
    overrides: dict[str, Any],
    train_cache: dict[str, torch.Tensor],
    test_cache: dict[str, torch.Tensor],
    device: torch.device,
    metric_key: str,
    trial_name: str,
) -> dict[str, Any]:
    trial_config = _clone_config(base_config)
    trial_config.update(overrides)
    _seed_everything(int(trial_config.seed))
    started = time.time()
    run = run_cached_msg_probe(
        config=trial_config,
        train_cache=train_cache,
        test_cache=test_cache,
        device=device,
    )
    best_metrics = _select_best_epoch(run["curve"], metric_key)
    result = {
        "name": trial_name,
        "params": dict(overrides),
        "elapsed_seconds": time.time() - started,
        "metric_key": metric_key,
        "best_metric_value": float(best_metrics[metric_key]),
        "best_epoch": int(best_metrics["msg_probe_epoch"]),
        "best_metrics": best_metrics,
        "final_metrics": run["final_metrics"],
        "curve": run["curve"],
    }
    log.info(
        "%s best %s=%.4f at epoch %d (final %.4f, %.1fs)",
        trial_name,
        metric_key,
        result["best_metric_value"],
        result["best_epoch"],
        float(run["final_metrics"][metric_key]),
        result["elapsed_seconds"],
    )
    return result


def _serialise_trial_rows(
    trials: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ranked_trials = _rank_trials(trials)
    rows: list[dict[str, Any]] = []
    for rank, trial in enumerate(ranked_trials, start=1):
        row = {
            "rank": rank,
            "name": trial["name"],
            "metric_key": trial["metric_key"],
            "best_metric_value": trial["best_metric_value"],
            "best_epoch": trial["best_epoch"],
            "final_metric_value": float(trial["final_metrics"][trial["metric_key"]]),
            "elapsed_seconds": trial["elapsed_seconds"],
            "params_json": json.dumps(trial["params"], sort_keys=True),
        }
        for key, value in trial["params"].items():
            row[key] = value
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _rank_trials(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not trials:
        return []
    metric_key = str(trials[0]["metric_key"])
    return sorted(
        trials,
        key=lambda item: item["best_metric_value"],
        reverse=msg_probe_metric_higher_is_better(metric_key),
    )


def _plot_trial_ranking(path: Path, trials: list[dict[str, Any]]) -> None:
    ranked = _rank_trials(trials)
    xs = np.arange(1, len(ranked) + 1)
    ys = np.asarray([trial["best_metric_value"] for trial in ranked], dtype=np.float32)
    labels = [trial["name"] for trial in ranked]
    plt.figure(figsize=(11, 5))
    plt.plot(xs, ys, marker="o", linewidth=1.5)
    for idx, label in enumerate(labels[:10], start=1):
        plt.annotate(label, (idx, ys[idx - 1]), fontsize=8, xytext=(0, 6), textcoords="offset points")
    plt.xlabel("Trial rank")
    plt.ylabel(ranked[0]["metric_key"])
    plt.title("Fixed-checkpoint MSG probe sweep")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _find_online_probe_log(workdir: Path) -> Path | None:
    candidates = sorted(workdir.glob("wandb/run-*/files/output.log"))
    return candidates[-1] if candidates else None


def _parse_online_probe_curve(workdir: Path) -> list[dict[str, float]]:
    log_path = _find_online_probe_log(workdir)
    if log_path is None:
        return []
    pattern = re.compile(
        r"step=(?P<step>\d+) msg_probe\(test_r2_mol_weight=(?P<r2>[0-9.]+) "
        r"test_auc_fg_mean=(?P<auc>[0-9.]+)"
    )
    rows: list[dict[str, float]] = []
    for line in log_path.read_text().splitlines():
        match = pattern.search(line)
        if match is None:
            continue
        rows.append(
            {
                "checkpoint_step": float(match.group("step")),
                "online_test_auc_fg_mean": float(match.group("auc")),
                "online_test_r2_mol_weight": float(match.group("r2")),
            }
        )
    return rows


def _plot_checkpoint_curve(
    path: Path,
    checkpoint_rows: list[dict[str, Any]],
    online_rows: list[dict[str, float]],
    metric_key: str,
) -> None:
    plt.figure(figsize=(11, 5))
    if online_rows:
        steps = [row["checkpoint_step"] for row in online_rows]
        values = [row["online_test_auc_fg_mean"] for row in online_rows]
        plt.plot(steps, values, marker="o", label="online final probe")
    by_setting: dict[str, list[dict[str, Any]]] = {}
    for row in checkpoint_rows:
        by_setting.setdefault(str(row["setting"]), []).append(row)
    for setting, rows in sorted(by_setting.items()):
        rows.sort(key=lambda item: item["checkpoint_step"])
        plt.plot(
            [row["checkpoint_step"] for row in rows],
            [row["best_metric_value"] for row in rows],
            marker="o",
            label=setting,
        )
    plt.xlabel("Checkpoint step")
    plt.ylabel(metric_key)
    plt.title("Checkpoint sweep: online vs offline probe")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _write_summary(
    path: Path,
    *,
    checkpoint_path: Path,
    metric_key: str,
    ranked_trials: list[dict[str, Any]],
    checkpoint_rows: list[dict[str, Any]],
    online_rows: list[dict[str, float]],
) -> None:
    best = max(ranked_trials, key=lambda item: item["best_metric_value"])
    default = next(trial for trial in ranked_trials if trial["name"] == "default")
    lines = [
        "# MSG Probe Sweep",
        "",
        f"- Fixed checkpoint: `{checkpoint_path}`",
        f"- Ranking metric: `{metric_key}`",
        f"- Trials: {len(ranked_trials)}",
        "",
        "## Fixed Checkpoint",
        "",
        f"- Default best: {default['best_metric_value']:.4f} at epoch {default['best_epoch']}",
        f"- Best tuned: {best['best_metric_value']:.4f} at epoch {best['best_epoch']}",
        f"- Improvement: {best['best_metric_value'] - default['best_metric_value']:+.4f}",
        f"- Best params: `{json.dumps(best['params'], sort_keys=True)}`",
        "",
        "## Top Trials",
        "",
        "| Rank | Trial | Best metric | Best epoch | Final metric |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    sorted_trials = sorted(
        ranked_trials,
        key=lambda item: item["best_metric_value"],
        reverse=True,
    )
    for rank, trial in enumerate(sorted_trials[:10], start=1):
        lines.append(
            "| "
            f"{rank} | {trial['name']} | {trial['best_metric_value']:.4f} | "
            f"{trial['best_epoch']} | {float(trial['final_metrics'][metric_key]):.4f} |"
        )
    if checkpoint_rows:
        lines.extend(
            [
                "",
                "## Checkpoint Sweep",
                "",
                "| Step | Setting | Best metric | Best epoch |",
                "| ---: | --- | ---: | ---: |",
            ]
        )
        for row in sorted(
            checkpoint_rows,
            key=lambda item: (item["checkpoint_step"], item["setting"]),
        ):
            lines.append(
                f"| {row['checkpoint_step']} | {row['setting']} | "
                f"{row['best_metric_value']:.4f} | {row['best_epoch']} |"
            )
    if online_rows:
        lines.extend(
            [
                "",
                "## Online Probe",
                "",
                f"- Parsed {len(online_rows)} online probe points from the existing run log.",
            ]
        )
    path.write_text("\n".join(lines) + "\n")


def _resolve_device(name: str) -> torch.device:
    if name == "cuda":
        return torch.device("cuda")
    return torch.device("cpu")


def _collect_checkpoint_paths(
    workdir: Path,
    pattern: str,
) -> list[Path]:
    checkpoint_dir = workdir / "checkpoints"
    paths = sorted(checkpoint_dir.glob(pattern))
    return [path for path in paths if path.is_file()]


# ---------------------------------------------------------------------------
# Bayesian optimisation (sklearn GP + scipy EI)
# ---------------------------------------------------------------------------

from dataclasses import dataclass


@dataclass
class RealDim:
    name: str
    low: float
    high: float
    log_scale: bool = False


@dataclass
class CategoricalDim:
    name: str
    categories: list[Any]


SearchDim = RealDim | CategoricalDim

DEFAULT_BAYESIAN_DIMS: list[SearchDim] = [
    RealDim("msg_probe_learning_rate", low=5e-5, high=3e-3, log_scale=True),
    RealDim("msg_probe_weight_decay", low=0.0, high=0.15),
    RealDim("msg_probe_dropout", low=0.0, high=0.5),
    CategoricalDim("msg_probe_hidden_dim", [256, 512, 1024]),
    CategoricalDim("msg_probe_num_layers", [1, 2, 3]),
    CategoricalDim("msg_probe_activation", ["gelu", "silu", "relu"]),
    CategoricalDim("msg_probe_init", ["default", "xavier_uniform", "xavier_normal", "kaiming_normal", "orthogonal"]),
]


def _encode_point(dims: list[SearchDim], params: dict[str, Any]) -> np.ndarray:
    """Encode a parameter dict to a [0, 1]^d vector for the GP."""
    x: list[float] = []
    for dim in dims:
        v = params[dim.name]
        if isinstance(dim, RealDim):
            if dim.log_scale:
                v = (np.log(v) - np.log(dim.low)) / (np.log(dim.high) - np.log(dim.low))
            else:
                span = dim.high - dim.low
                v = (v - dim.low) / span if span > 0 else 0.5
            x.append(float(np.clip(v, 0.0, 1.0)))
        else:
            idx = dim.categories.index(v)
            x.append(idx / max(1, len(dim.categories) - 1))
    return np.asarray(x, dtype=np.float64)


def _decode_point(dims: list[SearchDim], x: np.ndarray) -> dict[str, Any]:
    """Decode a [0, 1]^d vector back to a parameter dict."""
    params: dict[str, Any] = {}
    for i, dim in enumerate(dims):
        v = float(x[i])
        if isinstance(dim, RealDim):
            if dim.log_scale:
                v = float(np.exp(np.log(dim.low) + v * (np.log(dim.high) - np.log(dim.low))))
            else:
                v = dim.low + v * (dim.high - dim.low)
            params[dim.name] = v
        else:
            idx = int(round(v * (len(dim.categories) - 1)))
            idx = int(np.clip(idx, 0, len(dim.categories) - 1))
            params[dim.name] = dim.categories[idx]
    return params


def _random_samples(
    dims: list[SearchDim],
    n: int,
    rng: np.random.RandomState,
) -> list[dict[str, Any]]:
    """Latin Hypercube sampling in [0, 1]^d, decoded to param dicts."""
    d = len(dims)
    # Stratified LHS
    cuts = np.linspace(0, 1, n + 1)
    samples = np.empty((n, d), dtype=np.float64)
    for j in range(d):
        order = rng.permutation(n)
        for i in range(n):
            samples[order[i], j] = rng.uniform(cuts[i], cuts[i + 1])
    return [_decode_point(dims, samples[i]) for i in range(n)]


def _expected_improvement(
    X_new: np.ndarray,
    gp,
    y_best: float,
) -> np.ndarray:
    """Compute Expected Improvement (maximise)."""
    from scipy.stats import norm as _norm

    mu, sigma = gp.predict(X_new, return_std=True)
    sigma = np.maximum(sigma, 1e-8)
    z = (mu - y_best) / sigma
    return (mu - y_best) * _norm.cdf(z) + sigma * _norm.pdf(z)


def _bayesian_suggest(
    dims: list[SearchDim],
    X_obs: np.ndarray,
    y_obs: np.ndarray,
    rng: np.random.RandomState,
    n_restarts: int = 64,
) -> dict[str, Any]:
    """Fit GP, maximise EI via multi-start L-BFGS-B, return next params."""
    from scipy.optimize import minimize as _scipy_minimize
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        nu=2.5,
        length_scale=np.ones(X_obs.shape[1]),
        length_scale_bounds=(1e-3, 1e2),
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        normalize_y=True,
        random_state=int(rng.randint(0, 2**31)),
    )
    gp.fit(X_obs, y_obs)
    y_best = float(np.max(y_obs))
    d = X_obs.shape[1]
    bounds = [(0.0, 1.0)] * d

    best_x: np.ndarray | None = None
    best_ei = -np.inf
    for _ in range(n_restarts):
        x0 = rng.uniform(0, 1, size=d)
        result = _scipy_minimize(
            lambda x: -float(_expected_improvement(x.reshape(1, -1), gp, y_best)[0]),
            x0,
            bounds=bounds,
            method="L-BFGS-B",
        )
        ei = -result.fun
        if ei > best_ei:
            best_ei = ei
            best_x = result.x
    assert best_x is not None
    return _decode_point(dims, best_x)


def _parse_bayesian_dims(payload: list[dict[str, Any]]) -> list[SearchDim]:
    """Parse a JSON search-space spec into SearchDim objects."""
    dims: list[SearchDim] = []
    for entry in payload:
        if entry.get("type") == "real":
            dims.append(RealDim(
                name=entry["name"],
                low=float(entry["low"]),
                high=float(entry["high"]),
                log_scale=bool(entry.get("log_scale", False)),
            ))
        else:
            dims.append(CategoricalDim(
                name=entry["name"],
                categories=list(entry["categories"]),
            ))
    return dims


# ---------------------------------------------------------------------------
# Bayesian-specific visualisation
# ---------------------------------------------------------------------------

def _plot_convergence(path: Path, trials: list[dict[str, Any]], metric_key: str) -> None:
    """Objective value vs. trial number + running best."""
    ys = [t["best_metric_value"] for t in trials]
    running_best = np.maximum.accumulate(ys)
    xs = np.arange(1, len(ys) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(xs, ys, c="#4C72B0", s=30, alpha=0.7, label="Trial result", zorder=3)
    ax.plot(xs, running_best, color="#C44E52", linewidth=2, label="Running best", zorder=4)
    ax.set_xlabel("Trial number")
    ax.set_ylabel(metric_key)
    ax.set_title("Bayesian Optimisation Convergence")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_param_importance(
    path: Path,
    trials: list[dict[str, Any]],
    dims: list[SearchDim],
    metric_key: str,
) -> None:
    """Scatter plots of each hyperparameter vs. objective value."""
    n_dims = len(dims)
    cols = min(4, n_dims)
    rows = math.ceil(n_dims / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.5 * rows), squeeze=False)

    ys = np.asarray([t["best_metric_value"] for t in trials])
    y_min, y_max = float(ys.min()), float(ys.max())

    for idx, dim in enumerate(dims):
        ax = axes[idx // cols][idx % cols]
        xs_raw = [t["params"].get(dim.name) for t in trials]

        if isinstance(dim, CategoricalDim):
            cats = dim.categories
            cat_indices = [cats.index(v) if v in cats else -1 for v in xs_raw]
            ax.scatter(cat_indices, ys, c=ys, cmap="RdYlGn", s=30, alpha=0.7, vmin=y_min, vmax=y_max)
            ax.set_xticks(range(len(cats)))
            ax.set_xticklabels([str(c) for c in cats], rotation=45, ha="right", fontsize=8)
        else:
            xs_float = [float(v) for v in xs_raw]
            ax.scatter(xs_float, ys, c=ys, cmap="RdYlGn", s=30, alpha=0.7, vmin=y_min, vmax=y_max)
            if dim.log_scale:
                ax.set_xscale("log")
        ax.set_xlabel(dim.name.replace("msg_probe_", ""), fontsize=9)
        ax.set_ylabel(metric_key.split("/")[-1], fontsize=9)
        ax.grid(alpha=0.3)

    for idx in range(n_dims, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)
    fig.suptitle("Hyperparameter vs. Objective", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_epoch_curves(
    path: Path,
    trials: list[dict[str, Any]],
    top_n: int = 8,
) -> None:
    """Train/test AUC + R2 curves for the top-N trials (2x2 subplots)."""
    ranked = _rank_trials(trials)[:top_n]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = [
        ("msg_probe/test/auc_fg_mean", "msg_probe/train/auc_fg_mean", "AUC (fg_mean)"),
        (
            "msg_probe/test/r2_mean_wo_num_rings",
            "msg_probe/train/r2_mean_wo_num_rings",
            "R² (mean, no num_rings)",
        ),
        ("msg_probe/test/r2_mol_weight", "msg_probe/train/r2_mol_weight", "R² (mol_weight)"),
        ("msg_probe/test/mae_num_rings", "msg_probe/train/mae_num_rings", "MAE (num_rings)"),
    ]
    cmap = plt.get_cmap("tab10")

    for ax, (test_key, train_key, title) in zip(axes.flat, metrics):
        for rank, trial in enumerate(ranked):
            color = cmap(rank)
            epochs = [int(e["msg_probe_epoch"]) for e in trial["curve"]]
            test_vals = [e.get(test_key, float("nan")) for e in trial["curve"]]
            train_vals = [e.get(train_key, float("nan")) for e in trial["curve"]]
            short_name = trial["name"][:30]
            ax.plot(epochs, test_vals, color=color, linewidth=1.5, label=f"#{rank+1} test")
            ax.plot(epochs, train_vals, color=color, linewidth=1, linestyle="--", alpha=0.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7, ncol=2, loc="lower right")
    fig.suptitle(f"Top-{top_n} Trials: Train (dashed) vs Test (solid)", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_best_trial_all_metrics(
    path: Path,
    trial: dict[str, Any],
) -> None:
    """Bar chart of all test metrics at best epoch for the winning trial."""
    best_epoch_idx = trial["best_epoch"] - 1
    ep = trial["curve"][best_epoch_idx]

    test_keys = sorted(k for k in ep if k.startswith("msg_probe/test/") and k != "msg_probe/test/samples")
    train_keys = [k.replace("/test/", "/train/") for k in test_keys]

    labels = [k.split("/")[-1] for k in test_keys]
    test_vals = [ep.get(k, 0) for k in test_keys]
    train_vals = [ep.get(k, 0) for k in train_keys]

    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.8), 6))
    ax.bar(x - width / 2, test_vals, width, label="Test", color="#4C72B0", alpha=0.85)
    ax.bar(x + width / 2, train_vals, width, label="Train", color="#DD8452", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Metric value")
    ax.set_title(f"Best Trial (epoch {trial['best_epoch']}): {trial['name'][:60]}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_bayesian_summary(
    path: Path,
    *,
    checkpoint_path: Path,
    metric_key: str,
    dims: list[SearchDim],
    trials: list[dict[str, Any]],
    default_trial: dict[str, Any],
) -> None:
    ranked = _rank_trials(trials)
    best = ranked[0]
    best_ep_data = best["curve"][best["best_epoch"] - 1]

    lines = [
        "# Bayesian Probe Sweep",
        "",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Ranking metric: `{metric_key}`",
        f"- Total trials: {len(trials)}",
        f"- Search dims: {len(dims)}",
        "",
        "## Best Result",
        "",
        f"- **{metric_key}**: {best['best_metric_value']:.4f} at epoch {best['best_epoch']}",
        f"- Improvement over default: {best['best_metric_value'] - default_trial['best_metric_value']:+.4f}",
        f"- Params: `{json.dumps(best['params'], sort_keys=True)}`",
        "",
        "## All Test Metrics at Best Epoch",
        "",
        "| Metric | Test | Train | Gap |",
        "| --- | ---: | ---: | ---: |",
    ]
    test_keys = sorted(k for k in best_ep_data if k.startswith("msg_probe/test/") and k != "msg_probe/test/samples")
    for tk in test_keys:
        train_k = tk.replace("/test/", "/train/")
        tv = best_ep_data.get(tk, 0)
        trv = best_ep_data.get(train_k, 0)
        label = tk.split("/")[-1]
        lines.append(f"| {label} | {tv:.4f} | {trv:.4f} | {trv - tv:.4f} |")

    lines.extend(["", "## Top 10 Trials", ""])
    lines.append("| Rank | " + metric_key.split("/")[-1] + " | Epoch | Params |")
    lines.append("| ---: | ---: | ---: | --- |")
    for rank, t in enumerate(ranked[:10], 1):
        pstr = ", ".join(f"{k.replace('msg_probe_','')}={_format_param_value(v)}" for k, v in sorted(t["params"].items()))
        lines.append(f"| {rank} | {t['best_metric_value']:.4f} | {t['best_epoch']} | {pstr} |")

    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Config file path.")
    parser.add_argument("--workdir", required=True, help="Training workdir.")
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Fixed checkpoint path. Defaults to the latest checkpoint under --workdir.",
    )
    parser.add_argument(
        "--results-dir",
        default="",
        help="Output directory. Defaults to <workdir>/probe_sweeps/<checkpoint-stem>.",
    )
    parser.add_argument(
        "--metric",
        default="",
        help="Metric key used to rank trials. Defaults to cfg.msg_probe_tune_metric.",
    )
    parser.add_argument(
        "--param-space-json",
        default="",
        help="Optional JSON override for cfg.msg_probe_tune_param_space.",
    )
    parser.add_argument(
        "--config-override-json",
        default="",
        help="Optional JSON dict of config overrides applied before extraction and tuning.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Default device used for both extraction and probe fitting.",
    )
    parser.add_argument(
        "--extract-device",
        choices=("cpu", "cuda"),
        default="",
        help="Optional override for embedding extraction device.",
    )
    parser.add_argument(
        "--probe-device",
        choices=("cpu", "cuda"),
        default="",
        help="Optional override for probe fitting device.",
    )
    parser.add_argument(
        "--checkpoint-glob",
        default="step-*.pt",
        help="Checkpoint glob used for the post-tuning checkpoint sweep.",
    )
    parser.add_argument(
        "--skip-checkpoint-sweep",
        action="store_true",
        help="Only run the fixed-checkpoint sweep.",
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=0,
        help="Optional limit on the number of checkpoints reevaluated after tuning.",
    )
    parser.add_argument(
        "--search-method",
        choices=("grid", "bayesian"),
        default="grid",
        help="Search strategy: exhaustive grid or Bayesian (GP-EI).",
    )
    parser.add_argument(
        "--n-calls",
        type=int,
        default=50,
        help="Total evaluations for Bayesian search.",
    )
    parser.add_argument(
        "--n-initial",
        type=int,
        default=15,
        help="Initial random (LHS) evaluations before GP kicks in.",
    )
    parser.add_argument(
        "--bayesian-dims-json",
        default="",
        help="Optional JSON override for Bayesian search dimensions.",
    )
    args = parser.parse_args()

    workdir = Path(args.workdir).expanduser().resolve()
    config = load_config(Path(args.config).expanduser().resolve())
    if args.config_override_json:
        config.update(json.loads(args.config_override_json))
    metric_key = args.metric or str(
        config.get("msg_probe_tune_metric", "msg_probe/test/auc_fg_mean")
    )
    fixed_checkpoint = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint
        else Path(latest_ckpt_path(workdir) or "")
    )
    if not fixed_checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {fixed_checkpoint}")
    results_dir = (
        Path(args.results_dir).expanduser().resolve()
        if args.results_dir
        else workdir / "probe_sweeps" / fixed_checkpoint.stem
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    extract_device = _resolve_device(args.extract_device or args.device)
    probe_device = _resolve_device(args.probe_device or args.device)

    log.info("Fixed checkpoint: %s", fixed_checkpoint)
    log.info("Results dir: %s", results_dir)
    log.info("Ranking metric: %s", metric_key)
    log.info("Search method: %s", args.search_method)

    fixed_cache = extract_checkpoint_cache(
        config=config,
        checkpoint_path=fixed_checkpoint,
        device=extract_device,
    )

    # Always run default trial first
    default_trial = _evaluate_trial(
        base_config=config,
        overrides={},
        train_cache=fixed_cache["train"],
        test_cache=fixed_cache["test"],
        device=probe_device,
        metric_key=metric_key,
        trial_name="default",
    )

    if args.search_method == "bayesian":
        # --- Bayesian search ---
        if args.bayesian_dims_json:
            dims = _parse_bayesian_dims(json.loads(args.bayesian_dims_json))
        else:
            dims = list(DEFAULT_BAYESIAN_DIMS)
        n_calls = args.n_calls
        n_initial = args.n_initial
        rng = np.random.RandomState(int(config.seed))

        log.info("Bayesian search: %d calls (%d initial LHS), %d dims", n_calls, n_initial, len(dims))

        trial_results: list[dict[str, Any]] = [default_trial]
        X_obs: list[np.ndarray] = []
        y_obs: list[float] = []

        # Initial LHS samples
        initial_params = _random_samples(dims, n_initial, rng)
        for idx, overrides in enumerate(initial_params):
            trial = _evaluate_trial(
                base_config=config,
                overrides=overrides,
                train_cache=fixed_cache["train"],
                test_cache=fixed_cache["test"],
                device=probe_device,
                metric_key=metric_key,
                trial_name=_trial_name(idx, overrides),
            )
            trial_results.append(trial)
            X_obs.append(_encode_point(dims, overrides))
            y_obs.append(trial["best_metric_value"])

        # GP-EI loop
        for call_idx in range(n_initial, n_calls):
            overrides = _bayesian_suggest(
                dims,
                np.array(X_obs),
                np.array(y_obs),
                rng,
            )
            trial = _evaluate_trial(
                base_config=config,
                overrides=overrides,
                train_cache=fixed_cache["train"],
                test_cache=fixed_cache["test"],
                device=probe_device,
                metric_key=metric_key,
                trial_name=_trial_name(call_idx, overrides),
            )
            trial_results.append(trial)
            X_obs.append(_encode_point(dims, overrides))
            y_obs.append(trial["best_metric_value"])
            current_best = (
                max(y_obs)
                if msg_probe_metric_higher_is_better(metric_key)
                else min(y_obs)
            )
            log.info(
                "Bayesian call %d/%d: %.4f (running best %.4f)",
                call_idx + 1, n_calls, trial["best_metric_value"], current_best,
            )

        ranked_trials = _rank_trials(trial_results)
        best_trial = ranked_trials[0]

        # Save results
        trial_rows = _serialise_trial_rows(trial_results)
        _write_csv(results_dir / "bayesian_trials.csv", trial_rows)
        _write_json(results_dir / "bayesian_trials.json", trial_results)

        # Visualisations
        non_default = [t for t in trial_results if t["name"] != "default"]
        _plot_convergence(results_dir / "convergence.png", non_default, metric_key)
        _plot_param_importance(results_dir / "param_importance.png", non_default, dims, metric_key)
        _plot_epoch_curves(results_dir / "top_epoch_curves.png", trial_results, top_n=8)
        _plot_best_trial_all_metrics(results_dir / "best_trial_metrics.png", best_trial)
        _write_bayesian_summary(
            results_dir / "summary.md",
            checkpoint_path=fixed_checkpoint,
            metric_key=metric_key,
            dims=dims,
            trials=trial_results,
            default_trial=default_trial,
        )

    else:
        # --- Grid search (original behaviour) ---
        param_space = _load_param_space(config, args.param_space_json)
        trial_overrides = _expand_grid_param_space(param_space)
        log.info("Grid trials: %d", len(trial_overrides))

        trial_results = [default_trial]
        for idx, overrides in enumerate(trial_overrides):
            trial_results.append(
                _evaluate_trial(
                    base_config=config,
                    overrides=overrides,
                    train_cache=fixed_cache["train"],
                    test_cache=fixed_cache["test"],
                    device=probe_device,
                    metric_key=metric_key,
                    trial_name=_trial_name(idx, overrides),
                )
            )
        ranked_trials = sorted(
            trial_results,
            key=lambda item: item["best_metric_value"],
            reverse=True,
        )
        best_trial = ranked_trials[0]

        trial_rows = _serialise_trial_rows(trial_results)
        _write_csv(results_dir / "fixed_checkpoint_trials.csv", trial_rows)
        _write_json(results_dir / "fixed_checkpoint_trials.json", trial_results)
        _plot_trial_ranking(results_dir / "fixed_checkpoint_ranking.png", trial_results)

    # --- Checkpoint sweep (shared by both methods) ---
    checkpoint_rows: list[dict[str, Any]] = []
    if not args.skip_checkpoint_sweep:
        checkpoint_paths = _collect_checkpoint_paths(workdir, args.checkpoint_glob)
        if args.max_checkpoints > 0:
            checkpoint_paths = checkpoint_paths[-args.max_checkpoints :]
        log.info("Checkpoint sweep count: %d", len(checkpoint_paths))
        for checkpoint_path in checkpoint_paths:
            cache = (
                fixed_cache
                if checkpoint_path == fixed_checkpoint
                else extract_checkpoint_cache(
                    config=config,
                    checkpoint_path=checkpoint_path,
                    device=extract_device,
                )
            )
            for setting, overrides in (
                ("default", {}),
                ("tuned", best_trial["params"]),
            ):
                trial = _evaluate_trial(
                    base_config=config,
                    overrides=overrides,
                    train_cache=cache["train"],
                    test_cache=cache["test"],
                    device=probe_device,
                    metric_key=metric_key,
                    trial_name=f"{setting}@{checkpoint_path.stem}",
                )
                checkpoint_rows.append(
                    {
                        "checkpoint": str(checkpoint_path),
                        "checkpoint_step": _checkpoint_step(checkpoint_path),
                        "setting": setting,
                        "metric_key": metric_key,
                        "best_metric_value": trial["best_metric_value"],
                        "best_epoch": trial["best_epoch"],
                        "final_metric_value": float(
                            trial["final_metrics"][metric_key]
                        ),
                    }
                )
    online_rows = _parse_online_probe_curve(workdir)
    if checkpoint_rows:
        _write_csv(results_dir / "checkpoint_sweep.csv", checkpoint_rows)
        _write_json(results_dir / "checkpoint_sweep.json", checkpoint_rows)
        _plot_checkpoint_curve(
            results_dir / "checkpoint_curve.png",
            checkpoint_rows,
            online_rows,
            metric_key,
        )
    if online_rows:
        _write_csv(results_dir / "online_probe_curve.csv", online_rows)
    if args.search_method == "grid":
        _write_summary(
            results_dir / "summary.md",
            checkpoint_path=fixed_checkpoint,
            metric_key=metric_key,
            ranked_trials=trial_results,
            checkpoint_rows=checkpoint_rows,
            online_rows=online_rows,
        )

    print(f"Saved probe sweep results to {results_dir}")
    print(
        f"Best trial: {best_trial['name']}  "
        f"({metric_key}={best_trial['best_metric_value']:.4f} at epoch {best_trial['best_epoch']})"
    )


if __name__ == "__main__":
    main()
