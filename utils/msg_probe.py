import logging
import math
from typing import Callable, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from ml_collections import config_dict
from sklearn.metrics import r2_score, roc_auc_score

from input_pipeline import numpy_batch_to_torch
from models.model import PeakSetEncoder, PeakSetSIGReg
from utils.massspec_probe_data import MassSpecProbeData
from utils.massspec_probe_targets import FG_SMARTS, REGRESSION_TARGET_KEYS
from utils.schedulers import learning_rate_at_step


log = logging.getLogger(__name__)


class MsgProbeTaskSpec(NamedTuple):
    regression_tasks: tuple[str, ...]
    classification_tasks: tuple[str, ...]
    num_rings_classes: tuple[int, ...]
    regression_means: dict[str, float]
    regression_stds: dict[str, float]


class MsgProbeSplitTargets(NamedTuple):
    regression: dict[str, np.ndarray]
    classification: dict[str, np.ndarray]


def build_msg_probe_inputs(
    peak_embeddings: torch.Tensor,
    cls_embeddings: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    mask = valid_mask.unsqueeze(-1).to(dtype=peak_embeddings.dtype)
    mean_readout = (peak_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(
        min=1.0
    )
    return torch.cat((mean_readout, cls_embeddings), dim=-1)


_NUM_RINGS_TASK = "num_rings"
_REGRESSION_PROBE_TASKS = tuple(
    name for name in REGRESSION_TARGET_KEYS if name != _NUM_RINGS_TASK
)


class MsgLinearProbe(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        task_names: tuple[str, ...],
        task_output_dims: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.heads = torch.nn.ModuleDict(
            {
                name: torch.nn.Linear(
                    input_dim,
                    1 if task_output_dims is None else task_output_dims.get(name, 1),
                )
                for name in task_names
            }
        )

    def forward(
        self,
        probe_inputs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {name: head(probe_inputs) for name, head in self.heads.items()}


class DreamsLinearProbe(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        task_names: tuple[str, ...],
        task_output_dims: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.heads = torch.nn.ModuleDict(
            {
                name: torch.nn.Linear(
                    input_dim,
                    1 if task_output_dims is None else task_output_dims.get(name, 1),
                )
                for name in task_names
            }
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {name: head(x) for name, head in self.heads.items()}


def _probe_task_names(task_spec: MsgProbeTaskSpec) -> tuple[str, ...]:
    task_names = task_spec.regression_tasks + task_spec.classification_tasks
    if task_spec.num_rings_classes:
        task_names += (_NUM_RINGS_TASK,)
    return task_names


def _probe_task_output_dims(task_spec: MsgProbeTaskSpec) -> dict[str, int]:
    if not task_spec.num_rings_classes:
        return {}
    return {_NUM_RINGS_TASK: len(task_spec.num_rings_classes)}


def iter_massspec_probe(
    probe_data: MassSpecProbeData,
    split: str,
    *,
    seed: int,
    peak_ordering: str,
    drop_remainder: bool,
    max_samples: int | None = None,
):
    dataset = probe_data.build_dataset(
        split,
        seed=seed,
        peak_ordering=peak_ordering,
        shuffle=(split == "massspec_train"),
        drop_remainder=drop_remainder,
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


def probe_steps_per_epoch(
    probe_data: MassSpecProbeData,
    *,
    split: str,
    drop_remainder: bool,
    max_samples: int | None = None,
) -> int:
    size = int(probe_data.info[f"{split}_size"])
    if max_samples is not None:
        size = min(size, int(max_samples))
    batch_size = int(probe_data.batch_size)
    return size // batch_size if drop_remainder else math.ceil(size / batch_size)


def _collect_split_targets(
    *,
    probe_data: MassSpecProbeData,
    split: str,
    peak_ordering: str,
    seed: int,
    max_samples: int | None = None,
) -> MsgProbeSplitTargets:
    regression = {name: [] for name in REGRESSION_TARGET_KEYS}
    classification = {name: [] for name in FG_SMARTS}
    for batch in iter_massspec_probe(
        probe_data=probe_data,
        split=split,
        seed=seed,
        peak_ordering=peak_ordering,
        drop_remainder=False,
        max_samples=max_samples,
    ):
        valid_mask = (
            batch["probe_valid_mol"].detach().cpu().numpy().astype(bool, copy=False)
        )
        if not valid_mask.any():
            continue
        for name in REGRESSION_TARGET_KEYS:
            regression[name].append(
                batch[f"probe_{name}"][valid_mask].detach().cpu().numpy()
            )
        for name in FG_SMARTS:
            classification[name].append(
                batch[f"probe_fg_{name}"][valid_mask].detach().cpu().numpy()
            )

    def _cat(d, dt):
        return {
            n: np.concatenate(c) if c else np.empty(0, dtype=dt) for n, c in d.items()
        }

    return MsgProbeSplitTargets(
        regression=_cat(regression, np.float32),
        classification=_cat(classification, np.int32),
    )


def _build_task_spec(
    *,
    train_targets: MsgProbeSplitTargets,
    test_targets: MsgProbeSplitTargets,
) -> MsgProbeTaskSpec:
    regression_means, regression_stds = {}, {}
    for name in _REGRESSION_PROBE_TASKS:
        values = train_targets.regression[name].astype(np.float32)
        regression_means[name] = float(values.mean())
        regression_stds[name] = float(np.clip(values.std(), 1e-8, None))
    classification_tasks: list[str] = []
    for name in FG_SMARTS:
        tp = float(train_targets.classification[name].mean())
        ep = float(test_targets.classification[name].mean())
        if 0.01 <= tp <= 0.99 and 0.0 < ep < 1.0:
            classification_tasks.append(name)
    num_rings_classes = tuple(
        sorted(np.unique(train_targets.regression[_NUM_RINGS_TASK].astype(np.int32)).tolist())
    )
    return MsgProbeTaskSpec(
        regression_tasks=_REGRESSION_PROBE_TASKS,
        classification_tasks=tuple(classification_tasks),
        num_rings_classes=num_rings_classes,
        regression_means=regression_means,
        regression_stds=regression_stds,
    )


def _probe_step(
    probe: MsgLinearProbe,
    batch: dict[str, torch.Tensor],
    *,
    task_spec: MsgProbeTaskSpec,
    device: torch.device,
    feature_extractor: Callable[[dict[str, torch.Tensor]], torch.Tensor],
) -> dict[str, object] | None:
    probe_inputs = feature_extractor(batch)
    valid_mask = batch["probe_valid_mol"].to(device=device, dtype=torch.bool)
    if not bool(valid_mask.any()):
        return None
    probe_inputs = probe_inputs[valid_mask]
    logits = probe(probe_inputs)
    losses, predictions, task_targets = {}, {}, {}
    for name in task_spec.regression_tasks:
        target = batch[f"probe_{name}"][valid_mask].to(dtype=torch.float32)
        mean, std = task_spec.regression_means[name], task_spec.regression_stds[name]
        pred = logits[name].squeeze(-1)
        losses[name] = F.mse_loss(pred, (target - mean) / std)
        predictions[name] = pred.detach() * std + mean
        task_targets[name] = target
    for name in task_spec.classification_tasks:
        target = batch[f"probe_fg_{name}"][valid_mask].to(dtype=torch.float32)
        pred = logits[name].squeeze(-1)
        losses[name] = F.binary_cross_entropy_with_logits(pred, target)
        predictions[name] = torch.sigmoid(pred.detach())
        task_targets[name] = target
    if task_spec.num_rings_classes:
        target = batch["probe_num_rings"][valid_mask].to(dtype=torch.long)
        class_values = torch.tensor(
            task_spec.num_rings_classes,
            device=device,
            dtype=torch.long,
        )
        target_idx = torch.searchsorted(class_values, target)
        pred = logits[_NUM_RINGS_TASK]
        losses[_NUM_RINGS_TASK] = F.cross_entropy(pred, target_idx)
        predictions[_NUM_RINGS_TASK] = class_values[pred.detach().argmax(dim=-1)].to(
            dtype=torch.float32
        )
        task_targets[_NUM_RINGS_TASK] = target.to(dtype=torch.float32)
    return {
        "loss_total": torch.stack(list(losses.values())).mean(),
        "losses": losses,
        "predictions": predictions,
        "targets": task_targets,
        "batch_size": int(probe_inputs.shape[0]),
    }


def _new_epoch_state(task_spec: MsgProbeTaskSpec) -> dict[str, object]:
    task_names = _probe_task_names(task_spec)
    return {
        "count": 0,
        "predictions": {name: [] for name in task_names},
        "targets": {name: [] for name in task_names},
    }


def _update_epoch_state(
    epoch_state: dict[str, object],
    result: dict[str, object],
    task_spec: MsgProbeTaskSpec,
) -> None:
    batch_size = int(result["batch_size"])
    epoch_state["count"] += batch_size
    predictions = epoch_state["predictions"]
    targets = epoch_state["targets"]
    for name in _probe_task_names(task_spec):
        predictions[name].append(result["predictions"][name].detach().cpu().numpy())
        targets[name].append(result["targets"][name].detach().cpu().numpy())


def resolve_msg_probe_select_metric(
    config: config_dict.ConfigDict,
) -> str:
    return str(
        config.get(
            "msg_probe_select_metric",
            config.get("msg_probe_tune_metric", "msg_probe/test/auc_fg_mean"),
        )
    )


def msg_probe_metric_higher_is_better(metric_key: str) -> bool:
    return "/mae_" not in metric_key


def _score_epoch_state(
    *,
    prefix: str,
    epoch_state: dict[str, object],
    task_spec: MsgProbeTaskSpec,
) -> dict[str, float]:
    count = int(epoch_state["count"])
    metrics: dict[str, float] = {
        f"{prefix}/samples": float(count),
    }
    regression_r2_values, regression_mae_values = [], []
    classification_auc_values = []
    predictions = epoch_state["predictions"]
    targets = epoch_state["targets"]
    for name in task_spec.regression_tasks:
        pred = np.concatenate(predictions[name], axis=0)
        target = np.concatenate(targets[name], axis=0)
        metrics[f"{prefix}/r2_{name}"] = float(r2_score(target, pred))
        metrics[f"{prefix}/mae_{name}"] = float(np.mean(np.abs(target - pred)))
        regression_r2_values.append(metrics[f"{prefix}/r2_{name}"])
        regression_mae_values.append(metrics[f"{prefix}/mae_{name}"])
    for name in task_spec.classification_tasks:
        pred = np.concatenate(predictions[name], axis=0)
        target = np.concatenate(targets[name], axis=0)
        metrics[f"{prefix}/auc_fg_{name}"] = float(roc_auc_score(target, pred))
        classification_auc_values.append(metrics[f"{prefix}/auc_fg_{name}"])
    if task_spec.num_rings_classes:
        pred = np.concatenate(predictions[_NUM_RINGS_TASK], axis=0)
        target = np.concatenate(targets[_NUM_RINGS_TASK], axis=0)
        metrics[f"{prefix}/mae_num_rings"] = float(np.mean(np.abs(target - pred)))
        metrics[f"{prefix}/acc_num_rings_exact"] = float(np.mean(pred == target))
        metrics[f"{prefix}/acc_num_rings_within_1"] = float(
            np.mean(np.abs(pred - target) <= 1.0)
        )
    metrics[f"{prefix}/r2_mean"] = float(np.mean(regression_r2_values))
    metrics[f"{prefix}/mae_mean"] = float(np.mean(regression_mae_values))
    metrics[f"{prefix}/r2_mean_wo_num_rings"] = metrics[f"{prefix}/r2_mean"]
    metrics[f"{prefix}/mae_mean_wo_num_rings"] = metrics[f"{prefix}/mae_mean"]
    metrics[f"{prefix}/auc_fg_mean"] = float(np.mean(classification_auc_values))
    return metrics


def run_msg_probe(
    *,
    config: config_dict.ConfigDict,
    model: PeakSetSIGReg,
    device: torch.device,
    on_epoch_end: Callable[[dict[str, float]], None] | None = None,
) -> dict[str, float]:
    num_probe_epochs = int(config.get("msg_probe_num_epochs", 5))
    probe_lr = float(config.get("msg_probe_learning_rate", 1e-3))
    probe_weight_decay = float(config.get("msg_probe_weight_decay", 1e-2))
    probe_warmup_steps = int(config.get("msg_probe_warmup_steps", 100))
    _mts = config.get("msg_probe_max_train_samples", None)
    max_train_samples = int(_mts) if _mts is not None else None
    _mte = config.get("msg_probe_max_test_samples", None)
    max_test_samples = int(_mte) if _mte is not None else None
    peak_ordering = str(config.get("peak_ordering", "intensity"))
    probe_data = MassSpecProbeData.from_config(config)

    @torch.no_grad()
    def feature_extractor(
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        embeddings = model.encoder(
            batch["peak_mz"],
            batch["peak_intensity"],
            valid_mask=batch["peak_valid_mask"],
        )
        peak_embeddings, cls_embeddings = PeakSetEncoder.split_peak_and_cls(embeddings)
        return build_msg_probe_inputs(
            peak_embeddings,
            cls_embeddings,
            batch["peak_valid_mask"],
        )

    train_seed_base = int(config.seed) + 1_100_000
    test_seed_base = int(config.seed) + 1_200_000
    train_targets = _collect_split_targets(
        probe_data=probe_data,
        split="massspec_train",
        peak_ordering=peak_ordering,
        seed=train_seed_base,
        max_samples=max_train_samples,
    )
    test_targets = _collect_split_targets(
        probe_data=probe_data,
        split="massspec_test",
        peak_ordering=peak_ordering,
        seed=test_seed_base,
        max_samples=max_test_samples,
    )
    task_spec = _build_task_spec(train_targets=train_targets, test_targets=test_targets)
    was_training = model.training
    model.eval()
    probe = MsgLinearProbe(
        input_dim=2 * int(config.model_dim),
        task_names=_probe_task_names(task_spec),
        task_output_dims=_probe_task_output_dims(task_spec),
    ).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=probe_lr,
        weight_decay=probe_weight_decay,
    )
    steps_per_epoch = probe_steps_per_epoch(
        probe_data,
        split="massspec_train",
        drop_remainder=False,
        max_samples=max_train_samples,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step_idx: (
            learning_rate_at_step(
                step_idx + 1,
                base_lr=probe_lr,
                total_steps=num_probe_epochs * steps_per_epoch,
                warmup_steps=probe_warmup_steps,
            )
            / probe_lr
        ),
    )

    def move_batch(batch: dict[str, object]) -> dict[str, object]:
        return {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    compiled_probe_step = torch.compile(_probe_step)
    probe_select_metric = resolve_msg_probe_select_metric(config)
    higher_is_better = msg_probe_metric_higher_is_better(probe_select_metric)

    best_metrics: dict[str, float] = {}
    best_metric_value = -float("inf") if higher_is_better else float("inf")
    for epoch_idx in range(num_probe_epochs):
        probe.train()
        train_state = _new_epoch_state(task_spec)
        for batch in iter_massspec_probe(
            probe_data,
            "massspec_train",
            seed=train_seed_base + epoch_idx,
            peak_ordering=peak_ordering,
            drop_remainder=False,
            max_samples=max_train_samples,
        ):
            batch = move_batch(batch)
            optimizer.zero_grad(set_to_none=True)
            result = compiled_probe_step(
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
            for batch in iter_massspec_probe(
                probe_data,
                "massspec_test",
                seed=test_seed_base + epoch_idx,
                peak_ordering=peak_ordering,
                drop_remainder=False,
                max_samples=max_test_samples,
            ):
                batch = move_batch(batch)
                result = compiled_probe_step(
                    probe,
                    batch,
                    task_spec=task_spec,
                    device=device,
                    feature_extractor=feature_extractor,
                )
                if result is None:
                    continue
                _update_epoch_state(test_state, result, task_spec)
        epoch_metrics = {
            **_score_epoch_state(
                prefix="msg_probe/train", epoch_state=train_state, task_spec=task_spec
            ),
            **_score_epoch_state(
                prefix="msg_probe/test", epoch_state=test_state, task_spec=task_spec
            ),
            "msg_probe/num_fg_tasks": float(len(task_spec.classification_tasks)),
            "msg_probe_epoch": float(epoch_idx + 1),
        }
        current_value = float(epoch_metrics[probe_select_metric])
        is_better = (
            current_value > best_metric_value
            if higher_is_better
            else current_value < best_metric_value
        )
        if is_better:
            best_metric_value = current_value
            best_metrics = dict(epoch_metrics)
        log.info(
            "MSG probe epoch %d/%d test_r2_mean_wo_num_rings=%.4f test_mae_num_rings=%.4f test_auc_fg_mean=%.4f fg_tasks=%d",
            epoch_idx + 1,
            num_probe_epochs,
            epoch_metrics["msg_probe/test/r2_mean_wo_num_rings"],
            epoch_metrics["msg_probe/test/mae_num_rings"],
            epoch_metrics["msg_probe/test/auc_fg_mean"],
            int(epoch_metrics["msg_probe/num_fg_tasks"]),
        )
        if on_epoch_end is not None:
            on_epoch_end(epoch_metrics)
    if best_metrics:
        log.info(
            "MSG probe best epoch %d: %s=%.4f test_r2_mean_wo_num_rings=%.4f test_mae_num_rings=%.4f test_auc_fg_mean=%.4f",
            int(best_metrics["msg_probe_epoch"]),
            probe_select_metric,
            best_metrics[probe_select_metric],
            best_metrics["msg_probe/test/r2_mean_wo_num_rings"],
            best_metrics["msg_probe/test/mae_num_rings"],
            best_metrics["msg_probe/test/auc_fg_mean"],
        )
    if was_training:
        model.train()
    return best_metrics


def _dreams_probe_step(
    probe: DreamsLinearProbe,
    batch: dict[str, torch.Tensor],
    *,
    task_spec: MsgProbeTaskSpec,
    device: torch.device,
) -> dict[str, object] | None:
    dreams_emb = batch["dreams_embedding"].to(device=device, dtype=torch.float32)
    valid_mask = batch["probe_valid_mol"].to(device=device, dtype=torch.bool)
    if not bool(valid_mask.any()):
        return None
    pooled = dreams_emb[valid_mask]
    logits = probe(pooled)
    losses, predictions, task_targets = {}, {}, {}
    for name in task_spec.regression_tasks:
        target = batch[f"probe_{name}"][valid_mask].to(dtype=torch.float32)
        mean, std = task_spec.regression_means[name], task_spec.regression_stds[name]
        pred = logits[name].squeeze(-1)
        losses[name] = F.mse_loss(pred, (target - mean) / std)
        predictions[name] = pred.detach() * std + mean
        task_targets[name] = target
    for name in task_spec.classification_tasks:
        target = batch[f"probe_fg_{name}"][valid_mask].to(dtype=torch.float32)
        pred = logits[name].squeeze(-1)
        losses[name] = F.binary_cross_entropy_with_logits(pred, target)
        predictions[name] = torch.sigmoid(pred.detach())
        task_targets[name] = target
    return {
        "loss_total": torch.stack(list(losses.values())).mean(),
        "losses": losses,
        "predictions": predictions,
        "targets": task_targets,
        "batch_size": int(pooled.shape[0]),
    }


def run_dreams_probe(
    *,
    config: config_dict.ConfigDict,
    device: torch.device,
    on_epoch_end: Callable[[dict[str, float]], None] | None = None,
) -> dict[str, float]:
    num_probe_epochs = int(config.get("msg_probe_num_epochs", 5))
    probe_lr = float(config.get("msg_probe_learning_rate", 1e-3))
    probe_weight_decay = float(config.get("msg_probe_weight_decay", 1e-2))
    probe_warmup_steps = int(config.get("msg_probe_warmup_steps", 100))
    _mts = config.get("msg_probe_max_train_samples", None)
    max_train_samples = int(_mts) if _mts is not None else None
    _mte = config.get("msg_probe_max_test_samples", None)
    max_test_samples = int(_mte) if _mte is not None else None
    peak_ordering = str(config.get("peak_ordering", "intensity"))
    probe_data = MassSpecProbeData.from_config(config)

    dreams_dim = probe_data.dreams_dim
    if dreams_dim == 0:
        log.warning("No DreaMS embeddings in probe data; skipping Dreams probe")
        return {}

    train_seed_base = int(config.seed) + 1_100_000
    test_seed_base = int(config.seed) + 1_200_000
    train_targets = _collect_split_targets(
        probe_data=probe_data,
        split="massspec_train",
        peak_ordering=peak_ordering,
        seed=train_seed_base,
        max_samples=max_train_samples,
    )
    test_targets = _collect_split_targets(
        probe_data=probe_data,
        split="massspec_test",
        peak_ordering=peak_ordering,
        seed=test_seed_base,
        max_samples=max_test_samples,
    )
    task_spec = _build_task_spec(train_targets=train_targets, test_targets=test_targets)

    probe = DreamsLinearProbe(
        input_dim=dreams_dim,
        task_names=_probe_task_names(task_spec),
        task_output_dims=_probe_task_output_dims(task_spec),
    ).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=probe_lr,
        weight_decay=probe_weight_decay,
    )
    steps_per_epoch = probe_steps_per_epoch(
        probe_data,
        split="massspec_train",
        drop_remainder=False,
        max_samples=max_train_samples,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step_idx: (
            learning_rate_at_step(
                step_idx + 1,
                base_lr=probe_lr,
                total_steps=num_probe_epochs * steps_per_epoch,
                warmup_steps=probe_warmup_steps,
            )
            / probe_lr
        ),
    )

    def move_batch(batch: dict[str, object]) -> dict[str, object]:
        return {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    compiled_dreams_probe_step = torch.compile(_dreams_probe_step)

    final_metrics: dict[str, float] = {}
    for epoch_idx in range(num_probe_epochs):
        probe.train()
        train_state = _new_epoch_state(task_spec)
        for batch in iter_massspec_probe(
            probe_data,
            "massspec_train",
            seed=train_seed_base + epoch_idx,
            peak_ordering=peak_ordering,
            drop_remainder=False,
            max_samples=max_train_samples,
        ):
            batch = move_batch(batch)
            optimizer.zero_grad(set_to_none=True)
            result = compiled_dreams_probe_step(
                probe,
                batch,
                task_spec=task_spec,
                device=device,
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
            for batch in iter_massspec_probe(
                probe_data,
                "massspec_test",
                seed=test_seed_base + epoch_idx,
                peak_ordering=peak_ordering,
                drop_remainder=False,
                max_samples=max_test_samples,
            ):
                batch = move_batch(batch)
                result = compiled_dreams_probe_step(
                    probe,
                    batch,
                    task_spec=task_spec,
                    device=device,
                )
                if result is None:
                    continue
                _update_epoch_state(test_state, result, task_spec)
        final_metrics = {
            **_score_epoch_state(
                prefix="dreams_probe/train", epoch_state=train_state, task_spec=task_spec
            ),
            **_score_epoch_state(
                prefix="dreams_probe/test", epoch_state=test_state, task_spec=task_spec
            ),
            "dreams_probe/num_fg_tasks": float(len(task_spec.classification_tasks)),
            "dreams_probe_epoch": float(epoch_idx + 1),
        }
        log.info(
            "DreaMS probe epoch %d/%d test_r2_mean=%.4f test_auc_fg_mean=%.4f fg_tasks=%d",
            epoch_idx + 1,
            num_probe_epochs,
            final_metrics["dreams_probe/test/r2_mean"],
            final_metrics["dreams_probe/test/auc_fg_mean"],
            int(final_metrics["dreams_probe/num_fg_tasks"]),
        )
        if on_epoch_end is not None:
            on_epoch_end(final_metrics)
    return final_metrics
