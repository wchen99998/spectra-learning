import logging
import math
from typing import Callable, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from ml_collections import config_dict
from sklearn.metrics import r2_score, roc_auc_score

from input_pipeline import numpy_batch_to_torch
from models.model import PeakSetSIGReg
from utils.massspec_probe_data import MassSpecProbeData
from utils.massspec_probe_targets import FG_SMARTS, REGRESSION_TARGET_KEYS
from utils.schedulers import learning_rate_at_step


log = logging.getLogger(__name__)


class MsgProbeTaskSpec(NamedTuple):
    regression_tasks: tuple[str, ...]
    classification_tasks: tuple[str, ...]
    regression_means: dict[str, float]
    regression_stds: dict[str, float]


class MsgProbeSplitTargets(NamedTuple):
    regression: dict[str, np.ndarray]
    classification: dict[str, np.ndarray]


class MsgProbePooler(torch.nn.Module):
    def __init__(
        self,
        *,
        model_dim: int,
        pooling_type: str = "mean",
        pma_num_heads: int = 8,
        pma_num_seeds: int = 1,
        norm_type: str = "rmsnorm",
    ) -> None:
        super().__init__()
        self.pooling_type = pooling_type
        if pooling_type == "pma":
            self.pool_query = torch.nn.Parameter(
                torch.empty(pma_num_seeds, model_dim)
            )
            torch.nn.init.xavier_normal_(self.pool_query)
            self.pool_mha = torch.nn.MultiheadAttention(
                embed_dim=model_dim,
                num_heads=pma_num_heads,
                batch_first=True,
            )
            kind = str(norm_type).lower()
            if kind == "rmsnorm":
                self.pool_norm = torch.nn.RMSNorm(model_dim, eps=1e-5)
            else:
                self.pool_norm = torch.nn.LayerNorm(model_dim, eps=1e-5)

    def forward(
        self,
        embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.pooling_type == "pma":
            pooled, _ = self.pool_mha(
                query=self.pool_query.unsqueeze(0).expand(
                    embeddings.shape[0], -1, -1
                ),
                key=embeddings,
                value=embeddings,
                key_padding_mask=~valid_mask,
                need_weights=False,
            )
            return self.pool_norm(pooled.mean(dim=1))
        mask = valid_mask.unsqueeze(-1).float()
        return (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


class MsgLinearProbe(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        task_names: tuple[str, ...],
        pooler: MsgProbePooler,
    ) -> None:
        super().__init__()
        self.pooler = pooler
        self.heads = torch.nn.ModuleDict(
            {name: torch.nn.Linear(input_dim, 1) for name in task_names}
        )

    def forward(
        self,
        pooled: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {name: head(pooled) for name, head in self.heads.items()}


class DreamsLinearProbe(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        task_names: tuple[str, ...],
    ) -> None:
        super().__init__()
        self.heads = torch.nn.ModuleDict(
            {name: torch.nn.Linear(input_dim, 1) for name in task_names}
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {name: head(x) for name, head in self.heads.items()}


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
    for name in REGRESSION_TARGET_KEYS:
        values = train_targets.regression[name].astype(np.float32)
        regression_means[name] = float(values.mean())
        regression_stds[name] = float(np.clip(values.std(), 1e-8, None))
    classification_tasks: list[str] = []
    for name in FG_SMARTS:
        tp = float(train_targets.classification[name].mean())
        ep = float(test_targets.classification[name].mean())
        if 0.01 <= tp <= 0.99 and 0.0 < ep < 1.0:
            classification_tasks.append(name)
    return MsgProbeTaskSpec(
        regression_tasks=REGRESSION_TARGET_KEYS,
        classification_tasks=tuple(classification_tasks),
        regression_means=regression_means,
        regression_stds=regression_stds,
    )


def _probe_step(
    probe: MsgLinearProbe,
    batch: dict[str, torch.Tensor],
    *,
    task_spec: MsgProbeTaskSpec,
    device: torch.device,
    feature_extractor: Callable[
        [dict[str, torch.Tensor]], tuple[torch.Tensor, torch.Tensor]
    ],
) -> dict[str, object] | None:
    token_emb, peak_valid_mask = feature_extractor(batch)
    valid_mask = batch["probe_valid_mol"].to(device=device, dtype=torch.bool)
    if not bool(valid_mask.any()):
        return None
    pooled = probe.pooler(token_emb, peak_valid_mask)
    pooled = pooled[valid_mask]
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


def _new_epoch_state(task_spec: MsgProbeTaskSpec) -> dict[str, object]:
    task_names = task_spec.regression_tasks + task_spec.classification_tasks
    return {
        "count": 0,
        "loss_total": 0.0,
        "losses": {name: 0.0 for name in task_names},
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
    epoch_state["loss_total"] += float(result["loss_total"].detach()) * batch_size
    losses, predictions = epoch_state["losses"], epoch_state["predictions"]
    targets = epoch_state["targets"]
    for name in task_spec.regression_tasks + task_spec.classification_tasks:
        losses[name] += float(result["losses"][name].detach()) * batch_size
        predictions[name].append(result["predictions"][name].detach().cpu().numpy())
        targets[name].append(result["targets"][name].detach().cpu().numpy())


def _score_epoch_state(
    *,
    prefix: str,
    epoch_state: dict[str, object],
    task_spec: MsgProbeTaskSpec,
) -> dict[str, float]:
    count = int(epoch_state["count"])
    metrics: dict[str, float] = {
        f"{prefix}/samples": float(count),
        f"{prefix}/loss_total": float(epoch_state["loss_total"]) / count,
    }
    regression_r2_values, classification_auc_values = [], []
    losses, predictions = epoch_state["losses"], epoch_state["predictions"]
    targets = epoch_state["targets"]
    for name in task_spec.regression_tasks:
        pred = np.concatenate(predictions[name], axis=0)
        target = np.concatenate(targets[name], axis=0)
        metrics[f"{prefix}/loss_{name}"] = float(losses[name]) / count
        metrics[f"{prefix}/r2_{name}"] = float(r2_score(target, pred))
        regression_r2_values.append(metrics[f"{prefix}/r2_{name}"])
    for name in task_spec.classification_tasks:
        pred = np.concatenate(predictions[name], axis=0)
        target = np.concatenate(targets[name], axis=0)
        metrics[f"{prefix}/loss_fg_{name}"] = float(losses[name]) / count
        metrics[f"{prefix}/auc_fg_{name}"] = float(roc_auc_score(target, pred))
        classification_auc_values.append(metrics[f"{prefix}/auc_fg_{name}"])
    metrics[f"{prefix}/r2_mean"] = float(np.mean(regression_r2_values))
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
    probe_pooling_type = str(config.get("msg_probe_pooling_type", "mean"))
    probe_pma_num_heads = int(config.get("msg_probe_pma_num_heads", config.num_heads))
    probe_pma_num_seeds = int(config.get("msg_probe_pma_num_seeds", 1))
    norm_type = str(config.get("norm_type", "rmsnorm"))
    peak_ordering = str(config.get("peak_ordering", "intensity"))
    probe_data = MassSpecProbeData.from_config(config)

    @torch.no_grad()
    def feature_extractor(
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = model.encoder(
            batch["peak_mz"],
            batch["peak_intensity"],
            valid_mask=batch["peak_valid_mask"],
        )
        return embeddings, batch["peak_valid_mask"]

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
    pooler = MsgProbePooler(
        model_dim=int(config.model_dim),
        pooling_type=probe_pooling_type,
        pma_num_heads=probe_pma_num_heads,
        pma_num_seeds=probe_pma_num_seeds,
        norm_type=norm_type,
    )
    probe = MsgLinearProbe(
        input_dim=int(config.model_dim),
        task_names=task_spec.regression_tasks + task_spec.classification_tasks,
        pooler=pooler,
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
        final_metrics = {
            **_score_epoch_state(
                prefix="msg_probe/train", epoch_state=train_state, task_spec=task_spec
            ),
            **_score_epoch_state(
                prefix="msg_probe/test", epoch_state=test_state, task_spec=task_spec
            ),
            "msg_probe/num_fg_tasks": float(len(task_spec.classification_tasks)),
            "msg_probe_epoch": float(epoch_idx + 1),
        }
        log.info(
            "MSG probe epoch %d/%d test_r2_mean=%.4f test_auc_fg_mean=%.4f fg_tasks=%d",
            epoch_idx + 1,
            num_probe_epochs,
            final_metrics["msg_probe/test/r2_mean"],
            final_metrics["msg_probe/test/auc_fg_mean"],
            int(final_metrics["msg_probe/num_fg_tasks"]),
        )
        if on_epoch_end is not None:
            on_epoch_end(final_metrics)
    if was_training:
        model.train()
    return final_metrics


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
        task_names=task_spec.regression_tasks + task_spec.classification_tasks,
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
