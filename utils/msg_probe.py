import logging
import math
from typing import Callable, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from ml_collections import config_dict
from sklearn.metrics import r2_score, roc_auc_score

from input_pipeline import numpy_batch_to_torch
from models.model import CrossAttention, PeakSetEncoder, PeakSetSIGReg
from utils.massspec_probe_data import MassSpecProbeData
from utils.massspec_probe_targets import (
    FG_SMARTS,
    MACCS_FINGERPRINT_BITS,
    REGRESSION_TARGET_KEYS,
)
from utils.schedulers import learning_rate_at_step


log = logging.getLogger(__name__)


class MsgProbeTaskSpec(NamedTuple):
    regression_tasks: tuple[str, ...]
    num_rings_classes: tuple[int, ...]
    maccs_bits: int
    regression_means: dict[str, float]
    regression_stds: dict[str, float]


class MsgProbeSplitTargets(NamedTuple):
    regression: dict[str, np.ndarray]
    maccs: np.ndarray


def build_msg_probe_inputs(
    peak_embeddings: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    mask = valid_mask.unsqueeze(-1).to(dtype=peak_embeddings.dtype)
    return (peak_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


_NUM_RINGS_TASK = "num_rings"
_MACCS_TASK = "maccs"
_REGRESSION_PROBE_TASKS = tuple(
    name for name in REGRESSION_TARGET_KEYS if name != _NUM_RINGS_TASK
)


def msg_probe_variants_from_config(
    config: config_dict.ConfigDict,
) -> tuple[str, ...]:
    raw_variants = config.get("msg_probe_variants", ("mean", "covariance", "pma"))
    if isinstance(raw_variants, str):
        return (raw_variants.lower(),)
    return tuple(str(variant).lower() for variant in raw_variants)


def resolve_msg_probe_sample_limits(
    config: config_dict.ConfigDict,
) -> tuple[int | None, int | None, bool]:
    probe_dataset = str(config.get("probe_dataset", "massspec"))
    raw_train = config.get("msg_probe_max_train_samples", None)
    raw_test = config.get("msg_probe_max_test_samples", None)
    if raw_train is None and probe_dataset == "nist-full":
        raw_train = config.get("nist_full_probe_train_samples", 4_000)
    if raw_test is None and probe_dataset == "nist-full":
        raw_test = config.get("nist_full_probe_test_samples", 1_000)
    max_train_samples = int(raw_train) if raw_train is not None else None
    max_test_samples = int(raw_test) if raw_test is not None else None
    randomize_test_subset = probe_dataset == "nist-full" and max_test_samples is not None
    return max_train_samples, max_test_samples, randomize_test_subset


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


class MsgProbeHeads(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        task_names: tuple[str, ...],
        task_output_dims: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.heads = torch.nn.ModuleDict(
            {
                name: torch.nn.Sequential(
                    torch.nn.Linear(input_dim, hidden_dim),
                    torch.nn.SiLU(),
                    torch.nn.Linear(
                        hidden_dim,
                        1 if task_output_dims is None else task_output_dims.get(name, 1),
                    ),
                )
                for name in task_names
            }
        )

    def forward(
        self,
        probe_inputs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {name: head(probe_inputs) for name, head in self.heads.items()}


class MsgMeanPool(torch.nn.Module):
    def forward(
        self,
        peak_embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        return build_msg_probe_inputs(peak_embeddings, valid_mask)


class MsgCovariancePool(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        compressed_dim: int,
    ) -> None:
        super().__init__()
        self.left_proj = torch.nn.Linear(input_dim, compressed_dim, bias=False)
        self.right_proj = torch.nn.Linear(input_dim, compressed_dim, bias=False)
        torch.nn.init.xavier_normal_(self.left_proj.weight)
        torch.nn.init.xavier_normal_(self.right_proj.weight)

    def forward(
        self,
        peak_embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = valid_mask.unsqueeze(-1).to(dtype=peak_embeddings.dtype)
        left = self.left_proj(peak_embeddings) * mask
        right = self.right_proj(peak_embeddings) * mask
        denom = mask.sum(dim=1).clamp(min=1.0)
        covariance = left.transpose(1, 2) @ right
        covariance = covariance / denom.unsqueeze(-1)
        return covariance.flatten(start_dim=1)


class MsgPmaPool(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        num_seeds: int,
        num_heads: int,
        qk_norm: bool = False,
        norm_type: str = "layernorm",
    ) -> None:
        super().__init__()
        self.seed_vectors = torch.nn.Parameter(torch.empty(num_seeds, input_dim))
        torch.nn.init.trunc_normal_(self.seed_vectors, std=0.02)
        self.cross_attention = CrossAttention(
            dim=input_dim,
            n_heads=num_heads,
            qk_norm=qk_norm,
            norm_type=norm_type,
        )

    def forward(
        self,
        peak_embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        seed_vectors = self.seed_vectors.unsqueeze(0).expand(peak_embeddings.shape[0], -1, -1)
        pooled = self.cross_attention(
            seed_vectors.to(dtype=peak_embeddings.dtype),
            peak_embeddings,
            memory_mask=valid_mask,
        )
        return pooled.mean(dim=1)


class MsgSequenceProbe(torch.nn.Module):
    def __init__(
        self,
        *,
        pooler: torch.nn.Module,
        pooled_dim: int,
        hidden_dim: int,
        task_names: tuple[str, ...],
        task_output_dims: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.pooler = pooler
        self.heads = MsgProbeHeads(
            input_dim=pooled_dim,
            hidden_dim=hidden_dim,
            task_names=task_names,
            task_output_dims=task_output_dims,
        )

    def forward(
        self,
        peak_embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.heads(self.pooler(peak_embeddings, valid_mask))


def _probe_task_names(task_spec: MsgProbeTaskSpec) -> tuple[str, ...]:
    task_names = task_spec.regression_tasks
    if task_spec.num_rings_classes:
        task_names += (_NUM_RINGS_TASK,)
    if task_spec.maccs_bits > 0:
        task_names += (_MACCS_TASK,)
    return task_names


def _probe_task_output_dims(task_spec: MsgProbeTaskSpec) -> dict[str, int]:
    output_dims: dict[str, int] = {}
    if task_spec.num_rings_classes:
        output_dims[_NUM_RINGS_TASK] = len(task_spec.num_rings_classes)
    if task_spec.maccs_bits > 0:
        output_dims[_MACCS_TASK] = task_spec.maccs_bits
    return output_dims


def _build_msg_sequence_probe(
    variant: str,
    *,
    config: config_dict.ConfigDict,
    task_spec: MsgProbeTaskSpec,
) -> MsgSequenceProbe:
    model_dim = int(config.model_dim)
    hidden_dim = int(config.get("msg_probe_mlp_hidden_dim", model_dim))
    task_names = _probe_task_names(task_spec)
    task_output_dims = _probe_task_output_dims(task_spec)
    if variant == "mean":
        pooler = MsgMeanPool()
        pooled_dim = model_dim
    elif variant == "covariance":
        compressed_dim = int(config.get("msg_probe_covariance_dim", 32))
        pooler = MsgCovariancePool(
            input_dim=model_dim,
            compressed_dim=compressed_dim,
        )
        pooled_dim = compressed_dim * compressed_dim
    elif variant == "pma":
        pooler = MsgPmaPool(
            input_dim=model_dim,
            num_seeds=int(config.get("msg_probe_pma_num_seeds", 4)),
            num_heads=int(config.get("msg_probe_pma_num_heads", config.get("encoder_num_heads", 8))),
            qk_norm=bool(config.get("encoder_qk_norm", False)),
            norm_type=str(config.get("norm_type", "layernorm")),
        )
        pooled_dim = model_dim
    else:
        raise ValueError(f"Unsupported MSG probe variant: {variant!r}")
    return MsgSequenceProbe(
        pooler=pooler,
        pooled_dim=pooled_dim,
        hidden_dim=hidden_dim,
        task_names=task_names,
        task_output_dims=task_output_dims,
    )


def iter_massspec_probe(
    probe_data: MassSpecProbeData,
    split: str,
    *,
    seed: int,
    peak_ordering: str,
    drop_remainder: bool,
    max_samples: int | None = None,
    sample_randomly: bool = False,
):
    dataset = probe_data.build_dataset(
        split,
        seed=seed,
        peak_ordering=peak_ordering,
        shuffle=(split == "massspec_train") or bool(sample_randomly),
        drop_remainder=drop_remainder,
    )
    size = int(probe_data.info[f"{split}_size"])
    if max_samples is not None:
        size = min(size, int(max_samples))
    seen = 0
    for batch in dataset:
        if seen >= size:
            break
        if not isinstance(batch["peak_mz"], torch.Tensor):
            batch = numpy_batch_to_torch(batch)
        take = min(int(batch["peak_mz"].shape[0]), size - seen)
        if take != batch["peak_mz"].shape[0]:
            batch = {key: value[:take] for key, value in batch.items()}
        seen += take
        yield batch


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
    sample_randomly: bool = False,
) -> MsgProbeSplitTargets:
    regression = {name: [] for name in REGRESSION_TARGET_KEYS}
    maccs = []
    for batch in iter_massspec_probe(
        probe_data=probe_data,
        split=split,
        seed=seed,
        peak_ordering=peak_ordering,
        drop_remainder=False,
        max_samples=max_samples,
        sample_randomly=sample_randomly,
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
        maccs.append(batch["probe_maccs"][valid_mask].detach().cpu().numpy())

    def _cat(d, dt):
        return {
            n: np.concatenate(c) if c else np.empty(0, dtype=dt) for n, c in d.items()
        }

    return MsgProbeSplitTargets(
        regression=_cat(regression, np.float32),
        maccs=(
            np.concatenate(maccs, axis=0)
            if maccs
            else np.empty((0, MACCS_FINGERPRINT_BITS), dtype=np.int32)
        ),
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
    num_rings_classes = tuple(
        sorted(np.unique(train_targets.regression[_NUM_RINGS_TASK].astype(np.int32)).tolist())
    )
    return MsgProbeTaskSpec(
        regression_tasks=_REGRESSION_PROBE_TASKS,
        num_rings_classes=num_rings_classes,
        maccs_bits=int(train_targets.maccs.shape[1]),
        regression_means=regression_means,
        regression_stds=regression_stds,
    )


def _build_probe_result(
    logits: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    valid_mask: torch.Tensor,
    *,
    task_spec: MsgProbeTaskSpec,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    losses, predictions, task_targets = {}, {}, {}
    for name in task_spec.regression_tasks:
        target = batch[f"probe_{name}"][valid_mask].to(dtype=torch.float32)
        mean, std = task_spec.regression_means[name], task_spec.regression_stds[name]
        pred = logits[name].squeeze(-1)
        losses[name] = F.mse_loss(pred, (target - mean) / std)
        predictions[name] = pred.detach() * std + mean
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
    if task_spec.maccs_bits > 0:
        target = batch["probe_maccs"][valid_mask].to(dtype=torch.float32)
        pred = logits[_MACCS_TASK]
        losses[_MACCS_TASK] = F.binary_cross_entropy_with_logits(pred, target)
        predictions[_MACCS_TASK] = torch.sigmoid(pred.detach())
        task_targets[_MACCS_TASK] = target
    return {
        "loss_total": torch.stack(list(losses.values())).mean(),
        "losses": losses,
        "predictions": predictions,
        "targets": task_targets,
        "batch_size": batch_size,
    }


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
    return _build_probe_result(
        logits,
        batch,
        valid_mask,
        task_spec=task_spec,
        device=device,
        batch_size=int(probe_inputs.shape[0]),
    )


def _sequence_probe_step(
    probe: MsgSequenceProbe,
    batch: dict[str, torch.Tensor],
    peak_embeddings: torch.Tensor,
    *,
    task_spec: MsgProbeTaskSpec,
    device: torch.device,
) -> dict[str, object] | None:
    valid_mask = batch["probe_valid_mol"].to(device=device, dtype=torch.bool)
    if not bool(valid_mask.any()):
        return None
    peak_embeddings = peak_embeddings[valid_mask]
    peak_valid_mask = batch["peak_valid_mask"][valid_mask].to(
        device=device,
        dtype=torch.bool,
    )
    logits = probe(peak_embeddings, peak_valid_mask)
    return _build_probe_result(
        logits,
        batch,
        valid_mask,
        task_spec=task_spec,
        device=device,
        batch_size=int(peak_embeddings.shape[0]),
    )


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
            config.get("msg_probe_tune_metric", "msg_probe/test/auc_maccs_mean"),
        )
    )


def msg_probe_metric_higher_is_better(metric_key: str) -> bool:
    return "/mae_" not in metric_key


def _msg_probe_variant_metric_key(
    variant: str,
    metric_key: str,
) -> str:
    variant_prefix = f"msg_probe/{variant}/"
    if metric_key.startswith(variant_prefix):
        return metric_key
    if metric_key.startswith("msg_probe/"):
        return variant_prefix + metric_key[len("msg_probe/"):]
    return metric_key


def _with_mean_probe_aliases(metrics: dict[str, float]) -> dict[str, float]:
    aliased = dict(metrics)
    for split in ("train", "test"):
        mean_prefix = f"msg_probe/mean/{split}/"
        legacy_prefix = f"msg_probe/{split}/"
        for key, value in metrics.items():
            if key.startswith(mean_prefix):
                aliased[legacy_prefix + key[len(mean_prefix):]] = value
    if "msg_probe/mean/num_maccs_bits" in metrics:
        aliased["msg_probe/num_maccs_bits"] = metrics["msg_probe/mean/num_maccs_bits"]
    if "msg_probe/mean/epoch" in metrics:
        aliased["msg_probe_epoch"] = metrics["msg_probe/mean/epoch"]
    return aliased


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
    predictions = epoch_state["predictions"]
    targets = epoch_state["targets"]
    for name in task_spec.regression_tasks:
        pred = np.concatenate(predictions[name], axis=0)
        target = np.concatenate(targets[name], axis=0)
        metrics[f"{prefix}/r2_{name}"] = float(r2_score(target, pred))
        metrics[f"{prefix}/mae_{name}"] = float(np.mean(np.abs(target - pred)))
        regression_r2_values.append(metrics[f"{prefix}/r2_{name}"])
        regression_mae_values.append(metrics[f"{prefix}/mae_{name}"])
    if task_spec.num_rings_classes:
        pred = np.concatenate(predictions[_NUM_RINGS_TASK], axis=0)
        target = np.concatenate(targets[_NUM_RINGS_TASK], axis=0)
        metrics[f"{prefix}/mae_num_rings"] = float(np.mean(np.abs(target - pred)))
        metrics[f"{prefix}/acc_num_rings_exact"] = float(np.mean(pred == target))
        metrics[f"{prefix}/acc_num_rings_within_1"] = float(
            np.mean(np.abs(pred - target) <= 1.0)
        )
    if task_spec.maccs_bits > 0:
        pred = np.concatenate(predictions[_MACCS_TASK], axis=0)
        target = np.concatenate(targets[_MACCS_TASK], axis=0)
        auc_values = []
        recall_values = []
        for bit_idx in range(task_spec.maccs_bits):
            bit_target = target[:, bit_idx]
            if np.unique(bit_target).size < 2:
                if np.count_nonzero(bit_target) > 0:
                    bit_pred = pred[:, bit_idx] >= 0.5
                    recall_values.append(
                        float(bit_pred[bit_target == 1].mean())
                    )
                continue
            bit_pred = pred[:, bit_idx] >= 0.5
            auc_values.append(float(roc_auc_score(bit_target, pred[:, bit_idx])))
            recall_values.append(float(bit_pred[bit_target == 1].mean()))
        metrics[f"{prefix}/num_maccs_auc_bits"] = float(len(auc_values))
        metrics[f"{prefix}/num_maccs_recall_bits"] = float(len(recall_values))
        metrics[f"{prefix}/auc_maccs_mean"] = (
            float(np.mean(auc_values)) if auc_values else float("nan")
        )
        metrics[f"{prefix}/recall_maccs_mean"] = (
            float(np.mean(recall_values)) if recall_values else float("nan")
        )
    metrics[f"{prefix}/r2_mean"] = float(np.mean(regression_r2_values))
    metrics[f"{prefix}/mae_mean"] = float(np.mean(regression_mae_values))
    metrics[f"{prefix}/r2_mean_wo_num_rings"] = metrics[f"{prefix}/r2_mean"]
    metrics[f"{prefix}/mae_mean_wo_num_rings"] = metrics[f"{prefix}/mae_mean"]
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
    max_train_samples, max_test_samples, randomize_test_subset = (
        resolve_msg_probe_sample_limits(config)
    )
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
        peak_embeddings, _ = PeakSetEncoder.split_peak_and_cls(embeddings)
        return peak_embeddings

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
        sample_randomly=randomize_test_subset,
    )
    task_spec = _build_task_spec(train_targets=train_targets, test_targets=test_targets)
    variants = msg_probe_variants_from_config(config)
    was_training = model.training
    model.eval()
    probes = {
        variant: _build_msg_sequence_probe(
            variant,
            config=config,
            task_spec=task_spec,
        ).to(device)
        for variant in variants
    }
    optimizers = {
        variant: torch.optim.AdamW(
            probe.parameters(),
            lr=probe_lr,
            weight_decay=probe_weight_decay,
        )
        for variant, probe in probes.items()
    }
    steps_per_epoch = probe_steps_per_epoch(
        probe_data,
        split="massspec_train",
        drop_remainder=False,
        max_samples=max_train_samples,
    )
    schedulers = {
        variant: torch.optim.lr_scheduler.LambdaLR(
            optimizers[variant],
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
        for variant in variants
    }

    def move_batch(batch: dict[str, object]) -> dict[str, object]:
        return {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    select_metric = resolve_msg_probe_select_metric(config)
    higher_is_better = msg_probe_metric_higher_is_better(select_metric)
    best_metrics_by_variant: dict[str, dict[str, float]] = {}
    best_metric_values = {
        variant: -float("inf") if higher_is_better else float("inf")
        for variant in variants
    }
    for epoch_idx in range(num_probe_epochs):
        for probe in probes.values():
            probe.train()
        train_states = {
            variant: _new_epoch_state(task_spec)
            for variant in variants
        }
        for batch in iter_massspec_probe(
            probe_data,
            "massspec_train",
            seed=train_seed_base + epoch_idx,
            peak_ordering=peak_ordering,
            drop_remainder=False,
            max_samples=max_train_samples,
        ):
            batch = move_batch(batch)
            peak_embeddings = feature_extractor(batch)
            for variant in variants:
                optimizers[variant].zero_grad(set_to_none=True)
                result = _sequence_probe_step(
                    probes[variant],
                    batch,
                    peak_embeddings,
                    task_spec=task_spec,
                    device=device,
                )
                if result is None:
                    continue
                result["loss_total"].backward()
                optimizers[variant].step()
                schedulers[variant].step()
                _update_epoch_state(train_states[variant], result, task_spec)
        for probe in probes.values():
            probe.eval()
        test_states = {
            variant: _new_epoch_state(task_spec)
            for variant in variants
        }
        with torch.no_grad():
            for batch in iter_massspec_probe(
                probe_data,
                "massspec_test",
                seed=test_seed_base + epoch_idx,
                peak_ordering=peak_ordering,
                drop_remainder=False,
                max_samples=max_test_samples,
                sample_randomly=randomize_test_subset,
            ):
                batch = move_batch(batch)
                peak_embeddings = feature_extractor(batch)
                for variant in variants:
                    result = _sequence_probe_step(
                        probes[variant],
                        batch,
                        peak_embeddings,
                        task_spec=task_spec,
                        device=device,
                    )
                    if result is None:
                        continue
                    _update_epoch_state(test_states[variant], result, task_spec)
        epoch_metrics: dict[str, float] = {}
        for variant in variants:
            variant_prefix = f"msg_probe/{variant}"
            variant_metrics = {
                **_score_epoch_state(
                    prefix=f"{variant_prefix}/train",
                    epoch_state=train_states[variant],
                    task_spec=task_spec,
                ),
                **_score_epoch_state(
                    prefix=f"{variant_prefix}/test",
                    epoch_state=test_states[variant],
                    task_spec=task_spec,
                ),
                f"{variant_prefix}/num_maccs_bits": float(task_spec.maccs_bits),
                f"{variant_prefix}/epoch": float(epoch_idx + 1),
            }
            epoch_metrics.update(variant_metrics)
            variant_select_metric = _msg_probe_variant_metric_key(variant, select_metric)
            current_value = float(variant_metrics[variant_select_metric])
            is_better = (
                current_value > best_metric_values[variant]
                if higher_is_better
                else current_value < best_metric_values[variant]
            )
            if is_better:
                best_metric_values[variant] = current_value
                best_metrics_by_variant[variant] = dict(variant_metrics)
            log.info(
                "MSG probe [%s] epoch %d/%d test_r2_mean_wo_num_rings=%.4f test_mae_num_rings=%.4f test_auc_maccs_mean=%.4f test_recall_maccs_mean=%.4f maccs_bits=%d",
                variant,
                epoch_idx + 1,
                num_probe_epochs,
                variant_metrics[f"{variant_prefix}/test/r2_mean_wo_num_rings"],
                variant_metrics[f"{variant_prefix}/test/mae_num_rings"],
                variant_metrics[f"{variant_prefix}/test/auc_maccs_mean"],
                variant_metrics[f"{variant_prefix}/test/recall_maccs_mean"],
                int(variant_metrics[f"{variant_prefix}/num_maccs_bits"]),
            )
        epoch_metrics = _with_mean_probe_aliases(epoch_metrics)
        if on_epoch_end is not None:
            on_epoch_end(epoch_metrics)
    best_metrics: dict[str, float] = {}
    for variant in variants:
        variant_metrics = best_metrics_by_variant.get(variant)
        if not variant_metrics:
            continue
        best_metrics.update(variant_metrics)
        variant_prefix = f"msg_probe/{variant}"
        variant_select_metric = _msg_probe_variant_metric_key(variant, select_metric)
        log.info(
            "MSG probe [%s] best epoch %d: %s=%.4f test_r2_mean_wo_num_rings=%.4f test_mae_num_rings=%.4f test_auc_maccs_mean=%.4f test_recall_maccs_mean=%.4f",
            variant,
            int(variant_metrics[f"{variant_prefix}/epoch"]),
            variant_select_metric,
            variant_metrics[variant_select_metric],
            variant_metrics[f"{variant_prefix}/test/r2_mean_wo_num_rings"],
            variant_metrics[f"{variant_prefix}/test/mae_num_rings"],
            variant_metrics[f"{variant_prefix}/test/auc_maccs_mean"],
            variant_metrics[f"{variant_prefix}/test/recall_maccs_mean"],
        )
    if was_training:
        model.train()
    return _with_mean_probe_aliases(best_metrics)


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
    max_train_samples, max_test_samples, randomize_test_subset = (
        resolve_msg_probe_sample_limits(config)
    )
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
        sample_randomly=randomize_test_subset,
    )
    task_spec = _build_task_spec(train_targets=train_targets, test_targets=test_targets)

    probe = MsgLinearProbe(
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

    def feature_extractor(
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return batch["dreams_embedding"].to(device=device, dtype=torch.float32)

    def move_batch(batch: dict[str, object]) -> dict[str, object]:
        return {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    compiled_probe_step = torch.compile(_probe_step)
    probe_select_metric = resolve_msg_probe_select_metric(config).replace(
        "msg_probe/",
        "dreams_probe/",
    )
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
                sample_randomly=randomize_test_subset,
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
                prefix="dreams_probe/train", epoch_state=train_state, task_spec=task_spec
            ),
            **_score_epoch_state(
                prefix="dreams_probe/test", epoch_state=test_state, task_spec=task_spec
            ),
            "dreams_probe/num_maccs_bits": float(task_spec.maccs_bits),
            "dreams_probe_epoch": float(epoch_idx + 1),
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
            "DreaMS probe epoch %d/%d test_r2_mean_wo_num_rings=%.4f test_mae_num_rings=%.4f test_auc_maccs_mean=%.4f test_recall_maccs_mean=%.4f maccs_bits=%d",
            epoch_idx + 1,
            num_probe_epochs,
            epoch_metrics["dreams_probe/test/r2_mean_wo_num_rings"],
            epoch_metrics["dreams_probe/test/mae_num_rings"],
            epoch_metrics["dreams_probe/test/auc_maccs_mean"],
            epoch_metrics["dreams_probe/test/recall_maccs_mean"],
            int(epoch_metrics["dreams_probe/num_maccs_bits"]),
        )
        if on_epoch_end is not None:
            on_epoch_end(epoch_metrics)
    if best_metrics:
        log.info(
            "DreaMS probe best epoch %d: %s=%.4f test_r2_mean_wo_num_rings=%.4f test_mae_num_rings=%.4f test_auc_maccs_mean=%.4f test_recall_maccs_mean=%.4f",
            int(best_metrics["dreams_probe_epoch"]),
            probe_select_metric,
            best_metrics[probe_select_metric],
            best_metrics["dreams_probe/test/r2_mean_wo_num_rings"],
            best_metrics["dreams_probe/test/mae_num_rings"],
            best_metrics["dreams_probe/test/auc_maccs_mean"],
            best_metrics["dreams_probe/test/recall_maccs_mean"],
        )
    return best_metrics
