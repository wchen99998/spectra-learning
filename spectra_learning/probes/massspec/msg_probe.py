import logging
import math
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, cast

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from ml_collections import config_dict
from torch.nn.parallel import DistributedDataParallel

from spectra_learning.data.contracts import peak_preprocessing_contract
from spectra_learning.data.gems.conversion import numpy_batch_to_torch
from spectra_learning.data.loading import local_batch_size
from spectra_learning.models.pooling import CovariancePool
from spectra_learning.models.model import PeakSetJEPA
from spectra_learning.models.spectrum_metadata import torch_spectrum_metadata_from_batch
from spectra_learning.data.massspec_probe import MassSpecProbeData
from spectra_learning.probes.massspec.msg_modules import (
    MsgLinearProbe,
    _uses_pair_features,
    build_msg_sequence_probe as _build_msg_sequence_probe,
)
from spectra_learning.probes.massspec.msg_probe_common import (
    EpochState,
    merge_epoch_states as _merge_epoch_states,
    msg_probe_metric_higher_is_better,
    msg_probe_variant_metric_key as _msg_probe_variant_metric_key,
    new_epoch_state as _new_epoch_state,
    probe_prediction_names as _probe_prediction_names,
    probe_task_names as _probe_task_names,
    probe_task_output_dims as _probe_task_output_dims,
    resolve_msg_probe_select_metric,
    resolve_probe_warmup_steps as _resolve_probe_warmup_steps,
    run_repeated_probe as _run_repeated_probe,
    score_epoch_state as _score_epoch_state,
)
from spectra_learning.probes.massspec.msg_settings import (
    MACCS_TASK as _MACCS_TASK,
    MORGAN_TASK as _MORGAN_TASK,
    PROBE_FINGERPRINT_BITS as _PROBE_FINGERPRINT_BITS,
    BINARY_PROBE_TASKS as _BINARY_PROBE_TASKS,
    REGRESSION_PROBE_TASKS as _REGRESSION_PROBE_TASKS,
    MsgProbePairwiseAlignment,
    MsgProbeSplitTargets,
    MsgProbeTaskSpec,
    msg_probe_variants_from_config,
    resolve_msg_probe_fingerprint,
    resolve_msg_probe_num_repeats,
    resolve_msg_probe_pairwise_alignment_num_pairs,
)
from spectra_learning.data.massspec_targets import FG_SMARTS
from spectra_learning.training.distributed import DistributedContext
from spectra_learning.training.schedules import learning_rate_at_step


log = logging.getLogger(__name__)

ProbeBatch = dict[str, torch.Tensor]
ProbeStepResult = dict[str, Any]
EncoderFeatures = torch.Tensor | tuple[torch.Tensor, torch.Tensor]


def _is_distributed(distributed: DistributedContext | None) -> bool:
    return distributed is not None and distributed.is_distributed


def _is_main(distributed: DistributedContext | None) -> bool:
    return distributed is None or distributed.is_main


def _distributed_world_size(distributed: DistributedContext | None) -> int:
    return distributed.world_size if distributed is not None else 1


def _distributed_rank(distributed: DistributedContext | None) -> int:
    return distributed.rank if distributed is not None else 0


def _distributed_local_rank(distributed: DistributedContext | None) -> int:
    return distributed.local_rank if distributed is not None else 0


def _all_gather_object(value: object) -> list[object]:
    gathered: list[object] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, value)
    return gathered


def _feature_single(features: EncoderFeatures) -> torch.Tensor:
    if isinstance(features, tuple):
        return features[0]
    return features


def _feature_pair(features: EncoderFeatures) -> torch.Tensor | None:
    if isinstance(features, tuple):
        return features[1]
    return None


def _probe_call(
    probe: torch.nn.Module,
    peak_embeddings: torch.Tensor,
    peak_valid_mask: torch.Tensor,
    pair_embeddings: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    if pair_embeddings is None:
        return cast(dict[str, torch.Tensor], probe(peak_embeddings, peak_valid_mask))
    return cast(
        dict[str, torch.Tensor],
        probe(peak_embeddings, peak_valid_mask, pair_embeddings),
    )




def iter_massspec_probe(
    probe_data: Any,
    split: str,
    *,
    seed: int,
    peak_ordering: str,
    drop_remainder: bool,
    max_samples: int | None = None,
    sample_randomly: bool = False,
    distributed_world_size: int = 1,
    distributed_rank: int = 0,
    pad_distributed: bool = False,
) -> Iterator[dict[str, Any]]:
    dataset = probe_data.build_dataset(
        split,
        seed=seed,
        peak_ordering=peak_ordering,
        shuffle=(split == "massspec_train") or sample_randomly,
        drop_remainder=drop_remainder,
        max_samples=max_samples,
        distributed_world_size=distributed_world_size,
        distributed_rank=distributed_rank,
        pad_distributed=pad_distributed,
    )
    size = int(probe_data.info[f"{split}_size"])
    if max_samples is not None:
        size = min(size, max_samples)
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
    probe_data: Any,
    *,
    split: str,
    drop_remainder: bool,
    max_samples: int | None = None,
    distributed_world_size: int = 1,
) -> int:
    size = int(probe_data.info[f"{split}_size"])
    if max_samples is not None:
        size = min(size, max_samples)
    if distributed_world_size > 1:
        size = math.ceil(size / distributed_world_size)
    batch_size = local_batch_size(
        int(probe_data.batch_size),
        distributed_world_size,
    )
    return size // batch_size if drop_remainder else math.ceil(size / batch_size)


def _collect_split_targets(
    *,
    probe_data: Any,
    split: str,
    peak_ordering: str,
    seed: int,
    max_samples: int | None = None,
    sample_randomly: bool = False,
    fingerprint_task: str = _MACCS_TASK,
    regression_tasks: tuple[str, ...] = _REGRESSION_PROBE_TASKS,
    binary_tasks: tuple[str, ...] = _BINARY_PROBE_TASKS,
    distributed: DistributedContext | None = None,
) -> MsgProbeSplitTargets:
    regression = {name: [] for name in regression_tasks}
    binary = {name: [] for name in binary_tasks}
    maccs = []
    fingerprint_key = f"probe_{fingerprint_task}"
    for batch in iter_massspec_probe(
        probe_data=probe_data,
        split=split,
        seed=seed,
        peak_ordering=peak_ordering,
        drop_remainder=False,
        max_samples=max_samples,
        sample_randomly=sample_randomly,
        distributed_world_size=_distributed_world_size(distributed),
        distributed_rank=_distributed_rank(distributed),
    ):
        valid_mask = (
            batch["probe_valid_mol"].detach().cpu().numpy().astype(bool, copy=False)
        )
        if not valid_mask.any():
            continue
        for name in regression_tasks:
            regression[name].append(
                batch[f"probe_{name}"][valid_mask].detach().cpu().numpy()
            )
        for name in binary_tasks:
            binary[name].append(
                batch[f"probe_{name}"][valid_mask].detach().cpu().numpy()
            )
        maccs.append(batch[fingerprint_key][valid_mask].detach().cpu().numpy())

    def _cat(d, dt):
        return {
            n: np.concatenate(c) if c else np.empty(0, dtype=dt) for n, c in d.items()
        }

    targets = MsgProbeSplitTargets(
        regression=_cat(regression, np.float32),
        binary=_cat(binary, np.float32),
        maccs=(
            np.concatenate(maccs, axis=0)
            if maccs
            else np.empty(
                (0, _PROBE_FINGERPRINT_BITS[fingerprint_task]), dtype=np.int32
            )
        ),
    )
    if _is_distributed(distributed):
        targets = _merge_split_targets(
            _all_gather_object(targets),
            fingerprint_task=fingerprint_task,
            regression_tasks=regression_tasks,
            binary_tasks=binary_tasks,
        )
    return targets


def _merge_split_targets(
    targets_by_rank: list[object],
    *,
    fingerprint_task: str,
    regression_tasks: tuple[str, ...] = _REGRESSION_PROBE_TASKS,
    binary_tasks: tuple[str, ...] = _BINARY_PROBE_TASKS,
) -> MsgProbeSplitTargets:
    targets = [target for target in targets_by_rank if isinstance(target, MsgProbeSplitTargets)]
    regression = {
        name: (
            np.concatenate(
                [target.regression[name] for target in targets],
                axis=0,
            )
            if targets
            else np.empty(0, dtype=np.float32)
        )
        for name in regression_tasks
    }
    binary = {
        name: (
            np.concatenate(
                [target.binary[name] for target in targets],
                axis=0,
            )
            if targets
            else np.empty(0, dtype=np.float32)
        )
        for name in binary_tasks
    }
    maccs = (
        np.concatenate([target.maccs for target in targets], axis=0)
        if targets
        else np.empty((0, _PROBE_FINGERPRINT_BITS[fingerprint_task]), dtype=np.int32)
    )
    return MsgProbeSplitTargets(regression=regression, binary=binary, maccs=maccs)


def _build_task_spec(
    *,
    train_targets: MsgProbeSplitTargets,
    test_targets: MsgProbeSplitTargets,
    fingerprint_task: str = _MACCS_TASK,
    regression_tasks: tuple[str, ...] = _REGRESSION_PROBE_TASKS,
    binary_tasks: tuple[str, ...] = _BINARY_PROBE_TASKS,
) -> MsgProbeTaskSpec:
    regression_means, regression_stds = {}, {}
    for name in regression_tasks:
        values = train_targets.regression[name].astype(np.float32)
        regression_means[name] = float(values.mean())
        regression_stds[name] = float(np.clip(values.std(), 1e-8, None))
    return MsgProbeTaskSpec(
        regression_tasks=regression_tasks,
        binary_tasks=binary_tasks,
        maccs_bits=int(train_targets.maccs.shape[1]),
        regression_means=regression_means,
        regression_stds=regression_stds,
        fingerprint_task=fingerprint_task,
    )


def _build_probe_result(
    logits: dict[str, torch.Tensor],
    batch: ProbeBatch,
    valid_mask: torch.Tensor,
    *,
    task_spec: MsgProbeTaskSpec,
    batch_size: int,
) -> ProbeStepResult:
    losses, predictions, task_targets = {}, {}, {}
    joint_logits = (
        logits[task_spec.fingerprint_task]
        if task_spec.maccs_bits > 0
        else None
    )
    for regression_idx, name in enumerate(task_spec.regression_tasks):
        target = batch[f"probe_{name}"][valid_mask].to(dtype=torch.float32)
        mean, std = task_spec.regression_means[name], task_spec.regression_stds[name]
        if joint_logits is None:
            pred = logits[name].squeeze(-1)
        else:
            pred = joint_logits[:, regression_idx]
        losses[name] = F.mse_loss(pred, (target - mean) / std)
        predictions[name] = pred.detach() * std + mean
        task_targets[name] = target
    for name in task_spec.binary_tasks:
        target = batch[f"probe_{name}"][valid_mask].to(dtype=torch.float32)
        pred = logits[name].squeeze(-1)
        losses[name] = F.binary_cross_entropy_with_logits(pred, target)
        predictions[name] = torch.sigmoid(pred.detach())
        task_targets[name] = target
    if task_spec.maccs_bits > 0:
        fingerprint_task = task_spec.fingerprint_task
        target = batch[f"probe_{fingerprint_task}"][valid_mask].to(dtype=torch.float32)
        pred = cast(torch.Tensor, joint_logits)[:, len(task_spec.regression_tasks):]
        losses[fingerprint_task] = F.binary_cross_entropy_with_logits(pred, target)
        predictions[fingerprint_task] = torch.sigmoid(pred.detach())
        task_targets[fingerprint_task] = target
    return {
        "loss_total": torch.stack(list(losses.values())).mean(),
        "losses": losses,
        "predictions": predictions,
        "targets": task_targets,
        "batch_size": batch_size,
    }


def _binary_tanimoto_for_pairs(
    bits: np.ndarray,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    *,
    chunk_size: int = 4096,
) -> np.ndarray:
    bits = bits.astype(bool, copy=False)
    tanimoto = np.empty(len(left_idx), dtype=np.float32)
    for start in range(0, len(left_idx), chunk_size):
        end = min(start + chunk_size, len(left_idx))
        left_bits = bits[left_idx[start:end]]
        right_bits = bits[right_idx[start:end]]
        intersection = np.count_nonzero(left_bits & right_bits, axis=1)
        union = np.count_nonzero(left_bits | right_bits, axis=1)
        tanimoto[start:end] = intersection / np.maximum(union, 1)
    return tanimoto


def _compute_pairwise_similarity_alignment(
    *,
    embeddings: np.ndarray,
    morgan_bits: np.ndarray,
    num_pairs: int,
    seed: int,
) -> MsgProbePairwiseAlignment:
    rng = np.random.default_rng(seed)
    n = int(embeddings.shape[0])
    left_idx = rng.integers(0, n, size=num_pairs, dtype=np.int64)
    right_idx = rng.integers(0, n - 1, size=num_pairs, dtype=np.int64)
    right_idx += (right_idx >= left_idx).astype(np.int64)

    embeddings = embeddings.astype(np.float32, copy=False)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True).clip(
        min=1e-12
    )
    cosine = np.sum(embeddings[left_idx] * embeddings[right_idx], axis=1).astype(
        np.float32,
        copy=False,
    )
    tanimoto = _binary_tanimoto_for_pairs(morgan_bits, left_idx, right_idx)
    pearson = float(np.corrcoef(tanimoto, cosine)[0, 1])
    return MsgProbePairwiseAlignment(
        tanimoto=tanimoto,
        cosine=cosine,
        pearson=pearson,
    )


def _compute_pairwise_similarity_alignment_for_indices(
    *,
    embeddings: np.ndarray,
    tanimoto: np.ndarray,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
) -> MsgProbePairwiseAlignment:
    embeddings = embeddings.astype(np.float32, copy=False)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True).clip(
        min=1e-12
    )
    cosine = np.sum(embeddings[left_idx] * embeddings[right_idx], axis=1).astype(
        np.float32,
        copy=False,
    )
    tanimoto = tanimoto.astype(np.float32, copy=False)
    pearson = float(np.corrcoef(tanimoto, cosine)[0, 1])
    return MsgProbePairwiseAlignment(
        tanimoto=tanimoto,
        cosine=cosine,
        pearson=pearson,
    )


def _plot_pairwise_similarity_alignment(
    alignment: MsgProbePairwiseAlignment,
    output_stem: Path,
) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    with mpl.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "axes.linewidth": 0.6,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        cmap = LinearSegmentedColormap.from_list(
            "nature_teal",
            ["#ebe9fb", "#9bd6cf", "#1f9d8a"],
        )
        fig, ax = plt.subplots(figsize=(2.15, 2.15), constrained_layout=True)
        ax.hist2d(
            alignment.tanimoto,
            alignment.cosine,
            bins=120,
            range=[[0.0, 1.0], [-0.25, 1.02]],
            cmap=cmap,
            norm=LogNorm(),
            cmin=1,
        )
        slope, intercept = np.polyfit(alignment.tanimoto, alignment.cosine, deg=1)
        xs = np.asarray([0.0, 1.0], dtype=np.float32)
        ax.plot(xs, slope * xs + intercept, color="#188f88", linewidth=1.2)
        ax.axhline(0.0, color="#d7d7d7", linewidth=0.6, zorder=0)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.25, 1.02)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_xticklabels(["0", "0.5", "1.0"])
        ax.set_yticklabels(["0", "0.5", "1.0"])
        ax.set_xlabel("Morgan Tanimoto")
        ax.set_ylabel("Covariance cosine")
        ax.set_title(f"Pearson = {alignment.pearson:.2f}", pad=8)
        ax.tick_params(direction="in")
        fig.savefig(png_path, dpi=600, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
    return png_path, pdf_path


def _collect_covariance_morgan_alignment_inputs(
    *,
    probe_data: MassSpecProbeData,
    covariance_pooler: torch.nn.Module,
    feature_extractor: Callable[[ProbeBatch], EncoderFeatures],
    move_batch: Callable[[dict[str, Any]], ProbeBatch],
    split: str,
    peak_ordering: str,
    seed: int,
    max_samples: int | None,
    sample_randomly: bool,
    device: torch.device,
    distributed: DistributedContext | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    embeddings, morgan = [], []
    with torch.no_grad():
        for batch in iter_massspec_probe(
            probe_data,
            split,
            seed=seed,
            peak_ordering=peak_ordering,
            drop_remainder=False,
            max_samples=max_samples,
            sample_randomly=sample_randomly,
            distributed_world_size=_distributed_world_size(distributed),
            distributed_rank=_distributed_rank(distributed),
        ):
            batch = move_batch(batch)
            valid_mask = batch["probe_valid_mol"].to(device=device, dtype=torch.bool)
            if not bool(valid_mask.any()):
                continue
            peak_embeddings = _feature_single(feature_extractor(batch))[valid_mask]
            peak_valid_mask = batch["peak_valid_mask"][valid_mask].to(
                device=device,
                dtype=torch.bool,
            )
            covariance = covariance_pooler(
                peak_embeddings.float(),
                peak_valid_mask,
            )
            embeddings.append(covariance.detach().cpu().numpy())
            morgan.append(batch["probe_morgan"][valid_mask].detach().cpu().numpy())
    local = (np.concatenate(embeddings, axis=0), np.concatenate(morgan, axis=0))
    if not _is_distributed(distributed):
        return local
    gathered = _all_gather_object(local)
    gathered_inputs = cast(list[tuple[np.ndarray, np.ndarray]], gathered)
    return (
        np.concatenate([item[0] for item in gathered_inputs], axis=0),
        np.concatenate([item[1] for item in gathered_inputs], axis=0),
    )


def _collect_covariance_embeddings_for_indices(
    *,
    probe_data: MassSpecProbeData,
    covariance_pooler: torch.nn.Module,
    feature_extractor: Callable[[ProbeBatch], EncoderFeatures],
    move_batch: Callable[[dict[str, Any]], ProbeBatch],
    indices: np.ndarray,
    peak_ordering: str,
    distributed: DistributedContext | None = None,
) -> np.ndarray:
    local_positions = np.arange(len(indices))[
        _distributed_rank(distributed) :: _distributed_world_size(distributed)
    ]
    local_indices = indices[local_positions]
    embeddings = []
    with torch.no_grad():
        for batch in probe_data.build_indexed_dataset(
            "all",
            local_indices,
            peak_ordering=peak_ordering,
            drop_remainder=False,
            distributed_world_size=_distributed_world_size(distributed),
        ):
            batch = move_batch(batch)
            peak_embeddings = _feature_single(feature_extractor(batch))
            covariance = covariance_pooler(
                peak_embeddings.float(),
                batch["peak_valid_mask"].to(dtype=torch.bool),
            )
            embeddings.append(covariance.detach().cpu().numpy())
    local_embeddings = np.concatenate(embeddings, axis=0)
    if not _is_distributed(distributed):
        return local_embeddings
    gathered = _all_gather_object((local_positions, local_embeddings))
    gathered_embeddings = cast(list[tuple[np.ndarray, np.ndarray]], gathered)
    output = np.empty(
        (len(indices), local_embeddings.shape[1]),
        dtype=local_embeddings.dtype,
    )
    for positions, values in gathered_embeddings:
        output[positions] = values
    return output


def _run_prepared_covariance_morgan_pairwise_alignment(
    *,
    config: config_dict.ConfigDict,
    probe_data: MassSpecProbeData,
    covariance_pooler: torch.nn.Module,
    feature_extractor: Callable[[ProbeBatch], EncoderFeatures],
    move_batch: Callable[[dict[str, Any]], ProbeBatch],
    peak_ordering: str,
    num_pairs: int,
    distributed: DistributedContext | None = None,
) -> tuple[MsgProbePairwiseAlignment, int] | None:
    raw_pair_path = str(
        config.get(
            "msg_probe_pairwise_alignment_path",
            probe_data.pairwise_alignment_path,
        )
        or ""
    )
    if not raw_pair_path:
        return None
    pair_path = Path(raw_pair_path)
    if not pair_path.exists():
        return None

    pair_data = np.load(pair_path)
    tanimoto = pair_data["tanimoto"][:num_pairs]
    left_endpoint = pair_data["left_endpoint"][:num_pairs]
    right_endpoint = pair_data["right_endpoint"][:num_pairs]
    endpoint_indices = pair_data["endpoint_index"]
    endpoint_embeddings = _collect_covariance_embeddings_for_indices(
        probe_data=probe_data,
        covariance_pooler=covariance_pooler,
        feature_extractor=feature_extractor,
        move_batch=move_batch,
        indices=endpoint_indices,
        peak_ordering=peak_ordering,
        distributed=distributed,
    )
    return (
        _compute_pairwise_similarity_alignment_for_indices(
            embeddings=endpoint_embeddings,
            tanimoto=tanimoto,
            left_idx=left_endpoint,
            right_idx=right_endpoint,
        ),
        len(endpoint_indices),
    )


def _run_covariance_morgan_pairwise_alignment(
    *,
    config: config_dict.ConfigDict,
    probe_data: MassSpecProbeData,
    model: PeakSetJEPA,
    covariance_pooler: torch.nn.Module | None,
    feature_extractor: Callable[[ProbeBatch], EncoderFeatures],
    move_batch: Callable[[dict[str, Any]], ProbeBatch],
    device: torch.device,
    split: str,
    peak_ordering: str,
    seed: int,
    max_samples: int | None,
    sample_randomly: bool,
    plot_dir: Path | None,
    plot_step: int | None,
    repeat_index: int,
    distributed: DistributedContext | None = None,
) -> dict[str, float]:
    num_pairs = resolve_msg_probe_pairwise_alignment_num_pairs(config)
    if (
        num_pairs <= 0
        or covariance_pooler is None
        or int(probe_data.info.get("probe_morgan_bits", 0)) <= 0
    ):
        return {}
    prepared = _run_prepared_covariance_morgan_pairwise_alignment(
        config=config,
        probe_data=probe_data,
        covariance_pooler=covariance_pooler,
        feature_extractor=feature_extractor,
        move_batch=move_batch,
        peak_ordering=peak_ordering,
        num_pairs=num_pairs,
        distributed=distributed,
    )
    if prepared is None:
        embeddings, morgan = _collect_covariance_morgan_alignment_inputs(
            probe_data=probe_data,
            covariance_pooler=covariance_pooler,
            feature_extractor=feature_extractor,
            move_batch=move_batch,
            split=split,
            peak_ordering=peak_ordering,
            seed=seed,
            max_samples=max_samples,
            sample_randomly=sample_randomly,
            device=device,
            distributed=distributed,
        )
        alignment = _compute_pairwise_similarity_alignment(
            embeddings=embeddings,
            morgan_bits=morgan,
            num_pairs=num_pairs,
            seed=seed + 77_000,
        )
        num_samples = int(embeddings.shape[0])
        source = "random"
    else:
        alignment, num_samples = prepared
        source = "prepared"
    if _is_main(distributed) and plot_dir is not None and bool(
        config.get("msg_probe_pairwise_alignment_plot", True)
    ):
        step_label = "unknown" if plot_step is None else f"{plot_step:08d}"
        output_stem = (
            plot_dir
            / f"msg_probe_covariance_morgan_pairwise_step-{step_label}_repeat-{repeat_index:02d}"
        )
        _plot_pairwise_similarity_alignment(alignment, output_stem)
    prefix = "msg_probe/covariance_morgan_pairwise"
    return {
        f"{prefix}/pearson": alignment.pearson,
        f"{prefix}/num_pairs": float(len(alignment.tanimoto)),
        f"{prefix}/num_samples": float(num_samples),
        f"{prefix}/mean_tanimoto": float(alignment.tanimoto.mean()),
        f"{prefix}/mean_cosine": float(alignment.cosine.mean()),
        f"{prefix}/source_prepared": float(source == "prepared"),
    }


def _probe_step(
    probe: MsgLinearProbe,
    batch: ProbeBatch,
    *,
    task_spec: MsgProbeTaskSpec,
    device: torch.device,
    feature_extractor: Callable[[ProbeBatch], torch.Tensor],
) -> ProbeStepResult | None:
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
        batch_size=probe_inputs.shape[0],
    )


def _sequence_probe_step(
    probe: torch.nn.Module,
    batch: ProbeBatch,
    features: EncoderFeatures,
    *,
    task_spec: MsgProbeTaskSpec,
    device: torch.device,
    allow_empty: bool = False,
) -> ProbeStepResult | None:
    valid_mask = batch["probe_valid_mol"].to(device=device, dtype=torch.bool)
    peak_embeddings = _feature_single(features)
    pair_embeddings = _feature_pair(features)
    if not bool(valid_mask.any()):
        if allow_empty:
            empty_pair_embeddings = None
            if pair_embeddings is not None:
                empty_pair_embeddings = pair_embeddings[:1]
            logits = cast(
                dict[str, torch.Tensor],
                _probe_call(
                    probe,
                    peak_embeddings[:1],
                    batch["peak_valid_mask"][:1].to(device=device, dtype=torch.bool),
                    empty_pair_embeddings,
                ),
            )
            zero_loss = torch.stack(
                [value.float().sum() * 0.0 for value in logits.values()]
            ).sum()
            return {
                "loss_total": zero_loss,
                "predictions": {},
                "targets": {},
                "batch_size": 0,
            }
        return None
    peak_embeddings = peak_embeddings[valid_mask]
    peak_valid_mask = batch["peak_valid_mask"][valid_mask].to(
        device=device,
        dtype=torch.bool,
    )
    if pair_embeddings is not None:
        pair_embeddings = pair_embeddings[valid_mask]
    logits = _probe_call(
        probe,
        peak_embeddings,
        peak_valid_mask,
        pair_embeddings,
    )
    return _build_probe_result(
        logits,
        batch,
        valid_mask,
        task_spec=task_spec,
        batch_size=peak_embeddings.shape[0],
    )


def _evaluate_sequence_probe_split(
    *,
    probe_data: MassSpecProbeData,
    probes: dict[str, torch.nn.Module],
    task_spec: MsgProbeTaskSpec,
    feature_extractor: Callable[[ProbeBatch], EncoderFeatures],
    move_batch: Callable[[dict[str, Any]], ProbeBatch],
    split: str,
    seed: int,
    peak_ordering: str,
    max_samples: int | None,
    sample_randomly: bool,
    device: torch.device,
    distributed: DistributedContext | None = None,
) -> dict[str, EpochState]:
    states = {variant: _new_epoch_state(task_spec) for variant in probes}
    with torch.no_grad():
        for batch in iter_massspec_probe(
            probe_data,
            split,
            seed=seed,
            peak_ordering=peak_ordering,
            drop_remainder=False,
            max_samples=max_samples,
            sample_randomly=sample_randomly,
            distributed_world_size=_distributed_world_size(distributed),
            distributed_rank=_distributed_rank(distributed),
        ):
            batch = move_batch(batch)
            features = feature_extractor(batch)
            for variant, probe in probes.items():
                result = _sequence_probe_step(
                    probe,
                    batch,
                    features,
                    task_spec=task_spec,
                    device=device,
                )
                if result is None:
                    continue
                _update_epoch_state(states[variant], result, task_spec)
    return _gather_variant_states(states, task_spec, distributed)


def _evaluate_linear_probe_split(
    *,
    probe_data: MassSpecProbeData,
    probe: MsgLinearProbe,
    compiled_probe_step: Callable[..., ProbeStepResult | None],
    task_spec: MsgProbeTaskSpec,
    feature_extractor: Callable[[ProbeBatch], torch.Tensor],
    move_batch: Callable[[dict[str, Any]], ProbeBatch],
    split: str,
    seed: int,
    peak_ordering: str,
    max_samples: int | None,
    sample_randomly: bool,
    device: torch.device,
) -> EpochState:
    state = _new_epoch_state(task_spec)
    with torch.no_grad():
        for batch in iter_massspec_probe(
            probe_data,
            split,
            seed=seed,
            peak_ordering=peak_ordering,
            drop_remainder=False,
            max_samples=max_samples,
            sample_randomly=sample_randomly,
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
            _update_epoch_state(state, result, task_spec)
    return state


def _update_epoch_state(
    epoch_state: EpochState,
    result: ProbeStepResult,
    task_spec: MsgProbeTaskSpec,
) -> None:
    batch_size = int(result["batch_size"])
    epoch_state["count"] += batch_size
    predictions = epoch_state["predictions"]
    targets = epoch_state["targets"]
    for name in _probe_prediction_names(task_spec):
        predictions[name].append(result["predictions"][name].detach().cpu().numpy())
        targets[name].append(result["targets"][name].detach().cpu().numpy())


def _gather_variant_states(
    states: dict[str, EpochState],
    task_spec: MsgProbeTaskSpec,
    distributed: DistributedContext | None,
) -> dict[str, EpochState]:
    if not _is_distributed(distributed):
        return states
    return {
        variant: _merge_epoch_states(cast(list[EpochState], _all_gather_object(state)), task_spec)
        for variant, state in states.items()
    }


def _wrap_probe_for_distributed(
    probe: torch.nn.Module,
    distributed: DistributedContext | None,
) -> torch.nn.Module:
    if not _is_distributed(distributed):
        return probe
    assert distributed is not None
    return DistributedDataParallel(
        probe,
        device_ids=[distributed.local_rank] if distributed.device.type == "cuda" else None,
        output_device=distributed.local_rank if distributed.device.type == "cuda" else None,
        gradient_as_bucket_view=True,
    )


def _online_probe_covariance_pooler(
    variant: str,
    covariance_pooler: torch.nn.Module | None = None,
) -> torch.nn.Module | None:
    if variant not in ("covariance", "single_pair_covariance", "pair_covariance"):
        return None
    return covariance_pooler


@dataclass(frozen=True)
class _MsgProbeSetup:
    probe_data: MassSpecProbeData
    task_spec: MsgProbeTaskSpec
    variants: tuple[str, ...]
    feature_extractor: Callable[[ProbeBatch], EncoderFeatures]
    move_batch: Callable[[dict[str, Any]], ProbeBatch]
    device: torch.device
    distributed: DistributedContext | None
    num_epochs: int
    learning_rate: float
    weight_decay: float
    grad_clip_norm: float | None
    early_stopping: bool
    early_stopping_patience: int
    early_stopping_min_delta: float
    early_stopping_min_epochs: int
    peak_ordering: str
    fingerprint_task: str
    train_seed_base: int
    test_seed_base: int
    online_maccs_only: bool


@dataclass
class _MsgProbeTrainingState:
    probes: dict[str, torch.nn.Module]
    train_probes: dict[str, torch.nn.Module]
    optimizers: dict[str, torch.optim.Optimizer]
    schedulers: dict[str, torch.optim.lr_scheduler.LambdaLR]
    select_metric: str
    higher_is_better: bool
    best_metrics_by_variant: dict[str, dict[str, Any]]
    best_metric_values: dict[str, float]
    best_state_by_variant: dict[str, dict[str, torch.Tensor]]
    epochs_without_improvement: dict[str, int]


def _make_msg_probe_feature_extractor(
    model: Any,
    *,
    use_pair_features: bool,
) -> Callable[[ProbeBatch], EncoderFeatures]:
    @torch.no_grad()
    def feature_extractor(batch: ProbeBatch) -> EncoderFeatures:
        if use_pair_features:
            embeddings, pair_embeddings = model.encoder.forward_with_pair(
                batch["peak_mz"],
                batch["peak_intensity"],
                valid_mask=batch["peak_valid_mask"],
                precursor_mz=batch.get("precursor_mz", None),
                spectrum_metadata=torch_spectrum_metadata_from_batch(batch),
            )
            return embeddings, pair_embeddings
        return model.encoder(
            batch["peak_mz"],
            batch["peak_intensity"],
            valid_mask=batch["peak_valid_mask"],
            precursor_mz=batch.get("precursor_mz", None),
            spectrum_metadata=torch_spectrum_metadata_from_batch(batch),
        )

    return feature_extractor


def _move_msg_probe_batch(
    batch: dict[str, Any],
    *,
    device: torch.device,
) -> ProbeBatch:
    return cast(
        ProbeBatch,
        {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        },
    )


def _collect_msg_probe_task_spec(
    *,
    probe_data: MassSpecProbeData,
    peak_ordering: str,
    train_seed_base: int,
    test_seed_base: int,
    fingerprint_task: str,
    regression_tasks: tuple[str, ...],
    binary_tasks: tuple[str, ...],
    early_stopping: bool,
    distributed: DistributedContext | None,
) -> MsgProbeTaskSpec:
    del test_seed_base, early_stopping
    train_targets = _collect_split_targets(
        probe_data=probe_data,
        split="massspec_train",
        peak_ordering=peak_ordering,
        seed=train_seed_base,
        fingerprint_task=fingerprint_task,
        regression_tasks=regression_tasks,
        binary_tasks=binary_tasks,
        distributed=distributed,
    )
    return _build_task_spec(
        train_targets=train_targets,
        test_targets=train_targets,
        fingerprint_task=fingerprint_task,
        regression_tasks=regression_tasks,
        binary_tasks=binary_tasks,
    )


def _setup_msg_probe(
    *,
    config: config_dict.ConfigDict,
    model: Any,
    device: torch.device,
    repeat_index: int,
    distributed: DistributedContext | None,
    online_maccs_only: bool,
    on_probe_data: Callable[[MassSpecProbeData], None] | None = None,
) -> _MsgProbeSetup:
    num_epochs = int(config.get("msg_probe_num_epochs", 5))
    learning_rate = float(config.get("msg_probe_learning_rate", 1e-3))
    weight_decay = float(config.get("msg_probe_weight_decay", 1e-2))
    raw_grad_clip_norm = config.get("msg_probe_grad_clip_norm", None)
    grad_clip_norm = (
        None if raw_grad_clip_norm is None else float(raw_grad_clip_norm)
    )
    early_stopping = bool(config.get("msg_probe_early_stopping", False))
    early_stopping_patience = int(
        config.get("msg_probe_early_stopping_patience", 10)
    )
    early_stopping_min_delta = float(
        config.get("msg_probe_early_stopping_min_delta", 0.0)
    )
    early_stopping_min_epochs = int(
        config.get("msg_probe_early_stopping_min_epochs", 1)
    )
    fingerprint_task = (
        _MACCS_TASK if online_maccs_only else resolve_msg_probe_fingerprint(config)
    )
    regression_tasks = () if online_maccs_only else _REGRESSION_PROBE_TASKS
    binary_tasks = () if online_maccs_only else _BINARY_PROBE_TASKS
    probe_data_kwargs = {
        "distributed_world_size": _distributed_world_size(distributed),
        "distributed_rank": _distributed_rank(distributed),
        "distributed_local_rank": _distributed_local_rank(distributed),
    }
    if online_maccs_only:
        probe_data_kwargs["maccs_only"] = True
    probe_data = MassSpecProbeData.from_config(config, **probe_data_kwargs)
    if on_probe_data is not None:
        on_probe_data(probe_data)
    peak_ordering = peak_preprocessing_contract(config)["peak_ordering"]
    variants = msg_probe_variants_from_config(config)
    feature_extractor = _make_msg_probe_feature_extractor(
        model,
        use_pair_features=any(_uses_pair_features(variant) for variant in variants),
    )
    seed_offset = 100_000 * repeat_index
    train_seed_base = int(config.seed) + 1_100_000 + seed_offset
    test_seed_base = int(config.seed) + 1_200_000 + seed_offset
    task_spec = _collect_msg_probe_task_spec(
        probe_data=probe_data,
        peak_ordering=peak_ordering,
        train_seed_base=train_seed_base,
        test_seed_base=test_seed_base,
        fingerprint_task=fingerprint_task,
        regression_tasks=regression_tasks,
        binary_tasks=binary_tasks,
        early_stopping=early_stopping,
        distributed=distributed,
    )
    return _MsgProbeSetup(
        probe_data=probe_data,
        task_spec=task_spec,
        variants=variants,
        feature_extractor=feature_extractor,
        move_batch=lambda batch: _move_msg_probe_batch(batch, device=device),
        device=device,
        distributed=distributed,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm,
        early_stopping=early_stopping,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        early_stopping_min_epochs=early_stopping_min_epochs,
        peak_ordering=peak_ordering,
        fingerprint_task=fingerprint_task,
        train_seed_base=train_seed_base,
        test_seed_base=test_seed_base,
        online_maccs_only=online_maccs_only,
    )


def _initialize_msg_probe_training(
    *,
    config: config_dict.ConfigDict,
    setup: _MsgProbeSetup,
    covariance_pooler: torch.nn.Module | None,
) -> _MsgProbeTrainingState:
    probes = {
        variant: _build_msg_sequence_probe(
            variant,
            config=config,
            task_spec=setup.task_spec,
            covariance_pooler=_online_probe_covariance_pooler(
                variant,
                covariance_pooler,
            ),
        ).to(setup.device)
        for variant in setup.variants
    }
    train_probes = {
        variant: _wrap_probe_for_distributed(probe, setup.distributed)
        for variant, probe in probes.items()
    }
    optimizers = {
        variant: torch.optim.AdamW(
            train_probes[variant].parameters(),
            lr=setup.learning_rate,
            weight_decay=setup.weight_decay,
        )
        for variant in probes
    }
    steps_per_epoch = probe_steps_per_epoch(
        setup.probe_data,
        split="massspec_train",
        drop_remainder=False,
        distributed_world_size=_distributed_world_size(setup.distributed),
    )
    warmup_steps = _resolve_probe_warmup_steps(config, steps_per_epoch)
    schedulers = {
        variant: torch.optim.lr_scheduler.LambdaLR(
            optimizers[variant],
            lr_lambda=lambda step_idx: (
                learning_rate_at_step(
                    step_idx + 1,
                    base_lr=setup.learning_rate,
                    total_steps=setup.num_epochs * steps_per_epoch,
                    warmup_steps=warmup_steps,
                )
                / setup.learning_rate
            ),
        )
        for variant in setup.variants
    }
    select_metric = (
        "msg_probe/test/auc_maccs_mean"
        if setup.online_maccs_only
        else resolve_msg_probe_select_metric(config)
    )
    higher_is_better = msg_probe_metric_higher_is_better(select_metric)
    return _MsgProbeTrainingState(
        probes=probes,
        train_probes=train_probes,
        optimizers=optimizers,
        schedulers=schedulers,
        select_metric=select_metric,
        higher_is_better=higher_is_better,
        best_metrics_by_variant={},
        best_metric_values={
            variant: -float("inf") if higher_is_better else float("inf")
            for variant in setup.variants
        },
        best_state_by_variant={},
        epochs_without_improvement={variant: 0 for variant in setup.variants},
    )


def _train_and_evaluate_msg_probe_epoch(
    *,
    setup: _MsgProbeSetup,
    state: _MsgProbeTrainingState,
    epoch_idx: int,
) -> tuple[dict[str, EpochState], dict[str, EpochState]]:
    for probe in state.probes.values():
        probe.train()
    train_states = {
        variant: _new_epoch_state(setup.task_spec) for variant in setup.variants
    }
    train_iterator = iter_massspec_probe(
        setup.probe_data,
        "massspec_train",
        seed=setup.train_seed_base + epoch_idx,
        peak_ordering=setup.peak_ordering,
        drop_remainder=False,
        distributed_world_size=_distributed_world_size(setup.distributed),
        distributed_rank=_distributed_rank(setup.distributed),
        pad_distributed=True,
    )
    for batch in train_iterator:
        batch = setup.move_batch(batch)
        features = setup.feature_extractor(batch)
        for variant in setup.variants:
            state.optimizers[variant].zero_grad(set_to_none=True)
            result = _sequence_probe_step(
                state.train_probes[variant],
                batch,
                features,
                task_spec=setup.task_spec,
                device=setup.device,
                allow_empty=_is_distributed(setup.distributed),
            )
            if result is None:
                continue
            result["loss_total"].backward()
            if setup.grad_clip_norm is not None and setup.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    state.train_probes[variant].parameters(),
                    max_norm=setup.grad_clip_norm,
                )
            state.optimizers[variant].step()
            state.schedulers[variant].step()
            if int(result["batch_size"]) > 0:
                _update_epoch_state(train_states[variant], result, setup.task_spec)
    train_states = _gather_variant_states(
        train_states,
        setup.task_spec,
        setup.distributed,
    )
    for probe in state.probes.values():
        probe.eval()
    eval_states = _evaluate_sequence_probe_split(
        probe_data=setup.probe_data,
        probes=state.probes,
        task_spec=setup.task_spec,
        feature_extractor=setup.feature_extractor,
        move_batch=setup.move_batch,
        split="massspec_val",
        seed=setup.train_seed_base + 10_000,
        peak_ordering=setup.peak_ordering,
        max_samples=None,
        sample_randomly=False,
        device=setup.device,
        distributed=setup.distributed,
    )
    return train_states, eval_states


def _msg_probe_selection_metric_key(
    variant: str,
    *,
    select_metric: str,
) -> str:
    key = _msg_probe_variant_metric_key(variant, select_metric)
    return key.replace("/test/", "/val/")


def _log_msg_probe_epoch_metrics(
    *,
    setup: _MsgProbeSetup,
    variant: str,
    variant_metrics: dict[str, Any],
    epoch_idx: int,
) -> None:
    variant_prefix = f"msg_probe/{variant}"
    log.info(
        "MSG probe [%s] epoch %d/%d train_samples=%d %s_auc_%s_mean=%.4f %s_average_precision_%s_mean=%.4f %s_recall_%s_mean=%.4f %s_precision_%s_mean=%.4f %s_bits=%d",
        variant,
        epoch_idx + 1,
        setup.num_epochs,
        int(variant_metrics[f"{variant_prefix}/train/samples"]),
        "val",
        setup.fingerprint_task,
        variant_metrics[
            f"{variant_prefix}/val/auc_{setup.fingerprint_task}_mean"
        ],
        "val",
        setup.fingerprint_task,
        variant_metrics[
            f"{variant_prefix}/val/average_precision_{setup.fingerprint_task}_mean"
        ],
        "val",
        setup.fingerprint_task,
        variant_metrics[
            f"{variant_prefix}/val/recall_{setup.fingerprint_task}_mean"
        ],
        "val",
        setup.fingerprint_task,
        variant_metrics[
            f"{variant_prefix}/val/precision_{setup.fingerprint_task}_mean"
        ],
        setup.fingerprint_task,
        int(variant_metrics[f"{variant_prefix}/num_{setup.fingerprint_task}_bits"]),
    )


def _score_msg_probe_epoch(
    *,
    setup: _MsgProbeSetup,
    state: _MsgProbeTrainingState,
    train_states: dict[str, EpochState],
    eval_states: dict[str, EpochState],
    epoch_idx: int,
    on_epoch_end: Callable[[dict[str, float]], None] | None,
) -> bool:
    epoch_metrics: dict[str, float] = {}
    for variant in setup.variants:
        variant_prefix = f"msg_probe/{variant}"
        variant_metrics = {
            **_score_epoch_state(
                prefix=f"{variant_prefix}/train",
                epoch_state=train_states[variant],
                task_spec=setup.task_spec,
            ),
            **_score_epoch_state(
                prefix=f"{variant_prefix}/val",
                epoch_state=eval_states[variant],
                task_spec=setup.task_spec,
            ),
            f"{variant_prefix}/num_{setup.fingerprint_task}_bits": float(
                setup.task_spec.maccs_bits
            ),
            f"{variant_prefix}/epoch": float(epoch_idx + 1),
        }
        epoch_metrics.update(variant_metrics)
        select_metric = _msg_probe_selection_metric_key(
            variant,
            select_metric=state.select_metric,
        )
        current_value = variant_metrics[select_metric]
        previous_best = state.best_metric_values[variant]
        is_better = (
            current_value > previous_best + setup.early_stopping_min_delta
            if state.higher_is_better
            else current_value < previous_best - setup.early_stopping_min_delta
        )
        if is_better:
            state.best_metric_values[variant] = current_value
            state.best_metrics_by_variant[variant] = dict(variant_metrics)
            state.best_state_by_variant[variant] = copy.deepcopy(
                state.probes[variant].state_dict()
            )
            state.epochs_without_improvement[variant] = 0
        else:
            state.epochs_without_improvement[variant] += 1
        if _is_main(setup.distributed):
            _log_msg_probe_epoch_metrics(
                setup=setup,
                variant=variant,
                variant_metrics=variant_metrics,
                epoch_idx=epoch_idx,
            )
    if on_epoch_end is not None and _is_main(setup.distributed):
        on_epoch_end(epoch_metrics)
    should_stop = (
        setup.early_stopping
        and epoch_idx + 1 >= setup.early_stopping_min_epochs
        and all(
            state.epochs_without_improvement[variant]
            >= setup.early_stopping_patience
            for variant in setup.variants
        )
    )
    if should_stop and _is_main(setup.distributed):
        log.info(
            "MSG probe early stopping at epoch %d/%d after %d epochs without validation improvement",
            epoch_idx + 1,
            setup.num_epochs,
            setup.early_stopping_patience,
        )
    return should_stop


def _restore_best_msg_probe_states(
    setup: _MsgProbeSetup,
    state: _MsgProbeTrainingState,
) -> None:
    for variant in setup.variants:
        if variant in state.best_state_by_variant:
            state.probes[variant].load_state_dict(
                state.best_state_by_variant[variant]
            )


def _evaluate_final_msg_probe_split(
    *,
    setup: _MsgProbeSetup,
    state: _MsgProbeTrainingState,
) -> dict[str, dict[str, Any]]:
    selected_probes = {
        variant: state.probes[variant]
        for variant in setup.variants
        if variant in state.best_state_by_variant
    }
    final_test_states = _evaluate_sequence_probe_split(
        probe_data=setup.probe_data,
        probes=selected_probes,
        task_spec=setup.task_spec,
        feature_extractor=setup.feature_extractor,
        move_batch=setup.move_batch,
        split="massspec_test",
        seed=setup.test_seed_base,
        peak_ordering=setup.peak_ordering,
        max_samples=None,
        sample_randomly=False,
        device=setup.device,
        distributed=setup.distributed,
    )
    return {
        variant: _score_epoch_state(
            prefix=f"msg_probe/{variant}/test",
            epoch_state=epoch_state,
            task_spec=setup.task_spec,
            include_pr_curves=True,
        )
        for variant, epoch_state in final_test_states.items()
    }


def _merge_msg_probe_best_metrics(
    *,
    setup: _MsgProbeSetup,
    state: _MsgProbeTrainingState,
    final_test_metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    best_metrics: dict[str, Any] = {}
    for variant in setup.variants:
        variant_metrics = dict(state.best_metrics_by_variant.get(variant, {}))
        if not variant_metrics:
            continue
        variant_metrics.update(final_test_metrics.get(variant, {}))
        best_metrics.update(variant_metrics)
        variant_prefix = f"msg_probe/{variant}"
        select_metric = _msg_probe_selection_metric_key(
            variant,
            select_metric=state.select_metric,
        )
        if _is_main(setup.distributed):
            log.info(
                "MSG probe [%s] best epoch %d: %s=%.4f test_auc_%s_mean=%.4f test_average_precision_%s_mean=%.4f test_recall_%s_mean=%.4f test_precision_%s_mean=%.4f",
                variant,
                int(variant_metrics[f"{variant_prefix}/epoch"]),
                select_metric,
                variant_metrics[select_metric],
                setup.fingerprint_task,
                variant_metrics[
                    f"{variant_prefix}/test/auc_{setup.fingerprint_task}_mean"
                ],
                setup.fingerprint_task,
                variant_metrics[
                    f"{variant_prefix}/test/average_precision_{setup.fingerprint_task}_mean"
                ],
                setup.fingerprint_task,
                variant_metrics[
                    f"{variant_prefix}/test/recall_{setup.fingerprint_task}_mean"
                ],
                setup.fingerprint_task,
                variant_metrics[
                    f"{variant_prefix}/test/precision_{setup.fingerprint_task}_mean"
                ],
            )
    return best_metrics


def _finalize_msg_probe_once(
    *,
    config: config_dict.ConfigDict,
    setup: _MsgProbeSetup,
    state: _MsgProbeTrainingState,
    model: Any,
    covariance_pooler: torch.nn.Module | None,
    plot_dir: Path | None,
    plot_step: int | None,
    repeat_index: int,
) -> dict[str, Any]:
    _restore_best_msg_probe_states(setup, state)
    final_test_metrics = _evaluate_final_msg_probe_split(
        setup=setup,
        state=state,
    )
    best_metrics = _merge_msg_probe_best_metrics(
        setup=setup,
        state=state,
        final_test_metrics=final_test_metrics,
    )
    alignment_pooler = covariance_pooler
    if alignment_pooler is None and "covariance" in state.probes:
        alignment_pooler = state.probes["covariance"].pooler
    if not setup.online_maccs_only:
        best_metrics.update(
            _run_covariance_morgan_pairwise_alignment(
                config=config,
                probe_data=setup.probe_data,
                model=model,
                covariance_pooler=alignment_pooler,
                feature_extractor=setup.feature_extractor,
                move_batch=setup.move_batch,
                device=setup.device,
                split="massspec_test",
                peak_ordering=setup.peak_ordering,
                seed=setup.test_seed_base + 50_000,
                max_samples=None,
                sample_randomly=False,
                plot_dir=plot_dir,
                plot_step=plot_step,
                repeat_index=repeat_index,
                distributed=setup.distributed,
            )
        )
    return best_metrics


def _run_msg_probe_once(
    *,
    config: config_dict.ConfigDict,
    model: Any,
    device: torch.device,
    covariance_pooler: torch.nn.Module | None = None,
    on_epoch_end: Callable[[dict[str, float]], None] | None = None,
    repeat_index: int = 0,
    plot_dir: Path | None = None,
    plot_step: int | None = None,
    distributed: DistributedContext | None = None,
    online_maccs_only: bool = False,
    on_probe_data: Callable[[MassSpecProbeData], None] | None = None,
) -> dict[str, Any]:
    setup = _setup_msg_probe(
        config=config,
        model=model,
        device=device,
        repeat_index=repeat_index,
        distributed=distributed,
        online_maccs_only=online_maccs_only,
        on_probe_data=on_probe_data,
    )
    was_training = model.training
    model.eval()
    state = _initialize_msg_probe_training(
        config=config,
        setup=setup,
        covariance_pooler=covariance_pooler,
    )
    for epoch_idx in range(setup.num_epochs):
        train_states, eval_states = _train_and_evaluate_msg_probe_epoch(
            setup=setup,
            state=state,
            epoch_idx=epoch_idx,
        )
        should_stop = _score_msg_probe_epoch(
            setup=setup,
            state=state,
            train_states=train_states,
            eval_states=eval_states,
            epoch_idx=epoch_idx,
            on_epoch_end=on_epoch_end,
        )
        if should_stop:
            break
    best_metrics = _finalize_msg_probe_once(
        config=config,
        setup=setup,
        state=state,
        model=model,
        covariance_pooler=covariance_pooler,
        plot_dir=plot_dir,
        plot_step=plot_step,
        repeat_index=repeat_index,
    )
    if was_training:
        model.train()
    return best_metrics


def run_msg_probe(
    *,
    config: config_dict.ConfigDict,
    model: PeakSetJEPA,
    device: torch.device,
    covariance_pooler: torch.nn.Module | None = None,
    on_epoch_end: Callable[[dict[str, float]], None] | None = None,
    plot_dir: Path | None = None,
    plot_step: int | None = None,
    distributed: DistributedContext | None = None,
    online_maccs_only: bool = False,
    on_probe_data: Callable[[MassSpecProbeData], None] | None = None,
) -> dict[str, Any]:
    def run_once(
        repeat_index: int,
        repeat_on_epoch_end: Callable[[dict[str, float]], None] | None,
    ) -> dict[str, Any]:
        return _run_msg_probe_once(
            config=config,
            model=model,
            device=device,
            covariance_pooler=covariance_pooler,
            on_epoch_end=repeat_on_epoch_end,
            repeat_index=repeat_index,
            plot_dir=plot_dir,
            plot_step=plot_step,
            distributed=distributed,
            online_maccs_only=online_maccs_only,
            on_probe_data=on_probe_data,
        )

    metrics = _run_repeated_probe(
        repeat_count=resolve_msg_probe_num_repeats(config),
        metric_prefix="msg_probe",
        run_once=run_once,
        on_epoch_end=on_epoch_end,
        on_repeat_start=lambda repeat_index, repeat_count: log.info(
            "msg_probe repeat %d/%d",
            repeat_index + 1,
            repeat_count,
        ),
    )
    return metrics


@dataclass(frozen=True)
class _DreamsProbeSetup:
    probe_data: MassSpecProbeData
    task_spec: MsgProbeTaskSpec
    dreams_dim: int
    device: torch.device
    num_epochs: int
    learning_rate: float
    weight_decay: float
    grad_clip_norm: float | None
    early_stopping: bool
    early_stopping_patience: int
    early_stopping_min_delta: float
    early_stopping_min_epochs: int
    peak_ordering: str
    fingerprint_task: str
    train_seed_base: int
    test_seed_base: int
    feature_extractor: Callable[[ProbeBatch], torch.Tensor]
    move_batch: Callable[[dict[str, Any]], ProbeBatch]


@dataclass
class _DreamsProbeTrainingState:
    probe: MsgLinearProbe
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LambdaLR
    compiled_probe_step: Callable[..., ProbeStepResult | None]
    select_metric: str
    higher_is_better: bool
    best_metrics: dict[str, float]
    best_metric_value: float
    best_state: dict[str, torch.Tensor]
    epochs_without_improvement: int


def _setup_dreams_probe(
    *,
    config: config_dict.ConfigDict,
    device: torch.device,
    repeat_index: int,
) -> _DreamsProbeSetup | None:
    num_epochs = int(config.get("msg_probe_num_epochs", 5))
    learning_rate = float(config.get("msg_probe_learning_rate", 1e-3))
    weight_decay = float(config.get("msg_probe_weight_decay", 1e-2))
    raw_grad_clip_norm = config.get("msg_probe_grad_clip_norm", None)
    grad_clip_norm = (
        None if raw_grad_clip_norm is None else float(raw_grad_clip_norm)
    )
    early_stopping = bool(config.get("msg_probe_early_stopping", False))
    early_stopping_patience = int(
        config.get("msg_probe_early_stopping_patience", 10)
    )
    early_stopping_min_delta = float(
        config.get("msg_probe_early_stopping_min_delta", 0.0)
    )
    early_stopping_min_epochs = int(
        config.get("msg_probe_early_stopping_min_epochs", 1)
    )
    fingerprint_task = resolve_msg_probe_fingerprint(config)
    data_config = config.copy_and_resolve_references()
    data_config.nist_murcko_probe_include_dreams_auxiliary = True
    probe_data = MassSpecProbeData.from_config(data_config)
    peak_ordering = peak_preprocessing_contract(data_config)["peak_ordering"]
    dreams_dim = probe_data.dreams_dim
    if dreams_dim == 0:
        log.warning("No DreaMS embeddings in probe data; skipping Dreams probe")
        return None
    seed_offset = 100_000 * repeat_index
    train_seed_base = int(config.seed) + 1_100_000 + seed_offset
    test_seed_base = int(config.seed) + 1_200_000 + seed_offset
    task_spec = _collect_msg_probe_task_spec(
        probe_data=probe_data,
        peak_ordering=peak_ordering,
        train_seed_base=train_seed_base,
        test_seed_base=test_seed_base,
        fingerprint_task=fingerprint_task,
        regression_tasks=_REGRESSION_PROBE_TASKS,
        binary_tasks=_BINARY_PROBE_TASKS,
        early_stopping=early_stopping,
        distributed=None,
    )
    return _DreamsProbeSetup(
        probe_data=probe_data,
        task_spec=task_spec,
        dreams_dim=dreams_dim,
        device=device,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm,
        early_stopping=early_stopping,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        early_stopping_min_epochs=early_stopping_min_epochs,
        peak_ordering=peak_ordering,
        fingerprint_task=fingerprint_task,
        train_seed_base=train_seed_base,
        test_seed_base=test_seed_base,
        feature_extractor=lambda batch: batch["dreams_embedding"].to(
            device=device,
            dtype=torch.float32,
        ),
        move_batch=lambda batch: _move_msg_probe_batch(batch, device=device),
    )


def _initialize_dreams_probe_training(
    *,
    config: config_dict.ConfigDict,
    setup: _DreamsProbeSetup,
) -> _DreamsProbeTrainingState:
    probe = MsgLinearProbe(
        input_dim=setup.dreams_dim,
        task_names=_probe_task_names(setup.task_spec),
        task_output_dims=_probe_task_output_dims(setup.task_spec),
    ).to(setup.device)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=setup.learning_rate,
        weight_decay=setup.weight_decay,
    )
    steps_per_epoch = probe_steps_per_epoch(
        setup.probe_data,
        split="massspec_train",
        drop_remainder=False,
    )
    warmup_steps = _resolve_probe_warmup_steps(config, steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step_idx: (
            learning_rate_at_step(
                step_idx + 1,
                base_lr=setup.learning_rate,
                total_steps=setup.num_epochs * steps_per_epoch,
                warmup_steps=warmup_steps,
            )
            / setup.learning_rate
        ),
    )
    compiled_probe_step = torch.compile(_probe_step)
    select_metric = resolve_msg_probe_select_metric(config).replace(
        "msg_probe/",
        "dreams_probe/",
    )
    select_metric = select_metric.replace("/test/", "/val/").replace(
        "/mean/",
        "/",
    )
    higher_is_better = msg_probe_metric_higher_is_better(select_metric)
    return _DreamsProbeTrainingState(
        probe=probe,
        optimizer=optimizer,
        scheduler=scheduler,
        compiled_probe_step=compiled_probe_step,
        select_metric=select_metric,
        higher_is_better=higher_is_better,
        best_metrics={},
        best_metric_value=(
            -float("inf") if higher_is_better else float("inf")
        ),
        best_state={},
        epochs_without_improvement=0,
    )


def _train_and_evaluate_dreams_probe_epoch(
    *,
    setup: _DreamsProbeSetup,
    state: _DreamsProbeTrainingState,
    epoch_idx: int,
) -> tuple[EpochState, EpochState]:
    state.probe.train()
    train_state = _new_epoch_state(setup.task_spec)
    train_iterator = iter_massspec_probe(
        setup.probe_data,
        "massspec_train",
        seed=setup.train_seed_base + epoch_idx,
        peak_ordering=setup.peak_ordering,
        drop_remainder=False,
    )
    for batch in train_iterator:
        batch = setup.move_batch(batch)
        state.optimizer.zero_grad(set_to_none=True)
        result = state.compiled_probe_step(
            state.probe,
            batch,
            task_spec=setup.task_spec,
            device=setup.device,
            feature_extractor=setup.feature_extractor,
        )
        if result is None:
            continue
        result["loss_total"].backward()
        if setup.grad_clip_norm is not None and setup.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                state.probe.parameters(),
                max_norm=setup.grad_clip_norm,
            )
        state.optimizer.step()
        state.scheduler.step()
        _update_epoch_state(train_state, result, setup.task_spec)
    state.probe.eval()
    val_state = _evaluate_linear_probe_split(
        probe_data=setup.probe_data,
        probe=state.probe,
        compiled_probe_step=state.compiled_probe_step,
        task_spec=setup.task_spec,
        feature_extractor=setup.feature_extractor,
        move_batch=setup.move_batch,
        split="massspec_val",
        seed=setup.train_seed_base + 10_000,
        peak_ordering=setup.peak_ordering,
        max_samples=None,
        sample_randomly=False,
        device=setup.device,
    )
    return train_state, val_state


def _log_dreams_probe_epoch_metrics(
    *,
    setup: _DreamsProbeSetup,
    epoch_metrics: dict[str, float],
    epoch_idx: int,
) -> None:
    log.info(
        "DreaMS probe epoch %d/%d train_samples=%d val_auc_%s_mean=%.4f val_average_precision_%s_mean=%.4f val_recall_%s_mean=%.4f val_precision_%s_mean=%.4f %s_bits=%d",
        epoch_idx + 1,
        setup.num_epochs,
        int(epoch_metrics["dreams_probe/train/samples"]),
        setup.fingerprint_task,
        epoch_metrics[f"dreams_probe/val/auc_{setup.fingerprint_task}_mean"],
        setup.fingerprint_task,
        epoch_metrics[
            f"dreams_probe/val/average_precision_{setup.fingerprint_task}_mean"
        ],
        setup.fingerprint_task,
        epoch_metrics[f"dreams_probe/val/recall_{setup.fingerprint_task}_mean"],
        setup.fingerprint_task,
        epoch_metrics[
            f"dreams_probe/val/precision_{setup.fingerprint_task}_mean"
        ],
        setup.fingerprint_task,
        int(epoch_metrics[f"dreams_probe/num_{setup.fingerprint_task}_bits"]),
    )


def _score_dreams_probe_epoch(
    *,
    setup: _DreamsProbeSetup,
    state: _DreamsProbeTrainingState,
    train_state: EpochState,
    val_state: EpochState,
    epoch_idx: int,
    on_epoch_end: Callable[[dict[str, float]], None] | None,
) -> bool:
    epoch_metrics = {
        **_score_epoch_state(
            prefix="dreams_probe/train",
            epoch_state=train_state,
            task_spec=setup.task_spec,
        ),
        **_score_epoch_state(
            prefix="dreams_probe/val",
            epoch_state=val_state,
            task_spec=setup.task_spec,
        ),
        f"dreams_probe/num_{setup.fingerprint_task}_bits": float(
            setup.task_spec.maccs_bits
        ),
        "dreams_probe_epoch": float(epoch_idx + 1),
    }
    current_value = epoch_metrics[state.select_metric]
    is_better = (
        current_value > state.best_metric_value + setup.early_stopping_min_delta
        if state.higher_is_better
        else current_value < state.best_metric_value - setup.early_stopping_min_delta
    )
    if is_better:
        state.best_metric_value = current_value
        state.best_metrics = dict(epoch_metrics)
        state.best_state = copy.deepcopy(state.probe.state_dict())
        state.epochs_without_improvement = 0
    else:
        state.epochs_without_improvement += 1
    _log_dreams_probe_epoch_metrics(
        setup=setup,
        epoch_metrics=epoch_metrics,
        epoch_idx=epoch_idx,
    )
    if on_epoch_end is not None:
        on_epoch_end(epoch_metrics)
    should_stop = (
        setup.early_stopping
        and epoch_idx + 1 >= setup.early_stopping_min_epochs
        and state.epochs_without_improvement >= setup.early_stopping_patience
    )
    if should_stop:
        log.info(
            "DreaMS probe early stopping at epoch %d/%d after %d epochs without validation improvement",
            epoch_idx + 1,
            setup.num_epochs,
            setup.early_stopping_patience,
        )
    return should_stop


def _finalize_dreams_probe_once(
    *,
    setup: _DreamsProbeSetup,
    state: _DreamsProbeTrainingState,
) -> dict[str, float]:
    if not state.best_metrics:
        return state.best_metrics
    state.probe.load_state_dict(state.best_state)
    test_state = _evaluate_linear_probe_split(
        probe_data=setup.probe_data,
        probe=state.probe,
        compiled_probe_step=state.compiled_probe_step,
        task_spec=setup.task_spec,
        feature_extractor=setup.feature_extractor,
        move_batch=setup.move_batch,
        split="massspec_test",
        seed=setup.test_seed_base,
        peak_ordering=setup.peak_ordering,
        max_samples=None,
        sample_randomly=False,
        device=setup.device,
    )
    state.best_metrics.update(
        _score_epoch_state(
            prefix="dreams_probe/test",
            epoch_state=test_state,
            task_spec=setup.task_spec,
        )
    )
    log.info(
        "DreaMS probe best epoch %d: %s=%.4f test_auc_%s_mean=%.4f test_average_precision_%s_mean=%.4f test_recall_%s_mean=%.4f test_precision_%s_mean=%.4f",
        int(state.best_metrics["dreams_probe_epoch"]),
        state.select_metric,
        state.best_metrics[state.select_metric],
        setup.fingerprint_task,
        state.best_metrics[f"dreams_probe/test/auc_{setup.fingerprint_task}_mean"],
        setup.fingerprint_task,
        state.best_metrics[
            f"dreams_probe/test/average_precision_{setup.fingerprint_task}_mean"
        ],
        setup.fingerprint_task,
        state.best_metrics[
            f"dreams_probe/test/recall_{setup.fingerprint_task}_mean"
        ],
        setup.fingerprint_task,
        state.best_metrics[
            f"dreams_probe/test/precision_{setup.fingerprint_task}_mean"
        ],
    )
    return state.best_metrics


def _run_dreams_probe_once(
    *,
    config: config_dict.ConfigDict,
    device: torch.device,
    on_epoch_end: Callable[[dict[str, float]], None] | None = None,
    repeat_index: int = 0,
) -> dict[str, float]:
    setup = _setup_dreams_probe(
        config=config,
        device=device,
        repeat_index=repeat_index,
    )
    if setup is None:
        return {}
    state = _initialize_dreams_probe_training(
        config=config,
        setup=setup,
    )
    for epoch_idx in range(setup.num_epochs):
        train_state, val_state = _train_and_evaluate_dreams_probe_epoch(
            setup=setup,
            state=state,
            epoch_idx=epoch_idx,
        )
        should_stop = _score_dreams_probe_epoch(
            setup=setup,
            state=state,
            train_state=train_state,
            val_state=val_state,
            epoch_idx=epoch_idx,
            on_epoch_end=on_epoch_end,
        )
        if should_stop:
            break
    return _finalize_dreams_probe_once(setup=setup, state=state)


def run_dreams_probe(
    *,
    config: config_dict.ConfigDict,
    device: torch.device,
    on_epoch_end: Callable[[dict[str, float]], None] | None = None,
) -> dict[str, float]:
    return _run_repeated_probe(
        repeat_count=resolve_msg_probe_num_repeats(config),
        metric_prefix="dreams_probe",
        run_once=lambda repeat_index, repeat_on_epoch_end: _run_dreams_probe_once(
            config=config,
            device=device,
            on_epoch_end=repeat_on_epoch_end,
            repeat_index=repeat_index,
        ),
        on_epoch_end=on_epoch_end,
        on_repeat_start=lambda repeat_index, repeat_count: log.info(
            "dreams_probe repeat %d/%d",
            repeat_index + 1,
            repeat_count,
        ),
    )
