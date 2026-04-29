import logging
import math
import copy
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from ml_collections import config_dict
from sklearn.metrics import r2_score

from input_pipeline import numpy_batch_to_torch
from models.model import CovariancePool, CrossAttention, PeakSetEncoder, PeakSetSIGReg
from utils.massspec_probe_data import MassSpecProbeData
from utils.massspec_probe_targets import (
    FG_SMARTS,
    MACCS_FINGERPRINT_BITS,
    MORGAN_PROBE_FINGERPRINT_BITS,
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
    fingerprint_task: str = "maccs"


class MsgProbeSplitTargets(NamedTuple):
    regression: dict[str, np.ndarray]
    maccs: np.ndarray


class MsgProbePairwiseAlignment(NamedTuple):
    tanimoto: np.ndarray
    cosine: np.ndarray
    pearson: float


def build_msg_probe_inputs(
    peak_embeddings: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    mask = valid_mask.unsqueeze(-1).to(dtype=peak_embeddings.dtype)
    return (peak_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


_NUM_RINGS_TASK = "num_rings"
_MACCS_TASK = "maccs"
_MORGAN_TASK = "morgan"
_REGRESSION_PROBE_TASKS = tuple(
    name for name in REGRESSION_TARGET_KEYS if name != _NUM_RINGS_TASK
)
_PROBE_FINGERPRINT_BITS = {
    _MACCS_TASK: MACCS_FINGERPRINT_BITS,
    _MORGAN_TASK: MORGAN_PROBE_FINGERPRINT_BITS,
}


def msg_probe_variants_from_config(
    config: config_dict.ConfigDict,
) -> tuple[str, ...]:
    raw_variants = config.get("msg_probe_variants", ("mean", "covariance", "pma"))
    if isinstance(raw_variants, str):
        return (raw_variants.lower(),)
    return tuple(str(variant).lower() for variant in raw_variants)


def resolve_msg_probe_fingerprint(
    config: config_dict.ConfigDict,
) -> str:
    return str(
        config.get(
            "msg_probe_fingerprint",
            config.get("msg_probe_fingerprint_type", _MACCS_TASK),
        )
    ).lower()


def resolve_msg_probe_sample_limits(
    config: config_dict.ConfigDict,
) -> tuple[int | None, int | None, int | None, bool]:
    probe_dataset = str(config.get("probe_dataset", "massspec"))
    raw_sample_size = config.get("msg_probe_sample_size", None)
    raw_train = config.get("msg_probe_max_train_samples", None)
    raw_val = config.get("msg_probe_max_val_samples", None)
    raw_test = config.get("msg_probe_max_test_samples", None)
    if raw_train is None:
        raw_train = raw_sample_size
    if raw_val is None:
        raw_val = raw_sample_size
    if raw_test is None:
        raw_test = raw_sample_size
    if raw_train is None and probe_dataset == "nist-full":
        raw_train = config.get("nist_full_probe_train_samples", 4_000)
    if raw_val is None and probe_dataset == "nist-full":
        raw_val = config.get("nist_full_probe_val_samples", 1_000)
    if raw_test is None and probe_dataset == "nist-full":
        raw_test = config.get("nist_full_probe_test_samples", 1_000)
    max_train_samples = int(raw_train) if raw_train is not None else None
    max_val_samples = int(raw_val) if raw_val is not None else None
    max_test_samples = int(raw_test) if raw_test is not None else None
    randomize_test_subset = probe_dataset == "nist-full" and max_test_samples is not None
    return max_train_samples, max_val_samples, max_test_samples, randomize_test_subset


def resolve_msg_probe_num_repeats(
    config: config_dict.ConfigDict,
) -> int:
    probe_dataset = str(config.get("probe_dataset", "massspec"))
    raw_repeats = config.get("msg_probe_num_repeats", None)
    if raw_repeats is None and probe_dataset == "nist-full":
        raw_repeats = config.get("nist_full_probe_num_repeats", 1)
    return int(raw_repeats) if raw_repeats is not None else 1


def resolve_msg_probe_pairwise_alignment_num_pairs(
    config: config_dict.ConfigDict,
) -> int:
    return int(config.get("msg_probe_pairwise_alignment_num_pairs", 20_000))


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


class MsgCovariancePool(CovariancePool):
    pass


class FrozenPooler(torch.nn.Module):
    def __init__(self, pooler: torch.nn.Module) -> None:
        super().__init__()
        object.__setattr__(self, "_pooler", pooler)

    @property
    def pooler(self) -> torch.nn.Module:
        return self._pooler

    def forward(
        self,
        peak_embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.pooler(peak_embeddings, valid_mask)


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
        task_names += (task_spec.fingerprint_task,)
    return task_names


def _probe_task_output_dims(task_spec: MsgProbeTaskSpec) -> dict[str, int]:
    output_dims: dict[str, int] = {}
    if task_spec.num_rings_classes:
        output_dims[_NUM_RINGS_TASK] = len(task_spec.num_rings_classes)
    if task_spec.maccs_bits > 0:
        output_dims[task_spec.fingerprint_task] = task_spec.maccs_bits
    return output_dims


def _build_msg_sequence_probe(
    variant: str,
    *,
    config: config_dict.ConfigDict,
    task_spec: MsgProbeTaskSpec,
    covariance_pooler: CovariancePool | None = None,
) -> MsgSequenceProbe:
    model_dim = int(config.model_dim)
    hidden_dim = int(config.get("msg_probe_mlp_hidden_dim", model_dim))
    task_names = _probe_task_names(task_spec)
    task_output_dims = _probe_task_output_dims(task_spec)
    if variant == "mean":
        pooler = MsgMeanPool()
        pooled_dim = model_dim
    elif variant == "covariance":
        if covariance_pooler is None:
            compressed_dim = int(config.get("msg_probe_covariance_dim", 32))
            pooler = MsgCovariancePool(
                input_dim=model_dim,
                compressed_dim=compressed_dim,
            )
        else:
            compressed_dim = int(covariance_pooler.left_proj.out_features)
            pooler = FrozenPooler(covariance_pooler)
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
    fingerprint_task: str = _MACCS_TASK,
) -> MsgProbeSplitTargets:
    regression = {name: [] for name in REGRESSION_TARGET_KEYS}
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
        maccs.append(batch[fingerprint_key][valid_mask].detach().cpu().numpy())

    def _cat(d, dt):
        return {
            n: np.concatenate(c) if c else np.empty(0, dtype=dt) for n, c in d.items()
        }

    return MsgProbeSplitTargets(
        regression=_cat(regression, np.float32),
        maccs=(
            np.concatenate(maccs, axis=0)
            if maccs
            else np.empty(
                (0, _PROBE_FINGERPRINT_BITS[fingerprint_task]), dtype=np.int32
            )
        ),
    )


def _collect_num_rings_classes(
    probe_data: MassSpecProbeData,
) -> tuple[int, ...]:
    classes: set[int] = set()
    for shard_dir_str in (
        *probe_data.train_files,
        *probe_data.val_files,
        *probe_data.test_files,
    ):
        shard_dir = Path(shard_dir_str)
        valid_mask = np.load(shard_dir / "probe_valid_mol.npy", mmap_mode="r")
        if not bool(np.any(valid_mask)):
            continue
        num_rings = np.load(shard_dir / "probe_num_rings.npy", mmap_mode="r")
        classes.update(
            np.asarray(num_rings[valid_mask], dtype=np.int32).tolist()
        )
    return tuple(sorted(classes))


def _build_task_spec(
    *,
    train_targets: MsgProbeSplitTargets,
    test_targets: MsgProbeSplitTargets,
    num_rings_classes: tuple[int, ...] | None = None,
    fingerprint_task: str = _MACCS_TASK,
) -> MsgProbeTaskSpec:
    regression_means, regression_stds = {}, {}
    for name in _REGRESSION_PROBE_TASKS:
        values = train_targets.regression[name].astype(np.float32)
        regression_means[name] = float(values.mean())
        regression_stds[name] = float(np.clip(values.std(), 1e-8, None))
    if num_rings_classes is None:
        num_rings_classes = tuple(
            sorted(
                np.unique(
                    train_targets.regression[_NUM_RINGS_TASK].astype(np.int32)
                ).tolist()
            )
        )
    return MsgProbeTaskSpec(
        regression_tasks=_REGRESSION_PROBE_TASKS,
        num_rings_classes=num_rings_classes,
        maccs_bits=int(train_targets.maccs.shape[1]),
        regression_means=regression_means,
        regression_stds=regression_stds,
        fingerprint_task=fingerprint_task,
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
        fingerprint_task = task_spec.fingerprint_task
        target = batch[f"probe_{fingerprint_task}"][valid_mask].to(dtype=torch.float32)
        pred = logits[fingerprint_task]
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
    rng = np.random.default_rng(int(seed))
    n = int(embeddings.shape[0])
    left_idx = rng.integers(0, n, size=int(num_pairs), dtype=np.int64)
    right_idx = rng.integers(0, n - 1, size=int(num_pairs), dtype=np.int64)
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
    model: PeakSetSIGReg,
    feature_extractor: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    move_batch: Callable[[dict[str, object]], dict[str, object]],
    split: str,
    peak_ordering: str,
    seed: int,
    max_samples: int | None,
    sample_randomly: bool,
    device: torch.device,
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
        ):
            batch = move_batch(batch)
            valid_mask = batch["probe_valid_mol"].to(device=device, dtype=torch.bool)
            if not bool(valid_mask.any()):
                continue
            peak_embeddings = feature_extractor(batch)[valid_mask]
            peak_valid_mask = batch["peak_valid_mask"][valid_mask].to(
                device=device,
                dtype=torch.bool,
            )
            covariance = model.covariance_pooler(
                peak_embeddings.float(),
                peak_valid_mask,
            )
            embeddings.append(covariance.detach().cpu().numpy())
            morgan.append(batch["probe_morgan"][valid_mask].detach().cpu().numpy())
    return np.concatenate(embeddings, axis=0), np.concatenate(morgan, axis=0)


def _collect_covariance_embeddings_for_indices(
    *,
    probe_data: MassSpecProbeData,
    model: PeakSetSIGReg,
    feature_extractor: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    move_batch: Callable[[dict[str, object]], dict[str, object]],
    indices: np.ndarray,
    peak_ordering: str,
) -> np.ndarray:
    embeddings = []
    with torch.no_grad():
        for batch in probe_data.build_indexed_dataset(
            "all",
            indices,
            peak_ordering=peak_ordering,
            drop_remainder=False,
        ):
            batch = move_batch(batch)
            peak_embeddings = feature_extractor(batch)
            covariance = model.covariance_pooler(
                peak_embeddings.float(),
                batch["peak_valid_mask"].to(dtype=torch.bool),
            )
            embeddings.append(covariance.detach().cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def _run_prepared_covariance_morgan_pairwise_alignment(
    *,
    config: config_dict.ConfigDict,
    probe_data: MassSpecProbeData,
    model: PeakSetSIGReg,
    feature_extractor: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    move_batch: Callable[[dict[str, object]], dict[str, object]],
    peak_ordering: str,
    num_pairs: int,
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
        model=model,
        feature_extractor=feature_extractor,
        move_batch=move_batch,
        indices=endpoint_indices,
        peak_ordering=peak_ordering,
    )
    return (
        _compute_pairwise_similarity_alignment_for_indices(
            embeddings=endpoint_embeddings,
            tanimoto=tanimoto,
            left_idx=left_endpoint,
            right_idx=right_endpoint,
        ),
        int(len(endpoint_indices)),
    )


def _run_covariance_morgan_pairwise_alignment(
    *,
    config: config_dict.ConfigDict,
    probe_data: MassSpecProbeData,
    model: PeakSetSIGReg,
    feature_extractor: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    move_batch: Callable[[dict[str, object]], dict[str, object]],
    device: torch.device,
    split: str,
    peak_ordering: str,
    seed: int,
    max_samples: int | None,
    sample_randomly: bool,
    plot_dir: Path | None,
    plot_step: int | None,
    repeat_index: int,
) -> dict[str, float]:
    num_pairs = resolve_msg_probe_pairwise_alignment_num_pairs(config)
    if (
        num_pairs <= 0
        or not hasattr(model, "covariance_pooler")
        or int(probe_data.info.get("probe_morgan_bits", 0)) <= 0
    ):
        return {}
    prepared = _run_prepared_covariance_morgan_pairwise_alignment(
        config=config,
        probe_data=probe_data,
        model=model,
        feature_extractor=feature_extractor,
        move_batch=move_batch,
        peak_ordering=peak_ordering,
        num_pairs=num_pairs,
    )
    if prepared is None:
        embeddings, morgan = _collect_covariance_morgan_alignment_inputs(
            probe_data=probe_data,
            model=model,
            feature_extractor=feature_extractor,
            move_batch=move_batch,
            split=split,
            peak_ordering=peak_ordering,
            seed=seed,
            max_samples=max_samples,
            sample_randomly=sample_randomly,
            device=device,
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
    if plot_dir is not None and bool(
        config.get("msg_probe_pairwise_alignment_plot", True)
    ):
        step_label = "unknown" if plot_step is None else f"{int(plot_step):08d}"
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


def _evaluate_sequence_probe_split(
    *,
    probe_data: MassSpecProbeData,
    probes: dict[str, MsgSequenceProbe],
    task_spec: MsgProbeTaskSpec,
    feature_extractor: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    move_batch: Callable[[dict[str, object]], dict[str, object]],
    split: str,
    seed: int,
    peak_ordering: str,
    max_samples: int | None,
    sample_randomly: bool,
    device: torch.device,
) -> dict[str, dict[str, object]]:
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
        ):
            batch = move_batch(batch)
            peak_embeddings = feature_extractor(batch)
            for variant, probe in probes.items():
                result = _sequence_probe_step(
                    probe,
                    batch,
                    peak_embeddings,
                    task_spec=task_spec,
                    device=device,
                )
                if result is None:
                    continue
                _update_epoch_state(states[variant], result, task_spec)
    return states


def _evaluate_linear_probe_split(
    *,
    probe_data: MassSpecProbeData,
    probe: MsgLinearProbe,
    compiled_probe_step: Callable[..., dict[str, object] | None],
    task_spec: MsgProbeTaskSpec,
    feature_extractor: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    move_batch: Callable[[dict[str, object]], dict[str, object]],
    split: str,
    seed: int,
    peak_ordering: str,
    max_samples: int | None,
    sample_randomly: bool,
    device: torch.device,
) -> dict[str, object]:
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
    if (
        "msg_probe_select_metric" not in config
        and "msg_probe_tune_metric" not in config
    ):
        fingerprint_task = resolve_msg_probe_fingerprint(config)
        return f"msg_probe/test/auc_{fingerprint_task}_mean"
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
    for fingerprint_task in (_MACCS_TASK, _MORGAN_TASK):
        source_key = f"msg_probe/mean/num_{fingerprint_task}_bits"
        if source_key in metrics:
            aliased[f"msg_probe/num_{fingerprint_task}_bits"] = metrics[source_key]
    if "msg_probe/mean/epoch" in metrics:
        aliased["msg_probe_epoch"] = metrics["msg_probe/mean/epoch"]
    return aliased


def _average_metric_dicts(
    metric_dicts: list[dict[str, float]],
) -> dict[str, float]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for metrics in metric_dicts:
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return {key: totals[key] / counts[key] for key in totals}


def _run_repeated_probe(
    *,
    repeat_count: int,
    metric_prefix: str,
    run_once: Callable[
        [int, Callable[[dict[str, float]], None] | None],
        dict[str, float],
    ],
    on_epoch_end: Callable[[dict[str, float]], None] | None = None,
) -> dict[str, float]:
    if repeat_count == 1:
        metrics = dict(run_once(0, on_epoch_end))
        if metrics:
            metrics[f"{metric_prefix}/repeats"] = 1.0
        return metrics

    repeat_metrics: list[dict[str, float]] = []
    repeat_curves: list[list[dict[str, float]]] = []
    for repeat_idx in range(repeat_count):
        log.info("%s repeat %d/%d", metric_prefix, repeat_idx + 1, repeat_count)
        repeat_curve: list[dict[str, float]] = []
        metrics = run_once(repeat_idx, repeat_curve.append)
        if metrics:
            repeat_metrics.append(metrics)
        if repeat_curve:
            repeat_curves.append(repeat_curve)

    averaged_metrics = _average_metric_dicts(repeat_metrics)
    if averaged_metrics:
        averaged_metrics[f"{metric_prefix}/repeats"] = float(repeat_count)
    if on_epoch_end is not None and repeat_curves:
        num_epochs = max(len(curve) for curve in repeat_curves)
        for epoch_idx in range(num_epochs):
            epoch_metrics = _average_metric_dicts(
                [curve[epoch_idx] for curve in repeat_curves if epoch_idx < len(curve)]
            )
            if epoch_metrics:
                on_epoch_end(epoch_metrics)
    return averaged_metrics


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
        fingerprint_task = task_spec.fingerprint_task
        pred = np.concatenate(predictions[fingerprint_task], axis=0)
        target = np.concatenate(targets[fingerprint_task], axis=0)
        positives = target.sum(axis=0)
        valid_metric_mask = (positives > 0) & (positives < target.shape[0])
        if np.count_nonzero(valid_metric_mask) > 0:
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
            ranks = np.arange(
                1, target_descending.shape[0] + 1, dtype=np.float64
            )[:, None]
            average_precision_values = (
                (target_descending * true_positives_at_rank / ranks).sum(axis=0)
                / valid_positives
            )
        else:
            auc_values = np.asarray([], dtype=np.float64)
            average_precision_values = np.asarray([], dtype=np.float64)
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
        metrics[f"{prefix}/num_{fingerprint_task}_auc_bits"] = float(len(auc_values))
        metrics[f"{prefix}/num_{fingerprint_task}_average_precision_bits"] = float(
            len(average_precision_values)
        )
        metrics[f"{prefix}/num_{fingerprint_task}_recall_bits"] = float(
            len(recall_values)
        )
        metrics[f"{prefix}/num_{fingerprint_task}_precision_bits"] = float(
            len(precision_values)
        )
        metrics[f"{prefix}/auc_{fingerprint_task}_mean"] = (
            float(np.mean(auc_values)) if len(auc_values) else float("nan")
        )
        metrics[f"{prefix}/average_precision_{fingerprint_task}_mean"] = (
            float(np.mean(average_precision_values))
            if len(average_precision_values)
            else float("nan")
        )
        metrics[f"{prefix}/recall_{fingerprint_task}_mean"] = (
            float(np.mean(recall_values)) if len(recall_values) else float("nan")
        )
        metrics[f"{prefix}/precision_{fingerprint_task}_mean"] = (
            float(np.mean(precision_values)) if len(precision_values) else float("nan")
        )
    metrics[f"{prefix}/r2_mean"] = float(np.mean(regression_r2_values))
    metrics[f"{prefix}/mae_mean"] = float(np.mean(regression_mae_values))
    metrics[f"{prefix}/r2_mean_wo_num_rings"] = metrics[f"{prefix}/r2_mean"]
    metrics[f"{prefix}/mae_mean_wo_num_rings"] = metrics[f"{prefix}/mae_mean"]
    return metrics


def _run_msg_probe_once(
    *,
    config: config_dict.ConfigDict,
    model: PeakSetSIGReg,
    device: torch.device,
    on_epoch_end: Callable[[dict[str, float]], None] | None = None,
    repeat_index: int = 0,
    plot_dir: Path | None = None,
    plot_step: int | None = None,
) -> dict[str, float]:
    num_probe_epochs = int(config.get("msg_probe_num_epochs", 5))
    probe_lr = float(config.get("msg_probe_learning_rate", 1e-3))
    probe_weight_decay = float(config.get("msg_probe_weight_decay", 1e-2))
    probe_warmup_steps = int(config.get("msg_probe_warmup_steps", 100))
    early_stopping = bool(config.get("msg_probe_early_stopping", False))
    early_stopping_patience = int(config.get("msg_probe_early_stopping_patience", 10))
    early_stopping_min_delta = float(
        config.get("msg_probe_early_stopping_min_delta", 0.0)
    )
    early_stopping_min_epochs = int(
        config.get("msg_probe_early_stopping_min_epochs", 1)
    )
    max_train_samples, max_val_samples, max_test_samples, randomize_test_subset = (
        resolve_msg_probe_sample_limits(config)
    )
    peak_ordering = str(config.get("peak_ordering", "intensity"))
    fingerprint_task = resolve_msg_probe_fingerprint(config)
    probe_data = MassSpecProbeData.from_config(config)

    @torch.no_grad()
    def feature_extractor(
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        embeddings = model.encoder(
            batch["peak_mz"],
            batch["peak_intensity"],
            valid_mask=batch["peak_valid_mask"],
            precursor_mz=batch.get("precursor_mz", None),
        )
        peak_embeddings, _ = model.encoder.split_peak_and_cls(embeddings)
        return peak_embeddings

    seed_offset = 100_000 * int(repeat_index)
    train_seed_base = int(config.seed) + 1_100_000 + seed_offset
    test_seed_base = int(config.seed) + 1_200_000 + seed_offset
    train_targets = _collect_split_targets(
        probe_data=probe_data,
        split="massspec_train",
        peak_ordering=peak_ordering,
        seed=train_seed_base,
        max_samples=max_train_samples,
        fingerprint_task=fingerprint_task,
    )
    val_targets = _collect_split_targets(
        probe_data=probe_data,
        split="massspec_val",
        peak_ordering=peak_ordering,
        seed=train_seed_base + 10_000,
        max_samples=max_val_samples,
        sample_randomly=True,
        fingerprint_task=fingerprint_task,
    )
    selection_targets = val_targets if early_stopping else _collect_split_targets(
        probe_data=probe_data,
        split="massspec_test",
        peak_ordering=peak_ordering,
        seed=test_seed_base,
        max_samples=max_test_samples,
        sample_randomly=randomize_test_subset,
        fingerprint_task=fingerprint_task,
    )
    task_spec = _build_task_spec(
        train_targets=train_targets,
        test_targets=selection_targets,
        num_rings_classes=_collect_num_rings_classes(probe_data),
        fingerprint_task=fingerprint_task,
    )
    variants = msg_probe_variants_from_config(config)
    was_training = model.training
    model.eval()
    probes = {
        variant: _build_msg_sequence_probe(
            variant,
            config=config,
            task_spec=task_spec,
            covariance_pooler=(
                model.covariance_pooler
                if variant == "covariance"
                and bool(config.get("train_covariance_pooling", False))
                else None
            ),
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
    best_state_by_variant: dict[str, dict[str, torch.Tensor]] = {}
    epochs_without_improvement = {variant: 0 for variant in variants}
    for epoch_idx in range(num_probe_epochs):
        for probe in probes.values():
            probe.train()
        train_states = {
            variant: _new_epoch_state(task_spec)
            for variant in variants
        }
        train_iterator = iter_massspec_probe(
            probe_data,
            "massspec_train",
            seed=train_seed_base + epoch_idx,
            peak_ordering=peak_ordering,
            drop_remainder=False,
            max_samples=max_train_samples,
        )
        for batch in train_iterator:
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
        val_states = {variant: _new_epoch_state(task_spec) for variant in variants}
        if early_stopping:
            val_states = _evaluate_sequence_probe_split(
                probe_data=probe_data,
                probes=probes,
                task_spec=task_spec,
                feature_extractor=feature_extractor,
                move_batch=move_batch,
                split="massspec_val",
                seed=train_seed_base + 10_000,
                peak_ordering=peak_ordering,
                max_samples=max_val_samples,
                sample_randomly=True,
                device=device,
            )
        test_states = {variant: _new_epoch_state(task_spec) for variant in variants}
        if not early_stopping:
            test_states = _evaluate_sequence_probe_split(
                probe_data=probe_data,
                probes=probes,
                task_spec=task_spec,
                feature_extractor=feature_extractor,
                move_batch=move_batch,
                split="massspec_test",
                seed=test_seed_base,
                peak_ordering=peak_ordering,
                max_samples=max_test_samples,
                sample_randomly=randomize_test_subset,
                device=device,
            )
        epoch_metrics: dict[str, float] = {}
        for variant in variants:
            variant_prefix = f"msg_probe/{variant}"
            variant_metrics = {
                **_score_epoch_state(
                    prefix=f"{variant_prefix}/train",
                    epoch_state=train_states[variant],
                    task_spec=task_spec,
                ),
                **(
                    _score_epoch_state(
                        prefix=f"{variant_prefix}/val",
                        epoch_state=val_states[variant],
                        task_spec=task_spec,
                    )
                    if early_stopping
                    else {}
                ),
                **(
                    {}
                    if early_stopping
                    else _score_epoch_state(
                        prefix=f"{variant_prefix}/test",
                        epoch_state=test_states[variant],
                        task_spec=task_spec,
                    )
                ),
                f"{variant_prefix}/num_{fingerprint_task}_bits": float(
                    task_spec.maccs_bits
                ),
                f"{variant_prefix}/epoch": float(epoch_idx + 1),
            }
            epoch_metrics.update(variant_metrics)
            variant_select_metric = _msg_probe_variant_metric_key(variant, select_metric)
            if early_stopping:
                variant_select_metric = variant_select_metric.replace(
                    "/test/",
                    "/val/",
                )
            current_value = float(variant_metrics[variant_select_metric])
            previous_best = best_metric_values[variant]
            is_better = (
                current_value > previous_best + early_stopping_min_delta
                if higher_is_better
                else current_value < previous_best - early_stopping_min_delta
            )
            if is_better:
                best_metric_values[variant] = current_value
                best_metrics_by_variant[variant] = dict(variant_metrics)
                best_state_by_variant[variant] = copy.deepcopy(
                    probes[variant].state_dict()
                )
                epochs_without_improvement[variant] = 0
            else:
                epochs_without_improvement[variant] += 1
            if early_stopping:
                log.info(
                    "MSG probe [%s] epoch %d/%d train_samples=%d val_r2_mean_wo_num_rings=%.4f val_mae_num_rings=%.4f val_auc_%s_mean=%.4f val_average_precision_%s_mean=%.4f val_recall_%s_mean=%.4f val_precision_%s_mean=%.4f %s_bits=%d",
                    variant,
                    epoch_idx + 1,
                    num_probe_epochs,
                    int(variant_metrics[f"{variant_prefix}/train/samples"]),
                    variant_metrics[f"{variant_prefix}/val/r2_mean_wo_num_rings"],
                    variant_metrics[f"{variant_prefix}/val/mae_num_rings"],
                    fingerprint_task,
                    variant_metrics[f"{variant_prefix}/val/auc_{fingerprint_task}_mean"],
                    fingerprint_task,
                    variant_metrics[
                        f"{variant_prefix}/val/average_precision_{fingerprint_task}_mean"
                    ],
                    fingerprint_task,
                    variant_metrics[f"{variant_prefix}/val/recall_{fingerprint_task}_mean"],
                    fingerprint_task,
                    variant_metrics[f"{variant_prefix}/val/precision_{fingerprint_task}_mean"],
                    fingerprint_task,
                    int(variant_metrics[f"{variant_prefix}/num_{fingerprint_task}_bits"]),
                )
            else:
                log.info(
                    "MSG probe [%s] epoch %d/%d train_samples=%d test_r2_mean_wo_num_rings=%.4f test_mae_num_rings=%.4f test_auc_%s_mean=%.4f test_average_precision_%s_mean=%.4f test_recall_%s_mean=%.4f test_precision_%s_mean=%.4f %s_bits=%d",
                    variant,
                    epoch_idx + 1,
                    num_probe_epochs,
                    int(variant_metrics[f"{variant_prefix}/train/samples"]),
                    variant_metrics[f"{variant_prefix}/test/r2_mean_wo_num_rings"],
                    variant_metrics[f"{variant_prefix}/test/mae_num_rings"],
                    fingerprint_task,
                    variant_metrics[f"{variant_prefix}/test/auc_{fingerprint_task}_mean"],
                    fingerprint_task,
                    variant_metrics[
                        f"{variant_prefix}/test/average_precision_{fingerprint_task}_mean"
                    ],
                    fingerprint_task,
                    variant_metrics[
                        f"{variant_prefix}/test/recall_{fingerprint_task}_mean"
                    ],
                    fingerprint_task,
                    variant_metrics[
                        f"{variant_prefix}/test/precision_{fingerprint_task}_mean"
                    ],
                    fingerprint_task,
                    int(variant_metrics[f"{variant_prefix}/num_{fingerprint_task}_bits"]),
                )
        epoch_metrics = _with_mean_probe_aliases(epoch_metrics)
        if on_epoch_end is not None:
            on_epoch_end(epoch_metrics)
        if (
            early_stopping
            and epoch_idx + 1 >= early_stopping_min_epochs
            and all(
                epochs_without_improvement[variant] >= early_stopping_patience
                for variant in variants
            )
        ):
            log.info(
                "MSG probe early stopping at epoch %d/%d after %d epochs without validation improvement",
                epoch_idx + 1,
                num_probe_epochs,
                early_stopping_patience,
            )
            break
    for variant in variants:
        if variant in best_state_by_variant:
            probes[variant].load_state_dict(best_state_by_variant[variant])
    final_test_metrics_by_variant: dict[str, dict[str, float]] = {}
    if early_stopping:
        selected_probes = {
            variant: probes[variant]
            for variant in variants
            if variant in best_state_by_variant
        }
        final_test_states = _evaluate_sequence_probe_split(
            probe_data=probe_data,
            probes=selected_probes,
            task_spec=task_spec,
            feature_extractor=feature_extractor,
            move_batch=move_batch,
            split="massspec_test",
            seed=test_seed_base,
            peak_ordering=peak_ordering,
            max_samples=max_test_samples,
            sample_randomly=randomize_test_subset,
            device=device,
        )
        final_test_metrics_by_variant = {
            variant: _score_epoch_state(
                prefix=f"msg_probe/{variant}/test",
                epoch_state=state,
                task_spec=task_spec,
            )
            for variant, state in final_test_states.items()
        }
    best_metrics: dict[str, float] = {}
    for variant in variants:
        variant_metrics = dict(best_metrics_by_variant.get(variant, {}))
        if not variant_metrics:
            continue
        variant_metrics.update(final_test_metrics_by_variant.get(variant, {}))
        best_metrics.update(variant_metrics)
        variant_prefix = f"msg_probe/{variant}"
        variant_select_metric = _msg_probe_variant_metric_key(variant, select_metric)
        if early_stopping:
            variant_select_metric = variant_select_metric.replace("/test/", "/val/")
        log.info(
            "MSG probe [%s] best epoch %d: %s=%.4f test_r2_mean_wo_num_rings=%.4f test_mae_num_rings=%.4f test_auc_%s_mean=%.4f test_average_precision_%s_mean=%.4f test_recall_%s_mean=%.4f test_precision_%s_mean=%.4f",
            variant,
            int(variant_metrics[f"{variant_prefix}/epoch"]),
            variant_select_metric,
            variant_metrics[variant_select_metric],
            variant_metrics[f"{variant_prefix}/test/r2_mean_wo_num_rings"],
            variant_metrics[f"{variant_prefix}/test/mae_num_rings"],
            fingerprint_task,
            variant_metrics[f"{variant_prefix}/test/auc_{fingerprint_task}_mean"],
            fingerprint_task,
            variant_metrics[
                f"{variant_prefix}/test/average_precision_{fingerprint_task}_mean"
            ],
            fingerprint_task,
            variant_metrics[f"{variant_prefix}/test/recall_{fingerprint_task}_mean"],
            fingerprint_task,
            variant_metrics[f"{variant_prefix}/test/precision_{fingerprint_task}_mean"],
        )
    best_metrics.update(
        _run_covariance_morgan_pairwise_alignment(
            config=config,
            probe_data=probe_data,
            model=model,
            feature_extractor=feature_extractor,
            move_batch=move_batch,
            device=device,
            split="massspec_test",
            peak_ordering=peak_ordering,
            seed=test_seed_base + 50_000,
            max_samples=max_test_samples,
            sample_randomly=randomize_test_subset,
            plot_dir=plot_dir,
            plot_step=plot_step,
            repeat_index=repeat_index,
        )
    )
    if was_training:
        model.train()
    return _with_mean_probe_aliases(best_metrics)


def run_msg_probe(
    *,
    config: config_dict.ConfigDict,
    model: PeakSetSIGReg,
    device: torch.device,
    on_epoch_end: Callable[[dict[str, float]], None] | None = None,
    plot_dir: Path | None = None,
    plot_step: int | None = None,
) -> dict[str, float]:
    def run_once(
        repeat_index: int,
        repeat_on_epoch_end: Callable[[dict[str, float]], None] | None,
    ) -> dict[str, float]:
        kwargs = {
            "config": config,
            "model": model,
            "device": device,
            "on_epoch_end": repeat_on_epoch_end,
            "repeat_index": repeat_index,
        }
        if plot_dir is not None:
            kwargs["plot_dir"] = plot_dir
            kwargs["plot_step"] = plot_step
        return _run_msg_probe_once(**kwargs)

    metrics = _run_repeated_probe(
        repeat_count=resolve_msg_probe_num_repeats(config),
        metric_prefix="msg_probe",
        run_once=run_once,
        on_epoch_end=on_epoch_end,
    )
    return _with_mean_probe_aliases(metrics)


def _run_dreams_probe_once(
    *,
    config: config_dict.ConfigDict,
    device: torch.device,
    on_epoch_end: Callable[[dict[str, float]], None] | None = None,
    repeat_index: int = 0,
) -> dict[str, float]:
    num_probe_epochs = int(config.get("msg_probe_num_epochs", 5))
    probe_lr = float(config.get("msg_probe_learning_rate", 1e-3))
    probe_weight_decay = float(config.get("msg_probe_weight_decay", 1e-2))
    probe_warmup_steps = int(config.get("msg_probe_warmup_steps", 100))
    early_stopping = bool(config.get("msg_probe_early_stopping", False))
    early_stopping_patience = int(config.get("msg_probe_early_stopping_patience", 10))
    early_stopping_min_delta = float(
        config.get("msg_probe_early_stopping_min_delta", 0.0)
    )
    early_stopping_min_epochs = int(
        config.get("msg_probe_early_stopping_min_epochs", 1)
    )
    max_train_samples, max_val_samples, max_test_samples, randomize_test_subset = (
        resolve_msg_probe_sample_limits(config)
    )
    peak_ordering = str(config.get("peak_ordering", "intensity"))
    fingerprint_task = resolve_msg_probe_fingerprint(config)
    probe_data = MassSpecProbeData.from_config(config)

    dreams_dim = probe_data.dreams_dim
    if dreams_dim == 0:
        log.warning("No DreaMS embeddings in probe data; skipping Dreams probe")
        return {}

    seed_offset = 100_000 * int(repeat_index)
    train_seed_base = int(config.seed) + 1_100_000 + seed_offset
    test_seed_base = int(config.seed) + 1_200_000 + seed_offset
    train_targets = _collect_split_targets(
        probe_data=probe_data,
        split="massspec_train",
        peak_ordering=peak_ordering,
        seed=train_seed_base,
        max_samples=max_train_samples,
        fingerprint_task=fingerprint_task,
    )
    val_targets = _collect_split_targets(
        probe_data=probe_data,
        split="massspec_val",
        peak_ordering=peak_ordering,
        seed=train_seed_base + 10_000,
        max_samples=max_val_samples,
        sample_randomly=True,
        fingerprint_task=fingerprint_task,
    )
    selection_targets = val_targets if early_stopping else _collect_split_targets(
        probe_data=probe_data,
        split="massspec_test",
        peak_ordering=peak_ordering,
        seed=test_seed_base,
        max_samples=max_test_samples,
        sample_randomly=randomize_test_subset,
        fingerprint_task=fingerprint_task,
    )
    task_spec = _build_task_spec(
        train_targets=train_targets,
        test_targets=selection_targets,
        num_rings_classes=_collect_num_rings_classes(probe_data),
        fingerprint_task=fingerprint_task,
    )

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
    probe_select_metric = resolve_msg_probe_select_metric(config)
    probe_select_metric = probe_select_metric.replace("msg_probe/", "dreams_probe/")
    if early_stopping:
        probe_select_metric = probe_select_metric.replace("/test/", "/val/")
    else:
        probe_select_metric = probe_select_metric.replace("/mean/", "/")
    higher_is_better = msg_probe_metric_higher_is_better(probe_select_metric)

    best_metrics: dict[str, float] = {}
    best_metric_value = -float("inf") if higher_is_better else float("inf")
    best_state: dict[str, torch.Tensor] = {}
    epochs_without_improvement = 0
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
        if early_stopping:
            test_state = _new_epoch_state(task_spec)
        else:
            test_state = _evaluate_linear_probe_split(
                probe_data=probe_data,
                probe=probe,
                compiled_probe_step=compiled_probe_step,
                task_spec=task_spec,
                feature_extractor=feature_extractor,
                move_batch=move_batch,
                split="massspec_test",
                seed=test_seed_base,
                peak_ordering=peak_ordering,
                max_samples=max_test_samples,
                sample_randomly=randomize_test_subset,
                device=device,
            )
        val_state = _evaluate_linear_probe_split(
            probe_data=probe_data,
            probe=probe,
            compiled_probe_step=compiled_probe_step,
            task_spec=task_spec,
            feature_extractor=feature_extractor,
            move_batch=move_batch,
            split="massspec_val",
            seed=train_seed_base + 10_000,
            peak_ordering=peak_ordering,
            max_samples=max_val_samples,
            sample_randomly=True,
            device=device,
        )
        epoch_metrics = {
            **_score_epoch_state(
                prefix="dreams_probe/train", epoch_state=train_state, task_spec=task_spec
            ),
            **_score_epoch_state(
                prefix="dreams_probe/val", epoch_state=val_state, task_spec=task_spec
            ),
            **(
                {}
                if early_stopping
                else _score_epoch_state(
                    prefix="dreams_probe/test",
                    epoch_state=test_state,
                    task_spec=task_spec,
                )
            ),
            f"dreams_probe/num_{fingerprint_task}_bits": float(task_spec.maccs_bits),
            "dreams_probe_epoch": float(epoch_idx + 1),
        }
        current_value = float(epoch_metrics[probe_select_metric])
        is_better = (
            current_value > best_metric_value + early_stopping_min_delta
            if higher_is_better
            else current_value < best_metric_value - early_stopping_min_delta
        )
        if is_better:
            best_metric_value = current_value
            best_metrics = dict(epoch_metrics)
            best_state = copy.deepcopy(probe.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if early_stopping:
            log.info(
                "DreaMS probe epoch %d/%d train_samples=%d val_r2_mean_wo_num_rings=%.4f val_mae_num_rings=%.4f val_auc_%s_mean=%.4f val_average_precision_%s_mean=%.4f val_recall_%s_mean=%.4f val_precision_%s_mean=%.4f %s_bits=%d",
                epoch_idx + 1,
                num_probe_epochs,
                int(epoch_metrics["dreams_probe/train/samples"]),
                epoch_metrics["dreams_probe/val/r2_mean_wo_num_rings"],
                epoch_metrics["dreams_probe/val/mae_num_rings"],
                fingerprint_task,
                epoch_metrics[f"dreams_probe/val/auc_{fingerprint_task}_mean"],
                fingerprint_task,
                epoch_metrics[
                    f"dreams_probe/val/average_precision_{fingerprint_task}_mean"
                ],
                fingerprint_task,
                epoch_metrics[f"dreams_probe/val/recall_{fingerprint_task}_mean"],
                fingerprint_task,
                epoch_metrics[f"dreams_probe/val/precision_{fingerprint_task}_mean"],
                fingerprint_task,
                int(epoch_metrics[f"dreams_probe/num_{fingerprint_task}_bits"]),
            )
        else:
            log.info(
                "DreaMS probe epoch %d/%d train_samples=%d val_auc_%s_mean=%.4f test_r2_mean_wo_num_rings=%.4f test_mae_num_rings=%.4f test_auc_%s_mean=%.4f test_average_precision_%s_mean=%.4f test_recall_%s_mean=%.4f test_precision_%s_mean=%.4f %s_bits=%d",
                epoch_idx + 1,
                num_probe_epochs,
                int(epoch_metrics["dreams_probe/train/samples"]),
                fingerprint_task,
                float(
                    epoch_metrics.get(
                        f"dreams_probe/val/auc_{fingerprint_task}_mean",
                        float("nan"),
                    )
                ),
                epoch_metrics["dreams_probe/test/r2_mean_wo_num_rings"],
                epoch_metrics["dreams_probe/test/mae_num_rings"],
                fingerprint_task,
                epoch_metrics[f"dreams_probe/test/auc_{fingerprint_task}_mean"],
                fingerprint_task,
                epoch_metrics[
                    f"dreams_probe/test/average_precision_{fingerprint_task}_mean"
                ],
                fingerprint_task,
                epoch_metrics[f"dreams_probe/test/recall_{fingerprint_task}_mean"],
                fingerprint_task,
                epoch_metrics[f"dreams_probe/test/precision_{fingerprint_task}_mean"],
                fingerprint_task,
                int(epoch_metrics[f"dreams_probe/num_{fingerprint_task}_bits"]),
            )
        if on_epoch_end is not None:
            on_epoch_end(epoch_metrics)
        if (
            early_stopping
            and epoch_idx + 1 >= early_stopping_min_epochs
            and epochs_without_improvement >= early_stopping_patience
        ):
            log.info(
                "DreaMS probe early stopping at epoch %d/%d after %d epochs without validation improvement",
                epoch_idx + 1,
                num_probe_epochs,
                early_stopping_patience,
            )
            break
    if best_metrics:
        if early_stopping:
            probe.load_state_dict(best_state)
            test_state = _evaluate_linear_probe_split(
                probe_data=probe_data,
                probe=probe,
                compiled_probe_step=compiled_probe_step,
                task_spec=task_spec,
                feature_extractor=feature_extractor,
                move_batch=move_batch,
                split="massspec_test",
                seed=test_seed_base,
                peak_ordering=peak_ordering,
                max_samples=max_test_samples,
                sample_randomly=randomize_test_subset,
                device=device,
            )
            best_metrics.update(
                _score_epoch_state(
                    prefix="dreams_probe/test",
                    epoch_state=test_state,
                    task_spec=task_spec,
                )
            )
        log.info(
            "DreaMS probe best epoch %d: %s=%.4f test_r2_mean_wo_num_rings=%.4f test_mae_num_rings=%.4f test_auc_%s_mean=%.4f test_average_precision_%s_mean=%.4f test_recall_%s_mean=%.4f test_precision_%s_mean=%.4f",
            int(best_metrics["dreams_probe_epoch"]),
            probe_select_metric,
            best_metrics[probe_select_metric],
            best_metrics["dreams_probe/test/r2_mean_wo_num_rings"],
            best_metrics["dreams_probe/test/mae_num_rings"],
            fingerprint_task,
            best_metrics[f"dreams_probe/test/auc_{fingerprint_task}_mean"],
            fingerprint_task,
            best_metrics[
                f"dreams_probe/test/average_precision_{fingerprint_task}_mean"
            ],
            fingerprint_task,
            best_metrics[f"dreams_probe/test/recall_{fingerprint_task}_mean"],
            fingerprint_task,
            best_metrics[f"dreams_probe/test/precision_{fingerprint_task}_mean"],
        )
    return best_metrics


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
    )
