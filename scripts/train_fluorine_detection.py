from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, NamedTuple, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from ml_collections import config_dict
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset, Subset

from spectra_learning.config.loading import load_config
from spectra_learning.data.spectra import (
    DEFAULT_MAX_PRECURSOR_MZ,
    DEFAULT_MIN_PEAK_INTENSITY,
    NUM_PEAKS_INPUT,
    preprocess_peak_batch_torch,
)
from spectra_learning.models.factory import build_model_from_config
from spectra_learning.models.model import PeakSetSIGReg
from spectra_learning.models.pooling import CovariancePool
from spectra_learning.probes.massspec.data import _normalize_spectra_intensity
from spectra_learning.training.checkpointing import load_pretrained_weights


log = logging.getLogger(__name__)

HF_REPO_ID = "cjim8889/hr_msms_nist_fluorine_detection"
HF_SUBDIR = "fine_tuned"
CACHE_METADATA_VERSION = 1
EMBEDDING_CACHE_METADATA_VERSION = 1
SPLITS = ("train", "val", "test")


def _config_get(config: Any, key: str, default: Any) -> Any:
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


class FluorineData(NamedTuple):
    metadata: dict[str, Any]
    root: Path
    batch_size: int
    num_peaks: int
    max_precursor_mz: float
    min_peak_intensity: float
    peak_drop_min_intensity: float
    peak_ordering: str
    precursor_peak_exclusion_window_da: float


class TrialParams(NamedTuple):
    hidden_dim: int
    learning_rate: float
    weight_decay: float
    dropout: float


class TrialResult(NamedTuple):
    params: TrialParams
    best_epoch: int
    best_val: dict[str, float]
    classifier_state: dict[str, torch.Tensor]
    pooler_state: dict[str, torch.Tensor] | None


class EmbeddingData(NamedTuple):
    metadata: dict[str, Any]
    root: Path
    batch_size: int


def _split_files(metadata: dict[str, Any], split: str) -> list[Path]:
    return [Path(path) for path in metadata[f"{split}_files"]]


def _cache_valid(cache_dir: Path, repo_id: str, revision: str, subdir: str) -> dict[str, Any] | None:
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text())
    expected = {
        "metadata_version": CACHE_METADATA_VERSION,
        "repo_id": repo_id,
        "revision": revision,
        "subdir": subdir,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            return None
    for split in SPLITS:
        for shard_dir in _split_files(metadata, split):
            for filename in (
                "spectra.npy",
                "precursor_mz_raw.npy",
                "dreams_embedding.npy",
                "label.npy",
            ):
                if not (cache_dir / shard_dir / filename).exists():
                    return None
    return metadata


def _spectra_from_peak_lists(
    mz_lists: list[list[float]],
    intensity_lists: list[list[float]],
) -> np.ndarray:
    spectra = np.zeros((len(mz_lists), 2, NUM_PEAKS_INPUT), dtype=np.float32)
    for i, (mz, intensity) in enumerate(zip(mz_lists, intensity_lists, strict=True)):
        n = min(len(mz), NUM_PEAKS_INPUT)
        spectra[i, 0, :n] = np.asarray(mz[:n], dtype=np.float32)
        spectra[i, 1, :n] = np.asarray(intensity[:n], dtype=np.float32)
    return _normalize_spectra_intensity(spectra)


def _write_payload_shard(
    buffers: dict[str, list[np.ndarray]],
    output_dir: Path,
    shard_name: str,
) -> int:
    shard_dir = output_dir / shard_name
    shard_dir.mkdir(parents=True, exist_ok=True)
    length = len(buffers["label"][0]) if len(buffers["label"]) == 1 else sum(
        len(value) for value in buffers["label"]
    )
    for key, values in buffers.items():
        np.save(shard_dir / f"{key}.npy", np.concatenate(values, axis=0))
    return length


def _payload_from_parquet_rows(rows: dict[str, list[Any]]) -> dict[str, np.ndarray]:
    return {
        "spectra": _spectra_from_peak_lists(
            rows["spectrum_mz"],
            rows["spectrum_intensity"],
        ),
        "precursor_mz_raw": np.asarray(rows["precursor_mz"], dtype=np.float32),
        "dreams_embedding": np.asarray(rows["dreams_embedding"], dtype=np.float32),
        "label": np.asarray(rows["has_fluorine"], dtype=np.float32),
    }


def _write_split_cache_from_parquet(
    path: Path,
    output_dir: Path,
    split: str,
    num_shards: int,
    parquet_batch_size: int,
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    n = int(pf.metadata.num_rows)
    shard_count = max(1, min(num_shards, n))
    shard_size = int(np.ceil(n / shard_count))
    shard_names, shard_lengths = [], []
    buffers: dict[str, list[np.ndarray]] = {
        "spectra": [],
        "precursor_mz_raw": [],
        "dreams_embedding": [],
        "label": [],
    }
    buffer_len = 0
    positive = 0
    dreams_dim = 0
    shard_idx = 0

    def flush() -> None:
        nonlocal buffers, buffer_len, shard_idx
        shard_name = f"{split}/shard-{shard_idx:05d}-of-{shard_count:05d}"
        shard_lengths.append(_write_payload_shard(buffers, output_dir, shard_name))
        shard_names.append(shard_name)
        buffers = {key: [] for key in buffers}
        buffer_len = 0
        shard_idx += 1

    columns = [
        "dreams_embedding",
        "spectrum_mz",
        "spectrum_intensity",
        "precursor_mz",
        "has_fluorine",
    ]
    for batch in pf.iter_batches(batch_size=parquet_batch_size, columns=columns):
        payload = _payload_from_parquet_rows(batch.to_pydict())
        positive += int(payload["label"].sum())
        dreams_dim = int(payload["dreams_embedding"].shape[1])
        offset = 0
        while offset < len(payload["label"]):
            take = min(shard_size - buffer_len, len(payload["label"]) - offset)
            for key, value in payload.items():
                buffers[key].append(value[offset : offset + take])
            buffer_len += take
            offset += take
            if buffer_len == shard_size:
                flush()
    if buffer_len:
        flush()
    return {
        "files": shard_names,
        "lengths": shard_lengths,
        "size": n,
        "positive": positive,
        "dreams_dim": dreams_dim,
    }


def ensure_fluorine_cache(
    cache_dir: Path,
    *,
    repo_id: str,
    revision: str,
    subdir: str,
    num_shards: int,
    parquet_batch_size: int,
) -> dict[str, Any]:
    cached = _cache_valid(cache_dir, repo_id, revision, subdir)
    if cached is not None:
        return cached

    source_dir = cache_dir / "source"
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=source_dir,
        allow_patterns=[
            f"{subdir}/metadata.json",
            f"{subdir}/train.parquet",
            f"{subdir}/val.parquet",
            f"{subdir}/test.parquet",
        ],
    )
    metadata: dict[str, Any] = {
        "metadata_version": CACHE_METADATA_VERSION,
        "repo_id": repo_id,
        "revision": revision,
        "subdir": subdir,
    }
    for split in SPLITS:
        split_metadata = _write_split_cache_from_parquet(
            source_dir / subdir / f"{split}.parquet",
            cache_dir,
            split,
            max(1, num_shards if split == "train" else num_shards // 4),
            parquet_batch_size,
        )
        metadata[f"{split}_files"] = split_metadata["files"]
        metadata[f"{split}_lengths"] = split_metadata["lengths"]
        metadata[f"{split}_size"] = split_metadata["size"]
        metadata[f"{split}_positive"] = split_metadata["positive"]
        metadata["dreams_dim"] = split_metadata["dreams_dim"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


class _FluorineShardDataset(Dataset):
    def __init__(self, shard_entries: list[dict[str, Any]]) -> None:
        self._shards = [
            {"dir": Path(entry["dir"]), "length": int(entry["length"])}
            for entry in shard_entries
        ]
        lengths = np.asarray([entry["length"] for entry in self._shards], dtype=np.int64)
        self._starts = np.zeros(len(lengths) + 1, dtype=np.int64)
        np.cumsum(lengths, out=self._starts[1:])
        self._arrays: list[dict[str, np.ndarray]] | None = None

    def __len__(self) -> int:
        return int(self._starts[-1])

    def _ensure_arrays(self) -> list[dict[str, np.ndarray]]:
        if self._arrays is not None:
            return self._arrays
        arrays_by_shard: list[dict[str, np.ndarray]] = []
        for entry in self._shards:
            shard_dir = entry["dir"]
            arrays_by_shard.append(
                {
                    "spectra": np.load(shard_dir / "spectra.npy", mmap_mode="r"),
                    "precursor_mz_raw": np.load(
                        shard_dir / "precursor_mz_raw.npy",
                        mmap_mode="r",
                    ),
                    "dreams_embedding": np.load(
                        shard_dir / "dreams_embedding.npy",
                        mmap_mode="r",
                    ),
                    "label": np.load(shard_dir / "label.npy", mmap_mode="r"),
                }
            )
        self._arrays = arrays_by_shard
        return arrays_by_shard

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        arrays_by_shard = self._ensure_arrays()
        index = index
        shard_idx = int(np.searchsorted(self._starts, index, side="right") - 1)
        local_idx = index - int(self._starts[shard_idx])
        arrays = arrays_by_shard[shard_idx]
        return {
            "spectra": torch.from_numpy(arrays["spectra"][local_idx].copy()),
            "precursor_mz_raw": torch.tensor(
                float(arrays["precursor_mz_raw"][local_idx]),
                dtype=torch.float32,
            ),
            "dreams_embedding": torch.from_numpy(
                arrays["dreams_embedding"][local_idx].copy()
            ),
            "label": torch.tensor(float(arrays["label"][local_idx]), dtype=torch.float32),
        }


class _EmbeddingShardDataset(Dataset):
    def __init__(self, shard_entries: list[dict[str, Any]]) -> None:
        self._shards = [
            {"dir": Path(entry["dir"]), "length": int(entry["length"])}
            for entry in shard_entries
        ]
        lengths = np.asarray([entry["length"] for entry in self._shards], dtype=np.int64)
        self._starts = np.zeros(len(lengths) + 1, dtype=np.int64)
        np.cumsum(lengths, out=self._starts[1:])
        self._arrays: list[dict[str, np.ndarray]] | None = None

    def __len__(self) -> int:
        return int(self._starts[-1])

    def _ensure_arrays(self) -> list[dict[str, np.ndarray]]:
        if self._arrays is not None:
            return self._arrays
        arrays_by_shard: list[dict[str, np.ndarray]] = []
        for entry in self._shards:
            shard_dir = entry["dir"]
            arrays_by_shard.append(
                {
                    "peak_embeddings": np.load(
                        shard_dir / "peak_embeddings.npy",
                        mmap_mode="r",
                    ),
                    "peak_valid_mask": np.load(
                        shard_dir / "peak_valid_mask.npy",
                        mmap_mode="r",
                    ),
                    "label": np.load(shard_dir / "label.npy", mmap_mode="r"),
                }
            )
        self._arrays = arrays_by_shard
        return arrays_by_shard

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        arrays_by_shard = self._ensure_arrays()
        index = index
        shard_idx = int(np.searchsorted(self._starts, index, side="right") - 1)
        local_idx = index - int(self._starts[shard_idx])
        arrays = arrays_by_shard[shard_idx]
        return {
            "peak_embeddings": torch.from_numpy(
                arrays["peak_embeddings"][local_idx].copy()
            ),
            "peak_valid_mask": torch.from_numpy(
                arrays["peak_valid_mask"][local_idx].copy()
            ).to(torch.bool),
            "label": torch.tensor(float(arrays["label"][local_idx]), dtype=torch.float32),
        }


class _EmbeddingCollator:
    def __call__(self, samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {
            "peak_embeddings": torch.stack(
                [sample["peak_embeddings"] for sample in samples],
                dim=0,
            ),
            "peak_valid_mask": torch.stack(
                [sample["peak_valid_mask"] for sample in samples],
                dim=0,
            ).to(torch.bool),
            "label": torch.stack([sample["label"] for sample in samples]).to(
                torch.float32
            ),
        }


class _FastEmbeddingLoader:
    def __init__(
        self,
        *,
        shard_entries: list[dict[str, Any]],
        batch_size: int,
        shuffle: bool,
        seed: int,
        max_samples: int | None,
    ) -> None:
        self._shards = [
            {"dir": Path(entry["dir"]), "length": int(entry["length"])}
            for entry in shard_entries
        ]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.max_samples = max_samples
        self._epoch = 0

    def __len__(self) -> int:
        n = sum(entry["length"] for entry in self._shards)
        if self.max_samples is not None:
            n = min(n, self.max_samples)
        return int(np.ceil(n / self.batch_size))

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        shard_order = np.arange(len(self._shards))
        if self.shuffle:
            rng.shuffle(shard_order)
        remaining = self.max_samples
        for shard_idx in shard_order:
            entry = self._shards[int(shard_idx)]
            shard_dir = entry["dir"]
            peak_embeddings = np.load(shard_dir / "peak_embeddings.npy", mmap_mode="r")
            peak_valid_mask = np.load(shard_dir / "peak_valid_mask.npy", mmap_mode="r")
            labels = np.load(shard_dir / "label.npy", mmap_mode="r")
            order = np.arange(entry["length"])
            if self.shuffle:
                rng.shuffle(order)
            if remaining is not None:
                if remaining <= 0:
                    break
                order = order[: min(len(order), remaining)]
                remaining -= len(order)
            for start in range(0, len(order), self.batch_size):
                batch_idx = order[start : start + self.batch_size]
                if len(batch_idx) == 0:
                    continue
                if not self.shuffle:
                    lo = int(batch_idx[0])
                    hi = int(batch_idx[-1]) + 1
                    yield {
                        "peak_embeddings": torch.from_numpy(
                            np.array(peak_embeddings[lo:hi], copy=True)
                        ),
                        "peak_valid_mask": torch.from_numpy(
                            np.array(peak_valid_mask[lo:hi], copy=True)
                        ).to(torch.bool),
                        "label": torch.from_numpy(np.array(labels[lo:hi], copy=True)).to(
                            torch.float32
                        ),
                    }
                else:
                    yield {
                        "peak_embeddings": torch.from_numpy(
                            np.array(peak_embeddings[batch_idx], copy=True)
                        ),
                        "peak_valid_mask": torch.from_numpy(
                            np.array(peak_valid_mask[batch_idx], copy=True)
                        ).to(torch.bool),
                        "label": torch.from_numpy(np.array(labels[batch_idx], copy=True)).to(
                            torch.float32
                        ),
                    }


class _FluorineCollator:
    def __init__(
        self,
        *,
        num_peaks: int,
        max_precursor_mz: float,
        min_peak_intensity: float,
        peak_drop_min_intensity: float,
        peak_ordering: str,
        precursor_peak_exclusion_window_da: float,
    ) -> None:
        self.num_peaks = num_peaks
        self.max_precursor_mz = max_precursor_mz
        self.min_peak_intensity = min_peak_intensity
        self.peak_drop_min_intensity = peak_drop_min_intensity
        self.peak_ordering = peak_ordering
        self.precursor_peak_exclusion_window_da = precursor_peak_exclusion_window_da

    def __call__(self, samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        spectra = torch.stack([sample["spectra"] for sample in samples], dim=0)
        precursor_raw = torch.stack([sample["precursor_mz_raw"] for sample in samples])
        batch = preprocess_peak_batch_torch(
            spectra[:, 0, :],
            spectra[:, 1, :],
            precursor_raw,
            num_peaks=self.num_peaks,
            peak_drop_min_intensity=self.peak_drop_min_intensity,
            peak_ordering=self.peak_ordering,
            max_precursor_mz=self.max_precursor_mz,
            precursor_peak_exclusion_window_da=self.precursor_peak_exclusion_window_da,
            min_peak_intensity=self.min_peak_intensity,
        )
        batch["dreams_embedding"] = torch.stack(
            [sample["dreams_embedding"] for sample in samples],
            dim=0,
        ).to(torch.float32)
        batch["label"] = torch.stack([sample["label"] for sample in samples]).to(
            torch.float32
        )
        return batch


class _FluorineDreamsCollator:
    def __call__(self, samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {
            "dreams_embedding": torch.stack(
                [sample["dreams_embedding"] for sample in samples],
                dim=0,
            ).to(torch.float32),
            "label": torch.stack([sample["label"] for sample in samples]).to(
                torch.float32
            ),
        }


class MLPClassifier(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def binary_focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float,
    gamma: float,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    prob = torch.sigmoid(logits)
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_t * (1.0 - p_t).pow(gamma) * bce).mean()


def _metric_dict(targets: np.ndarray, logits: np.ndarray, prefix: str) -> dict[str, float]:
    probs = np.empty_like(logits, dtype=np.float64)
    positive = logits >= 0
    probs[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    probs[~positive] = exp_logits / (1.0 + exp_logits)
    pred = probs >= 0.5
    return {
        f"{prefix}/roc_auc": float(roc_auc_score(targets, probs)),
        f"{prefix}/average_precision": float(average_precision_score(targets, probs)),
        f"{prefix}/accuracy": float(accuracy_score(targets, pred)),
        f"{prefix}/balanced_accuracy": balanced_accuracy_score(targets, pred),
        f"{prefix}/f1": float(f1_score(targets, pred, zero_division=0)),
        f"{prefix}/precision": float(precision_score(targets, pred, zero_division=0)),
        f"{prefix}/recall": float(recall_score(targets, pred, zero_division=0)),
        f"{prefix}/positive_rate": float(np.mean(targets)),
    }


def _parse_int_grid(raw: str) -> tuple[int, ...]:
    return tuple(int(item) for item in raw.split(",") if item)


def _parse_float_grid(raw: str) -> tuple[float, ...]:
    return tuple(float(item) for item in raw.split(",") if item)


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _make_loader(
    data: FluorineData,
    split: str,
    *,
    shuffle: bool,
    seed: int,
    max_samples: int | None,
    dreams_only: bool = False,
) -> DataLoader:
    dataset = _FluorineShardDataset(
        [
            {"dir": data.root / path, "length": length}
            for path, length in zip(
                data.metadata[f"{split}_files"],
                data.metadata[f"{split}_lengths"],
                strict=True,
            )
        ]
    )
    if max_samples is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=generator)[
            : min(len(dataset), max_samples)
        ].tolist()
        dataset = Subset(dataset, indices)
    generator = torch.Generator()
    generator.manual_seed(seed)
    collator: Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]
    if dreams_only:
        collator = _FluorineDreamsCollator()
    else:
        collator = _FluorineCollator(
            num_peaks=data.num_peaks,
            max_precursor_mz=data.max_precursor_mz,
            min_peak_intensity=data.min_peak_intensity,
            peak_drop_min_intensity=data.peak_drop_min_intensity,
            peak_ordering=data.peak_ordering,
            precursor_peak_exclusion_window_da=data.precursor_peak_exclusion_window_da,
        )
    return DataLoader(
        dataset,
        batch_size=data.batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        collate_fn=collator,
        generator=generator,
    )


def _make_embedding_loader(
    data: EmbeddingData,
    split: str,
    *,
    shuffle: bool,
    seed: int,
    max_samples: int | None,
) -> _FastEmbeddingLoader:
    return _FastEmbeddingLoader(
        shard_entries=[
            {"dir": data.root / path, "length": length}
            for path, length in zip(
                data.metadata[f"{split}_files"],
                data.metadata[f"{split}_lengths"],
                strict=True,
            )
        ],
        batch_size=data.batch_size,
        shuffle=shuffle,
        seed=seed,
        max_samples=max_samples,
    )


@torch.no_grad()
def _evaluate(
    classifier: MLPClassifier,
    loader: Any,
    feature_fn: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    *,
    device: torch.device,
    prefix: str,
) -> dict[str, float]:
    classifier.eval()
    logits, targets = [], []
    for batch in loader:
        batch = _move_batch(batch, device)
        features = feature_fn(batch)
        logits.append(classifier(features).detach().cpu().numpy())
        targets.append(batch["label"].detach().cpu().numpy())
    return _metric_dict(
        np.concatenate(targets, axis=0),
        np.concatenate(logits, axis=0),
        prefix,
    )


def _select_metric_value(metrics: dict[str, float], select_metric: str) -> float:
    return metrics[select_metric]


def _train_trial(
    *,
    params: TrialParams,
    input_dim: int,
    train_loader: Any,
    val_loader: Any,
    device: torch.device,
    epochs: int,
    focal_alpha: float,
    focal_gamma: float,
    select_metric: str,
    higher_is_better: bool,
    patience: int,
    build_feature_fn: Callable[[bool], tuple[Callable[[dict[str, torch.Tensor]], torch.Tensor], torch.nn.Module | None]],
) -> TrialResult:
    classifier = MLPClassifier(
        input_dim=input_dim,
        hidden_dim=params.hidden_dim,
        dropout=params.dropout,
    ).to(device)
    feature_fn, trainable_pooler = build_feature_fn(True)
    trainable_params: list[torch.nn.Parameter] = list(classifier.parameters())
    if trainable_pooler is not None:
        trainable_params += list(trainable_pooler.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=params.learning_rate,
        weight_decay=params.weight_decay,
    )

    best_value = -float("inf") if higher_is_better else float("inf")
    best_epoch: int = 0
    best_val: dict[str, float] = {}
    best_classifier_state: dict[str, torch.Tensor] = {}
    best_pooler_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement: int = 0
    for epoch_idx in range(epochs):
        classifier.train()
        if trainable_pooler is not None:
            trainable_pooler.train()
        for batch in train_loader:
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            features = feature_fn(batch)
            logits = classifier(features)
            loss = binary_focal_loss_with_logits(
                logits,
                batch["label"],
                alpha=focal_alpha,
                gamma=focal_gamma,
            )
            loss.backward()
            optimizer.step()

        val_metrics = _evaluate(
            classifier,
            val_loader,
            feature_fn,
            device=device,
            prefix="val",
        )
        current_value = _select_metric_value(val_metrics, select_metric)
        improved = (
            current_value > best_value
            if higher_is_better
            else current_value < best_value
        )
        if improved:
            best_value = current_value
            best_epoch = epoch_idx + 1
            best_val = dict(val_metrics)
            best_classifier_state = copy.deepcopy(classifier.state_dict())
            best_pooler_state = (
                copy.deepcopy(trainable_pooler.state_dict())
                if trainable_pooler is not None
                else None
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        log.info(
            "trial hidden=%d lr=%.3g wd=%.3g dropout=%.2f epoch=%d/%d val_ap=%.4f val_auc=%.4f",
            params.hidden_dim,
            params.learning_rate,
            params.weight_decay,
            params.dropout,
            epoch_idx + 1,
            epochs,
            val_metrics["val/average_precision"],
            val_metrics["val/roc_auc"],
        )
        if epochs_without_improvement >= patience:
            break
    return TrialResult(
        params=params,
        best_epoch=best_epoch,
        best_val=best_val,
        classifier_state=best_classifier_state,
        pooler_state=best_pooler_state,
    )


def _build_dreams_feature_factory(
    device: torch.device,
) -> Callable[[bool], tuple[Callable[[dict[str, torch.Tensor]], torch.Tensor], None]]:
    def build_feature_fn(training: bool):
        def feature_fn(batch: dict[str, torch.Tensor]) -> torch.Tensor:
            return batch["dreams_embedding"].to(device=device, dtype=torch.float32)

        return feature_fn, None

    return build_feature_fn


def _peak_tokens_only(
    token_embeddings: torch.Tensor,
    peak_valid_mask: torch.Tensor,
) -> torch.Tensor:
    return token_embeddings[:, : peak_valid_mask.shape[1]]


def _embedding_cache_valid(
    cache_dir: Path,
    *,
    source_cache_dir: Path,
    config_path: Path,
    checkpoint_path: Path,
    embedding_dtype: str,
) -> dict[str, Any] | None:
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text())
    expected = {
        "metadata_version": EMBEDDING_CACHE_METADATA_VERSION,
        "source_cache_dir": str(source_cache_dir),
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "embedding_dtype": embedding_dtype,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            return None
    for split in SPLITS:
        for shard_dir in _split_files(metadata, split):
            for filename in ("peak_embeddings.npy", "peak_valid_mask.npy", "label.npy"):
                if not (cache_dir / shard_dir / filename).exists():
                    return None
    return metadata


def _numpy_embedding_dtype(name: str) -> np.dtype:
    if name == "float16":
        return np.dtype(np.float16)
    if name == "float32":
        return np.dtype(np.float32)
    raise ValueError("embedding dtype must be float16 or float32")


@torch.no_grad()
def _write_embedding_cache_split(
    *,
    model: PeakSetSIGReg,
    data: FluorineData,
    split: str,
    output_dir: Path,
    device: torch.device,
    embedding_dtype: str,
    seed: int,
) -> tuple[list[str], list[int]]:
    split_files = data.metadata[f"{split}_files"]
    split_lengths = data.metadata[f"{split}_lengths"]
    shard_names, shard_lengths = [], []
    loader = _make_loader(
        data,
        split,
        shuffle=False,
        seed=seed,
        max_samples=None,
        dreams_only=False,
    )
    np_dtype = _numpy_embedding_dtype(embedding_dtype)
    use_autocast = device.type == "cuda"
    current_shard: int = 0
    current_offset: int = 0
    peak_embeddings: np.memmap | None = None
    peak_valid_mask: np.memmap | None = None
    labels: np.memmap | None = None

    def open_shard() -> None:
        nonlocal peak_embeddings, peak_valid_mask, labels
        shard_name = str(split_files[current_shard])
        shard_dir = output_dir / shard_name
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_len = int(split_lengths[current_shard])
        peak_embeddings = np.lib.format.open_memmap(
            shard_dir / "peak_embeddings.npy",
            mode="w+",
            dtype=np_dtype,
            shape=(shard_len, model.num_peak_tokens, model.model_dim),
        )
        peak_valid_mask = np.lib.format.open_memmap(
            shard_dir / "peak_valid_mask.npy",
            mode="w+",
            dtype=bool,
            shape=(shard_len, model.num_peak_tokens),
        )
        labels = np.lib.format.open_memmap(
            shard_dir / "label.npy",
            mode="w+",
            dtype=np.float32,
            shape=(shard_len,),
        )

    open_shard()
    for batch in loader:
        batch = _move_batch(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_autocast,
        ):
            encoded = model.encoder(
                batch["peak_mz"],
                batch["peak_intensity"],
                valid_mask=batch["peak_valid_mask"],
                precursor_mz=batch.get("precursor_mz", None),
            )
            token_embeddings, _ = model.encoder.split_peak_and_cls(encoded)
            token_embeddings = _peak_tokens_only(token_embeddings, batch["peak_valid_mask"])
        values = token_embeddings.detach().cpu().numpy().astype(np_dtype, copy=False)
        masks = batch["peak_valid_mask"].detach().cpu().numpy().astype(bool, copy=False)
        batch_labels = batch["label"].detach().cpu().numpy().astype(np.float32, copy=False)
        source_offset: int = 0
        while source_offset < values.shape[0]:
            assert peak_embeddings is not None
            assert peak_valid_mask is not None
            assert labels is not None
            shard_remaining = peak_embeddings.shape[0] - current_offset
            take = min(shard_remaining, values.shape[0] - source_offset)
            target_slice = slice(current_offset, current_offset + take)
            source_slice = slice(source_offset, source_offset + take)
            peak_embeddings[target_slice] = values[source_slice]
            peak_valid_mask[target_slice] = masks[source_slice]
            labels[target_slice] = batch_labels[source_slice]
            current_offset += take
            source_offset += take
            if current_offset == peak_embeddings.shape[0]:
                peak_embeddings.flush()
                peak_valid_mask.flush()
                labels.flush()
                shard_names.append(str(split_files[current_shard]))
                shard_lengths.append(int(split_lengths[current_shard]))
                current_shard += 1
                current_offset = 0
                if current_shard < len(split_files):
                    open_shard()
    return shard_names, shard_lengths


def ensure_embedding_cache(
    *,
    cache_dir: Path,
    source_data: FluorineData,
    model: PeakSetSIGReg,
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    embedding_dtype: str,
    force: bool,
) -> dict[str, Any]:
    source_cache_dir = source_data.root.resolve()
    config_path = config_path.resolve()
    checkpoint_path = checkpoint_path.resolve()
    if not force:
        cached = _embedding_cache_valid(
            cache_dir,
            source_cache_dir=source_cache_dir,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            embedding_dtype=embedding_dtype,
        )
        if cached is not None:
            return cached
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "metadata_version": EMBEDDING_CACHE_METADATA_VERSION,
        "source_cache_dir": str(source_cache_dir),
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "embedding_dtype": embedding_dtype,
        "model_dim": model.model_dim,
        "num_peak_tokens": model.num_peak_tokens,
    }
    for split in SPLITS:
        log.info("writing cached encoder embeddings for split=%s", split)
        shard_names, shard_lengths = _write_embedding_cache_split(
            model=model,
            data=source_data,
            split=split,
            output_dir=cache_dir,
            device=device,
            embedding_dtype=embedding_dtype,
            seed=0,
        )
        metadata[f"{split}_files"] = shard_names
        metadata[f"{split}_lengths"] = shard_lengths
        metadata[f"{split}_size"] = int(source_data.metadata[f"{split}_size"])
        metadata[f"{split}_positive"] = int(source_data.metadata[f"{split}_positive"])
    (cache_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def _load_checkpoint_model(
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[config_dict.ConfigDict, PeakSetSIGReg]:
    config = load_config(config_path)
    model = build_model_from_config(config)
    load_pretrained_weights(model, str(checkpoint_path))
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return config, model


def _build_checkpoint_feature_factory(
    *,
    model: PeakSetSIGReg,
    config: config_dict.ConfigDict,
    device: torch.device,
    train_covariance_pooler: bool,
    covariance_dim: int | None,
) -> tuple[int, Callable[[bool], tuple[Callable[[dict[str, torch.Tensor]], torch.Tensor], torch.nn.Module | None]]]:
    if train_covariance_pooler:
        compressed_dim = (
            covariance_dim
            if covariance_dim is not None
            else int(_config_get(config, "covariance_pooling_dim", 32))
        )
    else:
        checkpoint_pooler = cast(CovariancePool, cast(Any, model).covariance_pooler)
        compressed_dim = checkpoint_pooler.left_proj.out_features
    input_dim = compressed_dim * compressed_dim

    @torch.no_grad()
    def encode(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        embeddings = model.encoder(
            batch["peak_mz"],
            batch["peak_intensity"],
            valid_mask=batch["peak_valid_mask"],
            precursor_mz=batch.get("precursor_mz", None),
        )
        peak_embeddings, _ = model.encoder.split_peak_and_cls(embeddings)
        return _peak_tokens_only(peak_embeddings, batch["peak_valid_mask"])

    def build_feature_fn(training: bool):
        if train_covariance_pooler:
            pooler = CovariancePool(
                input_dim=int(config.model_dim),
                compressed_dim=compressed_dim,
            ).to(device)
            trainable_pooler: torch.nn.Module | None = pooler
        else:
            pooler = cast(CovariancePool, cast(Any, model).covariance_pooler)
            pooler.eval()
            pooler.requires_grad_(False)
            trainable_pooler = None

        def feature_fn(batch: dict[str, torch.Tensor]) -> torch.Tensor:
            peak_embeddings = encode(batch)
            return pooler(
                peak_embeddings.float(),
                batch["peak_valid_mask"].to(dtype=torch.bool),
            )

        return feature_fn, trainable_pooler

    return input_dim, build_feature_fn


def _build_cached_embedding_feature_factory(
    *,
    config: config_dict.ConfigDict,
    device: torch.device,
    train_covariance_pooler: bool,
    covariance_dim: int | None,
) -> tuple[int, Callable[[bool], tuple[Callable[[dict[str, torch.Tensor]], torch.Tensor], torch.nn.Module | None]]]:
    compressed_dim = (
        covariance_dim
        if covariance_dim is not None
        else int(_config_get(config, "covariance_pooling_dim", 32))
    )
    input_dim = compressed_dim * compressed_dim

    def build_feature_fn(training: bool):
        pooler = CovariancePool(
            input_dim=int(config.model_dim),
            compressed_dim=compressed_dim,
        ).to(device)
        trainable_pooler: torch.nn.Module | None = pooler if train_covariance_pooler else None
        if trainable_pooler is None:
            pooler.requires_grad_(False)

        def feature_fn(batch: dict[str, torch.Tensor]) -> torch.Tensor:
            return pooler(
                batch["peak_embeddings"].to(device=device, dtype=torch.float32),
                batch["peak_valid_mask"].to(device=device, dtype=torch.bool),
            )

        return feature_fn, trainable_pooler

    return input_dim, build_feature_fn


def _instantiate_best_model(
    *,
    result: TrialResult,
    input_dim: int,
    device: torch.device,
    build_feature_fn: Callable[[bool], tuple[Callable[[dict[str, torch.Tensor]], torch.Tensor], torch.nn.Module | None]],
) -> tuple[MLPClassifier, Callable[[dict[str, torch.Tensor]], torch.Tensor]]:
    classifier = MLPClassifier(
        input_dim=input_dim,
        hidden_dim=result.params.hidden_dim,
        dropout=result.params.dropout,
    ).to(device)
    feature_fn, pooler = build_feature_fn(False)
    classifier.load_state_dict(result.classifier_state)
    if pooler is not None and result.pooler_state is not None:
        pooler.load_state_dict(result.pooler_state)
    classifier.eval()
    if pooler is not None:
        pooler.eval()
    return classifier, feature_fn


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    config = load_config(args.config) if args.config and args.source == "checkpoint" else None
    cache_dir = args.cache_dir.expanduser().resolve()
    metadata = ensure_fluorine_cache(
        cache_dir,
        repo_id=args.repo_id,
        revision=args.revision,
        subdir=args.subdir,
        num_shards=args.num_shards,
        parquet_batch_size=args.parquet_batch_size,
    )
    data = FluorineData(
        metadata=metadata,
        root=cache_dir,
        batch_size=int(args.batch_size),
        num_peaks=int(
            args.num_peaks
            if args.num_peaks is not None
            else (_config_get(config, "num_peaks", 60) if config is not None else 60)
        ),
        max_precursor_mz=float(
            _config_get(config, "max_precursor_mz", DEFAULT_MAX_PRECURSOR_MZ)
            if config is not None
            else DEFAULT_MAX_PRECURSOR_MZ
        ),
        min_peak_intensity=float(
            _config_get(config, "min_peak_intensity", DEFAULT_MIN_PEAK_INTENSITY)
            if config is not None
            else DEFAULT_MIN_PEAK_INTENSITY
        ),
        peak_drop_min_intensity=float(
            _config_get(
                config,
                "peak_drop_min_intensity",
                _config_get(config, "min_peak_intensity", DEFAULT_MIN_PEAK_INTENSITY),
            )
            if config is not None
            else DEFAULT_MIN_PEAK_INTENSITY
        ),
        peak_ordering=str(
            args.peak_ordering
            if args.peak_ordering
            else (
                _config_get(config, "peak_ordering", "intensity")
                if config is not None
                else "intensity"
            )
        ),
        precursor_peak_exclusion_window_da=float(
            _config_get(config, "precursor_peak_exclusion_window_da", 0.0)
            if config is not None
            else 0.0
        ),
    )
    if args.source == "dreams":
        train_loader = _make_loader(
            data,
            "train",
            shuffle=True,
            seed=args.seed,
            max_samples=args.max_train_samples,
            dreams_only=True,
        )
        val_loader = _make_loader(
            data,
            "val",
            shuffle=False,
            seed=args.seed + 10_000,
            max_samples=args.max_val_samples,
            dreams_only=True,
        )
        test_loader = _make_loader(
            data,
            "test",
            shuffle=False,
            seed=args.seed + 20_000,
            max_samples=args.max_test_samples,
            dreams_only=True,
        )
        build_feature_fn = _build_dreams_feature_factory(device)
        input_dim = int(metadata["dreams_dim"])
        embedding_cache_dir = ""
    else:
        checkpoint_config, model = _load_checkpoint_model(
            args.config.expanduser().resolve(),
            args.checkpoint.expanduser().resolve(),
            device,
        )
        if args.embedding_cache_dir is not None:
            resolved_embedding_cache_dir = args.embedding_cache_dir.expanduser().resolve()
            embedding_metadata = ensure_embedding_cache(
                cache_dir=resolved_embedding_cache_dir,
                source_data=data,
                model=model,
                config_path=args.config.expanduser().resolve(),
                checkpoint_path=args.checkpoint.expanduser().resolve(),
                device=device,
                embedding_dtype=args.embedding_dtype,
                force=args.force_embedding_cache,
            )
            embedding_data = EmbeddingData(
                metadata=embedding_metadata,
                root=resolved_embedding_cache_dir,
                batch_size=int(args.batch_size),
            )
            train_loader = _make_embedding_loader(
                embedding_data,
                "train",
                shuffle=True,
                seed=args.seed,
                max_samples=args.max_train_samples,
            )
            val_loader = _make_embedding_loader(
                embedding_data,
                "val",
                shuffle=False,
                seed=args.seed + 10_000,
                max_samples=args.max_val_samples,
            )
            test_loader = _make_embedding_loader(
                embedding_data,
                "test",
                shuffle=False,
                seed=args.seed + 20_000,
                max_samples=args.max_test_samples,
            )
            input_dim, build_feature_fn = _build_cached_embedding_feature_factory(
                config=checkpoint_config,
                device=device,
                train_covariance_pooler=bool(args.train_covariance_pooler),
                covariance_dim=args.covariance_dim,
            )
            embedding_cache_dir = str(resolved_embedding_cache_dir)
        else:
            train_loader = _make_loader(
                data,
                "train",
                shuffle=True,
                seed=args.seed,
                max_samples=args.max_train_samples,
                dreams_only=False,
            )
            val_loader = _make_loader(
                data,
                "val",
                shuffle=False,
                seed=args.seed + 10_000,
                max_samples=args.max_val_samples,
                dreams_only=False,
            )
            test_loader = _make_loader(
                data,
                "test",
                shuffle=False,
                seed=args.seed + 20_000,
                max_samples=args.max_test_samples,
                dreams_only=False,
            )
            input_dim, build_feature_fn = _build_checkpoint_feature_factory(
                model=model,
                config=checkpoint_config,
                device=device,
                train_covariance_pooler=bool(args.train_covariance_pooler),
                covariance_dim=args.covariance_dim,
            )
            embedding_cache_dir = ""

    train_positive = float(metadata["train_positive"])
    train_size = float(metadata["train_size"])
    if args.focal_alpha == "auto":
        focal_alpha = 1.0 - train_positive / train_size
    else:
        focal_alpha = float(args.focal_alpha)

    trial_params = [
        TrialParams(hidden_dim=hidden, learning_rate=lr, weight_decay=wd, dropout=dropout)
        for hidden, lr, wd, dropout in itertools.product(
            _parse_int_grid(args.hidden_dims),
            _parse_float_grid(args.learning_rates),
            _parse_float_grid(args.weight_decays),
            _parse_float_grid(args.dropouts),
        )
    ]
    select_metric = f"val/{args.select_metric}"
    higher_is_better = args.select_metric not in {"loss"}
    results = [
        _train_trial(
            params=params,
            input_dim=input_dim,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            focal_alpha=focal_alpha,
            focal_gamma=args.focal_gamma,
            select_metric=select_metric,
            higher_is_better=higher_is_better,
            patience=args.patience,
            build_feature_fn=build_feature_fn,
        )
        for params in trial_params
    ]
    best = max(
        results,
        key=lambda result: _select_metric_value(result.best_val, select_metric),
    )
    if not higher_is_better:
        best = min(
            results,
            key=lambda result: _select_metric_value(result.best_val, select_metric),
        )
    classifier, feature_fn = _instantiate_best_model(
        result=best,
        input_dim=input_dim,
        device=device,
        build_feature_fn=build_feature_fn,
    )
    test_metrics = _evaluate(
        classifier,
        test_loader,
        feature_fn,
        device=device,
        prefix="test",
    )
    payload: dict[str, Any] = {
        "source": args.source,
        "repo_id": args.repo_id,
        "revision": args.revision,
        "subdir": args.subdir,
        "cache_dir": str(cache_dir),
        "embedding_cache_dir": embedding_cache_dir,
        "input_dim": input_dim,
        "train_size": int(metadata["train_size"]),
        "train_positive": int(metadata["train_positive"]),
        "val_size": int(metadata["val_size"]),
        "test_size": int(metadata["test_size"]),
        "focal_alpha": focal_alpha,
        "focal_gamma": float(args.focal_gamma),
        "best_hparams": best.params._asdict(),
        "best_epoch": best.best_epoch,
        "best_val": best.best_val,
        "test": test_metrics,
        "trials": [
            {
                "hparams": result.params._asdict(),
                "best_epoch": result.best_epoch,
                "best_val": result.best_val,
            }
            for result in results
        ],
    }
    if args.output_json:
        output_path = args.output_json.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a fluorine-detection MLP on NIST HR-MS/MS spectra or DreaMS embeddings."
    )
    parser.add_argument("--source", choices=("checkpoint", "dreams"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--train-covariance-pooler", action="store_true")
    parser.add_argument("--covariance-dim", type=int, default=None)
    parser.add_argument("--embedding-cache-dir", type=Path, default=None)
    parser.add_argument("--force-embedding-cache", action="store_true")
    parser.add_argument(
        "--embedding-dtype",
        choices=("float16", "float32"),
        default="float16",
    )
    parser.add_argument("--repo-id", default=HF_REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--subdir", default=HF_SUBDIR)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/fluorine_detection_fine_tuned"),
    )
    parser.add_argument("--num-shards", type=int, default=16)
    parser.add_argument("--parquet-batch-size", type=int, default=50_000)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-peaks", type=int, default=None)
    parser.add_argument("--peak-ordering", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--hidden-dims", default="256,512")
    parser.add_argument("--learning-rates", default="0.001,0.0003")
    parser.add_argument("--weight-decays", default="0.0001")
    parser.add_argument("--dropouts", default="0.1")
    parser.add_argument("--focal-alpha", default="auto")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--select-metric",
        default="average_precision",
        choices=("average_precision", "roc_auc", "balanced_accuracy", "f1"),
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    args = parser.parse_args()
    if args.source == "checkpoint" and (args.config is None or args.checkpoint is None):
        parser.error("--source checkpoint requires --config and --checkpoint")
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    payload = run(parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
