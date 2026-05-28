from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

from spectra_learning.data.spectra import (
    DEFAULT_MAX_PRECURSOR_MZ,
    DEFAULT_MIN_PEAK_INTENSITY,
    NUM_PEAKS_INPUT,
    preprocess_peak_batch_torch,
)
from spectra_learning.models.pooling import CovariancePool, SinglePairCovariancePool
from spectra_learning.probes.massspec.data import _normalize_spectra_intensity
from spectra_learning.training.checkpointing import latest_ckpt_path

from scripts.train_fluorine_detection import (
    EmbeddingData,
    FluorineData,
    HF_REPO_ID,
    HF_SUBDIR,
    MLPClassifier,
    TrialParams,
    _build_cached_embedding_feature_factory,
    _load_checkpoint_model,
    _make_loader,
    _make_embedding_loader,
    _metric_dict,
    _move_batch,
    _peak_tokens_only,
    _train_trial,
    binary_focal_loss_with_logits,
    ensure_fluorine_cache,
)


log = logging.getLogger(__name__)
ATOM_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def _prepend_precursor_token_torch(
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    batch = dict(batch)
    batch["peak_mz"] = torch.cat(
        [batch["precursor_mz"][:, None], batch["peak_mz"]],
        dim=1,
    )
    batch["peak_intensity"] = torch.cat(
        [torch.ones_like(batch["precursor_mz"][:, None]), batch["peak_intensity"]],
        dim=1,
    )
    batch["peak_valid_mask"] = torch.cat(
        [torch.ones_like(batch["peak_valid_mask"][:, :1]), batch["peak_valid_mask"]],
        dim=1,
    )
    return batch


class MsnlibData:
    def __init__(
        self,
        *,
        spectra: np.ndarray,
        precursor_mz: np.ndarray,
        labels: np.ndarray,
        rows: list[dict[str, Any]],
        counts: dict[str, int],
    ) -> None:
        self.spectra = spectra
        self.precursor_mz = precursor_mz
        self.labels = labels
        self.rows = rows
        self.counts = counts


class MsnlibDataset(Dataset):
    def __init__(self, data: MsnlibData) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "spectra": torch.from_numpy(self.data.spectra[idx].copy()),
            "precursor_mz_raw": torch.tensor(
                float(self.data.precursor_mz[idx]),
                dtype=torch.float32,
            ),
            "label": torch.tensor(float(self.data.labels[idx]), dtype=torch.float32),
            "row_idx": torch.tensor(idx, dtype=torch.long),
        }


class MsnlibCollator:
    def __init__(
        self,
        *,
        num_peaks: int,
        max_precursor_mz: float,
        min_peak_intensity: float,
        peak_drop_min_intensity: float,
        peak_ordering: str,
        use_precursor_token: bool,
        precursor_peak_exclusion_window_da: float,
    ) -> None:
        self.num_peaks = int(num_peaks)
        self.max_precursor_mz = float(max_precursor_mz)
        self.min_peak_intensity = float(min_peak_intensity)
        self.peak_drop_min_intensity = float(peak_drop_min_intensity)
        self.peak_ordering = str(peak_ordering)
        self.use_precursor_token = bool(use_precursor_token)
        self.precursor_peak_exclusion_window_da = float(precursor_peak_exclusion_window_da)

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
        batch["label"] = torch.stack([sample["label"] for sample in samples]).to(torch.float32)
        batch["row_idx"] = torch.stack([sample["row_idx"] for sample in samples]).to(torch.long)
        if self.use_precursor_token:
            batch = _prepend_precursor_token_torch(batch)
        return batch


def _formula_has_fluorine(formula: str) -> bool:
    return any(
        atom == "F" and int(count or "1") > 0
        for atom, count in ATOM_RE.findall(formula or "")
    )


def _parse_charge(raw: str) -> int:
    match = re.match(r"([+-]?\d+)", raw or "")
    return int(match.group(1)) if match else 0


def _passes_filters(fields: dict[str, str], peaks: list[tuple[float, float]]) -> tuple[bool, dict[str, bool]]:
    charge_ok = _parse_charge(fields.get("CHARGE", "")) == 1
    precursor_mz = float(fields["PEPMASS"].split()[0])
    precursor_ok = precursor_mz <= 1000.0
    intensities = np.asarray([intensity for _, intensity in peaks], dtype=np.float32)
    amplitude_ok = bool(np.max(intensities) >= 20.0)
    normalized = intensities / np.max(intensities)
    high_peak_ok = bool(np.count_nonzero(normalized >= 0.1) >= 3)
    flags = {
        "charge1": charge_ok,
        "precursor_mz_le_1000": precursor_ok,
        "intensity_amplitude_ge_20": amplitude_ok,
        "num_high_peaks_ge_3": high_peak_ok,
    }
    return all(flags.values()), flags


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


def parse_mgf(path: Path) -> MsnlibData:
    counts = {
        "total": 0,
        "charge1": 0,
        "precursor_mz_le_1000": 0,
        "intensity_amplitude_ge_20": 0,
        "num_high_peaks_ge_3": 0,
        "kept": 0,
        "positive": 0,
    }
    rows: list[dict[str, Any]] = []
    mz_lists: list[list[float]] = []
    intensity_lists: list[list[float]] = []
    precursor_mz: list[float] = []
    labels: list[float] = []
    fields: dict[str, str] = {}
    peaks: list[tuple[float, float]] = []

    def flush() -> None:
        counts["total"] += 1
        keep, flags = _passes_filters(fields, peaks)
        for key, passed in flags.items():
            counts[key] += int(passed)
        if not keep:
            return
        formula = fields.get("FORMULA", "")
        label = float(_formula_has_fluorine(formula))
        counts["kept"] += 1
        counts["positive"] += int(label)
        mz_lists.append([mz for mz, _ in peaks])
        intensity_lists.append([intensity for _, intensity in peaks])
        precursor_mz.append(float(fields["PEPMASS"].split()[0]))
        labels.append(label)
        rows.append(
            {
                "source_index": counts["total"] - 1,
                "name": fields.get("NAME", ""),
                "formula": formula,
                "smiles": fields.get("SMILES", ""),
                "precursor_mz": precursor_mz[-1],
                "charge": fields.get("CHARGE", ""),
                "adduct": fields.get("ADDUCT", ""),
                "spectype": fields.get("SPECTYPE", ""),
                "instrument_type": fields.get("INSTRUMENT_TYPE", ""),
                "num_peaks_raw": len(peaks),
                "label": int(label),
            }
        )

    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line == "BEGIN IONS":
                fields = {}
                peaks = []
            elif line == "END IONS":
                flush()
            elif "=" in line:
                key, value = line.split("=", 1)
                fields[key] = value
            else:
                mz, intensity = line.split()[:2]
                peaks.append((float(mz), float(intensity)))

    spectra = _spectra_from_peak_lists(mz_lists, intensity_lists)
    return MsnlibData(
        spectra=spectra,
        precursor_mz=np.asarray(precursor_mz, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.float32),
        rows=rows,
        counts=counts,
    )


def train_or_load_head(
    *,
    head_state_path: Path,
    embedding_cache_dir: Path,
    config: Any,
    device: torch.device,
    batch_size: int,
    seed: int,
    epochs: int,
) -> dict[str, Any]:
    if head_state_path.exists():
        return torch.load(head_state_path, map_location=device)

    embedding_metadata = json.loads((embedding_cache_dir / "metadata.json").read_text())
    embedding_data = EmbeddingData(
        metadata=embedding_metadata,
        root=embedding_cache_dir,
        batch_size=batch_size,
    )
    input_dim, build_feature_fn = _build_cached_embedding_feature_factory(
        config=config,
        device=device,
        train_covariance_pooler=True,
        covariance_dim=64,
    )
    train_loader = _make_embedding_loader(
        embedding_data,
        "train",
        shuffle=True,
        seed=seed,
        max_samples=None,
    )
    val_loader = _make_embedding_loader(
        embedding_data,
        "val",
        shuffle=False,
        seed=seed + 10_000,
        max_samples=None,
    )
    params = TrialParams(
        hidden_dim=256,
        learning_rate=0.001,
        weight_decay=0.0001,
        dropout=0.1,
    )
    focal_alpha = 1.0 - float(embedding_metadata["train_positive"]) / float(
        embedding_metadata["train_size"]
    )
    result = _train_trial(
        params=params,
        input_dim=input_dim,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        focal_alpha=focal_alpha,
        focal_gamma=2.0,
        select_metric="val/average_precision",
        higher_is_better=True,
        patience=epochs,
        build_feature_fn=build_feature_fn,
    )
    state = {
        "mode": "probe",
        "input_dim": int(input_dim),
        "covariance_dim": 64,
        "pooler_state": result.pooler_state,
        "classifier_state": result.classifier_state,
        "best_epoch": int(result.best_epoch),
        "best_val": result.best_val,
        "hparams": result.params._asdict(),
        "focal_alpha": focal_alpha,
        "focal_gamma": 2.0,
        "embedding_cache_dir": str(embedding_cache_dir),
    }
    head_state_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, head_state_path)
    return state


def build_fluorine_data(
    *,
    config: Any,
    cache_dir: Path,
    batch_size: int,
    repo_id: str,
    revision: str,
    subdir: str,
) -> FluorineData:
    metadata = ensure_fluorine_cache(
        cache_dir,
        repo_id=repo_id,
        revision=revision,
        subdir=subdir,
        num_shards=16,
        parquet_batch_size=50_000,
    )
    return FluorineData(
        metadata=metadata,
        root=cache_dir,
        batch_size=batch_size,
        num_peaks=int(config.get("num_peaks", 64)),
        max_precursor_mz=float(config.get("max_precursor_mz", DEFAULT_MAX_PRECURSOR_MZ)),
        min_peak_intensity=float(config.get("min_peak_intensity", DEFAULT_MIN_PEAK_INTENSITY)),
        peak_drop_min_intensity=float(
            config.get(
                "peak_drop_min_intensity",
                config.get("min_peak_intensity", DEFAULT_MIN_PEAK_INTENSITY),
            )
        ),
        peak_ordering=str(config.get("peak_ordering", "mz")),
        precursor_peak_exclusion_window_da=float(
            config.get("precursor_peak_exclusion_window_da", 0.0)
        ),
    )


def _module_state_to_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def _finetune_logits(
    *,
    model: torch.nn.Module,
    pooler: torch.nn.Module,
    classifier: MLPClassifier,
    batch: dict[str, torch.Tensor],
    pooling: str,
) -> torch.Tensor:
    if pooling == "single_pair_covariance":
        peak_embeddings, _, pair_embeddings = model.encoder.forward_with_block_outputs(
            batch["peak_mz"],
            batch["peak_intensity"],
            valid_mask=batch["peak_valid_mask"],
            precursor_mz=batch.get("precursor_mz", None),
        )
        peak_embeddings = _peak_tokens_only(peak_embeddings, batch["peak_valid_mask"])
        features = pooler(
            peak_embeddings.float(),
            batch["peak_valid_mask"].to(dtype=torch.bool),
            pair_embeddings.float(),
        )
    else:
        encoded = model.encoder(
            batch["peak_mz"],
            batch["peak_intensity"],
            valid_mask=batch["peak_valid_mask"],
            precursor_mz=batch.get("precursor_mz", None),
        )
        peak_embeddings = _peak_tokens_only(encoded, batch["peak_valid_mask"])
        features = pooler(
            peak_embeddings.float(),
            batch["peak_valid_mask"].to(dtype=torch.bool),
        )
    return classifier(features)


@torch.no_grad()
def predict_finetuned(
    *,
    model: torch.nn.Module,
    pooler: torch.nn.Module,
    classifier: MLPClassifier,
    loader: DataLoader,
    device: torch.device,
    pooling: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    pooler.eval()
    classifier.eval()
    use_autocast = device.type == "cuda"
    logits, targets = [], []
    for batch in loader:
        batch = _move_batch(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_autocast,
        ):
            batch_logits = _finetune_logits(
                model=model,
                pooler=pooler,
                classifier=classifier,
                batch=batch,
                pooling=pooling,
            )
        logits.append(batch_logits.float().detach().cpu().numpy())
        targets.append(batch["label"].detach().cpu().numpy())
    return np.concatenate(targets, axis=0), np.concatenate(logits, axis=0)


def train_or_load_finetuned(
    *,
    state_path: Path,
    model: torch.nn.Module,
    config: Any,
    config_path: Path,
    checkpoint_path: Path,
    cache_dir: Path,
    device: torch.device,
    batch_size: int,
    seed: int,
    epochs: int,
    patience: int,
    model_learning_rate: float,
    head_learning_rate: float,
    weight_decay: float,
    hidden_dim: int,
    dropout: float,
    repo_id: str,
    revision: str,
    subdir: str,
    max_train_samples: int | None,
    max_val_samples: int | None,
    pooling: str,
) -> dict[str, Any]:
    if state_path.exists():
        state = torch.load(state_path, map_location=device)
        if (
            state.get("mode") == "finetune"
            and state.get("config_path") == str(config_path)
            and state.get("checkpoint_path") == str(checkpoint_path)
            and state.get("pooling", "covariance") == pooling
        ):
            model.load_state_dict(state["model_state"])
            return state

    data = build_fluorine_data(
        config=config,
        cache_dir=cache_dir,
        batch_size=batch_size,
        repo_id=repo_id,
        revision=revision,
        subdir=subdir,
    )
    train_loader = _make_loader(
        data,
        "train",
        shuffle=True,
        seed=seed,
        max_samples=max_train_samples,
        dreams_only=False,
    )
    val_loader = _make_loader(
        data,
        "val",
        shuffle=False,
        seed=seed + 10_000,
        max_samples=max_val_samples,
        dreams_only=False,
    )
    test_loader = _make_loader(
        data,
        "test",
        shuffle=False,
        seed=seed + 20_000,
        max_samples=None,
        dreams_only=False,
    )
    covariance_dim = int(config.get("covariance_pooling_dim", 64))
    input_dim = covariance_dim * covariance_dim
    if pooling == "single_pair_covariance":
        pooler: torch.nn.Module = SinglePairCovariancePool(
            single_dim=int(config.model_dim),
            pair_dim=int(config.get("pairformer_pair_dim", config.model_dim)),
            compressed_dim=covariance_dim,
        ).to(device)
    else:
        pooler = CovariancePool(
            input_dim=int(config.model_dim),
            compressed_dim=covariance_dim,
        ).to(device)
    classifier = MLPClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    model.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.encoder.parameters(),
                "lr": model_learning_rate,
                "weight_decay": weight_decay,
            },
            {
                "params": list(pooler.parameters()) + list(classifier.parameters()),
                "lr": head_learning_rate,
                "weight_decay": weight_decay,
            },
        ]
    )
    focal_alpha = 1.0 - float(data.metadata["train_positive"]) / float(
        data.metadata["train_size"]
    )
    focal_gamma = 2.0
    best_value = -float("inf")
    best_epoch = 0
    best_val: dict[str, float] = {}
    best_model_state: dict[str, torch.Tensor] = {}
    best_pooler_state: dict[str, torch.Tensor] = {}
    best_classifier_state: dict[str, torch.Tensor] = {}
    history: list[dict[str, Any]] = []
    epochs_without_improvement = 0
    use_autocast = device.type == "cuda"
    for epoch_idx in range(epochs):
        model.train()
        pooler.train()
        classifier.train()
        running_loss = 0.0
        seen = 0
        for batch in train_loader:
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_autocast,
            ):
                logits = _finetune_logits(
                    model=model,
                    pooler=pooler,
                    classifier=classifier,
                    batch=batch,
                    pooling=pooling,
                )
                loss = binary_focal_loss_with_logits(
                    logits.float(),
                    batch["label"],
                    alpha=focal_alpha,
                    gamma=focal_gamma,
                )
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu()) * int(batch["label"].shape[0])
            seen += int(batch["label"].shape[0])

        val_targets, val_logits = predict_finetuned(
            model=model,
            pooler=pooler,
            classifier=classifier,
            loader=val_loader,
            device=device,
            pooling=pooling,
        )
        val_metrics = _metric_dict(val_targets, val_logits, "val")
        history.append(
            {
                "epoch": epoch_idx + 1,
                "train_loss": running_loss / float(seen),
                "val": val_metrics,
            }
        )
        current_value = val_metrics["val/average_precision"]
        if current_value > best_value:
            best_value = current_value
            best_epoch = epoch_idx + 1
            best_val = dict(val_metrics)
            best_model_state = _module_state_to_cpu(model)
            best_pooler_state = _module_state_to_cpu(pooler)
            best_classifier_state = _module_state_to_cpu(classifier)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        log.info(
            "finetune epoch=%d/%d train_loss=%.5f val_ap=%.4f val_auc=%.4f",
            epoch_idx + 1,
            epochs,
            running_loss / float(seen),
            val_metrics["val/average_precision"],
            val_metrics["val/roc_auc"],
        )
        if epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_model_state)
    pooler.load_state_dict(best_pooler_state)
    classifier.load_state_dict(best_classifier_state)
    test_targets, test_logits = predict_finetuned(
        model=model,
        pooler=pooler,
        classifier=classifier,
        loader=test_loader,
        device=device,
        pooling=pooling,
    )
    test_metrics = _metric_dict(test_targets, test_logits, "test")

    state = {
        "mode": "finetune",
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "input_dim": int(input_dim),
        "covariance_dim": int(covariance_dim),
        "pooling": pooling,
        "pair_dim": int(config.get("pairformer_pair_dim", config.model_dim)),
        "model_state": best_model_state,
        "pooler_state": best_pooler_state,
        "classifier_state": best_classifier_state,
        "best_epoch": int(best_epoch),
        "best_val": best_val,
        "test": test_metrics,
        "history": history,
        "hparams": {
            "hidden_dim": int(hidden_dim),
            "dropout": float(dropout),
            "model_learning_rate": float(model_learning_rate),
            "head_learning_rate": float(head_learning_rate),
            "weight_decay": float(weight_decay),
        },
        "focal_alpha": focal_alpha,
        "focal_gamma": focal_gamma,
        "finetune_cache_dir": str(cache_dir),
        "train_size": int(data.metadata["train_size"]),
        "train_positive": int(data.metadata["train_positive"]),
        "val_size": int(data.metadata["val_size"]),
        "val_positive": int(data.metadata["val_positive"]),
        "max_train_samples": max_train_samples,
        "max_val_samples": max_val_samples,
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, state_path)
    return state


@torch.no_grad()
def evaluate_msnlib(
    *,
    model: torch.nn.Module,
    config: Any,
    head_state: dict[str, Any],
    data: MsnlibData,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    covariance_dim = int(head_state.get("covariance_dim", 64))
    pooling = str(head_state.get("pooling", "covariance"))
    if pooling == "single_pair_covariance":
        pooler = SinglePairCovariancePool(
            single_dim=int(config.model_dim),
            pair_dim=int(head_state.get("pair_dim", config.get("pairformer_pair_dim", config.model_dim))),
            compressed_dim=covariance_dim,
        ).to(device)
    else:
        pooler = CovariancePool(
            input_dim=int(config.model_dim),
            compressed_dim=covariance_dim,
        ).to(device)
    pooler.load_state_dict(head_state["pooler_state"])
    pooler.eval()
    classifier = MLPClassifier(
        input_dim=int(head_state["input_dim"]),
        hidden_dim=int(head_state["hparams"]["hidden_dim"]),
        dropout=float(head_state["hparams"]["dropout"]),
    ).to(device)
    classifier.load_state_dict(head_state["classifier_state"])
    classifier.eval()

    collator = MsnlibCollator(
        num_peaks=int(config.get("num_peaks", 64)),
        max_precursor_mz=float(config.get("max_precursor_mz", DEFAULT_MAX_PRECURSOR_MZ)),
        min_peak_intensity=float(config.get("min_peak_intensity", DEFAULT_MIN_PEAK_INTENSITY)),
        peak_drop_min_intensity=float(
            config.get(
                "peak_drop_min_intensity",
                config.get("min_peak_intensity", DEFAULT_MIN_PEAK_INTENSITY),
            )
        ),
        peak_ordering=str(config.get("peak_ordering", "mz")),
        use_precursor_token=bool(config.get("use_precursor_token", False)),
        precursor_peak_exclusion_window_da=float(
            config.get("precursor_peak_exclusion_window_da", 0.0)
        ),
    )
    loader = DataLoader(
        MsnlibDataset(data),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=collator,
    )
    use_autocast = device.type == "cuda"
    logits, targets, row_indices = [], [], []
    for batch in loader:
        batch = _move_batch(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_autocast,
        ):
            if pooling == "single_pair_covariance":
                peak_embeddings, _, pair_embeddings = model.encoder.forward_with_block_outputs(
                    batch["peak_mz"],
                    batch["peak_intensity"],
                    valid_mask=batch["peak_valid_mask"],
                    precursor_mz=batch.get("precursor_mz", None),
                )
                peak_embeddings = _peak_tokens_only(peak_embeddings, batch["peak_valid_mask"])
                features = pooler(
                    peak_embeddings.float(),
                    batch["peak_valid_mask"].to(dtype=torch.bool),
                    pair_embeddings.float(),
                )
            else:
                encoded = model.encoder(
                    batch["peak_mz"],
                    batch["peak_intensity"],
                    valid_mask=batch["peak_valid_mask"],
                    precursor_mz=batch.get("precursor_mz", None),
                )
                peak_embeddings = _peak_tokens_only(encoded, batch["peak_valid_mask"])
                features = pooler(
                    peak_embeddings.float(),
                    batch["peak_valid_mask"].to(dtype=torch.bool),
                )
            batch_logits = classifier(features)
        logits.append(batch_logits.float().detach().cpu().numpy())
        targets.append(batch["label"].detach().cpu().numpy())
        row_indices.append(batch["row_idx"].detach().cpu().numpy())
    return (
        np.concatenate(targets, axis=0),
        np.concatenate(logits, axis=0),
        np.concatenate(row_indices, axis=0),
    )


def sigmoid(logits: np.ndarray) -> np.ndarray:
    probs = np.empty_like(logits, dtype=np.float64)
    positive = logits >= 0
    probs[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    probs[~positive] = exp_logits / (1.0 + exp_logits)
    return probs


def summarize_metrics(targets: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    probs = sigmoid(logits)
    pred = probs >= 0.5
    precision, recall, thresholds = precision_recall_curve(targets, probs)
    threshold_precision = precision[:-1]
    threshold_recall = recall[:-1]
    f1_values = (
        2.0
        * threshold_precision
        * threshold_recall
        / np.maximum(threshold_precision + threshold_recall, 1e-12)
    )
    best_idx = int(np.argmax(f1_values))
    high_precision = np.flatnonzero(threshold_precision >= 0.9)
    metrics = {
        "roc_auc": float(roc_auc_score(targets, probs)),
        "average_precision": float(average_precision_score(targets, probs)),
        "accuracy_at_0_5": float(accuracy_score(targets, pred)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(targets, pred)),
        "f1_at_0_5": float(f1_score(targets, pred, zero_division=0)),
        "precision_at_0_5": float(precision_score(targets, pred, zero_division=0)),
        "recall_at_0_5": float(recall_score(targets, pred, zero_division=0)),
        "best_f1": float(f1_values[best_idx]),
        "best_f1_precision": float(threshold_precision[best_idx]),
        "best_f1_recall": float(threshold_recall[best_idx]),
        "best_f1_threshold": float(thresholds[best_idx]),
        "positive_rate": float(np.mean(targets)),
    }
    if len(high_precision):
        recall_idx = int(high_precision[np.argmax(threshold_recall[high_precision])])
        metrics["recall_at_precision_ge_0_9"] = float(threshold_recall[recall_idx])
        metrics["precision_at_precision_ge_0_9"] = float(threshold_precision[recall_idx])
        metrics["threshold_at_precision_ge_0_9"] = float(thresholds[recall_idx])
    return metrics


def write_outputs(
    *,
    output_prefix: Path,
    source_mgf: Path,
    config_path: Path,
    checkpoint_path: Path,
    head_state_path: Path,
    data: MsnlibData,
    targets: np.ndarray,
    logits: np.ndarray,
    row_indices: np.ndarray,
    head_state: dict[str, Any],
) -> dict[str, Any]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    probs = sigmoid(logits)
    precision, recall, thresholds = precision_recall_curve(targets, probs)
    metrics = summarize_metrics(targets, logits)
    train_cache_metrics = _metric_dict(targets, logits, "msnlib")
    summary = {
        "mode": head_state.get("mode", "probe"),
        "source_mgf": str(source_mgf),
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "head_state_path": str(head_state_path),
        "dataset": data.counts,
        "metrics": metrics,
        "metrics_prefixed": train_cache_metrics,
        "head": {
            "best_epoch": head_state["best_epoch"],
            "best_val": head_state["best_val"],
            "test": head_state.get("test", None),
            "hparams": head_state["hparams"],
            "pooling": head_state.get("pooling", "covariance"),
            "pair_dim": head_state.get("pair_dim", None),
            "focal_alpha": head_state["focal_alpha"],
            "focal_gamma": head_state["focal_gamma"],
            "embedding_cache_dir": head_state.get("embedding_cache_dir", ""),
            "finetune_cache_dir": head_state.get("finetune_cache_dir", ""),
            "train_size": head_state.get("train_size", None),
            "train_positive": head_state.get("train_positive", None),
            "val_size": head_state.get("val_size", None),
            "val_positive": head_state.get("val_positive", None),
        },
        "caveat": (
            "This is the reproducible public MCEBIO MGF subset after the notebook-style "
            "charge, precursor m/z, intensity amplitude, and high-peak filters. The "
            "paper notebook's smaller 17,052-row subset additionally depends on a local "
            "SIRIUS formula_identifications.tsv index that is not included in the public "
            "notebook or MGF."
        ),
    }
    summary_path = output_prefix.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    pr_path = output_prefix.with_suffix(".pr_curve.csv")
    with pr_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["precision", "recall", "threshold"])
        for i in range(len(precision)):
            threshold = thresholds[i] if i < len(thresholds) else ""
            writer.writerow([precision[i], recall[i], threshold])

    pred_path = output_prefix.with_suffix(".predictions.csv")
    with pred_path.open("w", newline="") as handle:
        fieldnames = [
            "eval_index",
            "source_index",
            "label",
            "score",
            "logit",
            "formula",
            "name",
            "smiles",
            "precursor_mz",
            "charge",
            "adduct",
            "spectype",
            "instrument_type",
            "num_peaks_raw",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for eval_i, row_i in enumerate(row_indices):
            row = data.rows[int(row_i)]
            writer.writerow(
                {
                    **{key: row[key] for key in fieldnames if key in row},
                    "eval_index": eval_i,
                    "label": int(targets[eval_i]),
                    "score": float(probs[eval_i]),
                    "logit": float(logits[eval_i]),
                }
            )

    fig_path = output_prefix.with_suffix(".pr_curve.png")
    fig, ax = plt.subplots(figsize=(6, 5), dpi=180)
    ax.plot(recall, precision, color="#2458a6", linewidth=2.0)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Our checkpoint on MCEBIO MGF (AP={metrics['average_precision']:.3f})")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)

    report_path = output_prefix.with_suffix(".report.md")
    report_path.write_text(
        "\n".join(
            [
                "# Our Checkpoint Fluorine Evaluation on Public MCEBIO MGF",
                "",
                f"- Source spectra: `{source_mgf}`",
                f"- Checkpoint: `{checkpoint_path}`",
                f"- Evaluation rows after filters: {data.counts['kept']}",
                f"- Positives after filters: {data.counts['positive']}",
                f"- Average precision: {metrics['average_precision']:.6f}",
                f"- ROC AUC: {metrics['roc_auc']:.6f}",
                f"- Best F1: {metrics['best_f1']:.6f} at threshold {metrics['best_f1_threshold']:.6f}",
                f"- F1 at 0.5: {metrics['f1_at_0_5']:.6f}",
                "",
                summary["caveat"],
            ]
        )
        + "\n"
    )
    return summary


def read_pr_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    recall, precision = [], []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            recall.append(float(row["recall"]))
            precision.append(float(row["precision"]))
    return np.asarray(recall), np.asarray(precision)


def write_dreams_comparison(
    *,
    output_prefix: Path,
    comparison_dir: Path,
    summary: dict[str, Any],
    previous_ours_prefix: Path | None,
) -> dict[str, Any]:
    fig, ax = plt.subplots(figsize=(6.4, 5.0), dpi=180)
    colors = {
        "ours_finetune": "#2458a6",
        "ours_probe": "#2563eb",
        "ours_previous_probe": "#16a34a",
        "dreams_embedding_model": "#111827",
        "dreams_ssl_backbone": "#b45309",
    }
    ours_key = "ours_finetune" if summary["mode"] == "finetune" else "ours_probe"
    ours_recall, ours_precision = read_pr_curve(output_prefix.with_suffix(".pr_curve.csv"))
    ours_label = "Ours full fine-tune" if summary["mode"] == "finetune" else "Ours probe"
    ax.plot(
        ours_recall,
        ours_precision,
        color=colors[ours_key],
        linewidth=2.2,
        label=f"{ours_label} (AP={summary['metrics']['average_precision']:.3f})",
    )

    previous_ours: dict[str, Any] | None = None
    if previous_ours_prefix is not None:
        previous_summary = json.loads(previous_ours_prefix.with_suffix(".summary.json").read_text())
        previous_recall, previous_precision = read_pr_curve(
            previous_ours_prefix.with_suffix(".pr_curve.csv")
        )
        ax.plot(
            previous_recall,
            previous_precision,
            color=colors["ours_previous_probe"],
            linewidth=2.0,
            linestyle="--",
            label=(
                "Ours previous probe "
                f"(AP={previous_summary['metrics']['average_precision']:.3f})"
            ),
        )
        previous_ours = {
            "mode": previous_summary.get("mode", "probe"),
            "summary": str(previous_ours_prefix.with_suffix(".summary.json")),
            "pr_curve": str(previous_ours_prefix.with_suffix(".pr_curve.csv")),
            "checkpoint_path": previous_summary["checkpoint_path"],
            "average_precision": previous_summary["metrics"]["average_precision"],
        }

    compared: list[dict[str, Any]] = []
    for summary_path in sorted(comparison_dir.glob("*.summary.json")):
        other = json.loads(summary_path.read_text())
        name = str(other["name"])
        pr_path = comparison_dir / f"{name}.pr_curve.csv"
        recall, precision = read_pr_curve(pr_path)
        ax.plot(
            recall,
            precision,
            color=colors.get(name, "#6b7280"),
            linewidth=2.0,
            label=f"{name.replace('_', ' ')} (AP={other['metrics']['average_precision']:.3f})",
        )
        compared.append(
            {
                "name": name,
                "summary": str(summary_path),
                "pr_curve": str(pr_path),
                "average_precision": other["metrics"]["average_precision"],
            }
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    ax.set_title("Fluorine PR Curves on Public MCEBIO MGF")
    fig.tight_layout()
    plot_path = output_prefix.with_suffix(".vs_dreams_actual_pr_curve.png")
    fig.savefig(plot_path)
    plt.close(fig)

    payload = {
        "comparison_plot": str(plot_path),
        "ours": {
            "mode": summary["mode"],
            "summary": str(output_prefix.with_suffix(".summary.json")),
            "pr_curve": str(output_prefix.with_suffix(".pr_curve.csv")),
            "average_precision": summary["metrics"]["average_precision"],
        },
        "previous_ours": previous_ours,
        "comparison_dir": str(comparison_dir),
        "compared": compared,
    }
    output_prefix.with_suffix(".vs_dreams_actual_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)
    )
    return payload


def resolve_checkpoint_path(checkpoint: Path | None, workdir: Path | None) -> Path:
    if checkpoint is not None:
        return checkpoint.expanduser().resolve()
    if workdir is not None:
        latest = latest_ckpt_path(workdir.expanduser().resolve())
        if latest is None:
            raise FileNotFoundError(f"no checkpoint found under {workdir}")
        return Path(latest).resolve()
    return Path("checkpoints/modal/no_fourier_embed_sentinel/step-01250000.pt").resolve()


def default_state_path(mode: str) -> Path:
    if mode == "finetune":
        return Path("results/fluorine_msnlib_latest_full_finetune_state.pt")
    return Path("results/fluorine_small_covariance_head_state.pt")


def default_output_prefix(mode: str) -> Path:
    if mode == "finetune":
        return Path("results/fluorine_msnlib_latest_full_finetune")
    return Path("results/fluorine_msnlib_ours_checkpoint")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the small Spectra checkpoint fluorine head on public MSnLib MCEBIO MGF."
    )
    parser.add_argument("--mode", choices=("probe", "finetune"), default="probe")
    parser.add_argument(
        "--pooling",
        choices=("covariance", "single_pair_covariance"),
        default="covariance",
    )
    parser.add_argument(
        "--mgf",
        type=Path,
        default=Path("data/massive_msv000094528/source/20240411_mcebio_library_pos_all_lib_MS2.mgf"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/wandb_pa645zxs_small.py"))
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--embedding-cache-dir",
        type=Path,
        default=Path("data/fluorine_small_encoder_embeddings"),
    )
    parser.add_argument(
        "--finetune-cache-dir",
        type=Path,
        default=Path("data/fluorine_detection_fine_tuned"),
    )
    parser.add_argument(
        "--head-state",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
    )
    parser.add_argument("--comparison-dir", type=Path, default=None)
    parser.add_argument(
        "--previous-ours-prefix",
        type=Path,
        default=Path("results/fluorine_msnlib_ours_checkpoint"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--finetune-model-lr", type=float, default=3e-6)
    parser.add_argument("--finetune-head-lr", type=float, default=1e-4)
    parser.add_argument("--finetune-weight-decay", type=float, default=0.0001)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--repo-id", default=HF_REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--subdir", default=HF_SUBDIR)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    args = parser.parse_args()
    if args.head_state is None:
        args.head_state = default_state_path(args.mode)
    if args.output_prefix is None:
        args.output_prefix = default_output_prefix(args.mode)
    if args.epochs is None:
        args.epochs = 15 if args.mode == "probe" else 3
    if args.patience is None:
        args.patience = args.epochs if args.mode == "probe" else 2
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint, args.workdir)
    config, model = _load_checkpoint_model(
        args.config.expanduser().resolve(),
        checkpoint_path,
        device,
    )
    if args.mode == "probe":
        head_state = train_or_load_head(
            head_state_path=args.head_state.expanduser().resolve(),
            embedding_cache_dir=args.embedding_cache_dir.expanduser().resolve(),
            config=config,
            device=device,
            batch_size=args.batch_size,
            seed=args.seed,
            epochs=args.epochs,
        )
    else:
        head_state = train_or_load_finetuned(
            state_path=args.head_state.expanduser().resolve(),
            model=model,
            config=config,
            config_path=args.config.expanduser().resolve(),
            checkpoint_path=checkpoint_path,
            cache_dir=args.finetune_cache_dir.expanduser().resolve(),
            device=device,
            batch_size=args.batch_size,
            seed=args.seed,
            epochs=args.epochs,
            patience=args.patience,
            model_learning_rate=args.finetune_model_lr,
            head_learning_rate=args.finetune_head_lr,
            weight_decay=args.finetune_weight_decay,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            repo_id=args.repo_id,
            revision=args.revision,
            subdir=args.subdir,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            pooling=args.pooling,
        )
    data = parse_mgf(args.mgf.expanduser().resolve())
    log.info(
        "parsed MGF total=%d kept=%d positive=%d",
        data.counts["total"],
        data.counts["kept"],
        data.counts["positive"],
    )
    targets, logits, row_indices = evaluate_msnlib(
        model=model,
        config=config,
        head_state=head_state,
        data=data,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    summary = write_outputs(
        output_prefix=args.output_prefix.expanduser().resolve(),
        source_mgf=args.mgf.expanduser().resolve(),
        config_path=args.config.expanduser().resolve(),
        checkpoint_path=checkpoint_path,
        head_state_path=args.head_state.expanduser().resolve(),
        data=data,
        targets=targets,
        logits=logits,
        row_indices=row_indices,
        head_state=head_state,
    )
    if args.comparison_dir is not None:
        comparison = write_dreams_comparison(
            output_prefix=args.output_prefix.expanduser().resolve(),
            comparison_dir=args.comparison_dir.expanduser().resolve(),
            summary=summary,
            previous_ours_prefix=(
                args.previous_ours_prefix.expanduser().resolve()
                if args.mode == "finetune"
                else None
            ),
        )
        summary["comparison"] = comparison
        args.output_prefix.expanduser().resolve().with_suffix(".summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True)
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
