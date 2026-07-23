from __future__ import annotations

import argparse
import copy
import csv
import io
import itertools
import json
import logging
from pathlib import Path
from typing import Any, Callable, Iterator, NamedTuple, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch._inductor.config as inductor_config
import torch.nn.functional as F
from ml_collections import config_dict
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
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from spectra_learning.config import load_config
from spectra_learning.data.spectra import (
    DEFAULT_MAX_PRECURSOR_MZ,
    DEFAULT_GROUPED_PEAK_ISOTOPE_CHARGES,
    DEFAULT_GROUPED_PEAK_SHOULDER_DA,
    DEFAULT_MIN_PEAK_INTENSITY,
    DEFAULT_PEAK_FILTERING,
)
from spectra_learning.models.factory import build_model_from_config
from spectra_learning.models.lora import (
    apply_lora_to_linear_modules,
    load_lora_state_dict,
    lora_parameters,
    lora_state_dict,
    module_state_to_cpu,
)
from spectra_learning.models.model import PeakSetJEPA
from spectra_learning.models.pooling import CovariancePool, SinglePairCovariancePool
from spectra_learning.models.spectrum_metadata import torch_spectrum_metadata_from_batch
from spectra_learning.data.murcko import (
    MCEBIO_MURCKO_PREPARED_SUBDIR,
    NIST_MURCKO_HF_REPO,
    NIST_MURCKO_PREPARED_SUBDIR,
    MurckoFluorineData as FluorineData,
    build_murcko_fluorine_data,
    build_murcko_fluorine_loader,
)
from spectra_learning.training.checkpointing import (
    latest_ckpt_path,
    load_pretrained_weights,
    load_resume_covariance_pooler_state,
    load_torch_checkpoint,
    save_torch_checkpoint,
)
from spectra_learning.training.distributed import (
    DistributedContext,
    barrier,
    cleanup_distributed,
    init_distributed_from_env,
    wrap_distributed_model,
)
from spectra_learning.training.runtime import build_grad_scaler, parse_autocast_dtype
from spectra_learning.training.storage import (
    StoragePath,
    is_remote_path,
    list_storage_files,
    local_cache_path,
    normalize_storage_path,
    read_text,
    storage_join,
    storage_name,
    storage_exists,
    storage_mkdir,
    storage_parent,
    storage_with_suffix,
    upload_local_file,
    write_text,
)


log = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")
inductor_config.triton.unique_kernel_names = True
inductor_config.fx_graph_cache = True
inductor_config.epilogue_fusion = True

HF_REPO_ID = NIST_MURCKO_HF_REPO
HF_TRAIN_SUBDIR = NIST_MURCKO_PREPARED_SUBDIR
HF_TEST_SUBDIR = MCEBIO_MURCKO_PREPARED_SUBDIR
LORA_ENCODER_TARGET_SUFFIXES = (
    "single_attention.wqkv",
    "single_attention.wo",
    "single_attention.qkv",
    "single_attention.o",
    "single_transition.fc1",
    "single_transition.fc2",
    "single_transition.fc3",
    "single_transition.w1",
    "single_transition.w2",
    "pair_transition.fc1",
    "pair_transition.fc2",
    "pair_transition.fc3",
    "pair_transition.w1",
    "pair_transition.w2",
)


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
    history: list[dict[str, Any]]


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


def _sigmoid_logits(logits: np.ndarray) -> np.ndarray:
    probs = np.empty_like(logits, dtype=np.float64)
    positive = logits >= 0
    probs[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    probs[~positive] = exp_logits / (1.0 + exp_logits)
    return probs


def _metric_dict(targets: np.ndarray, logits: np.ndarray, prefix: str) -> dict[str, float]:
    probs = _sigmoid_logits(logits)
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


def _precision_recall_curve_dict(
    targets: np.ndarray,
    logits: np.ndarray,
) -> dict[str, list[float]]:
    precision, recall, threshold = precision_recall_curve(
        targets,
        _sigmoid_logits(logits),
    )
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "threshold": threshold.tolist(),
    }


def _tensor_state_to_cpu(
    state: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor] | None:
    if state is None:
        return None
    return {key: value.detach().cpu().clone() for key, value in state.items()}


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
    num_workers: int = 0,
    distributed_world_size: int = 1,
    distributed_rank: int = 0,
) -> Any:
    return build_murcko_fluorine_loader(
        data,
        split,
        shuffle=shuffle,
        seed=seed,
        max_samples=max_samples,
        dreams_only=dreams_only,
        num_workers=num_workers,
        distributed_world_size=distributed_world_size,
        distributed_rank=distributed_rank,
    )


@torch.no_grad()
def _prediction_arrays(
    classifier: MLPClassifier,
    loader: Any,
    feature_fn: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    targets, logits, _ = _prediction_arrays_with_rows(
        classifier,
        loader,
        feature_fn,
        device=device,
    )
    return targets, logits


@torch.no_grad()
def _prediction_arrays_with_rows(
    classifier: MLPClassifier,
    loader: Any,
    feature_fn: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    classifier.eval()
    logits, targets, row_indices = [], [], []
    for batch in loader:
        batch = _move_batch(batch, device)
        features = feature_fn(batch)
        logits.append(classifier(features).detach().cpu().numpy())
        targets.append(batch["label"].detach().cpu().numpy())
        row_indices.append(batch["row_idx"].detach().cpu().numpy())
    return (
        np.concatenate(targets, axis=0),
        np.concatenate(logits, axis=0),
        np.concatenate(row_indices, axis=0),
    )


@torch.no_grad()
def _cache_frozen_features(
    loader: Any,
    feature_fn: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    *,
    device: torch.device,
    split: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    feature_chunks, label_chunks, row_chunks = [], [], []
    for batch in tqdm(loader, desc=f"cache {split} features", unit="batch", dynamic_ncols=True):
        batch = _move_batch(batch, device)
        feature_chunks.append(feature_fn(batch).detach().float().cpu())
        label_chunks.append(batch["label"].detach().float().cpu())
        row_chunks.append(batch["row_idx"].detach().cpu())
    return (
        torch.cat(feature_chunks, dim=0),
        torch.cat(label_chunks, dim=0),
        torch.cat(row_chunks, dim=0),
    )


def _cached_loader(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TensorDataset(features, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def _cached_prediction_arrays(
    classifier: MLPClassifier,
    loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    classifier.eval()
    logits, targets = [], []
    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        logits.append(classifier(features).detach().cpu().numpy())
        targets.append(labels.detach().cpu().numpy())
    return np.concatenate(targets, axis=0), np.concatenate(logits, axis=0)


@torch.no_grad()
def _evaluate(
    classifier: MLPClassifier,
    loader: Any,
    feature_fn: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    *,
    device: torch.device,
    prefix: str,
) -> dict[str, float]:
    targets, logits = _prediction_arrays(
        classifier,
        loader,
        feature_fn,
        device=device,
    )
    return _metric_dict(
        targets,
        logits,
        prefix,
    )


@torch.no_grad()
def _cached_evaluate(
    classifier: MLPClassifier,
    loader: DataLoader,
    *,
    device: torch.device,
    prefix: str,
) -> dict[str, float]:
    targets, logits = _cached_prediction_arrays(classifier, loader, device=device)
    return _metric_dict(targets, logits, prefix)


def _select_metric_value(metrics: dict[str, float], select_metric: str) -> float:
    return metrics[select_metric]


def _train_cached_trial(
    *,
    params: TrialParams,
    input_dim: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    focal_alpha: float,
    focal_gamma: float,
    select_metric: str,
    patience: int,
    progress_output_prefix: StoragePath | None = None,
) -> TrialResult:
    classifier = MLPClassifier(
        input_dim=input_dim,
        hidden_dim=params.hidden_dim,
        dropout=params.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=params.learning_rate,
        weight_decay=params.weight_decay,
    )

    best_value = -float("inf")
    best_epoch = 0
    best_val: dict[str, float] = {}
    best_classifier_state: dict[str, torch.Tensor] = {}
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    for epoch_idx in range(epochs):
        classifier.train()
        running_loss = 0.0
        seen = 0
        for features, labels in train_loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(features)
            loss = binary_focal_loss_with_logits(
                logits,
                labels,
                alpha=focal_alpha,
                gamma=focal_gamma,
            )
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu()) * int(labels.shape[0])
            seen += int(labels.shape[0])

        val_metrics = _cached_evaluate(
            classifier,
            val_loader,
            device=device,
            prefix="val",
        )
        history.append(
            {
                "epoch": epoch_idx + 1,
                "train_loss": running_loss / float(seen),
                "val": val_metrics,
            }
        )
        if progress_output_prefix is not None:
            write_training_history_outputs(
                output_prefix=progress_output_prefix,
                history=history,
            )
        current_value = _select_metric_value(val_metrics, select_metric)
        improved = current_value > best_value
        if improved:
            best_value = current_value
            best_epoch = epoch_idx + 1
            best_val = dict(val_metrics)
            best_classifier_state = copy.deepcopy(classifier.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        log.info(
            "cached trial hidden=%d lr=%.3g wd=%.3g dropout=%.2f epoch=%d/%d val_ap=%.4f val_auc=%.4f",
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
        pooler_state=None,
        history=history,
    )


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
    patience: int,
    build_feature_fn: Callable[[bool], tuple[Callable[[dict[str, torch.Tensor]], torch.Tensor], torch.nn.Module | None]],
    progress_output_prefix: StoragePath | None = None,
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

    best_value = -float("inf")
    best_epoch: int = 0
    best_val: dict[str, float] = {}
    best_classifier_state: dict[str, torch.Tensor] = {}
    best_pooler_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement: int = 0
    history: list[dict[str, Any]] = []
    for epoch_idx in range(epochs):
        classifier.train()
        if trainable_pooler is not None:
            trainable_pooler.train()
        running_loss = 0.0
        seen = 0
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
            running_loss += float(loss.detach().cpu()) * int(batch["label"].shape[0])
            seen += int(batch["label"].shape[0])

        val_metrics = _evaluate(
            classifier,
            val_loader,
            feature_fn,
            device=device,
            prefix="val",
        )
        history.append(
            {
                "epoch": epoch_idx + 1,
                "train_loss": running_loss / float(seen),
                "val": val_metrics,
            }
        )
        if progress_output_prefix is not None:
            write_training_history_outputs(
                output_prefix=progress_output_prefix,
                history=history,
            )
        current_value = _select_metric_value(val_metrics, select_metric)
        improved = current_value > best_value
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
        history=history,
    )


def _peak_tokens_only(
    token_embeddings: torch.Tensor,
    peak_valid_mask: torch.Tensor,
) -> torch.Tensor:
    return token_embeddings[:, : peak_valid_mask.shape[1]]


def _load_checkpoint_model(
    config_path: Path,
    checkpoint_path: StoragePath,
    device: torch.device,
) -> tuple[config_dict.ConfigDict, PeakSetJEPA]:
    config = load_config(config_path)
    model = build_model_from_config(config)
    load_pretrained_weights(model, checkpoint_path)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return config, model


def _build_checkpoint_feature_factory(
    *,
    model: PeakSetJEPA,
    config: config_dict.ConfigDict,
    checkpoint_path: StoragePath,
    device: torch.device,
    train_covariance_pooler: bool,
    covariance_dim: int | None,
    pooling: str,
) -> tuple[int, Callable[[bool], tuple[Callable[[dict[str, torch.Tensor]], torch.Tensor], torch.nn.Module | None]]]:
    compressed_dim = (
        covariance_dim
        if covariance_dim is not None
        else int(config.get("covariance_pooling_dim", 32))
    )
    if not train_covariance_pooler and pooling == "covariance":
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
            spectrum_metadata=torch_spectrum_metadata_from_batch(batch),
        )
        return _peak_tokens_only(embeddings, batch["peak_valid_mask"])

    @torch.no_grad()
    def encode_single_pair(
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        peak_embeddings, pair_embeddings = model.encoder.forward_with_pair(
            batch["peak_mz"],
            batch["peak_intensity"],
            valid_mask=batch["peak_valid_mask"],
            precursor_mz=batch.get("precursor_mz", None),
            spectrum_metadata=torch_spectrum_metadata_from_batch(batch),
        )
        return _peak_tokens_only(peak_embeddings, batch["peak_valid_mask"]), pair_embeddings

    def build_feature_fn(training: bool):
        if pooling == "single_pair_covariance":
            pooler = SinglePairCovariancePool(
                single_dim=int(config.model_dim),
                pair_dim=int(config.get("pairmixer_pair_dim", config.model_dim)),
                compressed_dim=compressed_dim,
            ).to(device)
            if not train_covariance_pooler:
                checkpoint = load_torch_checkpoint(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=True,
                )
                load_resume_covariance_pooler_state(
                    pooler,
                    checkpoint_path,
                    checkpoint,
                )
            trainable_pooler: torch.nn.Module | None = pooler if train_covariance_pooler else None
            if trainable_pooler is None:
                pooler.requires_grad_(False)

            def feature_fn(batch: dict[str, torch.Tensor]) -> torch.Tensor:
                peak_embeddings, pair_embeddings = encode_single_pair(batch)
                return pooler(
                    peak_embeddings.float(),
                    batch["peak_valid_mask"].to(dtype=torch.bool),
                    pair_embeddings.float(),
                )

            return feature_fn, trainable_pooler

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


def build_fluorine_data(
    *,
    config: Any,
    cache_dir: Path,
    batch_size: int,
    revision: str,
    distributed_world_size: int = 1,
    distributed_rank: int = 0,
    distributed_local_rank: int | None = None,
) -> FluorineData:
    return build_murcko_fluorine_data(
        cache_dir=cache_dir,
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
        peak_filtering=str(config.get("peak_filtering", DEFAULT_PEAK_FILTERING)),
        grouped_peak_shoulder_da=float(
            config.get("grouped_peak_shoulder_da", DEFAULT_GROUPED_PEAK_SHOULDER_DA)
        ),
        grouped_peak_isotope_charges=tuple(
            int(charge)
            for charge in config.get(
                "grouped_peak_isotope_charges",
                DEFAULT_GROUPED_PEAK_ISOTOPE_CHARGES,
            )
        ),
        repo_id=HF_REPO_ID,
        revision=revision,
        train_subdir=HF_TRAIN_SUBDIR,
        test_subdir=HF_TEST_SUBDIR,
        distributed_world_size=distributed_world_size,
        distributed_rank=distributed_rank,
        distributed_local_rank=distributed_local_rank,
    )


def _lora_config(
    *,
    rank: int,
    alpha: float,
    dropout: float,
) -> dict[str, Any]:
    return {
        "rank": int(rank),
        "alpha": float(alpha),
        "dropout": float(dropout),
        "target_suffixes": list(LORA_ENCODER_TARGET_SUFFIXES),
    }


def apply_fluorine_lora(
    encoder: torch.nn.Module,
    lora_config: dict[str, Any],
) -> tuple[str, ...]:
    return apply_lora_to_linear_modules(
        encoder,
        target_suffixes=tuple(lora_config["target_suffixes"]),
        rank=int(lora_config["rank"]),
        alpha=float(lora_config["alpha"]),
        dropout=float(lora_config["dropout"]),
    )


def load_fluorine_lora_state(
    encoder: torch.nn.Module,
    head_state: dict[str, Any],
) -> None:
    apply_fluorine_lora(encoder, head_state["lora_config"])
    load_lora_state_dict(encoder, head_state["lora_state"])


class FluorineFinetuneModule(torch.nn.Module):
    def __init__(
        self,
        *,
        encoder: torch.nn.Module,
        pooler: torch.nn.Module,
        classifier: MLPClassifier,
        pooling: str,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.pooler = pooler
        self.classifier = classifier
        self.pooling = pooling

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.pooling == "single_pair_covariance":
            peak_embeddings, pair_embeddings = self.encoder.forward_with_pair(
                batch["peak_mz"],
                batch["peak_intensity"],
                valid_mask=batch["peak_valid_mask"],
                precursor_mz=batch.get("precursor_mz", None),
                spectrum_metadata=torch_spectrum_metadata_from_batch(batch),
            )
            peak_embeddings = _peak_tokens_only(peak_embeddings, batch["peak_valid_mask"])
            features = self.pooler(
                peak_embeddings.float(),
                batch["peak_valid_mask"].to(dtype=torch.bool),
                pair_embeddings.float(),
            )
        else:
            encoded = self.encoder(
                batch["peak_mz"],
                batch["peak_intensity"],
                valid_mask=batch["peak_valid_mask"],
                precursor_mz=batch.get("precursor_mz", None),
                spectrum_metadata=torch_spectrum_metadata_from_batch(batch),
            )
            peak_embeddings = _peak_tokens_only(encoded, batch["peak_valid_mask"])
            features = self.pooler(
                peak_embeddings.float(),
                batch["peak_valid_mask"].to(dtype=torch.bool),
            )
        return self.classifier(features)


def compile_fluorine_module(module: torch.nn.Module, config: Any) -> None:
    compile_mode = str(config.get("compile_mode", "none"))
    if compile_mode.lower() == "none":
        return
    inductor_config.shape_padding = not compile_mode.startswith("max-autotune")
    module.compile(
        mode=compile_mode,
        fullgraph=False,
    )


def _autocast_dtype_name(dtype: torch.dtype | None) -> str:
    if dtype is None:
        return "none"
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "fp16"
    return str(dtype)


def _resolve_autocast_dtype(config: Any, raw: str | None) -> torch.dtype | None:
    value = config.get("autocast_dtype", "bf16") if raw is None else raw
    return parse_autocast_dtype(value)


@torch.no_grad()
def predict_finetuned(
    *,
    finetune_module: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
) -> tuple[np.ndarray, np.ndarray]:
    finetune_module.eval()
    use_autocast = device.type == "cuda" and autocast_dtype is not None
    logits, targets = [], []
    for batch in loader:
        batch = _move_batch(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype if autocast_dtype is not None else torch.bfloat16,
            enabled=use_autocast,
        ):
            batch_logits = finetune_module(batch)
        logits.append(batch_logits.float().detach().cpu().numpy())
        targets.append(batch["label"].detach().cpu().numpy())
    return np.concatenate(targets, axis=0), np.concatenate(logits, axis=0)


class _AdaptationLoaders(NamedTuple):
    data: FluorineData
    train: DataLoader
    val: DataLoader
    test: DataLoader | None


class _AdaptationTrainingResult(NamedTuple):
    best_epoch: int
    best_val: dict[str, float]
    best_state: dict[str, dict[str, torch.Tensor]]
    history: list[dict[str, Any]]


def _load_cached_adaptation_state(
    *,
    state_path: Path,
    device: torch.device,
    mode: str,
    config_path: Path,
    checkpoint_path: Path,
    pooling: str,
    requested_hparams: dict[str, Any],
    max_train_samples: int | None,
    max_val_samples: int | None,
) -> dict[str, Any] | None:
    if not state_path.exists():
        return None
    state = torch.load(state_path, map_location=device)
    state_hparams = state["hparams"]
    if (
        state["mode"] == mode
        and state["config_path"] == str(config_path)
        and state["checkpoint_path"] == str(checkpoint_path)
        and state["pooling"] == pooling
        and all(state_hparams.get(key) == value for key, value in requested_hparams.items())
        and state["max_train_samples"] == max_train_samples
        and state["max_val_samples"] == max_val_samples
    ):
        return state
    return None


def _build_adaptation_loaders(
    *,
    config: Any,
    cache_dir: Path,
    batch_size: int,
    revision: str,
    seed: int,
    max_train_samples: int | None,
    max_val_samples: int | None,
    max_test_samples: int | None,
    num_workers: int,
    eval_test_every_epoch: bool,
    distributed_world_size: int,
    distributed_rank: int,
    distributed_local_rank: int,
) -> _AdaptationLoaders:
    data = build_fluorine_data(
        config=config,
        cache_dir=cache_dir,
        batch_size=batch_size,
        revision=revision,
        distributed_world_size=distributed_world_size,
        distributed_rank=distributed_rank,
        distributed_local_rank=distributed_local_rank,
    )
    train_loader = _make_loader(
        data,
        "train",
        shuffle=True,
        seed=seed,
        max_samples=max_train_samples,
        dreams_only=False,
        num_workers=num_workers,
        distributed_world_size=distributed_world_size,
        distributed_rank=distributed_rank,
    )
    val_loader = _make_loader(
        data,
        "val",
        shuffle=False,
        seed=seed + 10_000,
        max_samples=max_val_samples,
        dreams_only=False,
        num_workers=num_workers,
    )
    test_loader = None
    if eval_test_every_epoch:
        test_loader = _make_loader(
            data,
            "test",
            shuffle=False,
            seed=seed + 20_000,
            max_samples=max_test_samples,
            dreams_only=False,
            num_workers=num_workers,
        )
    return _AdaptationLoaders(data, train_loader, val_loader, test_loader)


def _build_adaptation_pooler(
    *,
    config: Any,
    pooling: str,
    device: torch.device,
) -> tuple[torch.nn.Module, int, int]:
    covariance_dim = int(config.get("covariance_pooling_dim", 64))
    input_dim = covariance_dim * covariance_dim
    if pooling == "single_pair_covariance":
        pooler: torch.nn.Module = SinglePairCovariancePool(
            single_dim=int(config.model_dim),
            pair_dim=int(config.get("pairmixer_pair_dim", config.model_dim)),
            compressed_dim=covariance_dim,
        ).to(device)
    else:
        pooler = CovariancePool(
            input_dim=int(config.model_dim),
            compressed_dim=covariance_dim,
        ).to(device)
    return pooler, covariance_dim, input_dim


def _adaptation_state(
    *,
    metadata: dict[str, Any],
    result: _AdaptationTrainingResult,
    complete: bool,
) -> dict[str, Any]:
    state = {"mode": metadata["mode"], "complete": complete}
    state.update(metadata)
    state.update(result.best_state)
    state.update(
        best_epoch=int(result.best_epoch),
        best_val=result.best_val,
        test=None,
        history=result.history,
    )
    return state


def _train_adaptation_epoch(
    *,
    mode: str,
    epoch_idx: int,
    epochs: int,
    finetune_module: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    focal_alpha: float,
    focal_gamma: float,
    grad_scaler: Any,
    is_main: bool,
) -> float:
    sampler = getattr(train_loader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch_idx)
    finetune_module.train()
    running_loss = 0.0
    seen = 0
    pbar = tqdm(
        train_loader,
        desc=f"{mode} epoch {epoch_idx + 1}/{epochs}",
        unit="batch",
        dynamic_ncols=True,
        mininterval=5.0,
        disable=not is_main,
    )
    for batch in pbar:
        batch = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype if autocast_dtype is not None else torch.bfloat16,
            enabled=device.type == "cuda" and autocast_dtype is not None,
        ):
            logits = finetune_module(batch)
            loss = binary_focal_loss_with_logits(
                logits.float(),
                batch["label"],
                alpha=focal_alpha,
                gamma=focal_gamma,
            )
        if grad_scaler.is_enabled():
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            optimizer.step()
        batch_size = int(batch["label"].shape[0])
        running_loss += float(loss.detach().cpu()) * batch_size
        seen += batch_size
        if is_main:
            pbar.set_postfix(loss=f"{running_loss / float(seen):.5f}")
    return running_loss / float(seen)


def _evaluate_adaptation_epoch(
    *,
    epoch: int,
    train_loss: float,
    finetune_module: torch.nn.Module,
    val_loader: DataLoader,
    test_loader: DataLoader | None,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    eval_test_every_epoch: bool,
) -> tuple[dict[str, float], dict[str, Any]]:
    val_targets, val_logits = predict_finetuned(
        finetune_module=finetune_module,
        loader=val_loader,
        device=device,
        autocast_dtype=autocast_dtype,
    )
    val_metrics = _metric_dict(val_targets, val_logits, "val")
    history_row: dict[str, Any] = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val": val_metrics,
    }
    if eval_test_every_epoch:
        test_targets, test_logits = predict_finetuned(
            finetune_module=finetune_module,
            loader=cast(DataLoader, test_loader),
            device=device,
            autocast_dtype=autocast_dtype,
        )
        history_row["test"] = _metric_dict(test_targets, test_logits, "test")
    return val_metrics, history_row


def _run_adaptation_epochs(
    *,
    mode: str,
    state_path: Path,
    state_metadata: dict[str, Any],
    capture_best_state: Callable[[], dict[str, dict[str, torch.Tensor]]],
    finetune_module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loaders: _AdaptationLoaders,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    focal_alpha: float,
    focal_gamma: float,
    select_metric: str,
    epochs: int,
    patience: int,
    progress_output_prefix: StoragePath | None,
    eval_test_every_epoch: bool,
    is_main: bool,
) -> _AdaptationTrainingResult:
    best_value = -float("inf")
    best_epoch = 0
    best_val: dict[str, float] = {}
    best_state: dict[str, dict[str, torch.Tensor]] = {}
    history: list[dict[str, Any]] = []
    epochs_without_improvement = 0
    grad_scaler = build_grad_scaler(autocast_dtype, device)
    best_state_path = state_path.with_name(f"{state_path.stem}.best.pt")

    for epoch_idx in range(epochs):
        train_loss = _train_adaptation_epoch(
            mode=mode,
            epoch_idx=epoch_idx,
            epochs=epochs,
            finetune_module=finetune_module,
            train_loader=loaders.train,
            optimizer=optimizer,
            device=device,
            autocast_dtype=autocast_dtype,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            grad_scaler=grad_scaler,
            is_main=is_main,
        )
        val_metrics, history_row = _evaluate_adaptation_epoch(
            epoch=epoch_idx + 1,
            train_loss=train_loss,
            finetune_module=finetune_module,
            val_loader=loaders.val,
            test_loader=loaders.test,
            device=device,
            autocast_dtype=autocast_dtype,
            eval_test_every_epoch=eval_test_every_epoch,
        )
        history.append(history_row)
        if is_main and progress_output_prefix is not None:
            write_training_history_outputs(
                output_prefix=progress_output_prefix,
                history=history,
            )
        current_value = val_metrics[f"val/{select_metric}"]
        if current_value > best_value:
            best_value = current_value
            best_epoch = epoch_idx + 1
            best_val = dict(val_metrics)
            best_state = capture_best_state()
            epochs_without_improvement = 0
            if is_main:
                result = _AdaptationTrainingResult(
                    best_epoch,
                    best_val,
                    best_state,
                    history,
                )
                state_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    _adaptation_state(
                        metadata=state_metadata,
                        result=result,
                        complete=False,
                    ),
                    best_state_path,
                )
        else:
            epochs_without_improvement += 1
        if is_main:
            log.info(
                "%s epoch=%d/%d train_loss=%.5f val_ap=%.4f val_auc=%.4f",
                mode,
                epoch_idx + 1,
                epochs,
                train_loss,
                val_metrics["val/average_precision"],
                val_metrics["val/roc_auc"],
            )
        if epochs_without_improvement >= patience:
            break

    return _AdaptationTrainingResult(best_epoch, best_val, best_state, history)


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
    num_workers: int,
    seed: int,
    epochs: int,
    patience: int,
    model_learning_rate: float,
    pooler_learning_rate: float,
    head_learning_rate: float,
    weight_decay: float,
    focal_alpha: str,
    focal_gamma: float,
    autocast_dtype: torch.dtype | None,
    hidden_dim: int,
    dropout: float,
    revision: str,
    max_train_samples: int | None,
    max_val_samples: int | None,
    max_test_samples: int | None,
    pooling: str,
    select_metric: str,
    progress_output_prefix: StoragePath | None = None,
    eval_test_every_epoch: bool = False,
    distributed: DistributedContext | None = None,
) -> dict[str, Any]:
    distributed_world_size = distributed.world_size if distributed is not None else 1
    distributed_rank = distributed.rank if distributed is not None else 0
    distributed_local_rank = distributed.local_rank if distributed is not None else 0
    is_main = distributed is None or distributed.is_main
    requested_hparams = {
        "hidden_dim": int(hidden_dim),
        "dropout": float(dropout),
        "model_learning_rate": float(model_learning_rate),
        "pooler_learning_rate": float(pooler_learning_rate),
        "head_learning_rate": float(head_learning_rate),
        "weight_decay": float(weight_decay),
        "autocast_dtype": _autocast_dtype_name(autocast_dtype),
        "epochs": int(epochs),
        "patience": int(patience),
        "select_metric": select_metric,
        "focal_alpha": focal_alpha,
        "focal_gamma": float(focal_gamma),
    }
    state = _load_cached_adaptation_state(
        state_path=state_path,
        device=device,
        mode="finetune",
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        pooling=pooling,
        requested_hparams=requested_hparams,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
    )
    if state is not None:
        model.load_state_dict(state["model_state"])
        return state

    loaders = _build_adaptation_loaders(
        config=config,
        cache_dir=cache_dir,
        batch_size=batch_size,
        revision=revision,
        seed=seed,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        max_test_samples=max_test_samples,
        num_workers=num_workers,
        eval_test_every_epoch=eval_test_every_epoch,
        distributed_world_size=distributed_world_size,
        distributed_rank=distributed_rank,
        distributed_local_rank=distributed_local_rank,
    )
    pooler, covariance_dim, input_dim = _build_adaptation_pooler(
        config=config,
        pooling=pooling,
        device=device,
    )
    checkpoint = load_torch_checkpoint(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    load_resume_covariance_pooler_state(pooler, checkpoint_path, checkpoint)
    classifier = MLPClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    model.requires_grad_(True)
    finetune_module = FluorineFinetuneModule(
        encoder=model.encoder,
        pooler=pooler,
        classifier=classifier,
        pooling=pooling,
    ).to(device)
    compile_fluorine_module(finetune_module, config)
    if distributed is not None:
        finetune_module = wrap_distributed_model(
            finetune_module,
            distributed,
            static_graph=True,
        )
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.encoder.parameters(),
                "lr": model_learning_rate,
                "weight_decay": weight_decay,
            },
            {
                "params": pooler.parameters(),
                "lr": pooler_learning_rate,
                "weight_decay": weight_decay,
            },
            {
                "params": classifier.parameters(),
                "lr": head_learning_rate,
                "weight_decay": weight_decay,
            },
        ]
    )
    focal_alpha_value = (
        1.0
        - float(loaders.data.metadata["train_positive"])
        / float(loaders.data.metadata["train_size"])
        if focal_alpha == "auto"
        else float(focal_alpha)
    )
    state_metadata = {
        "mode": "finetune",
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "input_dim": int(input_dim),
        "covariance_dim": int(covariance_dim),
        "pooling": pooling,
        "pair_dim": int(config.get("pairmixer_pair_dim", config.model_dim)),
        "hparams": requested_hparams,
        "autocast_dtype": _autocast_dtype_name(autocast_dtype),
        "focal_alpha": focal_alpha_value,
        "focal_gamma": focal_gamma,
        "distributed_world_size": distributed_world_size,
        "train_size": int(loaders.data.metadata["train_size"]),
        "train_positive": int(loaders.data.metadata["train_positive"]),
        "val_size": int(loaders.data.metadata["val_size"]),
        "val_positive": int(loaders.data.metadata["val_positive"]),
        "max_train_samples": max_train_samples,
        "max_val_samples": max_val_samples,
    }

    def capture_best_state() -> dict[str, dict[str, torch.Tensor]]:
        return {
            "model_state": module_state_to_cpu(model),
            "pooler_state": module_state_to_cpu(pooler),
            "classifier_state": module_state_to_cpu(classifier),
        }

    result = _run_adaptation_epochs(
        mode="finetune",
        state_path=state_path,
        state_metadata=state_metadata,
        capture_best_state=capture_best_state,
        finetune_module=finetune_module,
        optimizer=optimizer,
        loaders=loaders,
        device=device,
        autocast_dtype=autocast_dtype,
        focal_alpha=focal_alpha_value,
        focal_gamma=focal_gamma,
        select_metric=select_metric,
        epochs=epochs,
        patience=patience,
        progress_output_prefix=progress_output_prefix,
        eval_test_every_epoch=eval_test_every_epoch,
        is_main=is_main,
    )
    model.load_state_dict(result.best_state["model_state"])
    pooler.load_state_dict(result.best_state["pooler_state"])
    classifier.load_state_dict(result.best_state["classifier_state"])

    state = _adaptation_state(metadata=state_metadata, result=result, complete=True)
    if is_main:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, state_path)
    return state


def train_or_load_lora(
    *,
    state_path: Path,
    model: torch.nn.Module,
    config: Any,
    config_path: Path,
    checkpoint_path: Path,
    cache_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
    epochs: int,
    patience: int,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float,
    lora_learning_rate: float,
    head_learning_rate: float,
    weight_decay: float,
    autocast_dtype: torch.dtype | None,
    hidden_dim: int,
    dropout: float,
    revision: str,
    max_train_samples: int | None,
    max_val_samples: int | None,
    max_test_samples: int | None,
    pooling: str,
    progress_output_prefix: StoragePath | None = None,
    eval_test_every_epoch: bool = False,
    distributed: DistributedContext | None = None,
) -> dict[str, Any]:
    distributed_world_size = distributed.world_size if distributed is not None else 1
    distributed_rank = distributed.rank if distributed is not None else 0
    distributed_local_rank = distributed.local_rank if distributed is not None else 0
    is_main = distributed is None or distributed.is_main
    requested_hparams = {
        "hidden_dim": int(hidden_dim),
        "dropout": float(dropout),
        "lora_rank": int(lora_rank),
        "lora_alpha": float(lora_alpha),
        "lora_dropout": float(lora_dropout),
        "lora_learning_rate": float(lora_learning_rate),
        "head_learning_rate": float(head_learning_rate),
        "weight_decay": float(weight_decay),
        "autocast_dtype": _autocast_dtype_name(autocast_dtype),
        "epochs": int(epochs),
        "patience": int(patience),
    }
    state = _load_cached_adaptation_state(
        state_path=state_path,
        device=device,
        mode="lora",
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        pooling=pooling,
        requested_hparams=requested_hparams,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
    )
    if state is not None:
        model.encoder.requires_grad_(False)
        load_fluorine_lora_state(model.encoder, state)
        return state

    loaders = _build_adaptation_loaders(
        config=config,
        cache_dir=cache_dir,
        batch_size=batch_size,
        revision=revision,
        seed=seed,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        max_test_samples=max_test_samples,
        num_workers=num_workers,
        eval_test_every_epoch=eval_test_every_epoch,
        distributed_world_size=distributed_world_size,
        distributed_rank=distributed_rank,
        distributed_local_rank=distributed_local_rank,
    )
    pooler, covariance_dim, input_dim = _build_adaptation_pooler(
        config=config,
        pooling=pooling,
        device=device,
    )
    classifier = MLPClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    model.encoder.requires_grad_(False)
    lora_config = _lora_config(
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
    )
    applied_modules = apply_fluorine_lora(model.encoder, lora_config)
    lora_config = {**lora_config, "applied_modules": list(applied_modules)}
    finetune_module = FluorineFinetuneModule(
        encoder=model.encoder,
        pooler=pooler,
        classifier=classifier,
        pooling=pooling,
    ).to(device)
    compile_fluorine_module(finetune_module, config)
    if distributed is not None:
        finetune_module = wrap_distributed_model(
            finetune_module,
            distributed,
            static_graph=True,
        )
    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(lora_parameters(model.encoder)),
                "lr": lora_learning_rate,
                "weight_decay": weight_decay,
            },
            {
                "params": list(pooler.parameters()) + list(classifier.parameters()),
                "lr": head_learning_rate,
                "weight_decay": weight_decay,
            },
        ]
    )
    focal_alpha = 1.0 - float(loaders.data.metadata["train_positive"]) / float(
        loaders.data.metadata["train_size"]
    )
    focal_gamma = 2.0
    state_metadata = {
        "mode": "lora",
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "input_dim": int(input_dim),
        "covariance_dim": int(covariance_dim),
        "pooling": pooling,
        "pair_dim": int(config.get("pairmixer_pair_dim", config.model_dim)),
        "lora_config": lora_config,
        "hparams": requested_hparams,
        "autocast_dtype": _autocast_dtype_name(autocast_dtype),
        "focal_alpha": focal_alpha,
        "focal_gamma": focal_gamma,
        "train_size": int(loaders.data.metadata["train_size"]),
        "train_positive": int(loaders.data.metadata["train_positive"]),
        "val_size": int(loaders.data.metadata["val_size"]),
        "val_positive": int(loaders.data.metadata["val_positive"]),
        "max_train_samples": max_train_samples,
        "max_val_samples": max_val_samples,
    }

    def capture_best_state() -> dict[str, dict[str, torch.Tensor]]:
        return {
            "lora_state": lora_state_dict(model.encoder),
            "pooler_state": module_state_to_cpu(pooler),
            "classifier_state": module_state_to_cpu(classifier),
        }

    result = _run_adaptation_epochs(
        mode="lora",
        state_path=state_path,
        state_metadata=state_metadata,
        capture_best_state=capture_best_state,
        finetune_module=finetune_module,
        optimizer=optimizer,
        loaders=loaders,
        device=device,
        autocast_dtype=autocast_dtype,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        select_metric="average_precision",
        epochs=epochs,
        patience=patience,
        progress_output_prefix=progress_output_prefix,
        eval_test_every_epoch=eval_test_every_epoch,
        is_main=is_main,
    )
    load_lora_state_dict(model.encoder, result.best_state["lora_state"])
    pooler.load_state_dict(result.best_state["pooler_state"])
    classifier.load_state_dict(result.best_state["classifier_state"])

    state = _adaptation_state(metadata=state_metadata, result=result, complete=True)
    if is_main:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, state_path)
    return state


@torch.no_grad()
def evaluate_fluorine_test_split(
    *,
    model: torch.nn.Module,
    config: Any,
    head_state: dict[str, Any],
    data: FluorineData,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_test_samples: int | None,
    autocast_dtype: torch.dtype | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    covariance_dim = int(head_state["covariance_dim"])
    pooling = str(head_state["pooling"])
    if head_state["mode"] == "lora":
        model.encoder.requires_grad_(False)
        load_fluorine_lora_state(model.encoder, head_state)
    if pooling == "single_pair_covariance":
        pooler = SinglePairCovariancePool(
            single_dim=int(config.model_dim),
            pair_dim=int(head_state["pair_dim"]),
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
    finetune_module = FluorineFinetuneModule(
        encoder=model.encoder,
        pooler=pooler,
        classifier=classifier,
        pooling=pooling,
    ).to(device)
    finetune_module.eval()

    eval_data = data._replace(batch_size=batch_size)
    loader = build_murcko_fluorine_loader(
        eval_data,
        "test",
        shuffle=False,
        seed=0,
        max_samples=max_test_samples,
        dreams_only=False,
        num_workers=num_workers,
    )
    use_autocast = device.type == "cuda" and autocast_dtype is not None
    logits, targets, row_indices = [], [], []
    for batch in loader:
        batch = _move_batch(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype if autocast_dtype is not None else torch.bfloat16,
            enabled=use_autocast,
        ):
            batch_logits = finetune_module(batch)
        logits.append(batch_logits.float().detach().cpu().numpy())
        targets.append(batch["label"].detach().cpu().numpy())
        row_indices.append(batch["row_idx"].detach().cpu().numpy())
    return (
        np.concatenate(targets, axis=0),
        np.concatenate(logits, axis=0),
        np.concatenate(row_indices, axis=0),
    )


def summarize_metrics(targets: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    probs = _sigmoid_logits(logits)
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


def _write_figure(path: StoragePath, fig: plt.Figure) -> None:
    if is_remote_path(path):
        local_path = local_cache_path(path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(local_path)
        upload_local_file(local_path, path)
        return
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)


def _write_csv_text(path: StoragePath, rows: list[list[Any]]) -> None:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerows(rows)
    write_text(path, buffer.getvalue())


def write_training_curve_plot(
    *,
    output_prefix: StoragePath,
    history: list[dict[str, Any]],
) -> str | None:
    if not history:
        return None
    epochs = [int(row["epoch"]) for row in history]
    train_loss = [float(row["train_loss"]) for row in history]
    val_ap = [float(row["val"]["val/average_precision"]) for row in history]
    val_auc = [float(row["val"]["val/roc_auc"]) for row in history]
    test_epochs = [int(row["epoch"]) for row in history if "test" in row]
    test_ap = [
        float(row["test"]["test/average_precision"])
        for row in history
        if "test" in row
    ]
    test_auc = [
        float(row["test"]["test/roc_auc"])
        for row in history
        if "test" in row
    ]

    fig, loss_ax = plt.subplots(figsize=(7, 4.6), dpi=180)
    metric_ax = loss_ax.twinx()
    loss_ax.plot(epochs, train_loss, color="#2458a6", marker="o", label="Train loss")
    metric_ax.plot(epochs, val_ap, color="#16a34a", marker="o", label="Val AP")
    metric_ax.plot(epochs, val_auc, color="#b45309", marker="o", label="Val ROC AUC")
    if test_epochs:
        metric_ax.plot(
            test_epochs,
            test_ap,
            color="#7c3aed",
            marker="s",
            linestyle="--",
            label="Test AP",
        )
        metric_ax.plot(
            test_epochs,
            test_auc,
            color="#dc2626",
            marker="s",
            linestyle="--",
            label="Test ROC AUC",
        )
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Train focal loss")
    metric_ax.set_ylabel("Validation metric")
    loss_ax.grid(True, alpha=0.25)
    loss_handles, loss_labels = loss_ax.get_legend_handles_labels()
    metric_handles, metric_labels = metric_ax.get_legend_handles_labels()
    loss_ax.legend(
        loss_handles + metric_handles,
        loss_labels + metric_labels,
        loc="best",
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout()
    plot_path = storage_with_suffix(output_prefix, ".training_curves.png")
    _write_figure(plot_path, fig)
    plt.close(fig)
    return str(plot_path)


def write_training_history_outputs(
    *,
    output_prefix: StoragePath,
    history: list[dict[str, Any]],
) -> dict[str, str | None]:
    history_path = storage_with_suffix(output_prefix, ".history.json")
    write_text(history_path, json.dumps(history, indent=2, sort_keys=True))
    return {
        "history": str(history_path),
        "training_curve_plot": write_training_curve_plot(
            output_prefix=output_prefix,
            history=history,
        ),
    }


def write_standard_fluorine_outputs(
    *,
    output_prefix: StoragePath,
    config_path: Path,
    checkpoint_path: Path,
    head_state_path: StoragePath,
    data: FluorineData,
    targets: np.ndarray,
    logits: np.ndarray,
    row_indices: np.ndarray,
    head_state: dict[str, Any],
) -> dict[str, Any]:
    storage_mkdir(storage_parent(output_prefix))
    probs = _sigmoid_logits(logits)
    precision, recall, thresholds = precision_recall_curve(targets, probs)
    metrics = summarize_metrics(targets, logits)
    history = list(head_state["history"])
    training_curve_plot = write_training_curve_plot(
        output_prefix=output_prefix,
        history=history,
    )
    summary = {
        "mode": head_state["mode"],
        "dataset": {
            "repo_id": HF_REPO_ID,
            "train_subdir": HF_TRAIN_SUBDIR,
            "test_subdir": HF_TEST_SUBDIR,
            "test_size": int(data.metadata["test_size"]),
            "test_positive": int(data.metadata["test_positive"]),
        },
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "head_state_path": str(head_state_path),
        "metrics": metrics,
        "head": {
            "best_epoch": head_state["best_epoch"],
            "best_val": head_state["best_val"],
            "test": head_state["test"],
            "hparams": head_state["hparams"],
            "pooling": head_state["pooling"],
            "pair_dim": head_state["pair_dim"],
            "focal_alpha": head_state["focal_alpha"],
            "focal_gamma": head_state["focal_gamma"],
            "train_size": head_state["train_size"],
            "train_positive": head_state["train_positive"],
            "val_size": head_state["val_size"],
            "val_positive": head_state["val_positive"],
            "history": history,
            "training_curve_plot": training_curve_plot,
        },
    }
    summary_path = storage_with_suffix(output_prefix, ".summary.json")
    write_text(summary_path, json.dumps(summary, indent=2, sort_keys=True))

    pr_path = storage_with_suffix(output_prefix, ".pr_curve.csv")
    pr_rows = [["precision", "recall", "threshold"]]
    for i in range(len(precision)):
        threshold = thresholds[i] if i < len(thresholds) else ""
        pr_rows.append([precision[i], recall[i], threshold])
    _write_csv_text(pr_path, pr_rows)

    pred_path = storage_with_suffix(output_prefix, ".predictions.csv")
    pred_buffer = io.StringIO()
    pred_writer = csv.DictWriter(
        pred_buffer,
        fieldnames=["eval_index", "row_idx", "label", "score", "logit"],
    )
    pred_writer.writeheader()
    for eval_i, row_i in enumerate(row_indices):
        pred_writer.writerow(
            {
                "eval_index": eval_i,
                "row_idx": int(row_i),
                "label": int(targets[eval_i]),
                "score": float(probs[eval_i]),
                "logit": float(logits[eval_i]),
            }
        )
    write_text(pred_path, pred_buffer.getvalue())

    fig_path = storage_with_suffix(output_prefix, ".pr_curve.png")
    fig, ax = plt.subplots(figsize=(6, 5), dpi=180)
    ax.plot(recall, precision, color="#2458a6", linewidth=2.0)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Fluorine on MCEBIO Murcko test (AP={metrics['average_precision']:.3f})")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _write_figure(fig_path, fig)
    plt.close(fig)

    report_path = storage_with_suffix(output_prefix, ".report.md")
    write_text(
        report_path,
        "\n".join(
            [
                "# Fluorine Evaluation on MCEBIO Murcko Test",
                "",
                f"- Dataset: `{HF_REPO_ID}`",
                f"- Train/validation source: `{HF_TRAIN_SUBDIR}`",
                f"- Test source: `{HF_TEST_SUBDIR}`",
                f"- Checkpoint: `{checkpoint_path}`",
                f"- Evaluation rows: {int(data.metadata['test_size'])}",
                f"- Positives: {int(data.metadata['test_positive'])}",
                f"- Average precision: {metrics['average_precision']:.6f}",
                f"- ROC AUC: {metrics['roc_auc']:.6f}",
                f"- Best F1: {metrics['best_f1']:.6f} at threshold {metrics['best_f1_threshold']:.6f}",
                f"- F1 at 0.5: {metrics['f1_at_0_5']:.6f}",
            ]
        )
        + "\n"
    )
    return summary


def read_pr_curve(path: StoragePath) -> tuple[np.ndarray, np.ndarray]:
    recall, precision = [], []
    for row in csv.DictReader(io.StringIO(read_text(path))):
        recall.append(float(row["recall"]))
        precision.append(float(row["precision"]))
    return np.asarray(recall), np.asarray(precision)


def _pr_curve_summary_path(pr_curve_path: StoragePath) -> StoragePath:
    raw = str(pr_curve_path)
    return f"{raw.removesuffix('.pr_curve.csv')}.summary.json"


def _pr_curve_label(summary: dict[str, Any], pr_curve_path: StoragePath) -> str:
    name = str(summary.get("name") or Path(str(pr_curve_path)).name.removesuffix(".pr_curve.csv"))
    mode = str(summary.get("mode", "")).replace("_", " ")
    ap = float(summary["metrics"]["average_precision"])
    test_size = summary.get("dataset", {}).get("test_size")
    label = name.replace("_", " ")
    if mode:
        label = f"{label} [{mode}]"
    if test_size is not None:
        label = f"{label} n={int(test_size)}"
    return f"{label} AP={ap:.3f}"


def write_all_pr_curve_comparison(
    *,
    output_prefix: StoragePath,
    curve_dirs: list[StoragePath],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for directory in curve_dirs:
        for item in list_storage_files(directory):
            if not item.name.endswith(".pr_curve.csv"):
                continue
            pr_curve_path = item.path
            key = str(pr_curve_path)
            if key in seen:
                continue
            seen.add(key)
            summary_path = _pr_curve_summary_path(pr_curve_path)
            if not storage_exists(summary_path):
                continue
            summary = json.loads(read_text(summary_path))
            metrics = summary.get("metrics", {})
            if "average_precision" not in metrics:
                continue
            recall, precision = read_pr_curve(pr_curve_path)
            entries.append(
                {
                    "name": str(summary.get("name") or item.name.removesuffix(".pr_curve.csv")),
                    "mode": summary.get("mode"),
                    "summary": str(summary_path),
                    "pr_curve": str(pr_curve_path),
                    "average_precision": float(metrics["average_precision"]),
                    "roc_auc": metrics.get("roc_auc"),
                    "positive_rate": metrics.get("positive_rate"),
                    "test_size": summary.get("dataset", {}).get("test_size"),
                    "recall": recall,
                    "precision": precision,
                    "label": _pr_curve_label(summary, pr_curve_path),
                }
            )

    entries.sort(key=lambda row: row["average_precision"], reverse=True)
    fig, ax = plt.subplots(figsize=(8.4, 6.0), dpi=180)
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, max(len(entries), 1)))
    current_pr_curve = str(storage_with_suffix(output_prefix, ".pr_curve.csv"))
    for idx, entry in enumerate(entries):
        is_current = entry["pr_curve"] == current_pr_curve
        ax.plot(
            entry["recall"],
            entry["precision"],
            color=colors[idx],
            linewidth=2.8 if is_current else 1.6,
            alpha=1.0 if is_current else 0.78,
            label=entry["label"],
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", frameon=False, fontsize=7)
    ax.set_title("All Fluorine PR Curves on MCEBIO")
    fig.tight_layout()
    plot_path = storage_with_suffix(output_prefix, ".all_pr_curves.png")
    _write_figure(plot_path, fig)
    plt.close(fig)

    payload = {
        "comparison_plot": str(plot_path),
        "curve_dirs": [str(directory) for directory in curve_dirs],
        "curves": [
            {
                key: value
                for key, value in entry.items()
                if key not in {"recall", "precision", "label"}
            }
            for entry in entries
        ],
    }
    write_text(
        storage_with_suffix(output_prefix, ".all_pr_curves.summary.json"),
        json.dumps(payload, indent=2, sort_keys=True),
    )
    return payload


def write_dreams_comparison(
    *,
    output_prefix: StoragePath,
    comparison_dir: Path,
    summary: dict[str, Any],
    previous_ours_prefix: Path | None,
) -> dict[str, Any]:
    fig, ax = plt.subplots(figsize=(6.4, 5.0), dpi=180)
    colors = {
        "ours_finetune": "#2458a6",
        "ours_lora": "#7c3aed",
        "ours_probe": "#2563eb",
        "ours_previous_probe": "#16a34a",
        "dreams_embedding_model": "#111827",
        "dreams_ssl_backbone": "#b45309",
    }
    ours_key = {
        "finetune": "ours_finetune",
        "lora": "ours_lora",
    }.get(summary["mode"], "ours_probe")
    ours_recall, ours_precision = read_pr_curve(storage_with_suffix(output_prefix, ".pr_curve.csv"))
    ours_label = {
        "finetune": "Ours full fine-tune",
        "lora": "Ours LoRA",
    }.get(summary["mode"], "Ours probe")
    ax.plot(
        ours_recall,
        ours_precision,
        color=colors[ours_key],
        linewidth=2.2,
        label=f"{ours_label} (AP={summary['metrics']['average_precision']:.3f})",
    )

    previous_ours: dict[str, Any] | None = None
    if previous_ours_prefix is not None:
        previous_summary = json.loads(read_text(previous_ours_prefix.with_suffix(".summary.json")))
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
    ax.set_title("Fluorine PR Curves on MCEBIO Murcko Test")
    fig.tight_layout()
    plot_path = storage_with_suffix(output_prefix, ".vs_dreams_actual_pr_curve.png")
    _write_figure(plot_path, fig)
    plt.close(fig)

    payload = {
        "comparison_plot": str(plot_path),
        "ours": {
            "mode": summary["mode"],
            "summary": str(storage_with_suffix(output_prefix, ".summary.json")),
            "pr_curve": str(storage_with_suffix(output_prefix, ".pr_curve.csv")),
            "average_precision": summary["metrics"]["average_precision"],
        },
        "previous_ours": previous_ours,
        "comparison_dir": str(comparison_dir),
        "compared": compared,
    }
    write_text(
        storage_with_suffix(output_prefix, ".vs_dreams_actual_summary.json"),
        json.dumps(payload, indent=2, sort_keys=True),
    )
    return payload


def resolve_checkpoint_path(
    checkpoint: StoragePath | None,
    workdir: StoragePath | None,
) -> StoragePath:
    if checkpoint is not None:
        return normalize_storage_path(checkpoint)
    if workdir is not None:
        latest = latest_ckpt_path(normalize_storage_path(workdir))
        if latest is None:
            raise FileNotFoundError(f"no checkpoint found under {workdir}")
        return normalize_storage_path(latest)
    raise ValueError("checkpoint or workdir is required")


def default_state_path(mode: str) -> Path:
    if mode == "finetune":
        return Path("results/fluorine_detection_full_finetune_state.pt")
    if mode == "lora":
        return Path("results/fluorine_detection_lora_state.pt")
    return Path("results/fluorine_detection_probe_state.pt")


def default_output_prefix(mode: str) -> Path:
    if mode == "finetune":
        return Path("results/fluorine_detection_full_finetune")
    if mode == "lora":
        return Path("results/fluorine_detection_lora")
    return Path("results/fluorine_detection_probe")


def _resolve_jax_checkpoint_dir_and_step(
    checkpoint: StoragePath | None,
    workdir: StoragePath | None,
    checkpoint_step: int | None,
) -> tuple[StoragePath, int | None]:
    if checkpoint is not None:
        checkpoint_path = normalize_storage_path(checkpoint)
        name = storage_name(checkpoint_path)
        if name.isdigit():
            step = int(name)
            if checkpoint_step is not None and checkpoint_step != step:
                raise ValueError(
                    f"--jax-checkpoint-step={checkpoint_step} does not match checkpoint path step {step}"
                )
            return storage_parent(storage_parent(checkpoint_path)), step
        return checkpoint_path, checkpoint_step
    if workdir is None:
        raise ValueError("checkpoint or workdir is required")
    return storage_join(normalize_storage_path(workdir), "checkpoints"), checkpoint_step


def _jax_tree_to_numpy(tree: Any) -> Any:
    import jax
    import numpy as np

    return jax.tree.map(lambda value: np.asarray(jax.device_get(value)), tree)


class _FluorineProbePaths(NamedTuple):
    cache_dir: Path
    output_prefix: StoragePath
    head_state_path: StoragePath


def _resolve_fluorine_probe_paths(args: argparse.Namespace) -> _FluorineProbePaths:
    cache_dir = args.cache_dir.expanduser().resolve()
    output_prefix = (
        normalize_storage_path(args.output_prefix)
        if getattr(args, "output_prefix", None) is not None
        else default_output_prefix("probe").resolve()
    )
    head_state_path = (
        normalize_storage_path(args.output_state)
        if getattr(args, "output_state", None)
        else default_state_path("probe").resolve()
    )
    return _FluorineProbePaths(cache_dir, output_prefix, head_state_path)


def _build_fluorine_probe_data(
    *,
    args: argparse.Namespace,
    checkpoint_config: Any,
    cache_dir: Path,
) -> FluorineData:
    return build_murcko_fluorine_data(
        cache_dir=cache_dir,
        batch_size=int(args.batch_size),
        num_peaks=int(
            args.num_peaks
            if args.num_peaks is not None
            else checkpoint_config.get("num_peaks", 60)
        ),
        max_precursor_mz=float(
            checkpoint_config.get("max_precursor_mz", DEFAULT_MAX_PRECURSOR_MZ)
        ),
        min_peak_intensity=float(
            checkpoint_config.get(
                "min_peak_intensity",
                DEFAULT_MIN_PEAK_INTENSITY,
            )
        ),
        peak_drop_min_intensity=float(
            checkpoint_config.get(
                "peak_drop_min_intensity",
                checkpoint_config.get(
                    "min_peak_intensity",
                    DEFAULT_MIN_PEAK_INTENSITY,
                ),
            )
        ),
        peak_ordering=str(
            args.peak_ordering
            if args.peak_ordering
            else checkpoint_config.get("peak_ordering", "intensity")
        ),
        precursor_peak_exclusion_window_da=float(
            checkpoint_config.get("precursor_peak_exclusion_window_da", 0.0)
        ),
        peak_filtering=str(
            checkpoint_config.get("peak_filtering", DEFAULT_PEAK_FILTERING)
        ),
        grouped_peak_shoulder_da=float(
            checkpoint_config.get(
                "grouped_peak_shoulder_da",
                DEFAULT_GROUPED_PEAK_SHOULDER_DA,
            )
        ),
        grouped_peak_isotope_charges=tuple(
            int(charge)
            for charge in checkpoint_config.get(
                "grouped_peak_isotope_charges",
                DEFAULT_GROUPED_PEAK_ISOTOPE_CHARGES,
            )
        ),
        repo_id=HF_REPO_ID,
        revision=args.revision,
        train_subdir=HF_TRAIN_SUBDIR,
        test_subdir=HF_TEST_SUBDIR,
    )


def _write_fluorine_probe_artifacts(
    *,
    args: argparse.Namespace,
    payload: dict[str, Any],
    head_state: dict[str, Any],
    paths: _FluorineProbePaths,
    config_path: Path,
    checkpoint_path: StoragePath,
    data: FluorineData,
    targets: np.ndarray,
    logits: np.ndarray,
    row_indices: np.ndarray,
    summary_backend: str | None = None,
) -> dict[str, Any]:
    save_torch_checkpoint(head_state, paths.head_state_path)
    summary = write_standard_fluorine_outputs(
        output_prefix=paths.output_prefix,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        head_state_path=paths.head_state_path,
        data=data,
        targets=targets,
        logits=logits,
        row_indices=row_indices,
        head_state=head_state,
    )
    if summary_backend is not None:
        summary["backend"] = summary_backend
    if args.comparison_dir is not None:
        comparison = write_dreams_comparison(
            output_prefix=paths.output_prefix,
            comparison_dir=args.comparison_dir.expanduser().resolve(),
            summary=summary,
            previous_ours_prefix=None,
        )
        summary["comparison"] = comparison
    curve_dirs: list[StoragePath] = [storage_parent(paths.output_prefix)]
    if args.comparison_dir is not None:
        curve_dirs.append(args.comparison_dir.expanduser().resolve())
    summary["all_pr_curves"] = write_all_pr_curve_comparison(
        output_prefix=paths.output_prefix,
        curve_dirs=curve_dirs,
    )
    summary_path = storage_with_suffix(paths.output_prefix, ".summary.json")
    write_text(summary_path, json.dumps(summary, indent=2, sort_keys=True))
    payload["standard_outputs"] = {
        "summary": str(summary_path),
        "state": str(paths.head_state_path),
        "output_prefix": str(paths.output_prefix),
    }
    if args.output_json:
        write_text(
            normalize_storage_path(args.output_json),
            json.dumps(payload, indent=2, sort_keys=True),
        )
    return payload


class _JaxFluorineCheckpoint(NamedTuple):
    config_path: Path
    config: Any
    checkpoint_dir: StoragePath
    restore_step: int
    model: Any
    data_mesh: Any
    manager: Any


def _restore_jax_fluorine_checkpoint(
    args: argparse.Namespace,
) -> _JaxFluorineCheckpoint:
    import jax
    from flax import nnx

    from spectra_learning.models.model_jax import PeakSetJEPAJax
    from spectra_learning.models.settings import PeakSetJEPASettings
    from spectra_learning.probes.massspec.msg_probe_jax import (
        _full_visible_fastmixer_probe_model,
    )
    from spectra_learning.training.checkpointing_jax import (
        build_jax_checkpoint_manager,
        jax_training_checkpoint_metadata,
        restore_jax_training_state,
    )
    from spectra_learning.training.pretrain_jax import (
        _jax_data_mesh_for_device_count,
        _replicate_tree_on_data_mesh,
        init_pure_optax_train_state,
        jax_config_checkpoint_contract,
        prepare_jax_training_config,
    )

    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    prepare_jax_training_config(config)
    checkpoint_metadata = jax_training_checkpoint_metadata(
        "pretrain",
        jax_config_checkpoint_contract(config),
    )
    jax_device_count = jax.device_count()
    config.jax_mesh_devices = str(jax_device_count)
    checkpoint_dir, checkpoint_step = _resolve_jax_checkpoint_dir_and_step(
        args.checkpoint,
        getattr(args, "workdir", None),
        getattr(args, "jax_checkpoint_step", None),
    )
    manager = build_jax_checkpoint_manager(
        checkpoint_dir,
        max_to_keep=None,
        enable_async_checkpointing=False,
    )
    restore_step = (
        int(checkpoint_step) if checkpoint_step is not None else manager.latest_step()
    )
    if restore_step is None:
        raise FileNotFoundError(f"no JAX checkpoint found under {checkpoint_dir}")
    settings = PeakSetJEPASettings.from_config(config)
    model = PeakSetJEPAJax(settings, rngs=nnx.Rngs(int(config.seed)))
    graphdef, trainable_params, static_state, opt_state, _ = init_pure_optax_train_state(
        config,
        model,
        total_steps=int(config.training_max_steps),
    )
    data_mesh = _jax_data_mesh_for_device_count(jax_device_count)
    trainable_params = _replicate_tree_on_data_mesh(trainable_params, data_mesh)
    static_state = _replicate_tree_on_data_mesh(static_state, data_mesh)
    opt_state = _replicate_tree_on_data_mesh(opt_state, data_mesh)
    restored = restore_jax_training_state(
        manager,
        restore_step,
        {
            "trainable_params": trainable_params,
            "static_state": static_state,
            "opt_state": opt_state,
        },
        expected_metadata=checkpoint_metadata,
    )
    model = nnx.merge(
        graphdef,
        restored["trainable_params"],
        restored["static_state"],
    )
    model = _full_visible_fastmixer_probe_model(config, model)
    return _JaxFluorineCheckpoint(
        config_path,
        config,
        checkpoint_dir,
        int(restore_step),
        model,
        data_mesh,
        manager,
    )


def _extract_jax_fluorine_single_features(
    feature_model: Any,
    peak_mz: Any,
    peak_intensity: Any,
    peak_valid_mask: Any,
    precursor_mz: Any,
    spectrum_metadata: Any,
) -> Any:
    return feature_model.encoder(
        peak_mz,
        peak_intensity,
        valid_mask=peak_valid_mask,
        precursor_mz=precursor_mz,
        spectrum_metadata=spectrum_metadata,
    )


def _extract_jax_fluorine_pair_features(
    feature_model: Any,
    peak_mz: Any,
    peak_intensity: Any,
    peak_valid_mask: Any,
    precursor_mz: Any,
    spectrum_metadata: Any,
) -> tuple[Any, Any]:
    return feature_model.encoder.forward_with_pair(
        peak_mz,
        peak_intensity,
        valid_mask=peak_valid_mask,
        precursor_mz=precursor_mz,
        spectrum_metadata=spectrum_metadata,
    )


class _JaxFluorineRuntime(NamedTuple):
    args: argparse.Namespace
    config: Any
    model: Any
    data: FluorineData
    data_mesh: Any
    variant: str
    task_spec: Any
    extract_single: Callable[..., Any]
    extract_pair: Callable[..., Any]
    predict_step: Callable[..., Any]


def _make_jax_fluorine_runtime(
    *,
    args: argparse.Namespace,
    checkpoint: _JaxFluorineCheckpoint,
    data: FluorineData,
) -> _JaxFluorineRuntime:
    from flax import nnx

    from spectra_learning.probes.massspec.msg_settings import MsgProbeTaskSpec

    variant = args.pooling
    task_spec = MsgProbeTaskSpec(
        regression_tasks=(),
        maccs_bits=0,
        regression_means={},
        regression_stds={},
        binary_tasks=("fluorine",),
    )
    predict_step = _make_jax_fluorine_predict_step(variant, task_spec)
    return _JaxFluorineRuntime(
        args,
        checkpoint.config,
        checkpoint.model,
        data,
        checkpoint.data_mesh,
        variant,
        task_spec,
        nnx.jit(_extract_jax_fluorine_single_features),
        nnx.jit(_extract_jax_fluorine_pair_features),
        predict_step,
    )


def _iter_jax_fluorine_split(
    runtime: _JaxFluorineRuntime,
    split: str,
    *,
    shuffle: bool,
    seed: int,
    max_samples: int | None,
) -> Iterator[dict[str, Any]]:
    from spectra_learning.probes.massspec.msg_probe_jax import _probe_value_to_jax

    loader = build_murcko_fluorine_loader(
        runtime.data,
        split,
        shuffle=shuffle,
        seed=seed,
        max_samples=max_samples,
        dreams_only=False,
        num_workers=int(runtime.args.num_workers),
        output_format="numpy",
    )
    for batch in loader:
        yield {
            key: _probe_value_to_jax(value, data_mesh=None)
            for key, value in batch.items()
        }


def _extract_jax_fluorine_features(
    runtime: _JaxFluorineRuntime,
    batch: dict[str, Any],
) -> Any:
    from spectra_learning.models.spectrum_metadata import (
        jax_spectrum_metadata_from_batch,
    )
    from spectra_learning.probes.massspec.msg_probe_jax import (
        _feature_pair,
        _feature_single,
    )

    precursor_mz = batch.get("precursor_mz", None)
    spectrum_metadata = jax_spectrum_metadata_from_batch(batch)
    if runtime.variant == "single_pair_covariance":
        features = runtime.extract_pair(
            runtime.model,
            batch["peak_mz"],
            batch["peak_intensity"],
            batch["peak_valid_mask"],
            precursor_mz,
            spectrum_metadata,
        )
        return (
            _feature_single(features)[:, : batch["peak_valid_mask"].shape[1]],
            _feature_pair(features),
        )
    features = runtime.extract_single(
        runtime.model,
        batch["peak_mz"],
        batch["peak_intensity"],
        batch["peak_valid_mask"],
        precursor_mz,
        spectrum_metadata,
    )
    return features[:, : batch["peak_valid_mask"].shape[1]]


def _cache_jax_fluorine_split(
    runtime: _JaxFluorineRuntime,
    split: str,
    *,
    seed: int,
    max_samples: int | None,
) -> dict[str, np.ndarray]:
    from spectra_learning.probes.massspec.msg_probe_jax import _host_local_array

    feature_chunks, mask_chunks, label_chunks, row_chunks = [], [], [], []
    iterator = _iter_jax_fluorine_split(
        runtime,
        split,
        shuffle=False,
        seed=seed,
        max_samples=max_samples,
    )
    for batch in tqdm(
        iterator,
        desc=f"cache jax {split} features",
        unit="batch",
        dynamic_ncols=True,
        mininterval=5.0,
    ):
        feature_chunks.append(
            _host_local_array(_extract_jax_fluorine_features(runtime, batch)).astype(
                np.float32,
                copy=False,
            )
        )
        mask_chunks.append(
            _host_local_array(batch["peak_valid_mask"]).astype(bool, copy=False)
        )
        label_chunks.append(
            _host_local_array(batch["label"]).astype(np.float32, copy=False)
        )
        row_chunks.append(
            _host_local_array(batch["row_idx"]).astype(np.int64, copy=False)
        )
    return {
        "features": np.concatenate(feature_chunks, axis=0),
        "peak_valid_mask": np.concatenate(mask_chunks, axis=0),
        "label": np.concatenate(label_chunks, axis=0),
        "row_idx": np.concatenate(row_chunks, axis=0),
    }


def _iter_cached_jax_fluorine_split(
    cache: dict[str, np.ndarray],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Iterator[dict[str, Any]]:
    from spectra_learning.probes.massspec.msg_probe_jax import _probe_value_to_jax

    indices = np.arange(cache["label"].shape[0])
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    for start in range(0, indices.shape[0], batch_size):
        batch_indices = indices[start : start + batch_size]
        yield {
            key: _probe_value_to_jax(value[batch_indices], data_mesh=None)
            for key, value in cache.items()
        }


def _iter_jax_fluorine_batches(
    runtime: _JaxFluorineRuntime,
    cached_splits: dict[str, dict[str, np.ndarray]] | None,
    split: str,
    *,
    shuffle: bool,
    seed: int,
    max_samples: int | None,
) -> Iterator[dict[str, Any]]:
    if cached_splits is not None:
        yield from _iter_cached_jax_fluorine_split(
            cached_splits[split],
            batch_size=int(runtime.args.batch_size),
            shuffle=shuffle,
            seed=seed,
        )
    else:
        yield from _iter_jax_fluorine_split(
            runtime,
            split,
            shuffle=shuffle,
            seed=seed,
            max_samples=max_samples,
        )


def _init_jax_fluorine_probe_params(
    key: Any,
    params: TrialParams,
    *,
    variant: str,
    config: Any,
) -> tuple[dict[str, Any], int]:
    import jax

    from spectra_learning.probes.massspec.msg_probe_jax import _init_mlp, _init_pooler

    pool_key, head_key = jax.random.split(key)
    pooler, input_dim = _init_pooler(
        pool_key,
        variant=variant,
        config=config,
        model_dim=int(config.model_dim),
    )
    return {
        "pooler": pooler,
        "head": _init_mlp(
            head_key,
            input_dim=input_dim,
            hidden_dim=int(params.hidden_dim),
            output_dim=1,
            num_layers=3,
        ),
    }, input_dim


def _jax_fluorine_mlp_apply(
    layers: list[dict[str, Any]],
    x: Any,
    *,
    dropout: float,
    rng: Any,
    training: bool,
) -> Any:
    import jax
    import jax.numpy as jnp

    for idx, layer in enumerate(layers):
        x = jnp.matmul(x, layer["w"]) + layer["b"]
        if idx == len(layers) - 1:
            continue
        x = jax.nn.silu(x)
        if training:
            rng, layer_key = jax.random.split(rng)
            keep_prob = 1.0 - dropout
            keep = jax.random.bernoulli(layer_key, keep_prob, x.shape)
            x = jnp.where(keep, x / keep_prob, 0.0)
    return x


def _jax_fluorine_logits(
    params: dict[str, Any],
    features: Any,
    valid_mask: Any,
    *,
    variant: str,
    task_spec: Any,
    dropout: float = 0.0,
    rng: Any = None,
    training: bool = False,
) -> Any:
    import jax

    from spectra_learning.probes.massspec.msg_probe_jax import _pool_features

    pooled = _pool_features(
        params["pooler"],
        variant=variant,
        task_spec=task_spec,
        features=features,
        valid_mask=valid_mask,
    )
    if rng is None:
        rng = jax.random.PRNGKey(0)
    return _jax_fluorine_mlp_apply(
        params["head"],
        pooled,
        dropout=dropout,
        rng=rng,
        training=training,
    ).squeeze(-1)


def _jax_fluorine_focal_loss(
    logits: Any,
    targets: Any,
    *,
    focal_alpha: float,
    focal_gamma: float,
) -> Any:
    import jax
    import jax.numpy as jnp
    import optax

    targets = targets.astype(jnp.float32)
    bce = optax.sigmoid_binary_cross_entropy(logits, targets)
    prob = jax.nn.sigmoid(logits)
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    alpha_t = focal_alpha * targets + (1.0 - focal_alpha) * (1.0 - targets)
    return jnp.mean(alpha_t * jnp.power(1.0 - p_t, focal_gamma) * bce)


def _make_jax_fluorine_train_step(
    optimizer: Any,
    *,
    variant: str,
    task_spec: Any,
) -> Callable[..., Any]:
    import jax
    import optax

    @jax.jit
    def train_step(
        params: dict[str, Any],
        state: Any,
        rng: Any,
        batch: dict[str, Any],
        features: Any,
        focal_alpha: float,
        focal_gamma: float,
        dropout: float,
    ) -> tuple[dict[str, Any], Any, Any, Any]:
        rng, dropout_key = jax.random.split(rng)

        def loss_fn(probe_params: dict[str, Any]) -> Any:
            logits = _jax_fluorine_logits(
                probe_params,
                features,
                batch["peak_valid_mask"],
                variant=variant,
                task_spec=task_spec,
                dropout=dropout,
                rng=dropout_key,
                training=True,
            )
            return _jax_fluorine_focal_loss(
                logits,
                batch["label"],
                focal_alpha=focal_alpha,
                focal_gamma=focal_gamma,
            )

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, state = optimizer.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        return params, state, rng, loss

    return train_step


def _make_jax_fluorine_predict_step(
    variant: str,
    task_spec: Any,
) -> Callable[..., Any]:
    import jax

    @jax.jit
    def predict_step(
        params: dict[str, Any],
        batch: dict[str, Any],
        features: Any,
    ) -> Any:
        return _jax_fluorine_logits(
            params,
            features,
            batch["peak_valid_mask"],
            variant=variant,
            task_spec=task_spec,
        )

    return predict_step


def _predict_jax_fluorine_arrays(
    runtime: _JaxFluorineRuntime,
    cached_splits: dict[str, dict[str, np.ndarray]] | None,
    params: dict[str, Any],
    split: str,
    *,
    seed: int,
    max_samples: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from spectra_learning.probes.massspec.msg_probe_jax import _host_local_array

    targets, logits, row_indices = [], [], []
    iterator = _iter_jax_fluorine_batches(
        runtime,
        cached_splits,
        split,
        shuffle=False,
        seed=seed,
        max_samples=max_samples,
    )
    for batch in iterator:
        features = (
            batch["features"]
            if cached_splits is not None
            else _extract_jax_fluorine_features(runtime, batch)
        )
        batch_logits = runtime.predict_step(params, batch, features)
        logits.append(_host_local_array(batch_logits))
        targets.append(_host_local_array(batch["label"]))
        row_indices.append(_host_local_array(batch["row_idx"]))
    return (
        np.concatenate(targets, axis=0),
        np.concatenate(logits, axis=0),
        np.concatenate(row_indices, axis=0),
    )


class _JaxFluorineEpochResult(NamedTuple):
    params: dict[str, Any]
    opt_state: Any
    train_rng: Any
    train_loss: float
    val_metrics: dict[str, float]


def _run_jax_fluorine_epoch(
    *,
    runtime: _JaxFluorineRuntime,
    cached_splits: dict[str, dict[str, np.ndarray]] | None,
    train_step: Callable[..., Any],
    params: dict[str, Any],
    opt_state: Any,
    train_rng: Any,
    trial_idx: int,
    trial_count: int,
    epoch_idx: int,
    focal_alpha: float,
    focal_gamma: float,
    dropout: float,
) -> _JaxFluorineEpochResult:
    import jax

    from spectra_learning.probes.massspec.msg_probe_jax import _host_local_array

    seed = int(runtime.args.seed)
    train_iterator = _iter_jax_fluorine_batches(
        runtime,
        cached_splits,
        "train",
        shuffle=True,
        seed=seed + epoch_idx,
        max_samples=runtime.args.max_train_samples,
    )
    running_loss = 0.0
    seen = 0
    pbar = tqdm(
        train_iterator,
        desc=(
            f"jax probe trial {trial_idx + 1}/{trial_count} "
            f"epoch {epoch_idx + 1}/{runtime.args.epochs}"
        ),
        unit="batch",
        dynamic_ncols=True,
        mininterval=5.0,
    )
    for batch in pbar:
        features = (
            batch["features"]
            if cached_splits is not None
            else _extract_jax_fluorine_features(runtime, batch)
        )
        params, opt_state, train_rng, loss = train_step(
            params,
            opt_state,
            train_rng,
            batch,
            features,
            float(focal_alpha),
            float(focal_gamma),
            float(dropout),
        )
        batch_size = int(_host_local_array(batch["label"]).shape[0])
        running_loss += float(jax.device_get(loss)) * batch_size
        seen += batch_size
        pbar.set_postfix(loss=f"{running_loss / float(seen):.5f}")
    val_targets, val_logits, _ = _predict_jax_fluorine_arrays(
        runtime,
        cached_splits,
        params,
        "val",
        seed=seed + 10_000,
        max_samples=runtime.args.max_val_samples,
    )
    return _JaxFluorineEpochResult(
        params,
        opt_state,
        train_rng,
        running_loss / float(seen),
        _metric_dict(val_targets, val_logits, "val"),
    )


def _run_jax_fluorine_trial(
    *,
    runtime: _JaxFluorineRuntime,
    cached_splits: dict[str, dict[str, np.ndarray]] | None,
    params: TrialParams,
    trial_idx: int,
    trial_count: int,
    focal_alpha: float,
    select_metric: str,
) -> tuple[TrialResult, dict[str, Any], int]:
    import jax
    import jax.numpy as jnp
    import optax

    key = jax.random.PRNGKey(int(runtime.args.seed) + trial_idx)
    probe_params, input_dim = _init_jax_fluorine_probe_params(
        key,
        params,
        variant=runtime.variant,
        config=runtime.config,
    )
    optimizer = optax.adamw(
        learning_rate=float(params.learning_rate),
        weight_decay=float(params.weight_decay),
    )
    opt_state = optimizer.init(probe_params)
    train_step = _make_jax_fluorine_train_step(
        optimizer,
        variant=runtime.variant,
        task_spec=runtime.task_spec,
    )
    train_rng = jax.random.PRNGKey(
        int(runtime.args.seed) + 10_000 * (trial_idx + 1)
    )
    best_value = -float("inf")
    best_epoch = 0
    best_val: dict[str, float] = {}
    best_probe_params = probe_params
    history: list[dict[str, Any]] = []
    epochs_without_improvement = 0
    for epoch_idx in range(int(runtime.args.epochs)):
        epoch = _run_jax_fluorine_epoch(
            runtime=runtime,
            cached_splits=cached_splits,
            train_step=train_step,
            params=probe_params,
            opt_state=opt_state,
            train_rng=train_rng,
            trial_idx=trial_idx,
            trial_count=trial_count,
            epoch_idx=epoch_idx,
            focal_alpha=focal_alpha,
            focal_gamma=float(runtime.args.focal_gamma),
            dropout=float(params.dropout),
        )
        probe_params = epoch.params
        opt_state = epoch.opt_state
        train_rng = epoch.train_rng
        history.append(
            {
                "epoch": epoch_idx + 1,
                "train_loss": epoch.train_loss,
                "val": epoch.val_metrics,
            }
        )
        current_value = _select_metric_value(epoch.val_metrics, select_metric)
        if current_value > best_value:
            best_value = current_value
            best_epoch = epoch_idx + 1
            best_val = dict(epoch.val_metrics)
            best_probe_params = jax.tree.map(lambda value: jnp.array(value), probe_params)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        log.info(
            "jax probe trial hidden=%d lr=%.3g wd=%.3g dropout=%.2f epoch=%d/%d val_ap=%.4f val_auc=%.4f",
            params.hidden_dim,
            params.learning_rate,
            params.weight_decay,
            params.dropout,
            epoch_idx + 1,
            runtime.args.epochs,
            epoch.val_metrics["val/average_precision"],
            epoch.val_metrics["val/roc_auc"],
        )
        if epochs_without_improvement >= int(runtime.args.patience):
            break
    result = TrialResult(
        params=params,
        best_epoch=best_epoch,
        best_val=best_val,
        classifier_state={},
        pooler_state=None,
        history=history,
    )
    return result, best_probe_params, input_dim


def _run_jax_fluorine_trials(
    *,
    runtime: _JaxFluorineRuntime,
    cached_splits: dict[str, dict[str, np.ndarray]] | None,
    trial_params: list[TrialParams],
    focal_alpha: float,
    select_metric: str,
) -> tuple[list[TrialResult], list[dict[str, Any]], int]:
    results: list[TrialResult] = []
    best_params_by_trial: list[dict[str, Any]] = []
    input_dim = 0
    for trial_idx, params in enumerate(trial_params):
        result, best_params, input_dim = _run_jax_fluorine_trial(
            runtime=runtime,
            cached_splits=cached_splits,
            params=params,
            trial_idx=trial_idx,
            trial_count=len(trial_params),
            focal_alpha=focal_alpha,
            select_metric=select_metric,
        )
        results.append(result)
        best_params_by_trial.append(best_params)
    return results, best_params_by_trial, input_dim


def _fluorine_probe_focal_alpha(metadata: dict[str, Any], raw_alpha: str) -> float:
    if raw_alpha != "auto":
        return float(raw_alpha)
    return 1.0 - float(metadata["train_positive"]) / float(metadata["train_size"])


def _fluorine_probe_trial_params(args: argparse.Namespace) -> list[TrialParams]:
    return [
        TrialParams(hidden_dim=hidden, learning_rate=lr, weight_decay=wd, dropout=dropout)
        for hidden, lr, wd, dropout in itertools.product(
            _parse_int_grid(args.hidden_dims),
            _parse_float_grid(args.learning_rates),
            _parse_float_grid(args.weight_decays),
            _parse_float_grid(args.dropouts),
        )
    ]


def _build_jax_fluorine_state_and_payload(
    *,
    args: argparse.Namespace,
    checkpoint: _JaxFluorineCheckpoint,
    paths: _FluorineProbePaths,
    data: FluorineData,
    input_dim: int,
    focal_alpha: float,
    results: list[TrialResult],
    best: TrialResult,
    best_probe_params: dict[str, Any],
    test_targets: np.ndarray,
    test_logits: np.ndarray,
    test_metrics: dict[str, float],
) -> tuple[dict[str, Any], dict[str, Any], StoragePath]:
    metadata = data.metadata
    checkpoint_path = storage_join(
        storage_join(checkpoint.checkpoint_dir, "orbax"),
        str(checkpoint.restore_step),
    )
    payload: dict[str, Any] = {
        "backend": "jax",
        "repo_id": HF_REPO_ID,
        "revision": args.revision,
        "train_subdir": HF_TRAIN_SUBDIR,
        "test_subdir": HF_TEST_SUBDIR,
        "cache_dir": str(paths.cache_dir),
        "pooling": args.pooling,
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
        "test_pr_curve": _precision_recall_curve_dict(test_targets, test_logits),
        "trials": [
            {
                "hparams": result.params._asdict(),
                "best_epoch": result.best_epoch,
                "best_val": result.best_val,
            }
            for result in results
        ],
    }
    head_state = {
        "mode": "probe",
        "backend": "jax",
        "complete": True,
        "config_path": str(checkpoint.config_path),
        "checkpoint_path": str(checkpoint_path),
        "input_dim": int(input_dim),
        "covariance_dim": int(checkpoint.config.get("covariance_pooling_dim", 32)),
        "pooling": args.pooling,
        "pair_dim": int(
            checkpoint.config.get("pairmixer_pair_dim", checkpoint.config.model_dim)
        ),
        "jax_params": _jax_tree_to_numpy(best_probe_params),
        "pooler_state": None,
        "classifier_state": {},
        "best_epoch": int(best.best_epoch),
        "best_val": best.best_val,
        "test": test_metrics,
        "history": best.history,
        "hparams": best.params._asdict(),
        "focal_alpha": focal_alpha,
        "focal_gamma": float(args.focal_gamma),
        "train_size": int(metadata["train_size"]),
        "train_positive": int(metadata["train_positive"]),
        "val_size": int(metadata["val_size"]),
        "val_positive": int(metadata["val_positive"]),
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
    }
    return payload, head_state, checkpoint_path


def run_probe_jax(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = _restore_jax_fluorine_checkpoint(args)
    paths = _resolve_fluorine_probe_paths(args)
    data = _build_fluorine_probe_data(
        args=args,
        checkpoint_config=checkpoint.config,
        cache_dir=paths.cache_dir,
    )
    runtime = _make_jax_fluorine_runtime(
        args=args,
        checkpoint=checkpoint,
        data=data,
    )

    metadata = data.metadata
    focal_alpha = _fluorine_probe_focal_alpha(metadata, args.focal_alpha)
    trial_params = _fluorine_probe_trial_params(args)
    select_metric = f"val/{args.select_metric}"
    cached_splits: dict[str, dict[str, np.ndarray]] | None = None
    if runtime.variant == "covariance":
        cached_splits = {
            "train": _cache_jax_fluorine_split(
                runtime,
                "train",
                seed=int(args.seed),
                max_samples=args.max_train_samples,
            ),
            "val": _cache_jax_fluorine_split(
                runtime,
                "val",
                seed=int(args.seed) + 10_000,
                max_samples=args.max_val_samples,
            ),
            "test": _cache_jax_fluorine_split(
                runtime,
                "test",
                seed=int(args.seed) + 20_000,
                max_samples=args.max_test_samples,
            ),
        }
    results, best_params_by_trial, input_dim = _run_jax_fluorine_trials(
        runtime=runtime,
        cached_splits=cached_splits,
        trial_params=trial_params,
        focal_alpha=focal_alpha,
        select_metric=select_metric,
    )
    best_idx = max(
        range(len(results)),
        key=lambda idx: _select_metric_value(results[idx].best_val, select_metric),
    )
    best = results[best_idx]
    best_probe_params = best_params_by_trial[best_idx]
    test_targets, test_logits, test_row_indices = _predict_jax_fluorine_arrays(
        runtime,
        cached_splits,
        best_probe_params,
        "test",
        seed=int(args.seed) + 20_000,
        max_samples=args.max_test_samples,
    )
    test_metrics = _metric_dict(test_targets, test_logits, "test")
    payload, head_state, checkpoint_path = _build_jax_fluorine_state_and_payload(
        args=args,
        checkpoint=checkpoint,
        paths=paths,
        data=data,
        input_dim=input_dim,
        focal_alpha=focal_alpha,
        results=results,
        best=best,
        best_probe_params=best_probe_params,
        test_targets=test_targets,
        test_logits=test_logits,
        test_metrics=test_metrics,
    )
    return _write_fluorine_probe_artifacts(
        args=args,
        payload=payload,
        head_state=head_state,
        paths=paths,
        config_path=checkpoint.config_path,
        checkpoint_path=checkpoint_path,
        data=data,
        targets=test_targets,
        logits=test_logits,
        row_indices=test_row_indices,
        summary_backend="jax",
    )


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint, getattr(args, "workdir", None))
    config_path = args.config.expanduser().resolve()
    checkpoint_config, model = _load_checkpoint_model(
        config_path,
        checkpoint_path,
        device,
    )
    paths = _resolve_fluorine_probe_paths(args)
    data = _build_fluorine_probe_data(
        args=args,
        checkpoint_config=checkpoint_config,
        cache_dir=paths.cache_dir,
    )
    metadata = data.metadata
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
        checkpoint_path=checkpoint_path,
        device=device,
        train_covariance_pooler=bool(args.train_covariance_pooler),
        covariance_dim=args.covariance_dim,
        pooling=args.pooling,
    )

    train_positive = float(metadata["train_positive"])
    train_size = float(metadata["train_size"])
    focal_alpha = (
        1.0 - train_positive / train_size
        if args.focal_alpha == "auto"
        else float(args.focal_alpha)
    )
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
    progress_output_prefix = paths.output_prefix if len(trial_params) == 1 else None

    if bool(args.train_covariance_pooler):
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
                patience=args.patience,
                build_feature_fn=build_feature_fn,
                progress_output_prefix=progress_output_prefix,
            )
            for params in trial_params
        ]
        best = max(
            results,
            key=lambda result: _select_metric_value(result.best_val, select_metric),
        )
        classifier, feature_fn = _instantiate_best_model(
            result=best,
            input_dim=input_dim,
            device=device,
            build_feature_fn=build_feature_fn,
        )
        test_targets, test_logits, test_row_indices_np = _prediction_arrays_with_rows(
            classifier,
            test_loader,
            feature_fn,
            device=device,
        )
    else:
        feature_fn, _ = build_feature_fn(False)
        train_features, train_labels, _ = _cache_frozen_features(
            train_loader,
            feature_fn,
            device=device,
            split="train",
        )
        val_features, val_labels, _ = _cache_frozen_features(
            val_loader,
            feature_fn,
            device=device,
            split="val",
        )
        test_features, test_labels, test_row_indices = _cache_frozen_features(
            test_loader,
            feature_fn,
            device=device,
            split="test",
        )
        train_feature_loader = _cached_loader(
            train_features,
            train_labels,
            batch_size=int(args.batch_size),
            shuffle=True,
            seed=int(args.seed),
        )
        val_feature_loader = _cached_loader(
            val_features,
            val_labels,
            batch_size=int(args.batch_size),
            shuffle=False,
            seed=int(args.seed) + 10_000,
        )
        test_feature_loader = _cached_loader(
            test_features,
            test_labels,
            batch_size=int(args.batch_size),
            shuffle=False,
            seed=int(args.seed) + 20_000,
        )
        results = [
            _train_cached_trial(
                params=params,
                input_dim=input_dim,
                train_loader=train_feature_loader,
                val_loader=val_feature_loader,
                device=device,
                epochs=args.epochs,
                focal_alpha=focal_alpha,
                focal_gamma=args.focal_gamma,
                select_metric=select_metric,
                patience=args.patience,
                progress_output_prefix=progress_output_prefix,
            )
            for params in trial_params
        ]
        best = max(
            results,
            key=lambda result: _select_metric_value(result.best_val, select_metric),
        )
        classifier = MLPClassifier(
            input_dim=input_dim,
            hidden_dim=best.params.hidden_dim,
            dropout=best.params.dropout,
        ).to(device)
        classifier.load_state_dict(best.classifier_state)
        classifier.eval()
        test_targets, test_logits = _cached_prediction_arrays(
            classifier,
            test_feature_loader,
            device=device,
        )
        test_row_indices_np = test_row_indices.numpy()

    test_metrics = _metric_dict(test_targets, test_logits, "test")
    payload: dict[str, Any] = {
        "repo_id": HF_REPO_ID,
        "revision": args.revision,
        "train_subdir": HF_TRAIN_SUBDIR,
        "test_subdir": HF_TEST_SUBDIR,
        "cache_dir": str(paths.cache_dir),
        "pooling": args.pooling,
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
        "test_pr_curve": _precision_recall_curve_dict(test_targets, test_logits),
        "trials": [
            {
                "hparams": result.params._asdict(),
                "best_epoch": result.best_epoch,
                "best_val": result.best_val,
            }
            for result in results
        ],
    }
    head_state = {
        "mode": "probe",
        "complete": True,
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "input_dim": int(input_dim),
        "covariance_dim": int(
            args.covariance_dim
            if args.covariance_dim is not None
            else checkpoint_config.get("covariance_pooling_dim", 32)
        ),
        "pooling": args.pooling,
        "pair_dim": int(
            checkpoint_config.get("pairmixer_pair_dim", checkpoint_config.model_dim)
        ),
        "pooler_state": _tensor_state_to_cpu(best.pooler_state),
        "classifier_state": _tensor_state_to_cpu(best.classifier_state),
        "best_epoch": int(best.best_epoch),
        "best_val": best.best_val,
        "test": test_metrics,
        "history": best.history,
        "hparams": best.params._asdict(),
        "focal_alpha": focal_alpha,
        "focal_gamma": float(args.focal_gamma),
        "train_size": int(metadata["train_size"]),
        "train_positive": int(metadata["train_positive"]),
        "val_size": int(metadata["val_size"]),
        "val_positive": int(metadata["val_positive"]),
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
    }
    return _write_fluorine_probe_artifacts(
        args=args,
        payload=payload,
        head_state=head_state,
        paths=paths,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        data=data,
        targets=test_targets,
        logits=test_logits,
        row_indices=test_row_indices_np,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    mode = str(getattr(args, "mode", "probe"))
    torch.manual_seed(int(args.seed))
    if mode == "probe":
        backend = str(getattr(args, "backend", "auto")).lower()
        if backend == "auto":
            config = load_config(args.config.expanduser().resolve())
            backend = (
                "jax"
                if str(config.get("device_backend", "torch")).lower() == "jax"
                else "torch"
            )
        if backend == "jax":
            return run_probe_jax(args)
        return run_probe(args)

    distributed = init_distributed_from_env()
    device = distributed.device if distributed.is_distributed else torch.device(args.device)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint, getattr(args, "workdir", None))
    config, model = _load_checkpoint_model(
        args.config.expanduser().resolve(),
        checkpoint_path,
        device,
    )
    head_state_path = (
        args.head_state.expanduser().resolve()
        if getattr(args, "head_state", None) is not None
        else default_state_path(mode).resolve()
    )
    output_prefix = (
        normalize_storage_path(args.output_prefix)
        if getattr(args, "output_prefix", None) is not None
        else default_output_prefix(mode).resolve()
    )
    autocast_dtype = _resolve_autocast_dtype(
        config,
        getattr(args, "autocast_dtype", None),
    )
    epochs = int(args.epochs if args.epochs is not None else 3)
    patience = int(args.patience if args.patience is not None else 2)

    if mode == "finetune":
        head_state = train_or_load_finetuned(
            state_path=head_state_path,
            model=model,
            config=config,
            config_path=args.config.expanduser().resolve(),
            checkpoint_path=checkpoint_path,
            cache_dir=args.finetune_cache_dir.expanduser().resolve(),
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            epochs=epochs,
            patience=patience,
            model_learning_rate=args.finetune_model_lr,
            pooler_learning_rate=args.finetune_pooler_lr,
            head_learning_rate=args.finetune_head_lr,
            weight_decay=args.finetune_weight_decay,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            autocast_dtype=autocast_dtype,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            revision=args.revision,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            max_test_samples=args.max_test_samples,
            pooling=args.pooling,
            select_metric=args.select_metric,
            progress_output_prefix=output_prefix,
            eval_test_every_epoch=bool(getattr(args, "eval_test_every_epoch", False)),
            distributed=distributed,
        )
    elif mode == "lora":
        head_state = train_or_load_lora(
            state_path=head_state_path,
            model=model,
            config=config,
            config_path=args.config.expanduser().resolve(),
            checkpoint_path=checkpoint_path,
            cache_dir=args.finetune_cache_dir.expanduser().resolve(),
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            epochs=epochs,
            patience=patience,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_learning_rate=args.lora_learning_rate,
            head_learning_rate=args.finetune_head_lr,
            weight_decay=args.finetune_weight_decay,
            autocast_dtype=autocast_dtype,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            revision=args.revision,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            max_test_samples=args.max_test_samples,
            pooling=args.pooling,
            progress_output_prefix=output_prefix,
            eval_test_every_epoch=bool(getattr(args, "eval_test_every_epoch", False)),
            distributed=distributed,
        )
    else:
        raise ValueError(f"unsupported fluorine mode: {mode}")

    if distributed.is_distributed and not distributed.is_main:
        barrier(distributed)
        cleanup_distributed(distributed)
        return {}

    data = build_fluorine_data(
        config=config,
        cache_dir=args.finetune_cache_dir.expanduser().resolve(),
        batch_size=args.batch_size,
        revision=args.revision,
    )
    log.info(
        "using fluorine test split size=%d positive=%d",
        int(data.metadata["test_size"]),
        int(data.metadata["test_positive"]),
    )
    targets, logits, row_indices = evaluate_fluorine_test_split(
        model=model,
        config=config,
        head_state=head_state,
        data=data,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_test_samples=args.max_test_samples,
        autocast_dtype=autocast_dtype,
    )
    head_state["test"] = _metric_dict(targets, logits, "test")
    torch.save(head_state, head_state_path)
    summary = write_standard_fluorine_outputs(
        output_prefix=output_prefix,
        config_path=args.config.expanduser().resolve(),
        checkpoint_path=checkpoint_path,
        head_state_path=head_state_path,
        data=data,
        targets=targets,
        logits=logits,
        row_indices=row_indices,
        head_state=head_state,
    )
    if args.comparison_dir is not None:
        comparison = write_dreams_comparison(
            output_prefix=output_prefix,
            comparison_dir=args.comparison_dir.expanduser().resolve(),
            summary=summary,
            previous_ours_prefix=(
                args.previous_ours_prefix.expanduser().resolve()
                if mode in {"finetune", "lora"}
                else None
            ),
        )
        summary["comparison"] = comparison
    curve_dirs: list[StoragePath] = [storage_parent(output_prefix)]
    if args.comparison_dir is not None:
        curve_dirs.append(args.comparison_dir.expanduser().resolve())
    summary["all_pr_curves"] = write_all_pr_curve_comparison(
        output_prefix=output_prefix,
        curve_dirs=curve_dirs,
    )
    write_text(
        storage_with_suffix(output_prefix, ".summary.json"),
        json.dumps(summary, indent=2, sort_keys=True),
    )
    if args.output_json:
        write_text(
            normalize_storage_path(args.output_json),
            json.dumps(summary, indent=2, sort_keys=True),
        )
    if distributed.is_distributed:
        barrier(distributed)
        cleanup_distributed(distributed)
    return summary
