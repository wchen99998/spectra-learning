from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import logging
from pathlib import Path
from typing import Any, Callable, NamedTuple, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
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
from torch.utils.data import DataLoader

from spectra_learning.config.loading import load_config
from spectra_learning.data.spectra import (
    DEFAULT_MAX_PRECURSOR_MZ,
    DEFAULT_MIN_PEAK_INTENSITY,
)
from spectra_learning.models.factory import build_model_from_config
from spectra_learning.models.model import PeakSetJEPA
from spectra_learning.models.pooling import CovariancePool, SinglePairCovariancePool
from spectra_learning.probes.massspec.data import (
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
    save_torch_checkpoint,
)
from spectra_learning.training.storage import StoragePath, normalize_storage_path, write_text


log = logging.getLogger(__name__)

HF_REPO_ID = NIST_MURCKO_HF_REPO
HF_TRAIN_SUBDIR = NIST_MURCKO_PREPARED_SUBDIR
HF_TEST_SUBDIR = MCEBIO_MURCKO_PREPARED_SUBDIR


def _config_get(config: Any, key: str, default: Any) -> Any:
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


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
) -> Any:
    return build_murcko_fluorine_loader(
        data,
        split,
        shuffle=shuffle,
        seed=seed,
        max_samples=max_samples,
        dreams_only=dreams_only,
        num_workers=num_workers,
    )


@torch.no_grad()
def _prediction_arrays(
    classifier: MLPClassifier,
    loader: Any,
    feature_fn: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    classifier.eval()
    logits, targets = [], []
    for batch in loader:
        batch = _move_batch(batch, device)
        features = feature_fn(batch)
        logits.append(classifier(features).detach().cpu().numpy())
        targets.append(batch["label"].detach().cpu().numpy())
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
    device: torch.device,
    train_covariance_pooler: bool,
    covariance_dim: int | None,
    pooling: str,
) -> tuple[int, Callable[[bool], tuple[Callable[[dict[str, torch.Tensor]], torch.Tensor], torch.nn.Module | None]]]:
    compressed_dim = (
        covariance_dim
        if covariance_dim is not None
        else int(_config_get(config, "covariance_pooling_dim", 32))
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
        )
        return _peak_tokens_only(embeddings, batch["peak_valid_mask"])

    @torch.no_grad()
    def encode_single_pair(
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        peak_embeddings, _, pair_embeddings = model.encoder.forward_with_block_outputs(
            batch["peak_mz"],
            batch["peak_intensity"],
            valid_mask=batch["peak_valid_mask"],
            precursor_mz=batch.get("precursor_mz", None),
        )
        return _peak_tokens_only(peak_embeddings, batch["peak_valid_mask"]), pair_embeddings

    def build_feature_fn(training: bool):
        if pooling == "single_pair_covariance":
            pooler = SinglePairCovariancePool(
                single_dim=int(config.model_dim),
                pair_dim=int(_config_get(config, "pairformer_pair_dim", config.model_dim)),
                compressed_dim=compressed_dim,
            ).to(device)
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
        repo_id=HF_REPO_ID,
        revision=revision,
        train_subdir=HF_TRAIN_SUBDIR,
        test_subdir=HF_TEST_SUBDIR,
    )


def _module_state_to_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


class FluorineFinetuneModule(torch.nn.Module):
    def __init__(
        self,
        *,
        encoder: torch.nn.Module,
        pooler: torch.nn.Module,
        classifier: MLPClassifier,
        pooling: str,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.pooler = pooler
        self.classifier = classifier
        self.pooling = pooling
        self.freeze_encoder = freeze_encoder

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.pooling == "single_pair_covariance":
            if self.freeze_encoder:
                with torch.no_grad():
                    peak_embeddings, _, pair_embeddings = self.encoder.forward_with_block_outputs(
                        batch["peak_mz"],
                        batch["peak_intensity"],
                        valid_mask=batch["peak_valid_mask"],
                        precursor_mz=batch.get("precursor_mz", None),
                    )
                    peak_embeddings = _peak_tokens_only(peak_embeddings, batch["peak_valid_mask"])
            else:
                peak_embeddings, _, pair_embeddings = self.encoder.forward_with_block_outputs(
                    batch["peak_mz"],
                    batch["peak_intensity"],
                    valid_mask=batch["peak_valid_mask"],
                    precursor_mz=batch.get("precursor_mz", None),
                )
                peak_embeddings = _peak_tokens_only(peak_embeddings, batch["peak_valid_mask"])
            features = self.pooler(
                peak_embeddings.float(),
                batch["peak_valid_mask"].to(dtype=torch.bool),
                pair_embeddings.float(),
            )
        else:
            if self.freeze_encoder:
                with torch.no_grad():
                    encoded = self.encoder(
                        batch["peak_mz"],
                        batch["peak_intensity"],
                        valid_mask=batch["peak_valid_mask"],
                        precursor_mz=batch.get("precursor_mz", None),
                    )
                    peak_embeddings = _peak_tokens_only(encoded, batch["peak_valid_mask"])
            else:
                encoded = self.encoder(
                    batch["peak_mz"],
                    batch["peak_intensity"],
                    valid_mask=batch["peak_valid_mask"],
                    precursor_mz=batch.get("precursor_mz", None),
                )
                peak_embeddings = _peak_tokens_only(encoded, batch["peak_valid_mask"])
            features = self.pooler(
                peak_embeddings.float(),
                batch["peak_valid_mask"].to(dtype=torch.bool),
            )
        return self.classifier(features)


def _wrap_data_parallel(
    module: torch.nn.Module,
    device_ids: list[int] | None,
) -> torch.nn.Module:
    return module


@torch.no_grad()
def predict_finetuned(
    *,
    finetune_module: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    finetune_module.eval()
    use_autocast = device.type == "cuda"
    logits, targets = [], []
    for batch in loader:
        batch = _move_batch(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_autocast,
        ):
            batch_logits = finetune_module(batch)
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
    revision: str,
    max_train_samples: int | None,
    max_val_samples: int | None,
    pooling: str,
    device_ids: list[int] | None,
) -> dict[str, Any]:
    requested_hparams = {
        "hidden_dim": int(hidden_dim),
        "dropout": float(dropout),
        "model_learning_rate": float(model_learning_rate),
        "head_learning_rate": float(head_learning_rate),
        "weight_decay": float(weight_decay),
        "epochs": int(epochs),
        "patience": int(patience),
    }
    if state_path.exists():
        state = torch.load(state_path, map_location=device)
        state_hparams = state.get("hparams", {})
        if (
            state.get("mode") == "finetune"
            and state.get("config_path") == str(config_path)
            and state.get("checkpoint_path") == str(checkpoint_path)
            and state.get("pooling", "covariance") == pooling
            and all(state_hparams.get(key) == value for key, value in requested_hparams.items())
            and state.get("max_train_samples") == max_train_samples
            and state.get("max_val_samples") == max_val_samples
        ):
            model.load_state_dict(state["model_state"])
            return state

    data = build_fluorine_data(
        config=config,
        cache_dir=cache_dir,
        batch_size=batch_size,
        revision=revision,
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
    finetune_module = FluorineFinetuneModule(
        encoder=model.encoder,
        pooler=pooler,
        classifier=classifier,
        pooling=pooling,
    ).to(device)
    finetune_module = _wrap_data_parallel(finetune_module, device_ids)
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
    best_state_path = state_path.with_name(f"{state_path.stem}.best.pt")

    def make_state(
        *,
        test_metrics: dict[str, float] | None,
        complete: bool,
    ) -> dict[str, Any]:
        return {
            "mode": "finetune",
            "complete": complete,
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
            "hparams": requested_hparams,
            "focal_alpha": focal_alpha,
            "focal_gamma": focal_gamma,
            "finetune_cache_dir": str(cache_dir),
            "device_ids": device_ids if device_ids is not None else [],
            "train_size": int(data.metadata["train_size"]),
            "train_positive": int(data.metadata["train_positive"]),
            "val_size": int(data.metadata["val_size"]),
            "val_positive": int(data.metadata["val_positive"]),
            "max_train_samples": max_train_samples,
            "max_val_samples": max_val_samples,
        }

    for epoch_idx in range(epochs):
        finetune_module.train()
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
                logits = finetune_module(batch)
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
            finetune_module=finetune_module,
            loader=val_loader,
            device=device,
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
            state_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(make_state(test_metrics=None, complete=False), best_state_path)
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
        finetune_module=finetune_module,
        loader=test_loader,
        device=device,
    )
    test_metrics = _metric_dict(test_targets, test_logits, "test")

    state = make_state(test_metrics=test_metrics, complete=True)
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
    device_ids: list[int] | None,
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
    finetune_module = FluorineFinetuneModule(
        encoder=model.encoder,
        pooler=pooler,
        classifier=classifier,
        pooling=pooling,
    ).to(device)
    finetune_module = _wrap_data_parallel(finetune_module, device_ids)
    finetune_module.eval()

    eval_data = data._replace(batch_size=batch_size)
    loader = build_murcko_fluorine_loader(
        eval_data,
        "test",
        shuffle=False,
        seed=0,
        max_samples=None,
        dreams_only=False,
        num_workers=num_workers,
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
            batch_logits = finetune_module(batch)
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


def write_standard_fluorine_outputs(
    *,
    output_prefix: Path,
    config_path: Path,
    checkpoint_path: Path,
    head_state_path: Path,
    data: FluorineData,
    targets: np.ndarray,
    logits: np.ndarray,
    row_indices: np.ndarray,
    head_state: dict[str, Any],
) -> dict[str, Any]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    probs = sigmoid(logits)
    precision, recall, thresholds = precision_recall_curve(targets, probs)
    metrics = summarize_metrics(targets, logits)
    train_cache_metrics = _metric_dict(targets, logits, "mcebio")
    summary = {
        "mode": head_state.get("mode", "probe"),
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
        "metrics_prefixed": train_cache_metrics,
        "head": {
            "best_epoch": head_state["best_epoch"],
            "best_val": head_state["best_val"],
            "test": head_state.get("test", None),
            "hparams": head_state["hparams"],
            "pooling": head_state.get("pooling", "covariance"),
            "pair_dim": head_state.get("pair_dim", None),
            "device_ids": head_state.get("device_ids", []),
            "focal_alpha": head_state["focal_alpha"],
            "focal_gamma": head_state["focal_gamma"],
            "finetune_cache_dir": head_state.get("finetune_cache_dir", ""),
            "train_size": head_state.get("train_size", None),
            "train_positive": head_state.get("train_positive", None),
            "val_size": head_state.get("val_size", None),
            "val_positive": head_state.get("val_positive", None),
        },
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
        writer = csv.DictWriter(
            handle,
            fieldnames=["eval_index", "row_idx", "label", "score", "logit"],
        )
        writer.writeheader()
        for eval_i, row_i in enumerate(row_indices):
            writer.writerow(
                {
                    "eval_index": eval_i,
                    "row_idx": int(row_i),
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
    ax.set_title(f"Fluorine on MCEBIO Murcko test (AP={metrics['average_precision']:.3f})")
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
    ax.set_title("Fluorine PR Curves on MCEBIO Murcko Test")
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
    return Path("checkpoints/modal/no_fourier_embed_sentinel/step-01250000.pt").resolve()


def default_state_path(mode: str) -> Path:
    if mode == "finetune":
        return Path("results/fluorine_detection_full_finetune_state.pt")
    return Path("results/fluorine_detection_probe_state.pt")


def default_output_prefix(mode: str) -> Path:
    if mode == "finetune":
        return Path("results/fluorine_detection_full_finetune")
    return Path("results/fluorine_detection_probe")


def parse_device_ids(raw: str | None) -> list[int] | None:
    if raw is None or raw == "":
        return None
    return [int(item) for item in raw.split(",") if item]


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint, getattr(args, "workdir", None))
    checkpoint_config, model = _load_checkpoint_model(
        args.config.expanduser().resolve(),
        checkpoint_path,
        device,
    )
    cache_dir = args.cache_dir.expanduser().resolve()
    data = build_murcko_fluorine_data(
        cache_dir=cache_dir,
        batch_size=int(args.batch_size),
        num_peaks=int(
            args.num_peaks
            if args.num_peaks is not None
            else _config_get(checkpoint_config, "num_peaks", 60)
        ),
        max_precursor_mz=float(
            _config_get(checkpoint_config, "max_precursor_mz", DEFAULT_MAX_PRECURSOR_MZ)
        ),
        min_peak_intensity=float(
            _config_get(
                checkpoint_config,
                "min_peak_intensity",
                DEFAULT_MIN_PEAK_INTENSITY,
            )
        ),
        peak_drop_min_intensity=float(
            _config_get(
                checkpoint_config,
                "peak_drop_min_intensity",
                _config_get(
                    checkpoint_config,
                    "min_peak_intensity",
                    DEFAULT_MIN_PEAK_INTENSITY,
                ),
            )
        ),
        peak_ordering=str(
            args.peak_ordering
            if args.peak_ordering
            else _config_get(checkpoint_config, "peak_ordering", "intensity")
        ),
        precursor_peak_exclusion_window_da=float(
            _config_get(checkpoint_config, "precursor_peak_exclusion_window_da", 0.0)
        ),
        repo_id=HF_REPO_ID,
        revision=args.revision,
        train_subdir=HF_TRAIN_SUBDIR,
        test_subdir=HF_TEST_SUBDIR,
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
    test_targets, test_logits = _prediction_arrays(
        classifier,
        test_loader,
        feature_fn,
        device=device,
    )
    test_metrics = _metric_dict(test_targets, test_logits, "test")
    payload: dict[str, Any] = {
        "repo_id": HF_REPO_ID,
        "revision": args.revision,
        "train_subdir": HF_TRAIN_SUBDIR,
        "test_subdir": HF_TEST_SUBDIR,
        "cache_dir": str(cache_dir),
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
    if args.output_state:
        save_torch_checkpoint(
            {
                "mode": "probe",
                "input_dim": int(input_dim),
                "covariance_dim": int(
                    args.covariance_dim
                    if args.covariance_dim is not None
                    else _config_get(checkpoint_config, "covariance_pooling_dim", 32)
                ),
                "pooling": args.pooling,
                "pair_dim": int(_config_get(checkpoint_config, "pairformer_pair_dim", checkpoint_config.model_dim)),
                "pooler_state": _tensor_state_to_cpu(best.pooler_state),
                "classifier_state": _tensor_state_to_cpu(best.classifier_state),
                "best_epoch": int(best.best_epoch),
                "best_val": best.best_val,
                "hparams": best.params._asdict(),
                "focal_alpha": focal_alpha,
                "focal_gamma": float(args.focal_gamma),
                "train_size": int(metadata["train_size"]),
                "train_positive": int(metadata["train_positive"]),
                "val_size": int(metadata["val_size"]),
                "val_positive": int(metadata["val_positive"]),
            },
            normalize_storage_path(args.output_state),
        )
    if args.output_json:
        write_text(
            normalize_storage_path(args.output_json),
            json.dumps(payload, indent=2, sort_keys=True),
        )
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    mode = str(getattr(args, "mode", "probe"))
    torch.manual_seed(int(args.seed))
    if mode == "probe":
        return run_probe(args)

    device = torch.device(args.device)
    device_ids = parse_device_ids(getattr(args, "device_ids", None))
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
        args.output_prefix.expanduser().resolve()
        if getattr(args, "output_prefix", None) is not None
        else default_output_prefix(mode).resolve()
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
            seed=args.seed,
            epochs=epochs,
            patience=patience,
            model_learning_rate=args.finetune_model_lr,
            head_learning_rate=args.finetune_head_lr,
            weight_decay=args.finetune_weight_decay,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            revision=args.revision,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            pooling=args.pooling,
            device_ids=device_ids,
        )
    else:
        raise ValueError(f"unsupported fluorine mode: {mode}")

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
        device_ids=device_ids,
    )
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
                if mode == "finetune"
                else None
            ),
        )
        summary["comparison"] = comparison
        output_prefix.with_suffix(".summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True)
        )
    if args.output_json:
        write_text(
            normalize_storage_path(args.output_json),
            json.dumps(summary, indent=2, sort_keys=True),
        )
    return summary
