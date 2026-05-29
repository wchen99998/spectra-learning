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
    load_pretrained_weights,
    save_torch_checkpoint,
)
from spectra_learning.training.storage import (
    StoragePath,
    normalize_storage_path,
    write_text,
)


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
) -> Any:
    return build_murcko_fluorine_loader(
        data,
        split,
        shuffle=shuffle,
        seed=seed,
        max_samples=max_samples,
        dreams_only=dreams_only,
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    checkpoint_path = normalize_storage_path(args.checkpoint)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a fluorine-detection probe on the NIST Murcko train/val splits and MCEBIO Murcko test split."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--pooling",
        choices=("covariance", "single_pair_covariance"),
        default="covariance",
    )
    parser.add_argument("--train-covariance-pooler", action="store_true")
    parser.add_argument("--covariance-dim", type=int, default=None)
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/fluorine_detection_fine_tuned"),
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-state", default=None)
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
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    payload = run(parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
