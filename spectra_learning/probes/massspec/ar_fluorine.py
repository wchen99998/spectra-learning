from __future__ import annotations

import argparse
import copy
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from spectra_learning.data.ar_spectra import SpectraARTokenizer, SpectraARTokenizerConfig
from spectra_learning.models.ar_spectra import (
    SpectraARTransformer,
    SpectraARTransformerConfig,
)
from spectra_learning.models.lora import (
    apply_lora_to_linear_modules,
    load_lora_state_dict,
    lora_parameters,
    lora_state_dict,
)
from spectra_learning.probes.massspec.fluorine import (
    _autocast_dtype_name,
    _make_loader,
    _metric_dict,
    _resolve_autocast_dtype,
    binary_focal_loss_with_logits,
    build_fluorine_data,
    write_all_pr_curve_comparison,
    write_standard_fluorine_outputs,
    write_training_history_outputs,
)
from spectra_learning.training.checkpointing import load_torch_checkpoint
from spectra_learning.training.runtime import build_grad_scaler
from spectra_learning.training.storage import StoragePath, storage_parent


log = logging.getLogger(__name__)

AR_LORA_TARGET_SUFFIXES = (
    "attention.qkv",
    "attention.out_proj",
    "ffn.0",
    "ffn.3",
)


def _module_state_to_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def _move_tokenized_to_device(
    tokenized: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in tokenized.items()}


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
        "target_suffixes": list(AR_LORA_TARGET_SUFFIXES),
    }


def apply_ar_fluorine_lora(
    model: torch.nn.Module,
    lora_config: dict[str, Any],
) -> tuple[str, ...]:
    return apply_lora_to_linear_modules(
        model,
        target_suffixes=tuple(lora_config["target_suffixes"]),
        rank=int(lora_config["rank"]),
        alpha=float(lora_config["alpha"]),
        dropout=float(lora_config["dropout"]),
    )


def load_ar_checkpoint_model(
    *,
    config_path: Path,
    checkpoint_path: StoragePath,
    device: torch.device,
) -> tuple[Any, SpectraARTokenizer, SpectraARTransformer, dict[str, Any]]:
    del config_path
    checkpoint = load_torch_checkpoint(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    config = checkpoint["config"]
    tokenizer = SpectraARTokenizer(
        SpectraARTokenizerConfig(**checkpoint["tokenizer_config"])
    )
    model = SpectraARTransformer(
        SpectraARTransformerConfig.from_config(config, tokenizer)
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()
    return config, tokenizer, model, checkpoint


class SpectraARFluorineModule(torch.nn.Module):
    def __init__(
        self,
        *,
        model: SpectraARTransformer,
        tokenizer: SpectraARTokenizer,
        classifier: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.classifier = classifier

    def features(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        device = self.model.token_embedding.weight.device
        tokenized = _move_tokenized_to_device(
            self.tokenizer.tokenize_batch(batch),
            device,
        )
        full_ids, full_kinds = self._full_sequence(tokenized)
        hidden = self.model.hidden_states(full_ids, full_kinds)
        eos_indices = (
            full_ids.eq(self.tokenizer.eos_token_id)
            .to(dtype=torch.long)
            .argmax(dim=1)
        )
        return hidden[
            torch.arange(hidden.shape[0], device=hidden.device),
            eos_indices,
        ].float()

    def _full_sequence(
        self,
        tokenized: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.cat(
                [
                    tokenized["input_token_ids"][:, :1],
                    tokenized["target_token_ids"],
                ],
                dim=1,
            ),
            torch.cat(
                [
                    tokenized["input_token_kinds"][:, :1],
                    tokenized["target_token_kinds"],
                ],
                dim=1,
            ),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.classifier(self.features(batch))


class FluorineLabelTokenHead(torch.nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.label_tokens = torch.nn.Linear(input_dim, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.label_tokens(features)
        return logits[:, 1] - logits[:, 0]


@torch.no_grad()
def predict_ar_fluorine(
    *,
    module: SpectraARFluorineModule,
    loader: Any,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    module.eval()
    use_autocast = device.type == "cuda" and autocast_dtype is not None
    targets, logits, row_indices = [], [], []
    for batch in loader:
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype if autocast_dtype is not None else torch.bfloat16,
            enabled=use_autocast,
        ):
            batch_logits = module(batch)
        logits.append(batch_logits.float().detach().cpu().numpy())
        targets.append(batch["label"].detach().cpu().numpy())
        row_indices.append(batch["row_idx"].detach().cpu().numpy())
    return (
        np.concatenate(targets, axis=0),
        np.concatenate(logits, axis=0),
        np.concatenate(row_indices, axis=0),
    )


def _input_dim(model: SpectraARTransformer) -> int:
    return int(model.config.model_dim)


def train_ar_fluorine(
    *,
    mode: str,
    state_path: Path,
    model: SpectraARTransformer,
    tokenizer: SpectraARTokenizer,
    config: Any,
    config_path: Path,
    checkpoint_path: StoragePath,
    cache_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
    epochs: int,
    patience: int,
    model_learning_rate: float,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float,
    lora_learning_rate: float,
    head_learning_rate: float,
    weight_decay: float,
    autocast_dtype: torch.dtype | None,
    revision: str,
    max_train_samples: int | None,
    max_val_samples: int | None,
    max_test_samples: int | None,
    select_metric: str,
    progress_output_prefix: StoragePath | None,
    eval_test_every_epoch: bool,
) -> tuple[dict[str, Any], Any, np.ndarray, np.ndarray, np.ndarray]:
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
        num_workers=num_workers,
    )
    val_loader = _make_loader(
        data,
        "val",
        shuffle=False,
        seed=seed + 10_000,
        max_samples=max_val_samples,
        num_workers=num_workers,
    )
    test_loader = _make_loader(
        data,
        "test",
        shuffle=False,
        seed=seed + 20_000,
        max_samples=max_test_samples,
        num_workers=num_workers,
    )

    input_dim = _input_dim(model)
    classifier = FluorineLabelTokenHead(input_dim=input_dim).to(device)
    finetune_module = SpectraARFluorineModule(
        model=model,
        tokenizer=tokenizer,
        classifier=classifier,
    ).to(device)
    if mode == "lora":
        model.requires_grad_(False)
        lora_config: dict[str, Any] | None = _lora_config(
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )
        applied_modules = apply_ar_fluorine_lora(model, lora_config)
        model.to(device)
        lora_config = {**lora_config, "applied_modules": list(applied_modules)}
        model_param_group = {
            "params": list(lora_parameters(model)),
            "lr": lora_learning_rate,
            "weight_decay": weight_decay,
        }
    else:
        model.requires_grad_(True)
        lora_config = None
        model_param_group = {
            "params": model.parameters(),
            "lr": model_learning_rate,
            "weight_decay": weight_decay,
        }
    optimizer = torch.optim.AdamW(
        [
            model_param_group,
            {
                "params": classifier.parameters(),
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
    best_lora_state: dict[str, torch.Tensor] = {}
    best_model_state: dict[str, torch.Tensor] = {}
    best_classifier_state: dict[str, torch.Tensor] = {}
    history: list[dict[str, Any]] = []
    epochs_without_improvement = 0
    use_autocast = device.type == "cuda" and autocast_dtype is not None
    grad_scaler = build_grad_scaler(autocast_dtype, device)
    best_state_path = state_path.with_name(f"{state_path.stem}.best.pt")

    requested_hparams = {
        "head_type": "eos_label_tokens",
        "model_learning_rate": float(model_learning_rate),
        "head_learning_rate": float(head_learning_rate),
        "weight_decay": float(weight_decay),
        "autocast_dtype": _autocast_dtype_name(autocast_dtype),
        "epochs": int(epochs),
        "patience": int(patience),
        "select_metric": select_metric,
    }
    if mode == "lora":
        requested_hparams.update(
            {
                "lora_rank": int(lora_rank),
                "lora_alpha": float(lora_alpha),
                "lora_dropout": float(lora_dropout),
                "lora_learning_rate": float(lora_learning_rate),
            }
        )

    def make_state(
        *,
        test_metrics: dict[str, float] | None,
        complete: bool,
    ) -> dict[str, Any]:
        state = {
            "mode": f"ar_{mode}",
            "complete": complete,
            "config_path": str(config_path),
            "checkpoint_path": str(checkpoint_path),
            "input_dim": int(input_dim),
            "model_dim": int(model.config.model_dim),
            "pooling": "eos",
            "pair_dim": int(model.config.model_dim),
            "classifier_state": best_classifier_state,
            "best_epoch": int(best_epoch),
            "best_val": best_val,
            "test": test_metrics,
            "history": history,
            "hparams": requested_hparams,
            "autocast_dtype": _autocast_dtype_name(autocast_dtype),
            "focal_alpha": focal_alpha,
            "focal_gamma": focal_gamma,
            "finetune_cache_dir": str(cache_dir),
            "device_ids": [device.index] if device.type == "cuda" else [],
            "train_size": int(data.metadata["train_size"]),
            "train_positive": int(data.metadata["train_positive"]),
            "val_size": int(data.metadata["val_size"]),
            "val_positive": int(data.metadata["val_positive"]),
            "max_train_samples": max_train_samples,
            "max_val_samples": max_val_samples,
            "tokenizer_config": asdict(tokenizer.config),
        }
        if mode == "lora":
            state["lora_config"] = lora_config
            state["lora_state"] = best_lora_state
        else:
            state["model_state"] = best_model_state
        return state

    for epoch_idx in range(epochs):
        finetune_module.train()
        running_loss = 0.0
        seen = 0
        pbar = tqdm(
            train_loader,
            desc=f"ar {mode} epoch {epoch_idx + 1}/{epochs}",
            unit="batch",
            dynamic_ncols=True,
            mininterval=5.0,
        )
        for batch in pbar:
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype if autocast_dtype is not None else torch.bfloat16,
                enabled=use_autocast,
            ):
                logits = finetune_module(batch)
                loss = binary_focal_loss_with_logits(
                    logits.float(),
                    labels,
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
            running_loss += float(loss.detach().cpu()) * int(labels.shape[0])
            seen += int(labels.shape[0])
            pbar.set_postfix(loss=f"{running_loss / float(seen):.5f}")

        val_targets, val_logits, _ = predict_ar_fluorine(
            module=finetune_module,
            loader=val_loader,
            device=device,
            autocast_dtype=autocast_dtype,
        )
        val_metrics = _metric_dict(val_targets, val_logits, "val")
        history_row: dict[str, Any] = {
            "epoch": epoch_idx + 1,
            "train_loss": running_loss / float(seen),
            "val": val_metrics,
        }
        if eval_test_every_epoch:
            epoch_test_targets, epoch_test_logits, _ = predict_ar_fluorine(
                module=finetune_module,
                loader=test_loader,
                device=device,
                autocast_dtype=autocast_dtype,
            )
            history_row["test"] = _metric_dict(
                epoch_test_targets,
                epoch_test_logits,
                "test",
            )
        history.append(history_row)
        if progress_output_prefix is not None:
            write_training_history_outputs(
                output_prefix=progress_output_prefix,
                history=history,
            )
        current_value = val_metrics[f"val/{select_metric}"]
        if current_value > best_value:
            best_value = current_value
            best_epoch = epoch_idx + 1
            best_val = dict(val_metrics)
            if mode == "lora":
                best_lora_state = lora_state_dict(model)
            else:
                best_model_state = _module_state_to_cpu(model)
            best_classifier_state = copy.deepcopy(_module_state_to_cpu(classifier))
            epochs_without_improvement = 0
            state_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(make_state(test_metrics=None, complete=False), best_state_path)
        else:
            epochs_without_improvement += 1
        log.info(
            "ar_%s epoch=%d/%d train_loss=%.5f val_ap=%.4f val_auc=%.4f",
            mode,
            epoch_idx + 1,
            epochs,
            running_loss / float(seen),
            val_metrics["val/average_precision"],
            val_metrics["val/roc_auc"],
        )
        if epochs_without_improvement >= patience:
            break

    if mode == "lora":
        load_lora_state_dict(model, best_lora_state)
    else:
        model.load_state_dict(best_model_state)
    classifier.load_state_dict(best_classifier_state)
    test_targets, test_logits, test_row_indices = predict_ar_fluorine(
        module=finetune_module,
        loader=test_loader,
        device=device,
        autocast_dtype=autocast_dtype,
    )
    test_metrics = _metric_dict(test_targets, test_logits, "test")
    state = make_state(test_metrics=test_metrics, complete=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, state_path)
    return state, data, test_targets, test_logits, test_row_indices


def _resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def _default_output_prefix(global_step: int, mode: str) -> Path:
    return Path("results") / f"ar_fluorine_{mode}_step_{global_step}" / "mcebio"


def _default_state_path(output_prefix: StoragePath, mode: str) -> Path:
    return Path(storage_parent(output_prefix)) / f"ar_{mode}_state.pt"


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = _resolve_device(args.device)
    config, tokenizer, model, checkpoint = load_ar_checkpoint_model(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=device,
    )
    autocast_dtype = _resolve_autocast_dtype(config, args.autocast_dtype)
    global_step = int(checkpoint["global_step"])
    output_prefix: StoragePath = (
        args.output_prefix
        if args.output_prefix is not None
        else _default_output_prefix(global_step, args.mode)
    )
    state_path = (
        args.state_path
        if args.state_path is not None
        else _default_state_path(output_prefix, args.mode)
    )
    state, data, test_targets, test_logits, test_row_indices = train_ar_fluorine(
        mode=args.mode,
        state_path=state_path,
        model=model,
        tokenizer=tokenizer,
        config=config,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        cache_dir=args.cache_dir,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        model_learning_rate=args.model_learning_rate,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_learning_rate=args.lora_learning_rate,
        head_learning_rate=args.head_learning_rate,
        weight_decay=args.weight_decay,
        autocast_dtype=autocast_dtype,
        revision=args.revision,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        select_metric=args.select_metric,
        progress_output_prefix=output_prefix,
        eval_test_every_epoch=args.eval_test_every_epoch,
    )
    summary = write_standard_fluorine_outputs(
        output_prefix=output_prefix,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        head_state_path=state_path,
        data=data,
        targets=test_targets,
        logits=test_logits,
        row_indices=test_row_indices,
        head_state=state,
    )
    curve_dirs = [storage_parent(output_prefix), *args.curve_dirs]
    if args.include_results_curves:
        curve_dirs.append(Path("results"))
    comparison = write_all_pr_curve_comparison(
        output_prefix=output_prefix,
        curve_dirs=curve_dirs,
    )
    return {
        "summary": summary,
        "comparison": comparison,
        "output_prefix": str(output_prefix),
        "state_path": str(state_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune the AR spectra checkpoint with an EOS label-token "
            "fluorine head on NIST Murcko, using either LoRA adapters or full "
            "model fine-tuning, then evaluate MCEBIO Murcko."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("configs/ar_spectra_coarse_to_fine.py"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/fluorine_detection_fine_tuned"),
    )
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--state-path", type=Path, default=None)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("lora", "full"), default="lora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--model-learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-learning-rate", type=float, default=1e-4)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--autocast-dtype",
        choices=("bf16", "bfloat16", "fp16", "float16", "fp32", "float32", "none"),
        default=None,
    )
    parser.add_argument(
        "--select-metric",
        choices=("average_precision", "roc_auc", "balanced_accuracy", "f1"),
        default="average_precision",
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--eval-test-every-epoch", action="store_true")
    parser.add_argument("--curve-dir", dest="curve_dirs", action="append", default=[])
    parser.add_argument("--include-results-curves", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    payload = run(build_arg_parser().parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
