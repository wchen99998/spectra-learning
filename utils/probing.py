from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn.functional as F

from ml_collections import config_dict

from input_pipeline import TfLightningDataModule, numpy_batch_to_torch
from models.model import PeakSetSIGReg
from utils.schedulers import learning_rate_at_step


PROBE_TASK_NAMES = ("adduct", "precursor_bin", "instrument")


def iter_massspec_probe(
    datamodule: TfLightningDataModule,
    split: str,
    *,
    seed: int,
    peak_ordering: str,
    drop_remainder: bool,
):
    dataset = datamodule.build_massspec_probe_dataset(
        split,
        seed=seed,
        peak_ordering=peak_ordering,
        shuffle=(split == "massspec_train"),
        drop_remainder=drop_remainder,
    )
    size_key = {
        "massspec_train": "massspec_train_size",
        "massspec_val": "massspec_val_size",
        "massspec_test": "massspec_test_size",
    }[split]
    size = int(datamodule.info[size_key])
    seen = 0
    for batch in dataset.as_numpy_iterator():
        remaining = size - seen
        if remaining <= 0:
            break
        take = min(int(batch["peak_mz"].shape[0]), remaining)
        if take != batch["peak_mz"].shape[0]:
            batch = {key: value[:take] for key, value in batch.items()}
        seen += take
        yield numpy_batch_to_torch(batch)


def probe_steps_per_epoch(
    datamodule: TfLightningDataModule,
    *,
    split: str,
    drop_remainder: bool,
) -> int:
    size_key = {
        "massspec_train": "massspec_train_size",
        "massspec_val": "massspec_val_size",
        "massspec_test": "massspec_test_size",
    }[split]
    size = int(datamodule.info[size_key])
    batch_size = int(datamodule.batch_size)
    if drop_remainder:
        return size // batch_size
    return math.ceil(size / batch_size)


def precursor_mz_to_bins(
    precursor_mz: torch.Tensor,
    *,
    num_bins: int,
    max_mz: float,
) -> torch.Tensor:
    mz = precursor_mz * max_mz
    bin_width = max_mz / float(num_bins)
    return torch.floor(mz / bin_width).to(torch.long).clamp(min=0, max=num_bins - 1)


def _extract_probe_features(
    backbone: PeakSetSIGReg,
    batch: dict[str, torch.Tensor],
    *,
    feature_source: str,
    compiled_encoder: torch.nn.Module | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoder = compiled_encoder if compiled_encoder is not None else backbone.encoder
    embeddings = encoder(
        batch["peak_mz"],
        batch["peak_intensity"],
        valid_mask=batch["peak_valid_mask"],
    )
    if feature_source == "encoder":
        return embeddings, batch["peak_valid_mask"]
    if feature_source == "projector":
        pooled = backbone.pool(embeddings, batch["peak_valid_mask"])
        projected = backbone.projector(pooled)
        feature_tokens = projected.unsqueeze(1)
        feature_mask = torch.ones(
            projected.shape[0],
            1,
            dtype=torch.bool,
            device=projected.device,
        )
        return feature_tokens, feature_mask
    raise ValueError(f"Unknown feature_source: {feature_source!r}")


class FinalLinearProbe(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        head_dims: dict[str, int],
    ) -> None:
        super().__init__()
        self.heads = torch.nn.ModuleDict({
            name: torch.nn.Linear(input_dim, dim)
            for name, dim in head_dims.items()
        })

    def forward(
        self,
        feature_tokens: torch.Tensor,
        feature_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Mean-pool over valid tokens.
        mask = feature_mask.unsqueeze(-1).float()
        pooled = (feature_tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return {name: head(pooled) for name, head in self.heads.items()}


class FinalAttentiveProbe(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_attention_heads: int,
        head_dims: dict[str, int],
    ) -> None:
        super().__init__()
        self.heads = torch.nn.ModuleDict({
            name: torch.nn.Linear(hidden_dim, dim)
            for name, dim in head_dims.items()
        })
        num_tasks = len(self.heads)
        self.task_queries = torch.nn.Parameter(torch.empty(num_tasks, input_dim))
        torch.nn.init.xavier_normal_(self.task_queries)
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_attention_heads,
            batch_first=True,
        )
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.RMSNorm(hidden_dim),
            torch.nn.SiLU(),
        )

    def forward(
        self,
        feature_tokens: torch.Tensor,
        feature_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        query = self.task_queries.unsqueeze(0).expand(feature_tokens.shape[0], -1, -1)
        attended, _ = self.attention(
            query=query,
            key=feature_tokens,
            value=feature_tokens,
            key_padding_mask=~feature_mask,
            need_weights=False,
        )
        states = self.trunk(attended)
        return {
            name: head(states[:, i, :])
            for i, (name, head) in enumerate(self.heads.items())
        }


def _probe_step(
    probe: FinalAttentiveProbe | FinalLinearProbe,
    backbone: PeakSetSIGReg,
    batch: dict[str, torch.Tensor],
    *,
    feature_source: str,
    num_precursor_bins: int,
    precursor_max_mz: float,
    loss_weights: dict[str, float],
    compiled_encoder: torch.nn.Module | None = None,
) -> dict[str, Any]:
    feature_tokens, feature_mask = _extract_probe_features(
        backbone, batch, feature_source=feature_source,
        compiled_encoder=compiled_encoder,
    )
    logits = probe(feature_tokens, feature_mask)
    targets = {
        "adduct": batch["adduct_id"].to(torch.long),
        "precursor_bin": precursor_mz_to_bins(
            batch["precursor_mz"].to(torch.float32),
            num_bins=num_precursor_bins,
            max_mz=precursor_max_mz,
        ),
        "instrument": batch["instrument_type_id"].to(torch.long),
    }
    losses = {
        name: F.cross_entropy(logits[name], targets[name])
        for name in logits
    }
    loss_total = sum(loss_weights[name] * losses[name] for name in losses)
    return {
        "logits": logits,
        "targets": targets,
        "losses": losses,
        "loss_total": loss_total,
        "batch_size": feature_tokens.shape[0],
    }


def run_attentive_probe(
    *,
    config: config_dict.ConfigDict,
    datamodule: TfLightningDataModule,
    model: PeakSetSIGReg,
    device: torch.device,
    loggers: tuple[Any, ...] = (),
    global_step: int = 0,
) -> dict[str, float]:
    def log_trial_metrics(metrics: dict[str, float], step: int) -> None:
        for pl_logger in loggers:
            pl_logger.log_metrics(metrics, step=step)
            experiment = pl_logger.experiment
            if hasattr(experiment, "log"):
                experiment.log(metrics, step=step)

    def move_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v.to(device) for k, v in batch.items()}

    num_probe_epochs = int(config.final_probe_num_epochs)
    probe_lr = float(config.final_probe_learning_rate)
    probe_weight_decay = float(config.final_probe_weight_decay)
    probe_warmup_steps = int(config.final_probe_warmup_steps)
    probe_hidden_dim = int(config.final_probe_head_hidden_dim)
    probe_feature_source = str(config.final_probe_feature_source)
    probe_peak_ordering = str(config.peak_ordering)
    num_precursor_bins = int(config.final_probe_num_precursor_bins)
    precursor_max_mz = float(config.max_precursor_mz)
    num_attention_heads = int(config.final_probe_attention_heads)
    loss_weight_list = config.get("final_probe_loss_weights", [1.0, 1.0, 1.0])
    loss_weights = dict(zip(PROBE_TASK_NAMES, loss_weight_list))
    probe_drop_remainder = False

    num_adduct_classes = int(datamodule.info["massspec_adduct_vocab_size"])
    num_instrument_classes = int(datamodule.info["massspec_instrument_type_vocab_size"])
    use_projector = probe_feature_source == "projector" and bool(config.get("sigreg_use_projector", True))
    input_dim = int(config.get("sigreg_proj_output_dim", 128)) if use_projector else int(config.model_dim)

    freeze_backbone = bool(config.get("final_probe_freeze_backbone", True))

    model.to(device)
    backbone = model
    if freeze_backbone:
        backbone.eval()
        for param in backbone.parameters():
            param.requires_grad = False
    compiled_encoder = torch.compile(backbone.encoder)

    probe = FinalAttentiveProbe(
        input_dim=input_dim,
        hidden_dim=probe_hidden_dim,
        num_attention_heads=num_attention_heads,
        head_dims={
            "adduct": num_adduct_classes,
            "precursor_bin": num_precursor_bins,
            "instrument": num_instrument_classes,
        },
    ).to(device)
    optim_params = list(probe.parameters())
    if not freeze_backbone:
        optim_params += list(backbone.parameters())
    optimizer = torch.optim.AdamW(optim_params, lr=probe_lr, weight_decay=probe_weight_decay)
    steps_per_epoch = probe_steps_per_epoch(
        datamodule,
        split="massspec_train",
        drop_remainder=probe_drop_remainder,
    )
    total_steps = num_probe_epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step_idx: learning_rate_at_step(
            step_idx + 1,
            base_lr=probe_lr,
            total_steps=total_steps,
            warmup_steps=probe_warmup_steps,
            schedule_type="cosine",
            min_learning_rate=None,
        ) / probe_lr,
    )

    step_kwargs = dict(
        feature_source=probe_feature_source,
        num_precursor_bins=num_precursor_bins,
        precursor_max_mz=precursor_max_mz,
        loss_weights=loss_weights,
    )

    logging.info(
        "Final probe running on device: %s (probe=%s backbone=%s, frozen=%s)",
        device,
        next(probe.parameters()).device,
        next(model.parameters()).device,
        freeze_backbone,
    )

    final_metrics: dict[str, float] = {}
    for epoch_idx in range(num_probe_epochs):
        # --- Train ---
        probe.train()
        train_sums: dict[str, Any] = {}
        train_count = 0
        train_seed = int(config.seed) + 8_000_000 + epoch_idx
        for batch in iter_massspec_probe(
            datamodule,
            "massspec_train",
            seed=train_seed,
            peak_ordering=probe_peak_ordering,
            drop_remainder=probe_drop_remainder,
        ):
            batch = move_batch(batch)
            optimizer.zero_grad(set_to_none=True)
            result = _probe_step(probe, backbone, batch, compiled_encoder=compiled_encoder, **step_kwargs)
            result["loss_total"].backward()
            optimizer.step()
            scheduler.step()

            bs = int(result["batch_size"])
            train_count += bs
            for name in PROBE_TASK_NAMES:
                key = f"loss_{name}"
                weighted = result["losses"][name].detach() * bs
                train_sums[key] = train_sums[key] + weighted if key in train_sums else weighted
            weighted = result["loss_total"].detach() * bs
            train_sums["loss_total"] = train_sums["loss_total"] + weighted if "loss_total" in train_sums else weighted

        # --- Eval ---
        probe.eval()
        test_sums: dict[str, Any] = {}
        test_count = 0
        test_seed = int(config.seed) + 9_000_000 + epoch_idx
        with torch.no_grad():
            for batch in iter_massspec_probe(
                datamodule,
                "massspec_test",
                seed=test_seed,
                peak_ordering=probe_peak_ordering,
                drop_remainder=probe_drop_remainder,
            ):
                batch = move_batch(batch)
                result = _probe_step(probe, backbone, batch, compiled_encoder=compiled_encoder, **step_kwargs)
                bs = int(result["batch_size"])
                test_count += bs
                for name in PROBE_TASK_NAMES:
                    loss_key = f"loss_{name}"
                    acc_key = f"acc_{name}"
                    weighted_loss = result["losses"][name].detach() * bs
                    correct = (result["logits"][name].argmax(dim=1) == result["targets"][name]).sum()
                    test_sums[loss_key] = test_sums[loss_key] + weighted_loss if loss_key in test_sums else weighted_loss
                    test_sums[acc_key] = test_sums[acc_key] + correct if acc_key in test_sums else correct
                weighted_total = result["loss_total"].detach() * bs
                test_sums["loss_total"] = test_sums["loss_total"] + weighted_total if "loss_total" in test_sums else weighted_total

        epoch_metrics = {}
        for key, val in train_sums.items():
            epoch_metrics[f"final_probe/train/{key}"] = val.item() / train_count
        for key, val in test_sums.items():
            epoch_metrics[f"final_probe/test/{key}"] = val.item() / test_count

        log_step = global_step + epoch_idx + 1
        epoch_metrics["final_probe_epoch"] = float(epoch_idx + 1)
        log_trial_metrics(epoch_metrics, step=log_step)
        logging.info(
            "Final probe epoch %d/%d test_acc(adduct=%.4f precursor=%.4f instrument=%.4f)",
            epoch_idx + 1,
            num_probe_epochs,
            epoch_metrics["final_probe/test/acc_adduct"],
            epoch_metrics["final_probe/test/acc_precursor_bin"],
            epoch_metrics["final_probe/test/acc_instrument"],
        )
        final_metrics = epoch_metrics
    return final_metrics


def run_linear_probe(
    *,
    config: config_dict.ConfigDict,
    datamodule: TfLightningDataModule,
    model: PeakSetSIGReg,
    device: torch.device,
    loggers: tuple[Any, ...] = (),
    global_step: int = 0,
) -> dict[str, float]:
    def log_trial_metrics(metrics: dict[str, float], step: int) -> None:
        for pl_logger in loggers:
            pl_logger.log_metrics(metrics, step=step)
            experiment = pl_logger.experiment
            if hasattr(experiment, "log"):
                experiment.log(metrics, step=step)

    def move_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v.to(device) for k, v in batch.items()}

    num_probe_epochs = int(config.final_probe_num_epochs)
    probe_lr = float(config.final_probe_learning_rate)
    probe_weight_decay = float(config.final_probe_weight_decay)
    probe_warmup_steps = int(config.final_probe_warmup_steps)
    probe_feature_source = str(config.final_probe_feature_source)
    probe_peak_ordering = str(config.peak_ordering)
    num_precursor_bins = int(config.final_probe_num_precursor_bins)
    precursor_max_mz = float(config.max_precursor_mz)
    loss_weight_list = config.get("final_probe_loss_weights", [1.0, 1.0, 1.0])
    loss_weights = dict(zip(PROBE_TASK_NAMES, loss_weight_list))
    probe_drop_remainder = False

    num_adduct_classes = int(datamodule.info["massspec_adduct_vocab_size"])
    num_instrument_classes = int(datamodule.info["massspec_instrument_type_vocab_size"])
    use_projector = probe_feature_source == "projector" and bool(config.get("sigreg_use_projector", True))
    input_dim = int(config.get("sigreg_proj_output_dim", 128)) if use_projector else int(config.model_dim)

    freeze_backbone = bool(config.get("final_probe_freeze_backbone", True))

    model.to(device)
    backbone = model
    if freeze_backbone:
        backbone.eval()
        for param in backbone.parameters():
            param.requires_grad = False
    compiled_encoder = torch.compile(backbone.encoder)

    probe = FinalLinearProbe(
        input_dim=input_dim,
        head_dims={
            "adduct": num_adduct_classes,
            "precursor_bin": num_precursor_bins,
            "instrument": num_instrument_classes,
        },
    ).to(device)
    optim_params = list(probe.parameters())
    if not freeze_backbone:
        optim_params += list(backbone.parameters())
    optimizer = torch.optim.AdamW(optim_params, lr=probe_lr, weight_decay=probe_weight_decay)
    steps_per_epoch = probe_steps_per_epoch(
        datamodule,
        split="massspec_train",
        drop_remainder=probe_drop_remainder,
    )
    total_steps = num_probe_epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step_idx: learning_rate_at_step(
            step_idx + 1,
            base_lr=probe_lr,
            total_steps=total_steps,
            warmup_steps=probe_warmup_steps,
            schedule_type="cosine",
            min_learning_rate=None,
        ) / probe_lr,
    )

    step_kwargs = dict(
        feature_source=probe_feature_source,
        num_precursor_bins=num_precursor_bins,
        precursor_max_mz=precursor_max_mz,
        loss_weights=loss_weights,
    )

    logging.info(
        "Linear probe running on device: %s (probe=%s backbone=%s, frozen=%s)",
        device,
        next(probe.parameters()).device,
        next(model.parameters()).device,
        freeze_backbone,
    )

    final_metrics: dict[str, float] = {}
    for epoch_idx in range(num_probe_epochs):
        # --- Train ---
        probe.train()
        train_sums: dict[str, Any] = {}
        train_count = 0
        train_seed = int(config.seed) + 8_000_000 + epoch_idx
        for batch in iter_massspec_probe(
            datamodule,
            "massspec_train",
            seed=train_seed,
            peak_ordering=probe_peak_ordering,
            drop_remainder=probe_drop_remainder,
        ):
            batch = move_batch(batch)
            optimizer.zero_grad(set_to_none=True)
            result = _probe_step(probe, backbone, batch, compiled_encoder=compiled_encoder, **step_kwargs)
            result["loss_total"].backward()
            optimizer.step()
            scheduler.step()

            bs = int(result["batch_size"])
            train_count += bs
            for name in PROBE_TASK_NAMES:
                key = f"loss_{name}"
                weighted = result["losses"][name].detach() * bs
                train_sums[key] = train_sums[key] + weighted if key in train_sums else weighted
            weighted = result["loss_total"].detach() * bs
            train_sums["loss_total"] = train_sums["loss_total"] + weighted if "loss_total" in train_sums else weighted

        # --- Eval ---
        probe.eval()
        test_sums: dict[str, Any] = {}
        test_count = 0
        test_seed = int(config.seed) + 9_000_000 + epoch_idx
        with torch.no_grad():
            for batch in iter_massspec_probe(
                datamodule,
                "massspec_test",
                seed=test_seed,
                peak_ordering=probe_peak_ordering,
                drop_remainder=probe_drop_remainder,
            ):
                batch = move_batch(batch)
                result = _probe_step(probe, backbone, batch, compiled_encoder=compiled_encoder, **step_kwargs)
                bs = int(result["batch_size"])
                test_count += bs
                for name in PROBE_TASK_NAMES:
                    loss_key = f"loss_{name}"
                    acc_key = f"acc_{name}"
                    weighted_loss = result["losses"][name].detach() * bs
                    correct = (result["logits"][name].argmax(dim=1) == result["targets"][name]).sum()
                    test_sums[loss_key] = test_sums[loss_key] + weighted_loss if loss_key in test_sums else weighted_loss
                    test_sums[acc_key] = test_sums[acc_key] + correct if acc_key in test_sums else correct
                weighted_total = result["loss_total"].detach() * bs
                test_sums["loss_total"] = test_sums["loss_total"] + weighted_total if "loss_total" in test_sums else weighted_total

        epoch_metrics = {}
        for key, val in train_sums.items():
            epoch_metrics[f"linear_probe/train/{key}"] = val.item() / train_count
        for key, val in test_sums.items():
            epoch_metrics[f"linear_probe/test/{key}"] = val.item() / test_count

        log_step = global_step + epoch_idx + 1
        epoch_metrics["linear_probe_epoch"] = float(epoch_idx + 1)
        log_trial_metrics(epoch_metrics, step=log_step)
        logging.info(
            "Linear probe epoch %d/%d test_acc(adduct=%.4f precursor=%.4f instrument=%.4f)",
            epoch_idx + 1,
            num_probe_epochs,
            epoch_metrics["linear_probe/test/acc_adduct"],
            epoch_metrics["linear_probe/test/acc_precursor_bin"],
            epoch_metrics["linear_probe/test/acc_instrument"],
        )
        final_metrics = epoch_metrics
    return final_metrics
