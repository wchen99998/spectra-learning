from __future__ import annotations

import math
import logging
import warnings
from pathlib import Path
from typing import Any, Iterator

import lightning.pytorch as pl
import torch
import torch._inductor.config as inductor_config
import torch.nn.functional as F

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from ml_collections import config_dict

from input_pipeline import TfLightningDataModule, numpy_batch_to_torch
from models.model import PeakSetSIGReg
from utils.schedulers import learning_rate_at_step
from utils.training import build_logger, build_model_from_config, latest_ckpt_path


def _configure_runtime_warning_filters() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
        module=r"lightning\.pytorch\.utilities\._pytree",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"The '.*_dataloader' does not have many workers which may be a bottleneck.*",
        module=r"lightning\.pytorch\.trainer\.connectors\.data_connector",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Your `IterableDataset` has `__len__` defined.*",
        module=r"lightning\.pytorch\.utilities\.data",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Found \d+ module\(s\) in eval mode at the start of training.*",
        module=r"lightning\.pytorch\.loops\.fit_loop",
    )


_configure_runtime_warning_filters()

torch.set_float32_matmul_precision('high')
inductor_config.coordinate_descent_tuning = True
inductor_config.triton.unique_kernel_names = True
inductor_config.fx_graph_cache = True
inductor_config.epilogue_fusion = True
inductor_config.shape_padding = True


def _iter_massspec_probe(
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


def _probe_steps_per_epoch(
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


def _precursor_mz_to_bins(
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


class _FinalLinearProbe(torch.nn.Module):
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


_PROBE_TASK_NAMES = ("adduct", "precursor_bin", "instrument")


class _FinalAttentiveProbe(torch.nn.Module):
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
    probe: _FinalAttentiveProbe | _FinalLinearProbe,
    backbone: PeakSetSIGReg,
    batch: dict[str, torch.Tensor],
    *,
    feature_source: str,
    num_precursor_bins: int,
    precursor_max_mz: float,
    loss_weights: dict[str, float],
    compiled_encoder: torch.nn.Module | None = None,
) -> dict[str, torch.Tensor]:
    feature_tokens, feature_mask = _extract_probe_features(
        backbone, batch, feature_source=feature_source,
        compiled_encoder=compiled_encoder,
    )
    logits = probe(feature_tokens, feature_mask)
    targets = {
        "adduct": batch["adduct_id"].to(torch.long),
        "precursor_bin": _precursor_mz_to_bins(
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
    loggers: tuple = (),
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
    loss_weights = dict(zip(_PROBE_TASK_NAMES, loss_weight_list))
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

    probe = _FinalAttentiveProbe(
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
    steps_per_epoch = _probe_steps_per_epoch(
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
        train_sums: dict[str, float] = {}
        train_count = 0
        train_seed = int(config.seed) + 8_000_000 + epoch_idx
        for batch in _iter_massspec_probe(
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
            for name in _PROBE_TASK_NAMES:
                key = f"loss_{name}"
                weighted = result["losses"][name].detach() * bs
                train_sums[key] = train_sums[key] + weighted if key in train_sums else weighted
            weighted = result["loss_total"].detach() * bs
            train_sums["loss_total"] = train_sums["loss_total"] + weighted if "loss_total" in train_sums else weighted

        # --- Eval ---
        probe.eval()
        test_sums: dict[str, float] = {}
        test_count = 0
        test_seed = int(config.seed) + 9_000_000 + epoch_idx
        with torch.no_grad():
            for batch in _iter_massspec_probe(
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
                for name in _PROBE_TASK_NAMES:
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
    loggers: tuple = (),
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
    loss_weights = dict(zip(_PROBE_TASK_NAMES, loss_weight_list))
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

    probe = _FinalLinearProbe(
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
    steps_per_epoch = _probe_steps_per_epoch(
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
        train_sums: dict[str, float] = {}
        train_count = 0
        train_seed = int(config.seed) + 8_000_000 + epoch_idx
        for batch in _iter_massspec_probe(
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
            for name in _PROBE_TASK_NAMES:
                key = f"loss_{name}"
                weighted = result["losses"][name].detach() * bs
                train_sums[key] = train_sums[key] + weighted if key in train_sums else weighted
            weighted = result["loss_total"].detach() * bs
            train_sums["loss_total"] = train_sums["loss_total"] + weighted if "loss_total" in train_sums else weighted

        # --- Eval ---
        probe.eval()
        test_sums: dict[str, float] = {}
        test_count = 0
        test_seed = int(config.seed) + 9_000_000 + epoch_idx
        with torch.no_grad():
            for batch in _iter_massspec_probe(
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
                for name in _PROBE_TASK_NAMES:
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


class TorchProfilerCallback(pl.Callback):
    def __init__(
        self,
        *,
        trace_dir: Path,
        wait_steps: int,
        warmup_steps: int,
        active_steps: int,
        repeat: int,
        record_shapes: bool,
        with_stack: bool,
        profile_memory: bool,
    ) -> None:
        super().__init__()
        self.trace_dir = Path(trace_dir)
        self.wait_steps = int(wait_steps)
        self.warmup_steps = int(warmup_steps)
        self.active_steps = int(active_steps)
        self.repeat = int(repeat)
        self.record_shapes = bool(record_shapes)
        self.with_stack = bool(with_stack)
        self.profile_memory = bool(profile_memory)
        self._profiler = None
        self._trace_idx = 0

    def _on_trace_ready(self, profiler) -> None:
        trace_path = self.trace_dir / f"trace_{self._trace_idx:03d}.json"
        profiler.export_chrome_trace(str(trace_path))
        self._trace_idx += 1

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if trainer.strategy.root_device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        schedule = torch.profiler.schedule(
            wait=self.wait_steps,
            warmup=self.warmup_steps,
            active=self.active_steps,
            repeat=self.repeat,
        )
        self._profiler = torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=self._on_trace_ready,
            record_shapes=self.record_shapes,
            with_stack=self.with_stack,
            profile_memory=self.profile_memory,
        )
        self._profiler.start()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del trainer, pl_module, outputs, batch, batch_idx
        self._profiler.step()

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del trainer, pl_module
        self._profiler.stop()
        self._profiler = None


def _partition_params_for_muon(
    model: torch.nn.Module,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Split parameters into 2D (Muon-eligible) and non-2D (AdamW-only)."""
    muon_params = []
    adamw_params = []
    for param in model.parameters():
        if param.ndim == 2:
            muon_params.append(param)
        else:
            adamw_params.append(param)
    return muon_params, adamw_params


class MAELightningModule(pl.LightningModule):
    def __init__(
        self,
        config: config_dict.ConfigDict,
        *,
        total_steps: int,
    ) -> None:
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters(ignore=["config"])
        self.config = config
        self.total_steps = int(total_steps)

        self.base_lr = float(config.learning_rate)
        self.warmup_steps = int(config.get("warmup_steps", 0))
        self.schedule_type = str(config.get("learning_rate_schedule", "cosine"))
        self.min_learning_rate = config.get("min_learning_rate", None)
        self.b2 = float(config.get("b2", 0.999))
        self.checkpoint_every_steps = int(config.checkpoint_every_steps)
        self.clip_value = float(config.get("clip", 0.0))
        self.optimizer_type = str(config.get("optimizer", "adamw")).lower()
        self._prefetch_stream: torch.cuda.Stream | None = None
        self._prefetched_batch: dict[str, torch.Tensor] | None = None
        self._prefetch_iter: Iterator | None = None

        self.model = build_model_from_config(config)

        # Compile train forward with CUDA graphs for max throughput
        self._train_forward = torch.compile(
            self._train_forward_impl,
            mode="max-autotune",
            fullgraph=True,
        )

    def _lr_for_step(self, step: int) -> float:
        return learning_rate_at_step(
            step,
            base_lr=self.base_lr,
            total_steps=self.total_steps,
            warmup_steps=self.warmup_steps,
            schedule_type=self.schedule_type,
            min_learning_rate=self.min_learning_rate,
        )

    def _lr_lambda(self, step_idx: int) -> float:
        step = step_idx + 1
        return self._lr_for_step(step) / self.base_lr

    def _make_lr_lambda(self, base_lr: float):
        """Return a lr_lambda closure over a specific base_lr."""
        def lr_lambda(step_idx: int) -> float:
            return self._lr_for_step(step_idx + 1) / base_lr
        return lr_lambda

    def _train_forward_impl(
        self,
        batch: dict[str, torch.Tensor],
        bcs_projection: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.model.forward_augmented(batch, bcs_projection=bcs_projection)

    def _sample_bcs_projection(self, seed: int) -> torch.Tensor:
        return self.model.sample_bcs_projection(device=self.device, seed=seed)

    @staticmethod
    def _record_stream(batch: Any, stream: torch.cuda.Stream) -> None:
        if isinstance(batch, torch.Tensor):
            batch.record_stream(stream)
            return
        if isinstance(batch, dict):
            for value in batch.values():
                MAELightningModule._record_stream(value, stream)
            return
        if isinstance(batch, (list, tuple)):
            for value in batch:
                MAELightningModule._record_stream(value, stream)

    _TRAIN_BATCH_KEYS = frozenset({
        "fused_mz", "fused_intensity", "fused_precursor_mz",
        "fused_valid_mask", "fused_masked_positions",
        "view1_masked_fraction",
    })

    def _move_train_batch_to_device(
        self,
        batch: dict[str, torch.Tensor],
        dataloader_idx: int,
    ) -> dict[str, torch.Tensor]:
        batch = {k: v for k, v in batch.items() if k in self._TRAIN_BATCH_KEYS}
        batch = self._move_batch_to_device(batch, dataloader_idx=dataloader_idx)
        return batch

    def _move_batch_to_device(
        self,
        batch: dict[str, torch.Tensor],
        dataloader_idx: int,
    ) -> dict[str, torch.Tensor]:
        batch = self.trainer.precision_plugin.convert_input(batch)
        batch = self._on_before_batch_transfer(batch, dataloader_idx=dataloader_idx)
        return self.trainer.strategy.batch_to_device(batch, dataloader_idx=dataloader_idx)

    def _prefetch_next_train_batch(self, dataloader_iter: Iterator) -> None:
        try:
            cpu_batch, _, dataloader_idx = next(dataloader_iter)
        except StopIteration:
            self._prefetched_batch = None
            return
        if self.device.type == "cuda":
            assert self._prefetch_stream is not None
            with torch.cuda.stream(self._prefetch_stream):
                self._prefetched_batch = self._move_train_batch_to_device(cpu_batch, dataloader_idx)
            return
        self._prefetched_batch = self._move_train_batch_to_device(cpu_batch, dataloader_idx)

    def _next_train_batch(self, dataloader_iter: Iterator) -> dict[str, torch.Tensor]:
        if self.device.type == "cuda" and self._prefetch_stream is None:
            self._prefetch_stream = torch.cuda.Stream(device=self.device)
        if self._prefetch_iter is not dataloader_iter:
            self._prefetch_iter = dataloader_iter
            self._prefetched_batch = None
        if self._prefetched_batch is None:
            self._prefetch_next_train_batch(dataloader_iter)
        assert self._prefetched_batch is not None
        if self.device.type == "cuda":
            assert self._prefetch_stream is not None
            current_stream = torch.cuda.current_stream(device=self.device)
            current_stream.wait_stream(self._prefetch_stream)
            self._record_stream(self._prefetched_batch, current_stream)
        batch = self._prefetched_batch
        self._prefetch_next_train_batch(dataloader_iter)
        return batch

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if dataloader_idx != 1:
            return
        bcs_projection = self._sample_bcs_projection(
            int(self.config.seed) + 7_000_000 + self.current_epoch * 100_000 + batch_idx
        )
        torch.compiler.cudagraph_mark_step_begin()
        metrics = self._train_forward(batch, bcs_projection)
        batch_size = int(batch["fused_mz"].shape[0]) // 2
        for key, value in metrics.items():
            self.log(
                f"msg_eval/{key}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=(key == "loss"),
                add_dataloader_idx=False,
                batch_size=batch_size,
            )

    def training_step(self, dataloader_iter: Iterator) -> None:
        batch = self._next_train_batch(dataloader_iter)
        batch_size = int(batch["fused_mz"].shape[0]) // 2
        bcs_projection = self._sample_bcs_projection(
            int(self.config.seed) + 6_000_000 + self.global_step
        )
        torch.compiler.cudagraph_mark_step_begin()
        metrics = self._train_forward(batch, bcs_projection)

        self.manual_backward(metrics["loss"])

        # Lightning returns unwrapped object for single opt, list for multiple.
        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        schedulers = self.lr_schedulers()
        if not isinstance(schedulers, list):
            schedulers = [schedulers]

        if self.clip_value > 0 and self.optimizer_type != "muon":
            for opt in optimizers:
                self.clip_gradients(
                    opt,
                    gradient_clip_val=self.clip_value,
                    gradient_clip_algorithm="value",
                )
        for opt in optimizers:
            opt.step()
            opt.zero_grad()
        for sched in schedulers:
            sched.step()

        step = self.global_step + 1
        self.log(
            "train/learning_rate",
            self._lr_for_step(step),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=batch_size,
        )
        for key, value in metrics.items():
            if key == "loss":
                self.log(
                    "train/loss",
                    value,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                    batch_size=batch_size,
                )
                if step % self.checkpoint_every_steps == 0:
                    self.log(
                        "train/loss_checkpoint",
                        value,
                        on_step=True,
                        on_epoch=False,
                        prog_bar=False,
                        logger=False,
                        batch_size=batch_size,
                    )
                continue
            metric_name = f"train/{key}"
            self.log(
                metric_name,
                value,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                batch_size=batch_size,
            )

    def configure_optimizers(self):
        capturable = bool(self.config.get("optimizer_capturable", True))
        capturable = capturable and self.trainer.strategy.root_device.type == "cuda"
        fused_cfg = self.config.get("optimizer_fused", None)
        if fused_cfg is None:
            fused = self.trainer.strategy.root_device.type == "cuda"
        else:
            fused = bool(fused_cfg) and self.trainer.strategy.root_device.type == "cuda"
        weight_decay = float(self.config.weight_decay)

        if self.optimizer_type == "muon":
            muon_params, adamw_params = _partition_params_for_muon(self.model)
            muon_lr = float(self.config.get("muon_lr", None) or self.base_lr)
            adamw_lr = float(self.config.get("adamw_lr", None) or self.base_lr)
            muon_wd = float(self.config.get("muon_weight_decay", None) or weight_decay)

            muon_opt = torch.optim.Muon(
                muon_params,
                lr=muon_lr,
                momentum=float(self.config.get("muon_momentum", 0.95)),
                nesterov=bool(self.config.get("muon_nesterov", True)),
                ns_steps=int(self.config.get("muon_ns_steps", 5)),
                weight_decay=muon_wd,
                adjust_lr_fn=str(self.config.get("muon_adjust_lr_fn", "match_rms_adamw")),
            )
            adamw_opt = torch.optim.AdamW(
                adamw_params,
                lr=adamw_lr,
                betas=(0.9, self.b2),
                weight_decay=weight_decay,
                capturable=capturable,
                fused=fused,
            )
            muon_sched = torch.optim.lr_scheduler.LambdaLR(
                muon_opt, lr_lambda=self._make_lr_lambda(muon_lr),
            )
            adamw_sched = torch.optim.lr_scheduler.LambdaLR(
                adamw_opt, lr_lambda=self._make_lr_lambda(adamw_lr),
            )
            return [muon_opt, adamw_opt], [muon_sched, adamw_sched]

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.base_lr,
            betas=(0.9, self.b2),
            weight_decay=weight_decay,
            capturable=capturable,
            fused=fused,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=self._lr_lambda,
        )
        return [optimizer], [scheduler]


def train_and_evaluate(
    config: config_dict.ConfigDict,
    workdir: str | Path,
    *,
    extra_callbacks: list[pl.Callback] = (),
) -> dict[str, float]:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    pl.seed_everything(int(config.seed), workers=False)

    datamodule = TfLightningDataModule(config, seed=int(config.seed))
    num_epochs = int(config.num_epochs)
    steps_per_epoch = datamodule.train_steps
    total_steps = num_epochs * steps_per_epoch
    limit_train_batches = config.get("limit_train_batches", 1.0)
    configured_log_every_n_steps = int(config.get("log_every_n_steps", 50))
    configured_val_check_interval = config.get("val_check_interval", 1.0)

    # Update config with dataset-derived values
    info = datamodule.info
    config.num_peaks = info["num_peaks"]
    config.fingerprint_bits = int(info["fingerprint_bits"])

    logging.info("Training with Lightning for %d epochs.", num_epochs)
    logging.info("Steps per epoch: %d", steps_per_epoch)
    logging.info("Total steps: %d", total_steps)
    logging.info(
        "Trainer cadence: log_every_n_steps=%d, val_check_interval=%s.",
        configured_log_every_n_steps,
        configured_val_check_interval,
    )

    module = MAELightningModule(
        config,
        total_steps=total_steps,
    )

    checkpoint_dir = Path(str(workdir)) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="step-{step:08d}",
        every_n_train_steps=int(config.checkpoint_every_steps),
        monitor="train/loss_checkpoint",
        mode="min",
        save_last=True,
        save_top_k=5,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    logger = build_logger(config, Path(str(workdir)))

    callbacks: list[pl.Callback] = [checkpoint_cb, lr_monitor, *extra_callbacks]
    if bool(config.get("profile_enabled", False)):
        callbacks.append(
            TorchProfilerCallback(
                trace_dir=workdir / str(config.get("profile_trace_dir", "profiler")),
                wait_steps=int(config.get("profile_wait_steps", 20)),
                warmup_steps=int(config.get("profile_warmup_steps", 20)),
                active_steps=int(config.get("profile_active_steps", 40)),
                repeat=int(config.get("profile_repeat", 1)),
                record_shapes=bool(config.get("profile_record_shapes", True)),
                with_stack=bool(config.get("profile_with_stack", True)),
                profile_memory=bool(config.get("profile_profile_memory", True)),
            )
        )

    trainer = pl.Trainer(
        default_root_dir=str(workdir),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed",
        max_epochs=num_epochs,
        log_every_n_steps=configured_log_every_n_steps,
        val_check_interval=configured_val_check_interval,
        callbacks=callbacks,
        logger=logger,
        reload_dataloaders_every_n_epochs=1,
        limit_train_batches=limit_train_batches,
        limit_val_batches=config.get("limit_val_batches", 1.0),
        limit_test_batches=0,
        num_sanity_val_steps=int(config.get("num_sanity_val_steps", 0)),
    )

    ckpt_path = latest_ckpt_path(Path(str(workdir)))

    if ckpt_path is not None:
        logging.info("Resuming from checkpoint: %s", ckpt_path)

    trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)
    final_probe_metrics = run_attentive_probe(
        config=config,
        datamodule=datamodule,
        model=module.model,
        device=trainer.strategy.root_device,
        loggers=trainer.loggers,
        global_step=module.global_step,
    )
    for key, value in final_probe_metrics.items():
        logging.info("%s = %.6f", key, value)
    return final_probe_metrics
