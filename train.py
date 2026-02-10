from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
import torch._inductor.config as inductor_config
import torch.nn.functional as F

torch.set_float32_matmul_precision('high')
inductor_config.coordinate_descent_tuning = True
inductor_config.triton.unique_kernel_names = True
inductor_config.fx_graph_cache = True

import logging
from ml_collections import config_dict
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from input_pipeline import TfLightningDataModule, numpy_batch_to_torch
from models.model import PeakSetSIGReg
from utils import wandb_writer


def _parse_schedule(schedule_type: str) -> tuple[str, dict[str, str]]:
    parts = schedule_type.split(";")
    base = parts[0]
    parsed: dict[str, str] = {}
    for kv in parts[1:]:
        if "=" in kv:
            k, v = kv.split("=", 1)
            parsed[k.strip()] = v.strip()
    return base, parsed


def _cosine_decay(
    base_lr: float,
    step: int,
    total_steps: int,
    *,
    min_lr: float | None,
) -> float:
    total_steps = max(1, total_steps)
    ratio = max(0.0, step / total_steps)
    mult = 0.5 * (1.0 + math.cos(math.pi * ratio))
    decayed = mult * base_lr
    min_lr_value = min_lr if min_lr is not None else 0.1 * base_lr
    return max(min_lr_value, decayed)


def _learning_rate_at_step(
    step: int,
    *,
    base_lr: float,
    total_steps: int,
    warmup_steps: int,
    schedule_type: str,
    min_learning_rate: float | None,
) -> float:
    if warmup_steps > 0:
        warmup = min(1.0, step / warmup_steps)
        effective_step = max(0, step - warmup_steps)
        effective_total = max(1, total_steps - warmup_steps)
    else:
        warmup = 1.0
        effective_step = step
        effective_total = max(1, total_steps)

    schedule_base, parsed = _parse_schedule(schedule_type)

    if schedule_base == "cosine":
        lr = _cosine_decay(
            base_lr,
            effective_step,
            effective_total,
            min_lr=min_learning_rate,
        )
    elif schedule_base == "constant":
        lr = base_lr
    elif schedule_base == "cyclic_cosine":
        cycle_length = int(parsed.get("cycle_length", max(1, total_steps // 10)))
        min_lr = float(parsed.get("min_lr", 0.0))
        decay_factor = float(parsed.get("decay_factor", 1.0))

        cycle_index = effective_step // cycle_length
        pos_in_cycle = effective_step % cycle_length
        peak_lr = base_lr * (decay_factor ** cycle_index)
        cosine_ratio = pos_in_cycle / cycle_length
        lr = min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * cosine_ratio))
    else:
        raise NotImplementedError(f"Unknown schedule type: {schedule_base}")

    return lr * warmup


def _build_logger(config: config_dict.ConfigDict, workdir: Path) -> pl.loggers.Logger:
    if config.get("enable_wandb", False):
        from lightning.pytorch.loggers import WandbLogger

        wandb_kwargs = wandb_writer.build_wandb_init_kwargs(config)
        logger = WandbLogger(
            project=config.get("wandb_project", "md4"),
            save_dir=str(workdir),
            log_model=False,
            **wandb_kwargs,
        )
        logger.log_hyperparams(wandb_writer.config_to_wandb_dict(config))
        return logger

    return CSVLogger(save_dir=str(workdir), name="csv_logs")


def _latest_ckpt_path(directory: Path) -> str | None:
    ckpts = sorted(directory.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        return None
    return str(ckpts[-1])


def _build_model_from_config(config: config_dict.ConfigDict) -> PeakSetSIGReg:
    return PeakSetSIGReg(
        num_peaks=int(config.num_peaks),
        model_dim=int(config.model_dim),
        encoder_num_layers=int(config.num_layers),
        encoder_num_heads=int(config.num_heads),
        encoder_num_kv_heads=config.get("num_kv_heads", None),
        attention_mlp_multiple=float(config.attention_mlp_multiple),
        feature_mlp_hidden_dim=int(config.get("feature_mlp_hidden_dim", 128)),
        mz_fourier_num_frequencies=int(config.get("mz_fourier_num_frequencies", 32)),
        mz_fourier_min_freq=float(config.get("mz_fourier_min_freq", 1.0)),
        mz_fourier_max_freq=float(config.get("mz_fourier_max_freq", 100.0)),
        mz_fourier_learnable=bool(config.get("mz_fourier_learnable", False)),
        pooling_type=str(config.get("pooling_type", "pma")),
        pma_num_heads=config.get("pma_num_heads", int(config.num_heads)),
        pma_num_seeds=int(config.get("pma_num_seeds", 1)),
        sigreg_use_projector=bool(config.get("sigreg_use_projector", True)),
        sigreg_proj_hidden_dim=int(config.get("sigreg_proj_hidden_dim", 2048)),
        sigreg_proj_output_dim=int(config.get("sigreg_proj_output_dim", 128)),
        bcs_num_slices=int(config.get("sigreg_num_slices", 256)),
        sigreg_lambda=float(config.get("sigreg_lambda", 10.0)),
        sigreg_drop_prob=float(config.get("sigreg_drop_prob", 0.20)),
        sigreg_mz_jitter_std=float(config.get("sigreg_mz_jitter_std", 0.005)),
        sigreg_intensity_jitter_std=float(
            config.get("sigreg_intensity_jitter_std", 0.05)
        ),
    )


class _FingerprintMetricAccumulator:
    def __init__(self, fingerprint_bits: int) -> None:
        self.fingerprint_bits = int(fingerprint_bits)
        self.tp = torch.zeros(self.fingerprint_bits, dtype=torch.float32)
        self.fp = torch.zeros(self.fingerprint_bits, dtype=torch.float32)
        self.fn = torch.zeros(self.fingerprint_bits, dtype=torch.float32)
        self.correct_bits = 0.0
        self.total_bits = 0.0
        self.tanimoto_sum = 0.0
        self.cosine_sum = 0.0
        self.num_samples = 0.0
        self.pred_positives = 0.0
        self.target_positives = 0.0

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        probs = torch.sigmoid(logits)
        pred_bits = probs > 0.5
        target_bits = targets > 0.5

        intersection = (pred_bits & target_bits).sum(dim=1).to(dtype=torch.float32)
        union = (pred_bits | target_bits).sum(dim=1).to(dtype=torch.float32)
        tanimoto = torch.where(union > 0.0, intersection / union, torch.ones_like(union))
        self.tanimoto_sum += float(tanimoto.sum().cpu())

        cosine = F.cosine_similarity(probs, targets, dim=1)
        self.cosine_sum += float(cosine.sum().cpu())

        self.correct_bits += float((pred_bits == target_bits).sum().cpu())
        self.total_bits += float(target_bits.numel())
        self.num_samples += float(pred_bits.shape[0])
        self.pred_positives += float(pred_bits.sum().cpu())
        self.target_positives += float(target_bits.sum().cpu())

        self.tp += (pred_bits & target_bits).sum(dim=0).to(dtype=torch.float32).cpu()
        self.fp += (pred_bits & ~target_bits).sum(dim=0).to(dtype=torch.float32).cpu()
        self.fn += (~pred_bits & target_bits).sum(dim=0).to(dtype=torch.float32).cpu()

    def compute(self, device: torch.device | str) -> dict[str, torch.Tensor]:
        precision_per_bit = self.tp / (self.tp + self.fp).clamp(min=1e-8)
        recall_per_bit = self.tp / (self.tp + self.fn).clamp(min=1e-8)
        f1_per_bit = 2 * precision_per_bit * recall_per_bit / (precision_per_bit + recall_per_bit).clamp(min=1e-8)

        has_pred = (self.tp + self.fp) > 0
        has_target = (self.tp + self.fn) > 0
        precision_per_bit = torch.where(has_pred, precision_per_bit, torch.zeros_like(precision_per_bit))
        recall_per_bit = torch.where(has_target, recall_per_bit, torch.zeros_like(recall_per_bit))
        f1_per_bit = torch.where(has_pred | has_target, f1_per_bit, torch.zeros_like(f1_per_bit))

        metrics = {
            "tanimoto": self.tanimoto_sum / self.num_samples,
            "cosine_similarity": self.cosine_sum / self.num_samples,
            "bit_accuracy": self.correct_bits / self.total_bits,
            "precision": float(precision_per_bit.mean()),
            "recall": float(recall_per_bit.mean()),
            "f1": float(f1_per_bit.mean()),
            "pred_positive_rate": self.pred_positives / self.total_bits,
            "target_positive_rate": self.target_positives / self.total_bits,
        }
        return {
            key: torch.tensor(value, dtype=torch.float32, device=device)
            for key, value in metrics.items()
        }


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


class MAELightningModule(pl.LightningModule):
    def __init__(
        self,
        config: config_dict.ConfigDict,
        *,
        total_steps: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.total_steps = int(total_steps)

        self.base_lr = float(config.learning_rate)
        self.warmup_steps = int(config.get("warmup_steps", 0))
        self.schedule_type = str(config.get("learning_rate_schedule", "cosine"))
        self.min_learning_rate = config.get("min_learning_rate", None)
        self.b2 = float(config.get("b2", 0.999))
        self.fingerprint_bits = int(config.get("fingerprint_bits", 1024))
        self.eval_msg_finetune_num_epochs = int(config.get("eval_msg_finetune_num_epochs", 1))
        self.eval_msg_finetune_feature_source = str(config.get("eval_msg_finetune_feature_source", "encoder"))
        self.eval_msg_finetune_trainable_scope = str(config.get("eval_msg_finetune_trainable_scope", "full"))
        self.eval_msg_finetune_head_hidden_dim = int(config.get("eval_msg_finetune_head_hidden_dim", 512))
        self.eval_msg_finetune_learning_rate = float(config.get("eval_msg_finetune_learning_rate", 1e-4))
        self.eval_msg_finetune_weight_decay = float(config.get("eval_msg_finetune_weight_decay", 1e-4))
        self.eval_msg_finetune_warmup_steps = int(config.get("eval_msg_finetune_warmup_steps", 0))
        self.eval_msg_finetune_peak_ordering = str(
            config.get("eval_msg_finetune_peak_ordering", config.get("probe_peak_ordering", "intensity"))
        )
        self.non_blocking_device_transfer = bool(config.get("non_blocking_device_transfer", True))
        self.train_log_extra_metrics_on_step = bool(config.get("train_log_extra_metrics_on_step", False))
        self.train_step_log_interval = int(
            config.get("train_step_log_interval", config.get("log_every_n_steps", 1))
        )
        self.checkpoint_every_steps = int(config.checkpoint_every_steps)
        self.eval_msg_finetune_compile = bool(config.get("eval_msg_finetune_compile", True)) and torch.cuda.is_available()
        self.eval_msg_finetune_compile_mode = str(config.get("eval_msg_finetune_compile_mode", "reduce-overhead"))
        self.eval_msg_finetune_compile_fullgraph = bool(config.get("eval_msg_finetune_compile_fullgraph", False))
        self.eval_msg_finetune_compile_dynamic = bool(config.get("eval_msg_finetune_compile_dynamic", True))
        self._eval_backbone: PeakSetSIGReg | None = None
        self._eval_head: torch.nn.Module | None = None
        self._eval_head_init_state: dict[str, torch.Tensor] | None = None
        self._eval_feature_forward: Any = None
        self._eval_head_forward: Any = None

        self.model = _build_model_from_config(config)

        # Compile train forward with CUDA graphs for max throughput
        self._train_forward = torch.compile(
            self._train_forward_impl,
            mode="max-autotune",
            fullgraph=True,
        )

    def _lr_for_step(self, step: int) -> float:
        return _learning_rate_at_step(
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

    def _train_forward_impl(
        self,
        batch: dict[str, torch.Tensor],
        bcs_projection: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.model(batch, train=True, bcs_projection=bcs_projection)

    def _sample_bcs_projection(self, seed: int) -> torch.Tensor:
        return self.model.sample_bcs_projection(device=self.device, seed=seed)

    def _to_device(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        non_blocking = self.non_blocking_device_transfer
        return {
            k: v.to(self.device, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _iter_massspec_probe(self, split: str, *, seed: int):
        dm = self.trainer.datamodule
        ds = dm.build_massspec_probe_dataset(
            split,
            seed=seed,
            peak_ordering=self.eval_msg_finetune_peak_ordering,
        )
        size_key = "massspec_train_size" if split == "massspec_train" else "massspec_test_size"
        size = int(dm.info[size_key])
        seen = 0
        for batch in ds.as_numpy_iterator():
            remaining = size - seen
            if remaining <= 0:
                break
            take = min(int(batch["peak_mz"].shape[0]), remaining)
            if take != batch["peak_mz"].shape[0]:
                batch = {key: value[:take] for key, value in batch.items()}
            seen += take
            yield numpy_batch_to_torch(batch)

    def _extract_eval_features(
        self,
        model: PeakSetSIGReg,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        embeddings = model.encoder(
            batch["peak_mz"],
            batch["peak_intensity"],
            batch["precursor_mz"],
        )
        pooled = model._pool(embeddings, batch["peak_valid_mask"])
        if self.eval_msg_finetune_feature_source == "projector":
            return model.projector(pooled)
        return pooled

    def _build_eval_head(self, input_dim: int) -> torch.nn.Module:
        hidden = self.eval_msg_finetune_head_hidden_dim
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            torch.nn.RMSNorm(hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, self.fingerprint_bits),
        )

    def _compile_eval_callable(self, fn):
        if not self.eval_msg_finetune_compile:
            return fn
        return torch.compile(
            fn,
            mode=self.eval_msg_finetune_compile_mode,
            fullgraph=self.eval_msg_finetune_compile_fullgraph,
            dynamic=self.eval_msg_finetune_compile_dynamic,
        )

    def _eval_feature_forward_impl(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._extract_eval_features(self._eval_backbone, batch)

    def _eval_head_forward_impl(self, features: torch.Tensor) -> torch.Tensor:
        return self._eval_head(features)

    def _ensure_eval_runtime(self) -> None:
        if self._eval_backbone is not None:
            return

        backbone = _build_model_from_config(self.config).to(self.device)
        if self.eval_msg_finetune_trainable_scope == "head_only":
            for param in backbone.parameters():
                param.requires_grad = False

        use_projector = (
            self.eval_msg_finetune_feature_source == "projector"
            and bool(self.config.get("sigreg_use_projector", True))
        )
        input_dim = int(self.config.get("sigreg_proj_output_dim", 128)) if use_projector else int(self.config.model_dim)
        head = self._build_eval_head(input_dim).to(self.device)

        object.__setattr__(self, "_eval_backbone", backbone)
        object.__setattr__(self, "_eval_head", head)
        self._eval_head_init_state = {
            key: value.detach().clone()
            for key, value in head.state_dict().items()
        }
        self._eval_feature_forward = self._compile_eval_callable(self._eval_feature_forward_impl)
        self._eval_head_forward = self._compile_eval_callable(self._eval_head_forward_impl)

    def _run_msg_finetune_eval(self) -> dict[str, torch.Tensor]:
        self._ensure_eval_runtime()
        backbone = self._eval_backbone
        head = self._eval_head

        backbone.load_state_dict(self.model.state_dict())
        head.load_state_dict(self._eval_head_init_state)

        if self.eval_msg_finetune_trainable_scope == "head_only":
            for param in backbone.parameters():
                param.requires_grad = False
        else:
            for param in backbone.parameters():
                param.requires_grad = True

        if self.eval_msg_finetune_trainable_scope == "head_only":
            parameters = head.parameters()
        else:
            parameters = list(backbone.parameters()) + list(head.parameters())
        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.eval_msg_finetune_learning_rate,
            weight_decay=self.eval_msg_finetune_weight_decay,
        )
        steps_per_epoch = int(self.trainer.datamodule.steps["massspec_train"])
        total_steps = self.eval_msg_finetune_num_epochs * steps_per_epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step_idx: _learning_rate_at_step(
                step_idx + 1,
                base_lr=self.eval_msg_finetune_learning_rate,
                total_steps=total_steps,
                warmup_steps=self.eval_msg_finetune_warmup_steps,
                schedule_type="cosine",
                min_learning_rate=None,
            ) / self.eval_msg_finetune_learning_rate,
        )

        for finetune_epoch in range(self.eval_msg_finetune_num_epochs):
            backbone.train()
            head.train()
            seed = int(self.config.seed) + 5_000_000 + self.current_epoch * 1_000 + finetune_epoch
            for batch in self._iter_massspec_probe("massspec_train", seed=seed):
                batch = self._to_device(batch)
                optimizer.zero_grad(set_to_none=True)
                torch.compiler.cudagraph_mark_step_begin()
                features = self._eval_feature_forward(batch)
                logits = self._eval_head_forward(features)
                targets = batch["fingerprint"][:, : self.fingerprint_bits].to(dtype=torch.float32)
                loss = F.binary_cross_entropy_with_logits(logits, targets)
                loss.backward()
                optimizer.step()
                scheduler.step()

        accumulator = _FingerprintMetricAccumulator(self.fingerprint_bits)
        backbone.eval()
        head.eval()
        with torch.no_grad():
            eval_seed = int(self.config.seed) + 6_000_000 + self.current_epoch
            for batch in self._iter_massspec_probe("massspec_test", seed=eval_seed):
                batch = self._to_device(batch)
                torch.compiler.cudagraph_mark_step_begin()
                features = self._eval_feature_forward(batch)
                logits = self._eval_head_forward(features)
                targets = batch["fingerprint"][:, : self.fingerprint_bits].to(dtype=torch.float32)
                accumulator.update(logits, targets)
        return accumulator.compute(self.device)

    def on_train_epoch_end(self) -> None:
        metrics = self._run_msg_finetune_eval()
        for key, value in metrics.items():
            self.log(
                f"msg_eval/{key}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=(key == "tanimoto"),
            )

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx
        torch.compiler.cudagraph_mark_step_begin()
        batch = self._to_device(batch)
        bcs_projection = self._sample_bcs_projection(self.global_step)
        metrics = self._train_forward(batch, bcs_projection)
        step = self.global_step + 1
        should_log_step = step == 1 or step % self.train_step_log_interval == 0
        if should_log_step:
            self.log("train/learning_rate", self._lr_for_step(step), on_step=True, on_epoch=False, prog_bar=True)
        for key, value in metrics.items():
            if key == "loss":
                if should_log_step:
                    self.log("train/loss", value, on_step=True, on_epoch=False, prog_bar=True)
                self.log("train/loss_epoch", value, on_step=False, on_epoch=True, prog_bar=False)
                if step % self.checkpoint_every_steps == 0:
                    self.log(
                        "train/loss_checkpoint",
                        value,
                        on_step=True,
                        on_epoch=False,
                        prog_bar=False,
                        logger=False,
                    )
                continue
            metric_name = f"train/{key}"
            if key == "representation_variance":
                self.log(metric_name, value, on_step=True, on_epoch=True, prog_bar=False)
                continue
            self.log(metric_name, value, on_step=False, on_epoch=True, prog_bar=False)
            if should_log_step and self.train_log_extra_metrics_on_step:
                self.log(f"{metric_name}_step", value, on_step=True, on_epoch=False, prog_bar=False)
        return metrics["loss"]

    def configure_optimizers(self):
        capturable = bool(self.config.get("optimizer_capturable", True))
        capturable = capturable and self.trainer.strategy.root_device.type == "cuda"
        fused_cfg = self.config.get("optimizer_fused", None)
        if fused_cfg is None:
            fused = None
        else:
            fused = bool(fused_cfg) and self.trainer.strategy.root_device.type == "cuda"
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.base_lr,
            betas=(0.9, self.b2),
            weight_decay=float(self.config.weight_decay),
            capturable=capturable,
            fused=fused,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self._lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def train_and_evaluate(
    config: config_dict.ConfigDict,
    workdir: str | Path,
) -> None:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    pl.seed_everything(int(config.seed), workers=False)

    datamodule = TfLightningDataModule(config, seed=int(config.seed))
    num_epochs = int(config.num_epochs)
    steps_per_epoch = datamodule.train_steps
    total_steps = num_epochs * steps_per_epoch

    # Update config with dataset-derived values
    info = datamodule.info
    config.num_peaks = info["num_peaks"]
    config.fingerprint_bits = int(info["fingerprint_bits"])

    logging.info("Training with Lightning for %d epochs.", num_epochs)
    logging.info("Steps per epoch: %d", steps_per_epoch)
    logging.info("Total steps: %d", total_steps)

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
    logger = _build_logger(config, Path(str(workdir)))

    callbacks: list[pl.Callback] = [checkpoint_cb, lr_monitor]
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
        log_every_n_steps=int(config.log_every_n_steps),
        val_check_interval=1.0,
        gradient_clip_val=float(config.clip) if config.get("clip", 0.) > 0. else None,
        gradient_clip_algorithm="norm" if config.get("clip", 0.) > 0. else None,
        callbacks=callbacks,
        logger=logger,
        reload_dataloaders_every_n_epochs=0,
        limit_train_batches=config.get("limit_train_batches", 1.0),
        limit_val_batches=0,
        limit_test_batches=0,
        num_sanity_val_steps=0,
    )

    ckpt_path = _latest_ckpt_path(Path(str(workdir)))

    if ckpt_path is not None:
        logging.info("Resuming from checkpoint: %s", ckpt_path)

    trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)
