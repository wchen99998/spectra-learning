from __future__ import annotations

import math
from pathlib import Path

import lightning.pytorch as pl
import torch
import torch._inductor.config as inductor_config

torch.set_float32_matmul_precision('medium')
inductor_config.coordinate_descent_tuning = True
inductor_config.triton.unique_kernel_names = True
inductor_config.fx_graph_cache = True

import logging
from ml_collections import config_dict
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from input_pipeline import TfLightningDataModule, numpy_batch_to_torch
from linear_probe import run_linear_probe
from models.bert_torch import BERTTorch
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


class MAELightningModule(pl.LightningModule):
    def __init__(
        self,
        config: config_dict.ConfigDict,
        *,
        total_steps: int,
        eval_splits: list[str],
        test_splits: list[str],
    ) -> None:
        super().__init__()
        self.config = config
        self.total_steps = int(total_steps)
        self.eval_splits = eval_splits
        self.test_splits = test_splits

        self.base_lr = float(config.learning_rate)
        self.warmup_steps = int(config.get("warmup_steps", 0))
        self.schedule_type = str(config.get("learning_rate_schedule", "cosine"))
        self.min_learning_rate = config.get("min_learning_rate", None)
        self.b2 = float(config.get("b2", 0.999))
        self.probe_ridge = float(config.get("probe_ridge", 1e-3))

        self.model = BERTTorch(
            vocab_size=int(config.vocab_size),
            max_length=int(config.max_length),
            precursor_bins=int(config.precursor_bins),
            precursor_offset=int(config.precursor_offset),
            model_dim=int(config.model_dim),
            num_layers=int(config.num_layers),
            num_heads=int(config.num_heads),
            num_kv_heads=config.get("num_kv_heads", None),
            attention_mlp_multiple=float(config.attention_mlp_multiple),
            num_segments=int(config.num_segments),
            mask_ratio=float(config.mask_ratio),
            mask_token_id=int(config.mask_token_id),
            pad_token_id=int(config.pad_token_id),
            cls_token_id=int(config.cls_token_id),
            sep_token_id=int(config.sep_token_id),
        )

        # Compile train/eval forward with CUDA graphs for max throughput
        self._train_forward = torch.compile(
            self._train_forward_impl,
            mode="max-autotune",
            fullgraph=True,
        )
        self._eval_forward = torch.compile(
            self._eval_forward_impl,
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

    def _train_forward_impl(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.model(batch, train=True, apply_mask=True)

    def _eval_forward_impl(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.model(batch, train=False, apply_mask=False)

    def _iter_massspec_probe(self, split: str):
        dm = self.trainer.datamodule
        seed = int(self.config.seed) + (2_000_000 if split == "massspec_train" else 3_000_000)
        ds = dm.build_massspec_probe_dataset(split, seed=seed)
        size_key = "massspec_train_size" if split == "massspec_train" else "massspec_test_size"
        size = int(dm.info[size_key])
        seen = 0
        for batch in ds.as_numpy_iterator():
            remaining = size - seen
            if remaining <= 0:
                break
            take = min(int(batch["token_ids"].shape[0]), remaining)
            if take != batch["token_ids"].shape[0]:
                batch = {key: value[:take] for key, value in batch.items()}
            seen += take
            yield numpy_batch_to_torch(batch)

    def _iter_probe_features(self, split: str):
        device = self.device
        for batch in self._iter_massspec_probe(split):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            x = self.model.encode(batch, train=False)
            y = batch["fingerprint"].to(dtype=torch.float32)
            yield x, y

    def on_validation_epoch_start(self) -> None:
        dm = self.trainer.datamodule
        if int(dm.info.get("massspec_train_size", 0)) == 0:
            return
        if int(dm.info.get("massspec_test_size", 0)) == 0:
            return
        with torch.no_grad():
            metrics = run_linear_probe(
                self._iter_probe_features("massspec_train"),
                self._iter_probe_features("massspec_test"),
                ridge=self.probe_ridge,
            )
        self.log(
            "massspec_test/linear_probe_accuracy",
            metrics["accuracy"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "massspec_test/linear_probe_tanimoto",
            metrics["tanimoto"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx
        torch.compiler.cudagraph_mark_step_begin()
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        metrics = self._train_forward(batch)
        lr_value = torch.tensor(self._lr_for_step(self.global_step + 1), device=self.device)
        self.log("train/learning_rate", lr_value, on_step=True, prog_bar=True)
        for key, value in metrics.items():
            self.log(f"train/{key}", value, on_step=True, prog_bar=(key == "loss"))
        return metrics["loss"]

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        del batch_idx
        torch.compiler.cudagraph_mark_step_begin()
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        split = self.eval_splits[dataloader_idx]
        metrics = self._eval_forward(batch)
        for key, value in metrics.items():
            self.log(f"{split}/{key}", value, on_step=False, on_epoch=True)
        return metrics["loss"]

    def test_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        del batch_idx
        torch.compiler.cudagraph_mark_step_begin()
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        split = self.test_splits[dataloader_idx]
        metrics = self._eval_forward(batch)
        for key, value in metrics.items():
            self.log(f"{split}/{key}", value, on_step=False, on_epoch=True)
        return metrics["loss"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.base_lr,
            betas=(0.9, self.b2),
            weight_decay=float(self.config.weight_decay),
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
    total_steps = int(config.num_train_steps)

    # Update config with dataset-derived values
    info = datamodule.info
    config.vocab_size = info["vocab_size"]
    config.max_length = info["pair_sequence_length"]
    config.precursor_bins = info["precursor_bins"]
    config.precursor_offset = info["precursor_offset"]

    logging.info("Training with Lightning for %d steps.", total_steps)
    logging.info("Steps per epoch: %d", datamodule.train_steps)
    logging.info("Validation splits: %s", datamodule.eval_splits)
    logging.info("Test splits: %s", datamodule.test_splits)

    module = MAELightningModule(
        config,
        total_steps=total_steps,
        eval_splits=datamodule.eval_splits,
        test_splits=datamodule.test_splits,
    )

    checkpoint_dir = Path(str(workdir)) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="step-{step:08d}",
        every_n_train_steps=int(config.checkpoint_every_steps),
        save_last=True,
        save_top_k=-1,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    logger = _build_logger(config, Path(str(workdir)))

    callbacks: list[pl.Callback] = [checkpoint_cb, lr_monitor]

    trainer = pl.Trainer(
        default_root_dir=str(workdir),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="bf16-mixed",
        max_steps=total_steps,
        limit_train_batches=total_steps,
        log_every_n_steps=int(config.log_loss_every_steps),
        val_check_interval=int(config.eval_every_steps),
        gradient_clip_val=float(config.clip) if config.get("clip", 0.) > 0. else None,
        gradient_clip_algorithm="norm",
        callbacks=callbacks,
        logger=logger,
        reload_dataloaders_every_n_epochs=0,
    )

    ckpt_path = _latest_ckpt_path(Path(str(workdir)))

    if ckpt_path is not None:
        logging.info("Resuming from checkpoint: %s", ckpt_path)

    trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)
