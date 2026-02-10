from __future__ import annotations

import argparse
import importlib.util
import logging
import math
from pathlib import Path

import lightning.pytorch as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from ml_collections import config_dict
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from input_pipeline import TfLightningDataModule, numpy_batch_to_torch
from models.model import PeakSetSIGReg
from utils import wandb_writer


def _learning_rate_at_step(
    step: int,
    *,
    base_lr: float,
    total_steps: int,
    warmup_steps: int,
) -> float:
    if warmup_steps > 0:
        warmup = min(1.0, step / warmup_steps)
        effective_step = max(0, step - warmup_steps)
        effective_total = max(1, total_steps - warmup_steps)
    else:
        warmup = 1.0
        effective_step = step
        effective_total = max(1, total_steps)

    total = max(1, effective_total)
    ratio = max(0.0, effective_step / total)
    mult = 0.5 * (1.0 + math.cos(math.pi * ratio))
    lr = mult * base_lr
    min_lr = 0.1 * base_lr
    return max(min_lr, lr) * warmup


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


def _load_pretrained_weights(model: PeakSetSIGReg, checkpoint_path: str) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = ckpt["state_dict"]
    model_state = {
        k.removeprefix("model."): v
        for k, v in state_dict.items()
        if k.startswith("model.")
    }
    result = model.load_state_dict(model_state, strict=False)
    if result.missing_keys:
        raise ValueError(f"Missing keys when loading checkpoint: {result.missing_keys}")
    if result.unexpected_keys:
        raise ValueError(f"Unexpected keys when loading checkpoint: {result.unexpected_keys}")


class FinetuneLightningModule(pl.LightningModule):
    def __init__(
        self,
        config: config_dict.ConfigDict,
        *,
        total_steps: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.total_steps = int(total_steps)

        self.feature_source = str(config.finetune_feature_source)
        self.freeze_backbone = bool(config.finetune_freeze_backbone)
        self.base_lr = float(config.finetune_learning_rate)
        self.weight_decay = float(config.finetune_weight_decay)
        self.warmup_steps = int(config.get("warmup_steps", 0))
        self.fingerprint_bits = int(config.fingerprint_bits)

        self.model = _build_model_from_config(config)

        checkpoint_path = str(config.finetune_checkpoint)
        if checkpoint_path:
            _load_pretrained_weights(self.model, checkpoint_path)

        if self.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        if self.feature_source == "projector":
            input_dim = int(config.get("sigreg_proj_output_dim", 128))
        else:
            input_dim = int(config.model_dim)

        head_hidden = int(config.finetune_head_hidden_dim)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(input_dim, head_hidden),
            torch.nn.RMSNorm(head_hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(head_hidden, self.fingerprint_bits),
        )

        self._val_preds: list[torch.Tensor] = []
        self._val_targets: list[torch.Tensor] = []
        self._test_preds: list[torch.Tensor] = []
        self._test_targets: list[torch.Tensor] = []

    def _extract_features(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        peak_mz = batch["peak_mz"]
        peak_intensity = batch["peak_intensity"]
        peak_valid_mask = batch["peak_valid_mask"]
        precursor_mz = batch["precursor_mz"]

        if self.freeze_backbone:
            with torch.no_grad():
                embeddings = self.model.encoder(peak_mz, peak_intensity, precursor_mz)
                pooled = self.model._pool(embeddings, peak_valid_mask)
                if self.feature_source == "projector":
                    return self.model.projector(pooled)
                return pooled
        else:
            embeddings = self.model.encoder(peak_mz, peak_intensity, precursor_mz)
            pooled = self.model._pool(embeddings, peak_valid_mask)
            if self.feature_source == "projector":
                return self.model.projector(pooled)
            return pooled

    def _lr_for_step(self, step: int) -> float:
        return _learning_rate_at_step(
            step,
            base_lr=self.base_lr,
            total_steps=self.total_steps,
            warmup_steps=self.warmup_steps,
        )

    def _lr_lambda(self, step_idx: int) -> float:
        step = step_idx + 1
        return self._lr_for_step(step) / self.base_lr

    def _to_device(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx
        batch = self._to_device(batch)
        features = self._extract_features(batch)
        with torch.autocast(device_type=features.device.type, enabled=False):
            logits = self.head(features.float())
        targets = batch["fingerprint"][:, :self.fingerprint_bits].to(dtype=torch.float32)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        step = self.global_step + 1
        self.log("train/learning_rate", self._lr_for_step(step), on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        del batch_idx
        batch = self._to_device(batch)
        features = self._extract_features(batch)
        with torch.autocast(device_type=features.device.type, enabled=False):
            logits = self.head(features.float())
        targets = batch["fingerprint"][:, :self.fingerprint_bits].to(dtype=torch.float32)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self._val_preds.append(logits.detach().cpu())
        self._val_targets.append(targets.detach().cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        preds = torch.cat(self._val_preds, dim=0)
        targets = torch.cat(self._val_targets, dim=0)
        self._val_preds.clear()
        self._val_targets.clear()
        metrics = _compute_metrics(preds, targets)
        for key, value in metrics.items():
            self.log(f"val/{key}", value, on_step=False, on_epoch=True, prog_bar=(key == "tanimoto"))

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        del batch_idx
        batch = self._to_device(batch)
        features = self._extract_features(batch)
        with torch.autocast(device_type=features.device.type, enabled=False):
            logits = self.head(features.float())
        targets = batch["fingerprint"][:, :self.fingerprint_bits].to(dtype=torch.float32)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self._test_preds.append(logits.detach().cpu())
        self._test_targets.append(targets.detach().cpu())

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        preds = torch.cat(self._test_preds, dim=0)
        targets = torch.cat(self._test_targets, dim=0)
        self._test_preds.clear()
        self._test_targets.clear()
        metrics = _compute_metrics(preds, targets)
        for key, value in metrics.items():
            self.log(f"test/{key}", value, on_step=False, on_epoch=True, prog_bar=(key == "tanimoto"))

        workdir = Path(self.trainer.default_root_dir)
        _save_figures(preds, targets, workdir / "figures")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.head.parameters() if self.freeze_backbone else self.parameters(),
            lr=self.base_lr,
            weight_decay=self.weight_decay,
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


def _compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, torch.Tensor]:
    probs = torch.sigmoid(logits)
    pred_bits = (probs > 0.5)
    target_bits = (targets > 0.5)

    # Tanimoto similarity (Jaccard)
    intersection = (pred_bits & target_bits).sum(dim=1).float()
    union = (pred_bits | target_bits).sum(dim=1).float()
    tanimoto_per_sample = torch.where(union > 0, intersection / union, torch.ones_like(union))
    tanimoto = tanimoto_per_sample.mean()

    # Cosine similarity
    cosine = F.cosine_similarity(probs, targets, dim=1).mean()

    # Bit accuracy
    bit_accuracy = (pred_bits == target_bits).float().mean()

    # Per-bit precision, recall, F1
    tp = (pred_bits & target_bits).float().sum(dim=0)
    fp = (pred_bits & ~target_bits).float().sum(dim=0)
    fn = (~pred_bits & target_bits).float().sum(dim=0)

    precision_per_bit = tp / (tp + fp).clamp(min=1e-8)
    recall_per_bit = tp / (tp + fn).clamp(min=1e-8)
    f1_per_bit = 2 * precision_per_bit * recall_per_bit / (precision_per_bit + recall_per_bit).clamp(min=1e-8)

    # Zero out metrics for bits with no positive support
    has_pred = (tp + fp) > 0
    has_target = (tp + fn) > 0
    precision_per_bit = torch.where(has_pred, precision_per_bit, torch.zeros_like(precision_per_bit))
    recall_per_bit = torch.where(has_target, recall_per_bit, torch.zeros_like(recall_per_bit))
    f1_per_bit = torch.where(has_pred | has_target, f1_per_bit, torch.zeros_like(f1_per_bit))

    precision = precision_per_bit.mean()
    recall = recall_per_bit.mean()
    f1 = f1_per_bit.mean()

    pred_positive_rate = pred_bits.float().mean()
    target_positive_rate = target_bits.float().mean()

    return {
        "tanimoto": tanimoto,
        "cosine_similarity": cosine,
        "bit_accuracy": bit_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_positive_rate": pred_positive_rate,
        "target_positive_rate": target_positive_rate,
    }


def _save_figures(
    logits: torch.Tensor,
    targets: torch.Tensor,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)

    probs = torch.sigmoid(logits)
    pred_bits = (probs > 0.5)
    target_bits = (targets > 0.5)

    # Per-sample Tanimoto
    intersection = (pred_bits & target_bits).sum(dim=1).float()
    union = (pred_bits | target_bits).sum(dim=1).float()
    tanimoto_per_sample = torch.where(union > 0, intersection / union, torch.ones_like(union))

    # Per-sample cosine
    cosine_per_sample = F.cosine_similarity(probs, targets, dim=1)

    # Per-bit accuracy
    per_bit_correct = (pred_bits == target_bits).float()
    per_bit_accuracy = per_bit_correct.mean(dim=0)

    # Per-bit precision/recall
    tp = (pred_bits & target_bits).float().sum(dim=0)
    fp = (pred_bits & ~target_bits).float().sum(dim=0)
    fn = (~pred_bits & target_bits).float().sum(dim=0)
    precision_per_bit = tp / (tp + fp).clamp(min=1e-8)
    recall_per_bit = tp / (tp + fn).clamp(min=1e-8)
    has_pred = (tp + fp) > 0
    has_target = (tp + fn) > 0
    precision_per_bit = torch.where(has_pred, precision_per_bit, torch.zeros_like(precision_per_bit))
    recall_per_bit = torch.where(has_target, recall_per_bit, torch.zeros_like(recall_per_bit))

    # Move to CPU numpy
    tanimoto_np = tanimoto_per_sample.cpu().numpy()
    cosine_np = cosine_per_sample.cpu().numpy()
    per_bit_acc_np = per_bit_accuracy.cpu().numpy()
    precision_np = precision_per_bit.cpu().numpy()
    recall_np = recall_per_bit.cpu().numpy()

    # 1. Tanimoto distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(tanimoto_np, bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Tanimoto Similarity")
    ax.set_ylabel("Count")
    ax.set_title(f"Tanimoto Distribution (mean={tanimoto_np.mean():.4f})")
    fig.tight_layout()
    fig.savefig(figure_dir / "tanimoto_distribution.png", dpi=150)
    plt.close(fig)

    # 2. Cosine distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(cosine_np, bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Count")
    ax.set_title(f"Cosine Distribution (mean={cosine_np.mean():.4f})")
    fig.tight_layout()
    fig.savefig(figure_dir / "cosine_distribution.png", dpi=150)
    plt.close(fig)

    # 3. Per-bit accuracy
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(per_bit_acc_np)), per_bit_acc_np, width=1.0, alpha=0.7)
    ax.set_xlabel("Bit Index")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Per-Bit Accuracy (mean={per_bit_acc_np.mean():.4f})")
    ax.set_xlim(-0.5, len(per_bit_acc_np) - 0.5)
    fig.tight_layout()
    fig.savefig(figure_dir / "per_bit_accuracy.png", dpi=150)
    plt.close(fig)

    # 4. Per-bit precision/recall
    fig, ax = plt.subplots(figsize=(14, 5))
    bit_indices = range(len(precision_np))
    ax.bar(bit_indices, precision_np, width=1.0, alpha=0.5, label="Precision")
    ax.bar(bit_indices, recall_np, width=1.0, alpha=0.5, label="Recall")
    ax.set_xlabel("Bit Index")
    ax.set_ylabel("Score")
    ax.set_title("Per-Bit Precision & Recall")
    ax.set_xlim(-0.5, len(precision_np) - 0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "per_bit_precision_recall.png", dpi=150)
    plt.close(fig)

    logging.getLogger(__name__).info("Saved figures to %s", figure_dir)


class _FinetuneDataModule(pl.LightningDataModule):
    """DataModule that serves only MassSpecGym splits with fingerprints."""

    def __init__(self, config: config_dict.ConfigDict, seed: int) -> None:
        super().__init__()
        self._base = TfLightningDataModule(config, seed=seed)
        self.info = self._base.info
        self.batch_size = self._base.batch_size
        self.train_steps = self._base.steps["massspec_train"]
        self.val_steps = self._base.steps["massspec_val"]
        self.test_steps = self._base.steps["massspec_test"]

    def _build_loader(self, split: str, seed: int, shuffle: bool) -> torch.utils.data.DataLoader:
        ds = self._base.build_massspec_probe_dataset(split, seed=seed)
        from input_pipeline import _TfIterableDataset, _StatefulDataLoader, _identity_collate
        if split == "massspec_train":
            steps = self.train_steps
        elif split == "massspec_val":
            steps = self.val_steps
        else:
            steps = self.test_steps
        iterable = _TfIterableDataset(dataset=ds, steps_per_epoch=steps)
        loader_kwargs: dict = {
            "dataset": iterable,
            "batch_size": None,
            "num_workers": self._base.dataloader_num_workers,
            "pin_memory": self._base.pin_memory,
            "collate_fn": _identity_collate,
        }
        if self._base.dataloader_num_workers > 0:
            loader_kwargs["persistent_workers"] = self._base.dataloader_persistent_workers
            loader_kwargs["prefetch_factor"] = self._base.dataloader_prefetch_factor
        return _StatefulDataLoader(**loader_kwargs)

    def train_dataloader(self):
        return self._build_loader("massspec_train", seed=42, shuffle=True)

    def val_dataloader(self):
        return self._build_loader("massspec_val", seed=43, shuffle=False)

    def test_dataloader(self):
        return self._build_loader("massspec_test", seed=44, shuffle=False)


def _load_config(path: str) -> config_dict.ConfigDict:
    spec = importlib.util.spec_from_file_location("finetune_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()


def finetune(
    config: config_dict.ConfigDict,
    workdir: str | Path,
) -> None:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    pl.seed_everything(int(config.seed), workers=False)

    datamodule = _FinetuneDataModule(config, seed=int(config.seed))

    # Update config with dataset-derived values
    config.num_peaks = datamodule.info["num_peaks"]
    config.fingerprint_bits = int(datamodule.info["fingerprint_bits"])

    num_epochs = int(config.get("finetune_num_epochs", config.num_epochs))
    steps_per_epoch = datamodule.train_steps
    total_steps = num_epochs * steps_per_epoch

    logging.info("Finetuning for %d epochs (%d steps/epoch, %d total)", num_epochs, steps_per_epoch, total_steps)
    logging.info("Feature source: %s", config.finetune_feature_source)
    logging.info("Freeze backbone: %s", config.finetune_freeze_backbone)

    module = FinetuneLightningModule(config, total_steps=total_steps)

    checkpoint_dir = workdir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="step-{step:08d}",
        every_n_train_steps=int(config.checkpoint_every_steps),
        monitor="val/tanimoto",
        mode="max",
        save_last=True,
        save_top_k=3,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    logger = _build_logger(config, workdir)

    trainer = pl.Trainer(
        default_root_dir=str(workdir),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="bf16-mixed",
        max_epochs=num_epochs,
        log_every_n_steps=int(config.get("log_every_n_steps", 50)),
        val_check_interval=config.get("val_check_interval", 1.0),
        gradient_clip_val=float(config.clip) if config.get("clip", 0.0) > 0.0 else None,
        gradient_clip_algorithm="norm" if config.get("clip", 0.0) > 0.0 else None,
        callbacks=[checkpoint_cb, lr_monitor],
        logger=logger,
        limit_train_batches=config.get("limit_train_batches", 1.0),
        limit_val_batches=config.get("limit_val_batches", 1.0),
        limit_test_batches=config.get("limit_test_batches", 1.0),
        num_sanity_val_steps=int(config.get("num_sanity_val_steps", 0)),
    )

    trainer.fit(module, datamodule=datamodule)
    trainer.test(module, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Finetune SIGReg for fingerprint prediction")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--workdir", required=True, help="Output directory")
    parser.add_argument("--checkpoint", default="", help="Pretrained checkpoint path (overrides config)")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    if args.checkpoint:
        cfg.finetune_checkpoint = args.checkpoint

    finetune(cfg, args.workdir)
