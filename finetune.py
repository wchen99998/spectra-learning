from __future__ import annotations

import argparse
import importlib.util
import logging
import math
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import lightning.pytorch as pl
import tensorflow as tf
import torch
import torch._inductor.config as inductor_config
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from ml_collections import config_dict
from torch import nn
from torch.nn import functional as F

from input_pipeline import TfLightningDataModule
from models.bert_torch import BERTTorch
from utils import wandb_writer


torch.set_float32_matmul_precision("medium")
inductor_config.coordinate_descent_tuning = True
inductor_config.triton.unique_kernel_names = True
inductor_config.fx_graph_cache = True

tf.config.set_visible_devices([], "GPU")


def _load_config(path: str | Path) -> config_dict.ConfigDict:
    path = Path(path)
    spec = importlib.util.spec_from_file_location("experiment_config", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.get_config()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finetune BERT on MassSpecGym Morgan fingerprint prediction."
    )
    parser.add_argument("--config", required=True, help="Path to a config file (python).")
    parser.add_argument("--workdir", required=True, help="Output directory.")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to pretrained checkpoint used to initialize finetuning.",
    )
    return parser.parse_args()


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


def _is_compatible_finetune_checkpoint(path: Path, config: config_dict.ConfigDict) -> bool:
    checkpoint = torch.load(str(path), map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    if "fingerprint_head.weight" not in state_dict or "fingerprint_head.bias" not in state_dict:
        return False
    fingerprint_bits = int(config.get("fingerprint_bits", 1024))
    use_massspec_metadata = bool(config.get("use_massspec_metadata", True))
    metadata_hidden_dim = int(config.get("metadata_hidden_dim", 128))
    expected_input_dim = int(config.model_dim)
    if use_massspec_metadata:
        expected_input_dim += metadata_hidden_dim
    head_weight = state_dict["fingerprint_head.weight"]
    if int(head_weight.shape[0]) != fingerprint_bits:
        return False
    if int(head_weight.shape[1]) != expected_input_dim:
        return False
    has_metadata_encoder = "metadata_mlp.0.weight" in state_dict
    if use_massspec_metadata and not has_metadata_encoder:
        return False
    if not use_massspec_metadata and has_metadata_encoder:
        return False
    return True


def _latest_finetune_ckpt_path(
    directory: Path,
    config: config_dict.ConfigDict,
) -> str | None:
    ckpts = sorted(directory.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    for ckpt in ckpts:
        if _is_compatible_finetune_checkpoint(ckpt, config):
            return str(ckpt)
    return None


def _load_pretrained_model(model: BERTTorch, checkpoint_path: Path) -> None:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model_state = {
        key.removeprefix("model."): value
        for key, value in state_dict.items()
        if key.startswith("model.")
    }
    model.load_state_dict(model_state, strict=True)


class MassSpecFingerprintDataModule(pl.LightningDataModule):
    def __init__(self, config: config_dict.ConfigDict, seed: int) -> None:
        super().__init__()
        self.base = TfLightningDataModule(config, seed=seed)
        self.train_steps = self.base.steps["massspec_train"]
        self.val_steps = self.base.steps["massspec_val"]
        self.test_steps = self.base.steps["massspec_test"]
        self.info = self.base.info

    def state_dict(self) -> dict[str, int]:
        return self.base.state_dict()

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        self.base.load_state_dict(state_dict)

    def train_dataloader(self):
        seed = self.base.seed
        dataset = self.base._build_dataset_for_files(
            self.base.massspec_train_files,
            seed=seed,
            shuffle=True,
            drop_remainder=self.base.drop_remainder,
            include_fingerprint=True,
        )
        return self.base._make_loader(dataset=dataset, steps=self.train_steps)

    def val_dataloader(self):
        seed = self.base.seed + 1_000_000
        dataset = self.base._build_dataset_for_files(
            self.base.massspec_val_files,
            seed=seed,
            shuffle=False,
            drop_remainder=False,
            include_fingerprint=True,
        )
        return self.base._make_loader(dataset=dataset, steps=self.val_steps)

    def test_dataloader(self):
        seed = self.base.seed + 2_000_000
        dataset = self.base._build_dataset_for_files(
            self.base.massspec_test_files,
            seed=seed,
            shuffle=False,
            drop_remainder=False,
            include_fingerprint=True,
        )
        return self.base._make_loader(dataset=dataset, steps=self.test_steps)


class FingerprintFinetuneModule(pl.LightningModule):
    def __init__(
        self,
        config: config_dict.ConfigDict,
        *,
        total_steps: int,
        pretrained_ckpt_path: Path | None,
    ) -> None:
        super().__init__()
        self.config = config
        self.total_steps = int(total_steps)

        self.base_lr = float(config.get("finetune_learning_rate", config.learning_rate))
        self.weight_decay = float(config.get("finetune_weight_decay", config.weight_decay))
        self.warmup_steps = int(config.get("warmup_steps", 0))
        self.schedule_type = str(config.get("learning_rate_schedule", "cosine"))
        self.min_learning_rate = config.get("min_learning_rate", None)
        self.b2 = float(config.get("b2", 0.999))
        self.non_blocking_device_transfer = bool(config.get("non_blocking_device_transfer", True))
        self.train_step_log_interval = int(
            config.get("train_step_log_interval", config.get("log_every_n_steps", 1))
        )
        self.use_massspec_metadata = bool(config.get("use_massspec_metadata", True))
        self.metadata_hidden_dim = int(config.get("metadata_hidden_dim", 128))
        self.metadata_dropout = float(config.get("metadata_dropout", 0.0))

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
            pad_token_id=int(config.pad_token_id),
            cls_token_id=int(config.cls_token_id),
            sep_token_id=int(config.sep_token_id),
            cache_rope_frequencies=bool(config.get("cache_rope_frequencies", True)),
        )
        if pretrained_ckpt_path is not None:
            _load_pretrained_model(self.model, pretrained_ckpt_path)

        fingerprint_bits = int(config.get("fingerprint_bits", 1024))
        head_input_dim = int(config.model_dim)
        if self.use_massspec_metadata:
            self.adduct_embed = nn.Embedding(
                int(config.massspec_adduct_vocab_size),
                self.metadata_hidden_dim,
            )
            self.instrument_type_embed = nn.Embedding(
                int(config.massspec_instrument_type_vocab_size),
                self.metadata_hidden_dim,
            )
            self.metadata_mlp = nn.Sequential(
                nn.Linear(2 * self.metadata_hidden_dim + 2, self.metadata_hidden_dim),
                nn.SiLU(),
                nn.Dropout(self.metadata_dropout),
            )
            head_input_dim += self.metadata_hidden_dim
        self.fingerprint_head = nn.Linear(head_input_dim, fingerprint_bits)

        # Compile train/eval forward with CUDA graphs for max throughput
        self._train_forward = torch.compile(
            self._train_forward_impl,
            mode="max-autotune-no-cudagraphs",
            fullgraph=True,
        )
        self._eval_forward = torch.compile(
            self._eval_forward_impl,
            mode="max-autotune-no-cudagraphs",
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

    def _move_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            key: value.to(self.device, non_blocking=self.non_blocking_device_transfer)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in batch.items()
        }

    def _compute_metrics(
        self,
        batch: dict[str, torch.Tensor],
        *,
        train: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cls_embedding = self.model.encode(batch, train=train)
        if self.use_massspec_metadata:
            adduct_id = batch["adduct_id"].to(torch.long)
            instrument_type_id = batch["instrument_type_id"].to(torch.long)
            collision_energy = batch["collision_energy"].to(torch.float32).unsqueeze(-1) / 100.0
            collision_energy_present = (
                batch["collision_energy_present"].to(torch.float32).unsqueeze(-1)
            )
            adduct_embed = self.adduct_embed(adduct_id)
            instrument_type_embed = self.instrument_type_embed(instrument_type_id)
            metadata_input = torch.cat(
                [
                    adduct_embed,
                    instrument_type_embed,
                    collision_energy,
                    collision_energy_present,
                ],
                dim=-1,
            )
            metadata_embed = self.metadata_mlp(metadata_input)
            cls_embedding = torch.cat([cls_embedding, metadata_embed], dim=-1)
        logits = self.fingerprint_head(cls_embedding)
        target = batch["fingerprint"].to(dtype=torch.float32)

        loss = F.binary_cross_entropy_with_logits(logits, target)

        pred = logits >= 0
        truth = target >= 0.5
        bit_accuracy = (pred == truth).to(torch.float32).mean()

        intersection = (pred & truth).sum(dim=1).to(torch.float32)
        union = (pred | truth).sum(dim=1).to(torch.float32)
        tanimoto = (intersection / union).mean()
        return loss, bit_accuracy, tanimoto

    def _train_forward_impl(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._compute_metrics(batch, train=True)

    def _eval_forward_impl(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._compute_metrics(batch, train=False)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx
        torch.compiler.cudagraph_mark_step_begin()
        batch = self._move_batch(batch)
        loss, bit_accuracy, tanimoto = self._train_forward(batch)

        step = self.global_step + 1
        should_log_step = step == 1 or step % self.train_step_log_interval == 0
        if should_log_step:
            self.log(
                "train/learning_rate",
                self._lr_for_step(step),
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
            self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
            self.log("train/bit_accuracy", bit_accuracy, on_step=True, on_epoch=False)
            self.log("train/tanimoto", tanimoto, on_step=True, on_epoch=False)

        self.log("train/loss_epoch", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log(
            "train/bit_accuracy_epoch",
            bit_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "train/tanimoto_epoch",
            tanimoto,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx
        torch.compiler.cudagraph_mark_step_begin()
        batch = self._move_batch(batch)
        loss, bit_accuracy, tanimoto = self._eval_forward(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/bit_accuracy", bit_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/tanimoto", tanimoto, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx
        torch.compiler.cudagraph_mark_step_begin()
        batch = self._move_batch(batch)
        loss, bit_accuracy, tanimoto = self._eval_forward(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/bit_accuracy", bit_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/tanimoto", tanimoto, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        capturable = bool(self.config.get("optimizer_capturable", True))
        capturable = capturable and self.trainer.strategy.root_device.type == "cuda"
        fused_cfg = self.config.get("optimizer_fused", None)
        if fused_cfg is None:
            fused = None
        else:
            fused = bool(fused_cfg) and self.trainer.strategy.root_device.type == "cuda"

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.base_lr,
            betas=(0.9, self.b2),
            weight_decay=self.weight_decay,
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


def finetune_and_evaluate(
    config: config_dict.ConfigDict,
    *,
    workdir: str | Path,
    model_path: str | Path,
) -> None:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    pl.seed_everything(int(config.seed), workers=False)

    datamodule = MassSpecFingerprintDataModule(config, seed=int(config.seed))

    info = datamodule.info
    config.vocab_size = info["vocab_size"]
    config.max_length = info["pair_sequence_length"]
    config.precursor_bins = info["precursor_bins"]
    config.precursor_offset = info["precursor_offset"]
    config.fingerprint_bits = int(config.get("fingerprint_bits", info["fingerprint_bits"]))
    config.use_massspec_metadata = bool(config.get("use_massspec_metadata", True))
    config.metadata_hidden_dim = int(config.get("metadata_hidden_dim", 128))
    config.metadata_dropout = float(config.get("metadata_dropout", 0.0))
    config.massspec_adduct_vocab_size = int(info["massspec_adduct_vocab_size"])
    config.massspec_instrument_type_vocab_size = int(
        info["massspec_instrument_type_vocab_size"]
    )

    num_epochs = int(config.num_epochs)
    steps_per_epoch = datamodule.train_steps
    total_steps = num_epochs * steps_per_epoch

    logging.info("Finetuning for %d epochs.", num_epochs)
    logging.info("MSG train/val/test steps: %d / %d / %d", steps_per_epoch, datamodule.val_steps, datamodule.test_steps)
    logging.info("Total steps: %d", total_steps)

    checkpoint_dir = workdir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="step-{step:08d}",
        every_n_train_steps=int(config.checkpoint_every_steps),
        monitor="val/loss",
        mode="min",
        save_last=True,
        save_top_k=5,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    logger = _build_logger(config, workdir)

    trainer = pl.Trainer(
        default_root_dir=str(workdir),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed",
        max_epochs=num_epochs,
        log_every_n_steps=int(config.log_every_n_steps),
        val_check_interval=config.val_check_interval,
        gradient_clip_val=float(config.clip) if config.get("clip", 0.0) > 0.0 else None,
        gradient_clip_algorithm="norm" if config.get("clip", 0.0) > 0.0 else None,
        callbacks=[checkpoint_cb, lr_monitor],
        logger=logger,
        reload_dataloaders_every_n_epochs=0,
        limit_train_batches=config.get("limit_train_batches", 1.0),
        limit_val_batches=config.get("limit_val_batches", 1.0),
        limit_test_batches=config.get("limit_test_batches", 1.0),
        num_sanity_val_steps=int(config.get("num_sanity_val_steps", 0)),
    )

    resume_ckpt_path = _latest_finetune_ckpt_path(checkpoint_dir, config)
    if resume_ckpt_path is not None:
        logging.info("Resuming finetuning from checkpoint: %s", resume_ckpt_path)
        pretrained_ckpt_path = None
    else:
        pretrained_ckpt_path = Path(model_path)
        logging.info("Initializing finetuning model from: %s", pretrained_ckpt_path)

    module = FingerprintFinetuneModule(
        config,
        total_steps=total_steps,
        pretrained_ckpt_path=pretrained_ckpt_path,
    )

    trainer.fit(module, datamodule=datamodule, ckpt_path=resume_ckpt_path)

    best_ckpt = checkpoint_cb.best_model_path
    test_ckpt = best_ckpt if best_ckpt else "last"
    logging.info("Running final test with checkpoint: %s", test_ckpt)
    trainer.test(module, datamodule=datamodule, ckpt_path=test_ckpt)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    config = _load_config(args.config)
    workdir = Path(args.workdir).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()

    finetune_and_evaluate(
        config,
        workdir=workdir,
        model_path=model_path,
    )


if __name__ == "__main__":
    main()
