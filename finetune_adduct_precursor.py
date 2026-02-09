from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from ml_collections import config_dict
from torch import nn
from torch.nn import functional as F

from finetune import (
    MassSpecFingerprintDataModule,
    _build_logger,
    _learning_rate_at_step,
    _load_config,
    _load_pretrained_model,
)
from input_pipeline import GeMSFormulaLightningDataModule
from models.bert_torch import BERTTorch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finetune BERT on MassSpecGym precursor + adduct prediction."
    )
    parser.add_argument("--config", required=True, help="Path to a config file (python).")
    parser.add_argument("--workdir", required=True, help="Output directory.")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to pretrained checkpoint used to initialize finetuning.",
    )
    return parser.parse_args()


def _is_compatible_adduct_precursor_checkpoint(
    path: Path, config: config_dict.ConfigDict
) -> bool:
    checkpoint = torch.load(str(path), map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    if "adduct_head.weight" not in state_dict or "adduct_head.bias" not in state_dict:
        return False
    head_weight = state_dict["adduct_head.weight"]
    if int(head_weight.shape[0]) != int(config.massspec_adduct_vocab_size):
        return False
    if int(head_weight.shape[1]) != int(config.model_dim):
        return False
    return True


def _latest_adduct_precursor_ckpt_path(
    directory: Path,
    config: config_dict.ConfigDict,
) -> str | None:
    ckpts = sorted(directory.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    for ckpt in ckpts:
        if _is_compatible_adduct_precursor_checkpoint(ckpt, config):
            return str(ckpt)
    return None


class MassSpecAdductPrecursorDataModule(pl.LightningDataModule):
    def __init__(self, config: config_dict.ConfigDict, seed: int) -> None:
        super().__init__()
        self.data_source = str(config.get("finetune_data_source", "gems_formula_2m"))
        if self.data_source == "massspec":
            self.base = MassSpecFingerprintDataModule(config, seed=seed)
        else:
            self.base = GeMSFormulaLightningDataModule(config, seed=seed)
        self.train_steps = int(self.base.train_steps)
        self.val_steps = int(self.base.val_steps)
        self.test_steps = int(self.base.test_steps)
        self.info = self.base.info

    def state_dict(self) -> dict[str, int]:
        return self.base.state_dict()

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        self.base.load_state_dict(state_dict)

    def train_dataloader(self):
        return self.base.train_dataloader()

    def val_dataloader(self):
        return self.base.val_dataloader()

    def test_dataloader(self):
        return self.base.test_dataloader()


_ELEMENT_MASS = {
    "H": 1.00782503223,
    "C": 12.0,
    "N": 14.00307400443,
    "O": 15.99491461957,
    "B": 11.00930536,
    "P": 30.97376199842,
    "S": 31.9720711744,
    "F": 18.99840316273,
    "Mg": 23.985041697,
    "Si": 27.97692653465,
    "Ca": 39.962590863,
    "Fe": 55.93493633,
    "Cu": 62.92959772,
    "As": 74.92159457,
    "Se": 79.9165218,
    "Cl": 34.968852682,
    "Br": 78.9183376,
    "I": 126.9044719,
    "Na": 22.9897692820,
    "K": 38.9637064864,
}
_FORMULA_PATTERN = re.compile(r"([A-Z][a-z]?)(\d*)")
_ADDUCT_SHIFT = {
    "[M+H]+": 1.007276466812,
    "[M+Na]+": 22.989218,
    "[M+K]+": 38.963158,
    "[M+NH4]+": 18.033823,
    "[M-H]-": -1.007276466812,
    "[M+Cl]-": 34.968853,
}
_ADDUCT_CHARGE = {
    "[M+H]+": 1,
    "[M+Na]+": 1,
    "[M+K]+": 1,
    "[M+NH4]+": 1,
    "[M-H]-": -1,
    "[M+Cl]-": -1,
}


def _neutral_formula_mass(formula: str) -> float:
    mass = 0.0
    for element, count_text in _FORMULA_PATTERN.findall(formula):
        count = int(count_text) if count_text else 1
        mass += _ELEMENT_MASS[element] * count
    return mass


def _precursor_mz_from_formula(formula: str, adduct_name: str) -> float:
    neutral = _neutral_formula_mass(formula)
    shift = _ADDUCT_SHIFT[adduct_name] if adduct_name in _ADDUCT_SHIFT else 0.0
    charge = _ADDUCT_CHARGE[adduct_name] if adduct_name in _ADDUCT_CHARGE else 1
    return (neutral + shift) / abs(charge)


def _infer_vocab_size_from_checkpoint(checkpoint_path: Path) -> int:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    if "model.token_embed.weight" in state_dict:
        return int(state_dict["model.token_embed.weight"].shape[0])
    return int(state_dict["token_embed.weight"].shape[0])


class AdductPrecursorFinetuneModule(pl.LightningModule):
    def __init__(
        self,
        config: config_dict.ConfigDict,
        *,
        total_steps: int,
        pretrained_ckpt_path: Path | None,
        adduct_id_to_name: dict[int, str],
        data_source: str,
    ) -> None:
        super().__init__()
        self.config = config
        self.total_steps = int(total_steps)
        self.data_source = str(data_source)

        self.base_lr = float(config.get("finetune_learning_rate", config.learning_rate))
        self.weight_decay = float(config.get("finetune_weight_decay", config.weight_decay))
        self.warmup_steps = int(config.get("warmup_steps", 0))
        self.schedule_type = str(config.get("learning_rate_schedule", "cosine"))
        self.min_learning_rate = config.get("min_learning_rate", None)
        self.b2 = float(config.get("b2", 0.999))
        self.non_blocking_device_transfer = bool(
            config.get("non_blocking_device_transfer", True)
        )
        self.train_step_log_interval = int(
            config.get("train_step_log_interval", config.get("log_every_n_steps", 1))
        )
        self.adduct_loss_weight = float(config.get("adduct_loss_weight", 1.0))
        self.precursor_loss_weight = float(config.get("precursor_loss_weight", 1.0))
        self.adduct_label_smoothing = float(config.get("adduct_label_smoothing", 0.0))
        self.max_precursor_mz = float(config.get("max_precursor_mz", 1000.0))
        self.formula_token_offset = int(config.get("formula_token_offset", 4))
        self.formula_token_span = int(config.vocab_size) - self.formula_token_offset
        self.max_length = int(config.max_length)
        self.pad_token_id = int(config.pad_token_id)
        self.cls_token_id = int(config.cls_token_id)
        self.precursor_offset = int(config.precursor_offset)
        self.adduct_id_to_name = adduct_id_to_name

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
            cache_rope_frequencies=bool(config.get("cache_rope_frequencies", True)),
        )
        if pretrained_ckpt_path is not None:
            _load_pretrained_model(self.model, pretrained_ckpt_path)

        self.adduct_head = nn.Linear(
            int(config.model_dim),
            int(config.massspec_adduct_vocab_size),
        )

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

    def _build_model_batch_from_formula(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        formulas = batch["formula"]
        adduct_id = batch["adduct_id"].to(torch.long)
        batch_size = len(formulas)
        token_ids = torch.full(
            (batch_size, self.max_length),
            self.pad_token_id,
            dtype=torch.long,
        )
        token_ids[:, 0] = self.cls_token_id
        segment_ids = torch.zeros((batch_size, self.max_length), dtype=torch.long)
        precursor_tokens = torch.zeros((batch_size,), dtype=torch.long)

        for i, formula in enumerate(formulas):
            text = str(formula)
            chars = text[: self.max_length - 1]
            for j, c in enumerate(chars, start=1):
                token_ids[i, j] = self.formula_token_offset + (
                    ord(c) % self.formula_token_span
                )
            adduct_name = self.adduct_id_to_name[int(adduct_id[i].item())]
            precursor_mz = _precursor_mz_from_formula(text, adduct_name)
            precursor_bin = int(
                min(self.max_precursor_mz, max(0.0, precursor_mz))
            ) + self.precursor_offset
            precursor_tokens[i] = precursor_bin

        return {
            "token_ids": token_ids,
            "segment_ids": segment_ids,
            "precursor_mz": precursor_tokens,
            "adduct_id": adduct_id,
        }

    def _prepare_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if "token_ids" in batch:
            return batch
        return self._build_model_batch_from_formula(batch)

    def _compute_metrics(
        self,
        batch: dict[str, torch.Tensor],
        *,
        train: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cls_embedding = self.model.encode(batch, train=train)
        adduct_logits = self.adduct_head(cls_embedding)
        adduct_target = batch["adduct_id"].to(torch.long)
        adduct_loss = F.cross_entropy(
            adduct_logits,
            adduct_target,
            label_smoothing=self.adduct_label_smoothing,
        )
        adduct_accuracy = (
            adduct_logits.argmax(dim=-1) == adduct_target
        ).to(torch.float32).mean()

        precursor_tokens = batch["precursor_mz"].to(torch.long)
        precursor_loss, precursor_accuracy = self.model._precursor_metrics(
            cls_embedding,
            precursor_tokens,
        )
        loss = (
            self.adduct_loss_weight * adduct_loss
            + self.precursor_loss_weight * precursor_loss
        )
        return loss, adduct_loss, adduct_accuracy, precursor_loss, precursor_accuracy

    def _train_forward_impl(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._compute_metrics(batch, train=True)

    def _eval_forward_impl(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._compute_metrics(batch, train=False)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx
        torch.compiler.cudagraph_mark_step_begin()
        batch = self._prepare_batch(batch)
        batch = self._move_batch(batch)
        loss, adduct_loss, adduct_accuracy, precursor_loss, precursor_accuracy = (
            self._train_forward(batch)
        )

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
            self.log("train/adduct_loss", adduct_loss, on_step=True, on_epoch=False)
            self.log("train/adduct_accuracy", adduct_accuracy, on_step=True, on_epoch=False)
            self.log(
                "train/precursor_loss",
                precursor_loss,
                on_step=True,
                on_epoch=False,
            )
            self.log(
                "train/precursor_accuracy",
                precursor_accuracy,
                on_step=True,
                on_epoch=False,
            )

        self.log("train/loss_epoch", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log(
            "train/adduct_loss_epoch",
            adduct_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "train/adduct_accuracy_epoch",
            adduct_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "train/precursor_loss_epoch",
            precursor_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "train/precursor_accuracy_epoch",
            precursor_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx
        torch.compiler.cudagraph_mark_step_begin()
        batch = self._prepare_batch(batch)
        batch = self._move_batch(batch)
        loss, adduct_loss, adduct_accuracy, precursor_loss, precursor_accuracy = (
            self._eval_forward(batch)
        )
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/adduct_loss", adduct_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val/adduct_accuracy",
            adduct_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/precursor_loss",
            precursor_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/precursor_accuracy",
            precursor_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx
        torch.compiler.cudagraph_mark_step_begin()
        batch = self._prepare_batch(batch)
        batch = self._move_batch(batch)
        loss, adduct_loss, adduct_accuracy, precursor_loss, precursor_accuracy = (
            self._eval_forward(batch)
        )
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/adduct_loss", adduct_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "test/adduct_accuracy",
            adduct_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test/precursor_loss",
            precursor_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test/precursor_accuracy",
            precursor_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
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

    datamodule = MassSpecAdductPrecursorDataModule(config, seed=int(config.seed))

    info = datamodule.info
    if "vocab_size" in info:
        config.vocab_size = info["vocab_size"]
    else:
        config.vocab_size = _infer_vocab_size_from_checkpoint(Path(model_path))
    if "pair_sequence_length" in info:
        config.max_length = info["pair_sequence_length"]
    else:
        config.max_length = int(config.get("pair_sequence_length", 128))
    if "precursor_bins" in info:
        config.precursor_bins = info["precursor_bins"]
    else:
        config.precursor_bins = int(config.get("max_precursor_mz", 1000.0)) + 1
    if "precursor_offset" in info:
        config.precursor_offset = info["precursor_offset"]
    else:
        config.precursor_offset = int(config.get("precursor_offset", 4))
    if "massspec_adduct_vocab_size" in info:
        config.massspec_adduct_vocab_size = int(info["massspec_adduct_vocab_size"])
        adduct_vocab = info["massspec_adduct_vocab"]
    else:
        config.massspec_adduct_vocab_size = int(info["gems_formula_adduct_vocab_size"])
        adduct_vocab = info["gems_formula_adduct_vocab"]
    adduct_id_to_name = {int(v): str(k) for k, v in adduct_vocab.items()}
    config.adduct_loss_weight = float(config.get("adduct_loss_weight", 1.0))
    config.precursor_loss_weight = float(config.get("precursor_loss_weight", 1.0))
    config.adduct_label_smoothing = float(config.get("adduct_label_smoothing", 0.0))

    num_epochs = int(config.num_epochs)
    steps_per_epoch = datamodule.train_steps
    total_steps = num_epochs * steps_per_epoch

    logging.info(
        "Finetuning (adduct+precursor, source=%s) for %d epochs.",
        datamodule.data_source,
        num_epochs,
    )
    logging.info(
        "MSG train/val/test steps: %d / %d / %d",
        steps_per_epoch,
        datamodule.val_steps,
        datamodule.test_steps,
    )
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

    resume_ckpt_path = _latest_adduct_precursor_ckpt_path(checkpoint_dir, config)
    if resume_ckpt_path is not None:
        logging.info("Resuming finetuning from checkpoint: %s", resume_ckpt_path)
        pretrained_ckpt_path = None
    else:
        pretrained_ckpt_path = Path(model_path)
        logging.info("Initializing finetuning model from: %s", pretrained_ckpt_path)

    module = AdductPrecursorFinetuneModule(
        config,
        total_steps=total_steps,
        pretrained_ckpt_path=pretrained_ckpt_path,
        adduct_id_to_name=adduct_id_to_name,
        data_source=datamodule.data_source,
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
