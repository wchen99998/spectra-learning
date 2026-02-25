from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import torch

from input_pipeline import TfLightningDataModule
from utils.probing import run_attentive_probe, run_linear_probe
from utils.training import (
    build_model_from_config,
    latest_ckpt_path,
    load_config,
    load_pretrained_weights,
)

tf.config.set_visible_devices([], "GPU")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run probe evaluation on a pretrained checkpoint.")
    parser.add_argument("--config", required=True, help="Path to a config file (python).")
    parser.add_argument("--workdir", required=True, help="Path to training workdir (used to find checkpoint).")
    parser.add_argument("--checkpoint", default=None, help="Explicit checkpoint path; defaults to latest in workdir.")
    parser.add_argument("--probe-type", choices=["attentive", "linear"], default="attentive", help="Probe type: attentive (default) or linear.")
    parser.add_argument(
        "--probe-feature-source",
        choices=["encoder"],
        default=None,
        help="Override probe feature source from config. encoder=token-level encoder features.",
    )
    parser.add_argument(
        "--probe-precursor-target",
        choices=["categorical", "continuous"],
        default=None,
        help="Override precursor probe target. categorical=1000-bin classification, continuous=regression on normalized precursor m/z.",
    )
    parser.add_argument("--wandb-project", default=None, help="W&B project for logging.")
    parser.add_argument("--no-freeze-backbone", action="store_true", help="Fine-tune backbone during probe (default: frozen).")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    config = load_config(args.config)
    workdir = Path(args.workdir).expanduser().resolve()

    datamodule = TfLightningDataModule(config, seed=int(config.seed))
    info = datamodule.info
    config.num_peaks = info["num_peaks"]
    config.fingerprint_bits = int(info["fingerprint_bits"])

    model = build_model_from_config(config)

    ckpt_path = args.checkpoint or latest_ckpt_path(workdir)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found in {workdir}")
    logging.info("Loading checkpoint: %s", ckpt_path)
    load_pretrained_weights(model, ckpt_path)

    if args.no_freeze_backbone:
        config.final_probe_freeze_backbone = False
    if args.probe_feature_source is not None:
        config.final_probe_feature_source = args.probe_feature_source
    if args.probe_precursor_target is not None:
        config.final_probe_precursor_target = args.probe_precursor_target

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loggers: list = []
    if args.wandb_project:
        config.enable_wandb = True
        config.wandb_project = args.wandb_project
    if config.get("enable_wandb", False):
        from lightning.pytorch.loggers import WandbLogger
        from utils import wandb_writer

        wandb_kwargs = wandb_writer.build_wandb_init_kwargs(config)
        logger = WandbLogger(
            project=config.get("wandb_project", "md4"),
            save_dir=str(workdir),
            log_model=False,
            **wandb_kwargs,
        )
        logger.log_hyperparams(wandb_writer.config_to_wandb_dict(config))
        loggers.append(logger)
    else:
        from lightning.pytorch.loggers import CSVLogger

        loggers.append(CSVLogger(save_dir=str(workdir), name="eval_logs"))

    run_probe = run_attentive_probe if args.probe_type == "attentive" else run_linear_probe
    metrics = run_probe(
        config=config,
        datamodule=datamodule,
        model=model,
        device=device,
        loggers=tuple(loggers),
    )
    for key, value in metrics.items():
        logging.info("%s = %.6f", key, value)


if __name__ == "__main__":
    main()
