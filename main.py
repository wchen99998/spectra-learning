from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TF info/warning logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN messages

import tensorflow as tf

from train import train_and_evaluate
from utils.training import load_config


tf.config.set_visible_devices([], "GPU")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MAE/BERT with PyTorch Lightning.")
    parser.add_argument("--config", required=True, help="Path to a config file (python).")
    parser.add_argument("--workdir", required=True, help="Output directory.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    config = load_config(args.config)
    workdir = Path(args.workdir).expanduser().resolve()

    train_and_evaluate(config, workdir=workdir)


if __name__ == "__main__":
    main()
