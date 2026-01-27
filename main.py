from __future__ import annotations

import argparse
import importlib.util
import logging
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TF info/warning logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN messages

import tensorflow as tf
from ml_collections import config_dict

from train_mae import train_and_evaluate


tf.config.set_visible_devices([], "GPU")


def _load_config(path: str | Path) -> config_dict.ConfigDict:
    path = Path(path)
    spec = importlib.util.spec_from_file_location("experiment_config", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.get_config()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MAE/BERT with PyTorch Lightning.")
    parser.add_argument("--config", required=True, help="Path to a config file (python).")
    parser.add_argument("--workdir", required=True, help="Output directory.")
    parser.add_argument("--olddir", default=None, help="Optional checkpoint load directory.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    config = _load_config(args.config)
    workdir = Path(args.workdir).expanduser().resolve()
    olddir = None if args.olddir is None else Path(args.olddir).expanduser().resolve()

    train_and_evaluate(config, workdir=workdir, olddir=olddir)


if __name__ == "__main__":
    main()
