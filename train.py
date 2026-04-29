import argparse
import logging
from pathlib import Path

from spectra_learning.training.pretrain import train_and_evaluate
from utils.training import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train peak-set SIGReg model.")
    parser.add_argument("--config", required=True, help="Path to config file.")
    parser.add_argument("--workdir", required=True, help="Output directory.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    train_and_evaluate(
        load_config(args.config),
        workdir=Path(args.workdir).expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
