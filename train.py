import argparse
import logging
from pathlib import Path

from spectra_learning.training.pretrain import train_and_evaluate
from spectra_learning.training.api import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train peak-set SIGReg model.")
    parser.add_argument("--config", required=True, help="Path to config file.")
    parser.add_argument("--workdir", required=True, help="Output directory.")
    parser.add_argument(
        "--training-max-steps",
        type=int,
        default=None,
        help="Optional cap on training optimizer steps.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    config = load_config(args.config)
    if args.training_max_steps is not None:
        config.training_max_steps = int(args.training_max_steps)
    train_and_evaluate(
        config,
        workdir=Path(args.workdir).expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
