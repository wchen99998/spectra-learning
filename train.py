import argparse
import json
import logging
import os

from spectra_learning.training.pretrain import train_and_evaluate
from spectra_learning.training.api import load_config
from spectra_learning.training.storage import normalize_storage_path, write_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train peak-set JEPA model.")
    parser.add_argument("--config", required=True, help="Path to config file.")
    parser.add_argument("--workdir", required=True, help="Output directory.")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--training-max-steps",
        type=int,
        default=None,
        help="Optional cap on training optimizer steps.",
    )
    parser.add_argument(
        "--overrides-json",
        default="{}",
        help="JSON object of config overrides applied after loading --config.",
    )
    parser.add_argument(
        "--metrics-json",
        default="",
        help="Optional path where rank 0 writes final metrics as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    config = load_config(args.config)
    config.update(json.loads(args.overrides_json))
    if args.training_max_steps is not None:
        config.training_max_steps = int(args.training_max_steps)
    results = train_and_evaluate(
        config,
        workdir=normalize_storage_path(args.workdir),
    )
    if args.metrics_json and int(os.environ.get("RANK", "0")) == 0:
        write_text(
            normalize_storage_path(args.metrics_json),
            json.dumps(results, indent=2, sort_keys=True),
        )


if __name__ == "__main__":
    main()
