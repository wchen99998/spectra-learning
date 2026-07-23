import argparse
import json
import logging
import os

from spectra_learning.config import load_config
from spectra_learning.training.jax_runtime_flags import configure_jax_tpu_xla_flags
from spectra_learning.training.logging import _serialise_metrics
from spectra_learning.training.routing import resolve_training_route
from spectra_learning.training.storage import normalize_storage_path, write_text


def _train(config, workdir):
    task, backend = resolve_training_route(config)
    if task == "pretrain":
        if backend == "jax":
            configure_jax_tpu_xla_flags()
            from spectra_learning.training.pretrain_jax import train_and_evaluate_jax

            return train_and_evaluate_jax(config, workdir)
        from spectra_learning.training.pretrain import train_and_evaluate

        return train_and_evaluate(config, workdir=workdir)
    if task == "contrastive":
        from spectra_learning.training.contrastive import train_contrastive

        return train_contrastive(config, workdir=workdir)
    configure_jax_tpu_xla_flags()
    from spectra_learning.training.ar_spectra_jax import (
        train_and_evaluate_ar_spectra_jax,
    )

    return train_and_evaluate_ar_spectra_jax(config, workdir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train peak-set JEPA model.")
    parser.add_argument("--config", required=True, help="Path to config file.")
    parser.add_argument("--workdir", required=True, help="Output directory.")
    parser.add_argument("--local-rank", type=int, default=0)
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
    config = load_config(args.config, json.loads(args.overrides_json))
    results = _train(config, normalize_storage_path(args.workdir))
    process_index = int(results.get("run/jax_process_index", os.environ.get("RANK", "0")))
    if args.metrics_json and process_index == 0:
        write_text(
            normalize_storage_path(args.metrics_json),
            json.dumps(_serialise_metrics(results), indent=2, sort_keys=True),
        )


if __name__ == "__main__":
    main()
