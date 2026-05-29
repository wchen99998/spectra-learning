from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectra_learning.config.loading import load_config
from spectra_learning.probes.massspec.checkpoint_probe import run_checkpoint_msg_probe
from spectra_learning.training.checkpointing import load_torch_checkpoint
from spectra_learning.training.storage import normalize_storage_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the MSG online probe for a saved training checkpoint."
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to config file.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path or URI.")
    parser.add_argument("--workdir", required=True, help="Output directory or URI.")
    parser.add_argument(
        "--global-step",
        type=int,
        default=None,
        help="Step to log metrics at. Defaults to checkpoint['global_step'].",
    )
    parser.add_argument(
        "--overrides-json",
        default="{}",
        help="JSON object of config overrides applied after loading --config.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, float]:
    args = parse_args(argv)
    config = load_config(args.config)
    config.update(json.loads(args.overrides_json))

    global_step = args.global_step
    if global_step is None:
        checkpoint = load_torch_checkpoint(
            normalize_storage_path(args.checkpoint),
            map_location="cpu",
            weights_only=True,
        )
        global_step = int(checkpoint["global_step"])

    metrics = run_checkpoint_msg_probe(
        config_json=json.dumps(config.to_dict()),
        checkpoint_path=args.checkpoint,
        workdir=args.workdir,
        global_step=global_step,
    )
    logging.info("Wrote MSG probe metrics for step %d to %s", global_step, args.workdir)
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
