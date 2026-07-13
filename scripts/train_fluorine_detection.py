from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from spectra_learning.probes.massspec.fluorine import (
    FluorineData,
    HF_REPO_ID,
    HF_TEST_SUBDIR,
    HF_TRAIN_SUBDIR,
    MLPClassifier,
    TrialParams,
    TrialResult,
    _make_loader,
    binary_focal_loss_with_logits,
    default_state_path,
    run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train or evaluate fluorine-detection heads on the NIST Murcko "
            "train/val splits and MCEBIO Murcko test split."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("probe", "finetune", "lora"),
        default="probe",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "torch", "jax"),
        default="auto",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--jax-checkpoint-step", type=int, default=None)
    parser.add_argument(
        "--pooling",
        choices=("covariance", "single_pair_covariance"),
        default="covariance",
    )
    parser.add_argument("--train-covariance-pooler", action="store_true")
    parser.add_argument("--covariance-dim", type=int, default=None)
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/fluorine_detection_fine_tuned"),
    )
    parser.add_argument(
        "--finetune-cache-dir",
        type=Path,
        default=Path("data/fluorine_detection_fine_tuned"),
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-state", default=None)
    parser.add_argument("--head-state", type=Path, default=None)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--comparison-dir", type=Path, default=None)
    parser.add_argument(
        "--previous-ours-prefix",
        type=Path,
        default=Path("results/fluorine_detection_probe"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--device-ids", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-peaks", type=int, default=None)
    parser.add_argument("--peak-ordering", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--hidden-dims", default="256,512")
    parser.add_argument("--learning-rates", default="0.001,0.0003")
    parser.add_argument("--weight-decays", default="0.0001")
    parser.add_argument("--dropouts", default="0.1")
    parser.add_argument("--focal-alpha", default="auto")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--select-metric",
        default="average_precision",
        choices=("average_precision", "roc_auc", "balanced_accuracy", "f1"),
    )
    parser.add_argument("--finetune-model-lr", type=float, default=3e-6)
    parser.add_argument("--finetune-pooler-lr", type=float, default=3e-6)
    parser.add_argument("--finetune-head-lr", type=float, default=1e-4)
    parser.add_argument("--finetune-weight-decay", type=float, default=0.0001)
    parser.add_argument(
        "--autocast-dtype",
        choices=("bf16", "bfloat16", "fp16", "float16", "fp32", "float32", "none"),
        default=None,
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-learning-rate", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--eval-test-every-epoch", action="store_true")
    args = parser.parse_args()
    if args.epochs is None:
        args.epochs = 20 if args.mode == "probe" else 3
    if args.patience is None:
        args.patience = 5 if args.mode == "probe" else 2
    if args.head_state is None and args.mode != "probe":
        args.head_state = default_state_path(args.mode)
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    payload = run(parse_args())
    if payload:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
