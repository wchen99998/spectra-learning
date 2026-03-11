"""Run Dreams linear probe and print results."""

from __future__ import annotations

import logging
import sys

import torch
from ml_collections import config_dict

logging.basicConfig(level=logging.INFO, stream=sys.stdout)


def main() -> None:
    from configs.gems_a_masked_latent_index_small import get_config
    from utils.msg_probe import run_dreams_probe

    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Probe dataset: {cfg.probe_dataset}")
    print(f"Probe epochs: {cfg.msg_probe_num_epochs}")

    metrics = run_dreams_probe(config=cfg, device=device)

    if not metrics:
        print("ERROR: No Dreams embeddings available")
        sys.exit(1)

    print("\n=== DreaMS Probe Results ===")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
