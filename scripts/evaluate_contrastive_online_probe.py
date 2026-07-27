from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectra_learning.data.contracts import (
    data_provenance_contract,
    peak_preprocessing_contract,
    validate_data_provenance_contract,
    validate_peak_preprocessing_contract,
)
from spectra_learning.data.massspec_probe import MassSpecProbeData
from spectra_learning.data.massspec_targets import MACCS_FINGERPRINT_BITS
from spectra_learning.config import load_config
from spectra_learning.probes.massspec.msg_probe_common import fingerprint_metrics
from spectra_learning.training.runtime import parse_autocast_dtype
from spectra_learning.training.checkpointing import (
    load_resume_covariance_pooler_state,
    load_resume_model_state,
    load_torch_checkpoint,
)
from spectra_learning.training.contrastive import (
    ContrastiveOnlineBatchCollator,
    ContrastiveOnlineDataset,
    _load_contrastive_split,
    _peak_collator_kwargs,
    build_contrastive_module,
)
from spectra_learning.training.storage import normalize_storage_path, write_text


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the jointly trained contrastive online MACCS probe."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overrides-json", default="{}")
    parser.add_argument("--metrics-json", default="")
    parser.add_argument("--logits-npz", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, float]:
    args = parse_args(argv)
    config = load_config(
        args.config,
        {**json.loads(args.overrides_json), "training_mode": "contrastive"},
    )

    probe_data = MassSpecProbeData.from_config(config)
    module = build_contrastive_module(
        config,
        maccs_pos_weight=torch.ones(MACCS_FINGERPRINT_BITS),
    )

    checkpoint_path = normalize_storage_path(args.checkpoint)
    checkpoint = load_torch_checkpoint(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    validate_peak_preprocessing_contract(checkpoint, config)
    validate_data_provenance_contract(checkpoint, probe_data.info)
    load_resume_model_state(module.model, checkpoint["model"])
    load_resume_covariance_pooler_state(module.pooler, checkpoint_path, checkpoint)
    module.online_probe.load_state_dict(checkpoint["online_probe"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module.to(device).eval()
    autocast_dtype = parse_autocast_dtype(config.get("autocast_dtype", "bf16"))
    batch_size = args.batch_size or int(config.get("msg_probe_batch_size", 256))
    split_files = {
        "train": probe_data.train_files,
        "val": probe_data.val_files,
        "test": probe_data.test_files,
    }[args.split]
    split = _load_contrastive_split(split_files, max_samples=args.max_samples)
    loader = DataLoader(
        ContrastiveOnlineDataset(split),
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config.get("dataloader_num_workers", 0)),
        pin_memory=bool(config.get("dataloader_pin_memory", False)),
        collate_fn=ContrastiveOnlineBatchCollator(
            split,
            **_peak_collator_kwargs(config),
        ),
    )

    logits_by_batch = []
    targets_by_batch = []
    with torch.inference_mode():
        for batch in loader:
            batch = {
                key: (
                    value.to(device, non_blocking=True)
                    if isinstance(value, torch.Tensor)
                    else value
                )
                for key, value in batch.items()
            }
            with (
                torch.autocast(device_type=device.type, dtype=autocast_dtype)
                if autocast_dtype is not None
                else torch.inference_mode()
            ):
                pooled = module._pooled_features(batch)
                logits = module.online_probe(pooled)
            logits_by_batch.append(logits.float().cpu().numpy())
            targets_by_batch.append(batch["probe_maccs"].cpu().numpy())

    logits = np.concatenate(logits_by_batch, axis=0)
    predictions = 1.0 / (1.0 + np.exp(-logits))
    targets = np.concatenate(targets_by_batch, axis=0)
    metric_prefix = f"contrastive_online_probe/{args.split}"
    metrics = fingerprint_metrics(
        metric_prefix,
        fingerprint_task="maccs",
        pred=predictions,
        target=targets,
    )
    metrics[f"{metric_prefix}/bit_accuracy"] = float(np.mean((predictions >= 0.5) == (targets > 0.5)))
    metrics["contrastive_online_probe/global_step"] = float(checkpoint["global_step"])
    metrics["contrastive_online_probe/samples"] = float(targets.shape[0])
    metrics["contrastive_online_probe/maccs_bits"] = float(MACCS_FINGERPRINT_BITS)
    output = {
        "metrics": metrics,
        "split": args.split,
        "checkpoint_data_provenance": checkpoint["data_provenance"],
        "evaluation_data_provenance": data_provenance_contract(probe_data.info),
        "peak_preprocessing": peak_preprocessing_contract(config),
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    if args.metrics_json:
        write_text(
            normalize_storage_path(args.metrics_json),
            json.dumps(output, indent=2, sort_keys=True),
        )
    if args.logits_npz:
        np.savez_compressed(
            normalize_storage_path(args.logits_npz),
            logits=logits.astype(np.float32),
            targets=targets.astype(np.int32),
        )
    return metrics


if __name__ == "__main__":
    main()
