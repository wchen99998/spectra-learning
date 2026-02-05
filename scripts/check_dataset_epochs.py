from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch

from input_pipeline import TfLightningDataModule


class _DummyTrainer:
    def __init__(self) -> None:
        self.current_epoch = 0


def _load_config(path: str) -> object:
    spec = importlib.util.spec_from_file_location("experiment_config", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.get_config()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan full dataset epochs and compute checksums without training."
    )
    parser.add_argument("--config", required=True, help="Path to config file.")
    parser.add_argument(
        "--split",
        default="gems_train",
        choices=(
            "gems_train",
            "gems_val",
            "gems_test",
            "massspec_train",
            "massspec_val",
            "massspec_test",
        ),
        help="Dataset split to scan.",
    )
    parser.add_argument("--epochs", type=int, default=2, help="Epochs to scan.")
    return parser.parse_args()


def _batch_checksum(batch: dict[str, torch.Tensor]) -> int:
    token_ids = batch["token_ids"].to(torch.int64)
    segment_ids = batch["segment_ids"].to(torch.int64)
    precursor = batch["precursor_mz"].to(torch.int64)
    return int(
        (token_ids.sum() + 3 * segment_ids.sum() + 7 * precursor.sum()).item()
    )


def _expected_samples(size: int, batch_size: int, drop_remainder: bool) -> int:
    if drop_remainder:
        return (size // batch_size) * batch_size
    return size


def main() -> None:
    args = _parse_args()
    cfg = _load_config(args.config)
    if hasattr(cfg, "steps_per_epoch"):
        cfg.steps_per_epoch = 0
    if hasattr(cfg, "num_eval_steps"):
        cfg.num_eval_steps = 0

    dm = TfLightningDataModule(cfg, seed=int(cfg.seed))
    dm.trainer = _DummyTrainer()

    if args.split == "gems_train":
        dataset = dm._build_gems_train_dataset(int(cfg.seed))
        steps = dm.steps["gems_train"]
        size = dm.info["train_size"]
        drop_remainder = bool(cfg.get("drop_remainder", False))
    elif args.split == "gems_val":
        dataset = dm._build_gems_val_dataset(int(cfg.seed))
        steps = dm.steps["gems_val"]
        size = dm.info["validation_size"]
        drop_remainder = False
    elif args.split == "gems_test":
        dataset = dm._build_gems_test_dataset(int(cfg.seed))
        steps = dm.steps["gems_test"]
        size = dm.info["validation_size"]
        drop_remainder = False
    elif args.split == "massspec_train":
        dataset = dm._build_massspec_train_dataset(int(cfg.seed))
        steps = dm.steps["massspec_train"]
        size = dm.info["massspec_train_size"]
        drop_remainder = bool(cfg.get("drop_remainder", False))
    elif args.split == "massspec_val":
        dataset = dm._build_massspec_val_dataset(int(cfg.seed))
        steps = dm.steps["massspec_val"]
        size = dm.info["massspec_val_size"]
        drop_remainder = False
    else:
        dataset = dm._build_massspec_test_dataset(int(cfg.seed))
        steps = dm.steps["massspec_test"]
        size = dm.info["massspec_test_size"]
        drop_remainder = False

    loader = dm._make_loader(dataset=dataset, steps=steps)
    expected_samples = _expected_samples(size, int(cfg.batch_size), drop_remainder)

    checksums: list[tuple[int, int, int]] = []
    for epoch in range(int(args.epochs)):
        dm.trainer.current_epoch = epoch
        order_hash = 0
        content_hash = 0
        samples_seen = 0

        for batch in loader:
            remaining = expected_samples - samples_seen
            if remaining <= 0:
                break
            take = min(int(batch["token_ids"].shape[0]), remaining)
            if take == batch["token_ids"].shape[0]:
                batch_sum = _batch_checksum(batch)
            else:
                sliced = {key: value[:take] for key, value in batch.items()}
                batch_sum = _batch_checksum(sliced)
            content_hash = (content_hash + batch_sum) & ((1 << 64) - 1)
            order_hash = (order_hash * 1_000_003 + batch_sum) & ((1 << 64) - 1)
            samples_seen += take

        checksums.append((samples_seen, content_hash, order_hash))
        print(
            f"epoch={epoch} samples={samples_seen} expected={expected_samples} "
            f"content_hash={content_hash} order_hash={order_hash}"
        )

    first_samples, first_content, first_order = checksums[0]
    for epoch, (samples_seen, content_hash, order_hash) in enumerate(checksums[1:], start=1):
        print(
            f"compare epoch0 vs epoch{epoch}: "
            f"samples_equal={samples_seen == first_samples} "
            f"content_equal={content_hash == first_content} "
            f"order_equal={order_hash == first_order}"
        )


if __name__ == "__main__":
    main()
