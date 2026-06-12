from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectra_learning.data.spectra import (
    DEFAULT_GROUPED_PEAK_ISOTOPE_CHARGES,
    DEFAULT_GROUPED_PEAK_SHOULDER_DA,
    DEFAULT_MAX_PRECURSOR_MZ,
    DEFAULT_MIN_PEAK_INTENSITY,
    DEFAULT_PEAK_FILTERING,
    preprocess_peak_batch_torch,
)
from spectra_learning.probes.massspec.data import MassSpecProbeData
from spectra_learning.probes.massspec.msg_probe import iter_massspec_probe
from spectra_learning.probes.massspec.msg_settings import resolve_msg_probe_sample_limits
from spectra_learning.probes.massspec.targets import MACCS_FINGERPRINT_BITS
from spectra_learning.training.api import load_config, parse_autocast_dtype
from spectra_learning.training.checkpointing import (
    load_resume_covariance_pooler_state,
    load_resume_model_state,
    load_torch_checkpoint,
)
from spectra_learning.training.contrastive import (
    ContrastiveSplit,
    _load_contrastive_split,
    build_contrastive_module,
)
from spectra_learning.training.storage import normalize_storage_path, write_text


class ContrastiveOnlineEvalDataset(Dataset):
    def __init__(self, split: ContrastiveSplit) -> None:
        self.split = split

    def __len__(self) -> int:
        return int(self.split.spectra.shape[0])

    def __getitem__(self, index: int) -> int:
        return index


class ContrastiveOnlineEvalCollator:
    def __init__(
        self,
        split: ContrastiveSplit,
        *,
        num_peaks: int,
        max_precursor_mz: float,
        min_peak_intensity: float,
        peak_drop_min_intensity: float,
        peak_ordering: str,
        precursor_peak_exclusion_window_da: float,
        peak_filtering: str = DEFAULT_PEAK_FILTERING,
        grouped_peak_shoulder_da: float = DEFAULT_GROUPED_PEAK_SHOULDER_DA,
        grouped_peak_isotope_charges: tuple[int, ...] = (
            DEFAULT_GROUPED_PEAK_ISOTOPE_CHARGES
        ),
    ) -> None:
        self.split = split
        self.num_peaks = num_peaks
        self.max_precursor_mz = max_precursor_mz
        self.min_peak_intensity = min_peak_intensity
        self.peak_drop_min_intensity = peak_drop_min_intensity
        self.peak_ordering = peak_ordering
        self.precursor_peak_exclusion_window_da = precursor_peak_exclusion_window_da
        self.peak_filtering = peak_filtering
        self.grouped_peak_shoulder_da = grouped_peak_shoulder_da
        self.grouped_peak_isotope_charges = grouped_peak_isotope_charges

    def __call__(self, indices: list[int]) -> dict[str, torch.Tensor]:
        row_indices = np.asarray(indices, dtype=np.int64)
        spectra = torch.from_numpy(self.split.spectra[row_indices].copy())
        precursor_raw = torch.from_numpy(self.split.precursor_mz[row_indices].copy())
        batch = preprocess_peak_batch_torch(
            spectra[:, 0, :],
            spectra[:, 1, :],
            precursor_raw,
            num_peaks=self.num_peaks,
            peak_drop_min_intensity=self.peak_drop_min_intensity,
            peak_ordering=self.peak_ordering,
            max_precursor_mz=self.max_precursor_mz,
            precursor_peak_exclusion_window_da=self.precursor_peak_exclusion_window_da,
            min_peak_intensity=self.min_peak_intensity,
            peak_filtering=self.peak_filtering,
            grouped_peak_shoulder_da=self.grouped_peak_shoulder_da,
            grouped_peak_isotope_charges=self.grouped_peak_isotope_charges,
        )
        batch["probe_maccs"] = torch.from_numpy(
            self.split.probe_maccs[row_indices].copy()
        ).to(torch.float32)
        return batch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the jointly trained contrastive online MACCS probe."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--msg-probe-sampling", action="store_true")
    parser.add_argument("--overrides-json", default="{}")
    parser.add_argument("--metrics-json", default="")
    parser.add_argument("--logits-npz", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, float]:
    args = parse_args(argv)
    config = load_config(args.config)
    config.update(json.loads(args.overrides_json))
    config.training_mode = "contrastive"

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
    load_resume_model_state(module.model, checkpoint["model"])
    load_resume_covariance_pooler_state(module.pooler, checkpoint_path, checkpoint)
    module.online_probe.load_state_dict(checkpoint["online_probe"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module.to(device).eval()
    autocast_dtype = parse_autocast_dtype(config.get("autocast_dtype", "bf16"))
    batch_size = args.batch_size or int(config.get("msg_probe_batch_size", 256))
    if args.msg_probe_sampling:
        config.msg_probe_batch_size = batch_size
        max_train_samples, max_val_samples, max_test_samples, randomize_test_subset = (
            resolve_msg_probe_sample_limits(config)
        )
        max_samples_by_split = {
            "train": max_train_samples,
            "val": max_val_samples,
            "test": max_test_samples,
        }
        seeds_by_split = {
            "train": int(config.seed) + 1_100_000,
            "val": int(config.seed) + 1_110_000,
            "test": int(config.seed) + 1_200_000,
        }
        random_by_split = {
            "train": False,
            "val": True,
            "test": randomize_test_subset,
        }
        loader = iter_massspec_probe(
            probe_data,
            f"massspec_{args.split}",
            seed=seeds_by_split[args.split],
            peak_ordering=str(config.get("peak_ordering", "mz")),
            drop_remainder=False,
            max_samples=(
                args.max_samples
                if args.max_samples is not None
                else max_samples_by_split[args.split]
            ),
            sample_randomly=random_by_split[args.split],
        )
    else:
        split_files = {
            "train": probe_data.train_files,
            "val": probe_data.val_files,
            "test": probe_data.test_files,
        }[args.split]
        split = _load_contrastive_split(split_files, max_samples=args.max_samples)
        loader = DataLoader(
            ContrastiveOnlineEvalDataset(split),
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(config.get("dataloader_num_workers", 0)),
            pin_memory=bool(config.get("dataloader_pin_memory", False)),
            collate_fn=ContrastiveOnlineEvalCollator(
                split,
                num_peaks=int(config.get("num_peaks", 60)),
                max_precursor_mz=float(
                    config.get("max_precursor_mz", DEFAULT_MAX_PRECURSOR_MZ)
                ),
                min_peak_intensity=float(
                    config.get("min_peak_intensity", DEFAULT_MIN_PEAK_INTENSITY)
                ),
                peak_drop_min_intensity=float(
                    config.get(
                        "peak_drop_min_intensity",
                        config.get("min_peak_intensity", DEFAULT_MIN_PEAK_INTENSITY),
                    )
                ),
                peak_ordering=str(config.get("peak_ordering", "mz")),
                precursor_peak_exclusion_window_da=float(
                    config.get("precursor_peak_exclusion_window_da", 0.0)
                ),
                peak_filtering=str(config.get("peak_filtering", DEFAULT_PEAK_FILTERING)),
                grouped_peak_shoulder_da=float(
                    config.get(
                        "grouped_peak_shoulder_da",
                        DEFAULT_GROUPED_PEAK_SHOULDER_DA,
                    )
                ),
                grouped_peak_isotope_charges=tuple(
                    int(charge)
                    for charge in config.get(
                        "grouped_peak_isotope_charges",
                        DEFAULT_GROUPED_PEAK_ISOTOPE_CHARGES,
                    )
                ),
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
                peak_embeddings, _, pair_embeddings = (
                    module.model.encoder.forward_with_block_outputs(
                        batch["peak_mz"],
                        batch["peak_intensity"],
                        valid_mask=batch["peak_valid_mask"],
                        precursor_mz=batch.get("precursor_mz", None),
                    )
                )
                pooled = module.pooler(
                    peak_embeddings.float(),
                    batch["peak_valid_mask"].to(dtype=torch.bool),
                    pair_embeddings.float(),
                )
                logits = module.online_probe(pooled)
            logits_by_batch.append(logits.float().cpu().numpy())
            targets_by_batch.append(batch["probe_maccs"].cpu().numpy())

    logits = np.concatenate(logits_by_batch, axis=0)
    predictions = 1.0 / (1.0 + np.exp(-logits))
    targets = np.concatenate(targets_by_batch, axis=0)
    metrics = _maccs_metrics(
        predictions=predictions,
        targets=targets,
        prefix=f"contrastive_online_probe/{args.split}",
    )
    metrics["contrastive_online_probe/global_step"] = float(checkpoint["global_step"])
    metrics["contrastive_online_probe/samples"] = float(targets.shape[0])
    metrics["contrastive_online_probe/maccs_bits"] = float(MACCS_FINGERPRINT_BITS)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if args.metrics_json:
        write_text(
            normalize_storage_path(args.metrics_json),
            json.dumps(metrics, indent=2, sort_keys=True),
        )
    if args.logits_npz:
        np.savez_compressed(
            normalize_storage_path(args.logits_npz),
            logits=logits.astype(np.float32),
            targets=targets.astype(np.int32),
        )
    return metrics


def _maccs_metrics(
    *,
    predictions: np.ndarray,
    targets: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    positives = targets.sum(axis=0)
    valid_metric_mask = (positives > 0) & (positives < targets.shape[0])
    valid_target = targets[:, valid_metric_mask].astype(np.float64)
    valid_pred = predictions[:, valid_metric_mask].astype(np.float64)
    valid_positives = positives[valid_metric_mask].astype(np.float64)
    valid_negatives = float(targets.shape[0]) - valid_positives
    ascending = np.argsort(valid_pred, axis=0)
    target_ascending = np.take_along_axis(valid_target, ascending, axis=0)
    negatives_before = np.cumsum(1.0 - target_ascending, axis=0)
    auc_values = (
        (target_ascending * negatives_before).sum(axis=0)
        / (valid_positives * valid_negatives)
    )
    target_descending = target_ascending[::-1]
    true_positives_at_rank = np.cumsum(target_descending, axis=0)
    ranks = np.arange(1, target_descending.shape[0] + 1, dtype=np.float64)[:, None]
    average_precision_values = (
        (target_descending * true_positives_at_rank / ranks).sum(axis=0)
        / valid_positives
    )
    bit_pred = predictions >= 0.5
    target_bits = targets > 0.5
    bit_accuracy = np.mean(bit_pred == target_bits)
    intersection = np.count_nonzero(bit_pred & target_bits, axis=1)
    union = np.count_nonzero(bit_pred | target_bits, axis=1)
    tanimoto_values = intersection / np.maximum(union, 1)
    return {
        f"{prefix}/auc_maccs_mean": float(np.mean(auc_values)),
        f"{prefix}/average_precision_maccs_mean": float(
            np.mean(average_precision_values)
        ),
        f"{prefix}/num_maccs_auc_bits": float(len(auc_values)),
        f"{prefix}/bit_accuracy": float(bit_accuracy),
        f"{prefix}/tanimoto_maccs_mean": float(np.mean(tanimoto_values)),
    }


if __name__ == "__main__":
    main()
