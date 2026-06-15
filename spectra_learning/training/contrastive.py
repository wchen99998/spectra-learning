from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, NamedTuple, cast

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from ml_collections import config_dict
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.nn import functional as dist_nn
from tqdm import tqdm

from spectra_learning.data.spectra import (
    DEFAULT_MAX_PRECURSOR_MZ,
    DEFAULT_GROUPED_PEAK_ISOTOPE_CHARGES,
    DEFAULT_GROUPED_PEAK_SHOULDER_DA,
    DEFAULT_MIN_PEAK_INTENSITY,
    DEFAULT_PEAK_FILTERING,
    NUM_PEAKS_INPUT,
    preprocess_peak_batch_torch,
)
from spectra_learning.data.loading import local_batch_size
from spectra_learning.models.model import PeakSetJEPA
from spectra_learning.models.pooling import SinglePairCovariancePool
from spectra_learning.data.massspec_probe import MassSpecProbeData
from spectra_learning.data.massspec_targets import MACCS_FINGERPRINT_BITS
from spectra_learning.training.api import (
    build_grad_scaler,
    build_logger,
    build_model_from_config,
    collect_and_log_param_metrics,
    cumulative_training_flops,
    estimate_training_flops_per_optimizer_step,
    parse_autocast_dtype,
)
from spectra_learning.training.checkpointing import (
    covariance_pooler_checkpoint_path,
    load_grad_scaler_state,
    load_optimizer_state,
    load_pretrained_weights,
    load_resume_covariance_pooler_state,
    load_resume_model_state,
    load_torch_checkpoint,
    save_torch_checkpoint,
    training_checkpoint_paths,
)
from spectra_learning.training.distributed import (
    DistributedContext,
    barrier,
    cleanup_distributed,
    init_distributed_from_env,
    reduce_metric_tensors,
    unwrap_model,
    wrap_distributed_model,
)
from spectra_learning.training.optimization import build_optimizers, is_weight_decay_target
from spectra_learning.training.schedules import LRSchedulerLike, make_cosine_schedule
from spectra_learning.training.storage import (
    StoragePath,
    local_scratch_dir,
    normalize_storage_path,
    storage_join,
    storage_mkdir,
    storage_name,
)

log = logging.getLogger(__name__)


def _config_get(config: config_dict.ConfigDict, key: str, default: Any) -> Any:
    return config.get(key, default)


class ContrastiveSplit(NamedTuple):
    spectra: np.ndarray
    precursor_mz: np.ndarray
    smiles: np.ndarray
    collision_energy: np.ndarray
    collision_energy_present: np.ndarray
    probe_maccs: np.ndarray


class ContrastiveBatchCollator:
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

    def __call__(self, pairs: list[dict[str, int]]) -> dict[str, torch.Tensor]:
        has_explicit_negative = "negative_idx" in pairs[0]
        if has_explicit_negative:
            indices = np.asarray(
                [
                    idx
                    for pair in pairs
                    for idx in (
                        pair["left_idx"],
                        pair["right_idx"],
                        pair["negative_idx"],
                    )
                ],
                dtype=np.int64,
            )
            compound_ids = torch.tensor(
                [
                    compound_id
                    for pair in pairs
                    for compound_id in (
                        pair["left_compound_id"],
                        pair["right_compound_id"],
                        pair["negative_compound_id"],
                    )
                ],
                dtype=torch.long,
            )
            positive_index = torch.arange(len(indices), dtype=torch.long)
            anchors = torch.arange(0, len(indices), 3, dtype=torch.long)
            positive_index[anchors] = anchors + 1
            positive_index[anchors + 1] = anchors
            batch_indices = {
                "triplet_anchor_index": anchors,
                "triplet_positive_index": anchors + 1,
                "triplet_negative_index": anchors + 2,
            }
        else:
            indices = np.asarray(
                [idx for pair in pairs for idx in (pair["left_idx"], pair["right_idx"])],
                dtype=np.int64,
            )
            compound_ids = torch.tensor(
                [pair["compound_id"] for pair in pairs for _ in range(2)],
                dtype=torch.long,
            )
            positive_index = torch.arange(len(indices), dtype=torch.long) ^ 1
            batch_indices = {}
        spectra = torch.from_numpy(self.split.spectra[indices].copy())
        precursor_raw = torch.from_numpy(self.split.precursor_mz[indices].copy())
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
        batch["compound_id"] = compound_ids
        batch["positive_index"] = positive_index
        batch.update(batch_indices)
        batch["probe_maccs"] = torch.from_numpy(self.split.probe_maccs[indices].copy()).to(
            torch.float32
        )
        return batch


class ContrastiveOnlineDataset(Dataset):
    def __init__(self, split: ContrastiveSplit) -> None:
        self.split = split

    def __len__(self) -> int:
        return int(self.split.spectra.shape[0])

    def __getitem__(self, index: int) -> int:
        return index


class WeightedOnlineSampler(Sampler[int]):
    def __init__(
        self,
        weights: np.ndarray,
        *,
        num_replicas: int,
        rank: int,
        seed: int,
        drop_last: bool,
    ) -> None:
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        if drop_last:
            self.num_samples = len(weights) // num_replicas
        else:
            self.num_samples = math.ceil(len(weights) / num_replicas)
        self.total_size = self.num_samples * num_replicas

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=True,
            generator=generator,
        )
        indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class ContrastiveOnlineBatchCollator:
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


class GatheredContrastiveBatch(NamedTuple):
    anchor_features: torch.Tensor
    candidate_features: torch.Tensor
    positive_index: torch.Tensor
    anchor_index: torch.Tensor
    anchor_compound_id: torch.Tensor
    candidate_compound_id: torch.Tensor
    anchor_maccs: torch.Tensor
    candidate_maccs: torch.Tensor


class NistMurckoContrastivePairs(Dataset):
    def __init__(
        self,
        split: ContrastiveSplit,
        *,
        pairs_per_epoch: int,
        seed: int,
        negative_mass_tolerance_da: float | None = None,
    ) -> None:
        self.split = split
        self.pairs_per_epoch = pairs_per_epoch
        self.seed = seed
        self.epoch = 0
        groups: dict[str, dict[float, list[int]]] = {}
        for idx, (smiles, ce, present) in enumerate(
            zip(
                split.smiles.tolist(),
                split.collision_energy.tolist(),
                split.collision_energy_present.tolist(),
                strict=True,
            )
        ):
            if not int(present):
                continue
            by_energy = groups.setdefault(str(smiles), {})
            by_energy.setdefault(float(ce), []).append(idx)
        eligible: list[tuple[str, list[float]]] = []
        for smiles, by_energy in groups.items():
            energies = sorted(by_energy)
            if len(energies) >= 2:
                eligible.append((smiles, energies))
        self.groups = groups
        self.compound_to_id = {
            smiles: idx
            for idx, smiles in enumerate(sorted(set(split.smiles.astype(str).tolist())))
        }
        self.negative_by_idx = (
            self._mass_matched_negatives(negative_mass_tolerance_da)
            if negative_mass_tolerance_da is not None
            else None
        )
        self.eligible = (
            self._eligible_with_negatives(eligible)
            if negative_mass_tolerance_da is not None
            else eligible
        )

    def __len__(self) -> int:
        return self.pairs_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict[str, int]:
        rng = np.random.default_rng(self.seed + self.epoch * self.pairs_per_epoch + index)
        if self.negative_by_idx is not None:
            compound_pos = int(rng.integers(0, len(self.eligible)))
            smiles, anchors_by_energy, positive_energies = self.eligible[compound_pos]
            anchor_energy = rng.choice(np.asarray(sorted(anchors_by_energy)))
            left_bucket = anchors_by_energy[float(anchor_energy)]
            left_idx = int(left_bucket[int(rng.integers(0, len(left_bucket)))])
            right_energy = rng.choice(
                np.asarray(
                    [
                        energy
                        for energy in positive_energies
                        if energy != float(anchor_energy)
                    ]
                )
            )
            right_bucket = self.groups[smiles][float(right_energy)]
            right_idx = int(right_bucket[int(rng.integers(0, len(right_bucket)))])
            negative_bucket = self.negative_by_idx[left_idx]
            negative_idx = int(negative_bucket[int(rng.integers(0, len(negative_bucket)))])
            negative_smiles = str(self.split.smiles[negative_idx])
            return {
                "left_idx": left_idx,
                "right_idx": right_idx,
                "negative_idx": negative_idx,
                "left_compound_id": self.compound_to_id[smiles],
                "right_compound_id": self.compound_to_id[smiles],
                "negative_compound_id": self.compound_to_id[negative_smiles],
            }
        return self._pair_item(rng)

    def _pair_item(self, rng: np.random.Generator) -> dict[str, int]:
        compound_pos = int(rng.integers(0, len(self.eligible)))
        smiles, energies = self.eligible[compound_pos]
        energy_pair = rng.choice(np.asarray(energies), size=2, replace=False)
        by_energy = self.groups[smiles]
        left_bucket = by_energy[float(energy_pair[0])]
        right_bucket = by_energy[float(energy_pair[1])]
        return {
            "left_idx": int(left_bucket[int(rng.integers(0, len(left_bucket)))]),
            "right_idx": int(right_bucket[int(rng.integers(0, len(right_bucket)))]),
            "compound_id": self.compound_to_id[smiles],
        }

    def _mass_matched_negatives(
        self,
        tolerance_da: float | None,
    ) -> dict[int, np.ndarray]:
        assert tolerance_da is not None
        mz = self.split.precursor_mz
        smiles = self.split.smiles.astype(str)
        order = np.argsort(mz)
        sorted_mz = mz[order]
        negative_by_idx = {}
        for idx, precursor_mz in enumerate(mz):
            left = int(np.searchsorted(sorted_mz, precursor_mz - tolerance_da, side="left"))
            right = int(np.searchsorted(sorted_mz, precursor_mz + tolerance_da, side="right"))
            candidates = order[left:right]
            candidates = candidates[smiles[candidates] != smiles[idx]]
            if len(candidates) > 0:
                negative_by_idx[idx] = candidates.astype(np.int64)
        return negative_by_idx

    def _eligible_with_negatives(
        self,
        eligible: list[tuple[str, list[float]]],
    ) -> list[tuple[str, dict[float, list[int]], list[float]]]:
        assert self.negative_by_idx is not None
        with_negatives = []
        for smiles, energies in eligible:
            by_energy = self.groups[smiles]
            anchors_by_energy = {
                energy: [idx for idx in by_energy[energy] if idx in self.negative_by_idx]
                for energy in energies
            }
            anchors_by_energy = {
                energy: indices
                for energy, indices in anchors_by_energy.items()
                if indices and len(energies) > 1
            }
            if anchors_by_energy:
                with_negatives.append((smiles, anchors_by_energy, energies))
        return with_negatives


def _fixed_spectra_from_lists(
    mz_lists: list[list[float]],
    intensity_lists: list[list[float]],
) -> np.ndarray:
    spectra = np.zeros((len(mz_lists), 2, NUM_PEAKS_INPUT), dtype=np.float32)
    for idx, (mz, intensity) in enumerate(zip(mz_lists, intensity_lists, strict=True)):
        n = min(len(mz), NUM_PEAKS_INPUT)
        spectra[idx, 0, :n] = np.asarray(mz[:n], dtype=np.float32)
        spectra[idx, 1, :n] = np.asarray(intensity[:n], dtype=np.float32)
    max_intensity = spectra[:, 1].max(axis=1, keepdims=True)
    spectra[:, 1] = spectra[:, 1] / np.maximum(max_intensity, 1e-8)
    return spectra


def _load_contrastive_split(
    files: list[str],
    *,
    max_samples: int | None,
) -> ContrastiveSplit:
    import pyarrow.parquet as pq

    chunks: list[dict[str, Any]] = []
    remaining = max_samples
    columns = [
        "spectrum_mz",
        "spectrum_intensity",
        "precursor_mz",
        "canonical_smiles",
        "collision_energy",
        "collision_energy_present",
        "maccs_166",
    ]
    for path in files:
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=65_536, columns=columns):
            if remaining is not None:
                batch = batch.slice(0, remaining)
                remaining -= batch.num_rows
            chunks.append(batch.to_pydict())
            if remaining == 0:
                break
        if remaining == 0:
            break

    mz_lists = [mz for chunk in chunks for mz in chunk["spectrum_mz"]]
    intensity_lists = [
        intensity for chunk in chunks for intensity in chunk["spectrum_intensity"]
    ]
    return ContrastiveSplit(
        spectra=_fixed_spectra_from_lists(mz_lists, intensity_lists),
        precursor_mz=np.concatenate(
            [np.asarray(chunk["precursor_mz"], dtype=np.float32) for chunk in chunks],
            axis=0,
        ),
        smiles=np.concatenate(
            [np.asarray(chunk["canonical_smiles"], dtype=str) for chunk in chunks],
            axis=0,
        ),
        collision_energy=np.concatenate(
            [np.asarray(chunk["collision_energy"], dtype=np.float32) for chunk in chunks],
            axis=0,
        ),
        collision_energy_present=np.concatenate(
            [
                np.asarray(chunk["collision_energy_present"], dtype=np.int32)
                for chunk in chunks
            ],
            axis=0,
        ),
        probe_maccs=np.concatenate(
            [np.asarray(chunk["maccs_166"], dtype=np.int8) for chunk in chunks],
            axis=0,
        ),
    )


def _maccs_pos_weight(split: ContrastiveSplit) -> torch.Tensor:
    targets = split.probe_maccs.astype(np.float32)
    positive = targets.sum(axis=0)
    negative = targets.shape[0] - positive
    return torch.from_numpy(negative / np.maximum(positive, 1.0))


def _online_sample_weights(split: ContrastiveSplit, power: float) -> np.ndarray:
    targets = split.probe_maccs.astype(np.float32)
    bit_frequency = np.clip(targets.mean(axis=0), 0.01, 1.0)
    bit_weight = np.power(1.0 / bit_frequency, power).astype(np.float32)
    sample_score = targets @ bit_weight
    sample_score = sample_score / max(float(sample_score.mean()), 1e-6)
    return (1.0 + sample_score).astype(np.float64)


def build_contrastive_loader(
    config: config_dict.ConfigDict,
    split: ContrastiveSplit,
    *,
    split_name: str,
    pairs_per_epoch: int,
    global_pairs_per_batch: int,
    seed: int,
    distributed: DistributedContext,
) -> DataLoader:
    dataset = NistMurckoContrastivePairs(
        split,
        pairs_per_epoch=pairs_per_epoch,
        seed=seed,
        negative_mass_tolerance_da=_optional_float(
            _config_get(config, "contrastive_triplet_negative_mass_tolerance_da", None)
        ),
    )
    assert dataset.eligible
    local_pairs_per_batch = local_batch_size(
        global_pairs_per_batch,
        distributed.world_size,
    )
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=distributed.world_size,
            rank=distributed.rank,
            shuffle=(split_name == "train"),
            seed=seed,
            drop_last=(split_name == "train"),
        )
        if distributed.is_distributed
        else None
    )
    num_workers = int(_config_get(config, "dataloader_num_workers", 0))
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": local_pairs_per_batch,
        "shuffle": (split_name == "train" and sampler is None),
        "sampler": sampler,
        "drop_last": (split_name == "train"),
        "num_workers": num_workers,
        "pin_memory": bool(_config_get(config, "dataloader_pin_memory", False)),
        "collate_fn": ContrastiveBatchCollator(
            split,
            num_peaks=int(_config_get(config, "num_peaks", 60)),
            max_precursor_mz=float(
                _config_get(config, "max_precursor_mz", DEFAULT_MAX_PRECURSOR_MZ)
            ),
            min_peak_intensity=float(
                _config_get(config, "min_peak_intensity", DEFAULT_MIN_PEAK_INTENSITY)
            ),
            peak_drop_min_intensity=float(
                _config_get(
                    config,
                    "peak_drop_min_intensity",
                    _config_get(config, "min_peak_intensity", DEFAULT_MIN_PEAK_INTENSITY),
                )
            ),
            peak_ordering=str(_config_get(config, "peak_ordering", "mz")),
            precursor_peak_exclusion_window_da=float(
                _config_get(config, "precursor_peak_exclusion_window_da", 0.0)
            ),
            peak_filtering=str(
                _config_get(config, "peak_filtering", DEFAULT_PEAK_FILTERING)
            ),
            grouped_peak_shoulder_da=float(
                _config_get(
                    config,
                    "grouped_peak_shoulder_da",
                    DEFAULT_GROUPED_PEAK_SHOULDER_DA,
                )
            ),
            grouped_peak_isotope_charges=tuple(
                int(charge)
                for charge in _config_get(
                    config,
                    "grouped_peak_isotope_charges",
                    DEFAULT_GROUPED_PEAK_ISOTOPE_CHARGES,
                )
            ),
        ),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(
            _config_get(config, "dataloader_persistent_workers", False)
        )
        loader_kwargs["prefetch_factor"] = int(
            _config_get(config, "dataloader_prefetch_factor", 2)
        )
    return DataLoader(
        **loader_kwargs,
    )


def build_contrastive_online_loader(
    config: config_dict.ConfigDict,
    split: ContrastiveSplit,
    *,
    split_name: str,
    global_batch_size: int,
    seed: int,
    distributed: DistributedContext,
) -> DataLoader:
    per_rank_batch_size = local_batch_size(
        global_batch_size,
        distributed.world_size,
    )
    dataset = ContrastiveOnlineDataset(split)
    sample_weight_power = float(
        _config_get(config, "contrastive_online_sample_weight_power", 0.0)
    )
    if split_name == "train" and sample_weight_power > 0:
        sampler = WeightedOnlineSampler(
            _online_sample_weights(split, sample_weight_power),
            num_replicas=distributed.world_size,
            rank=distributed.rank,
            seed=seed,
            drop_last=True,
        )
    else:
        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=distributed.world_size,
                rank=distributed.rank,
                shuffle=(split_name == "train"),
                seed=seed,
                drop_last=(split_name == "train"),
            )
            if distributed.is_distributed
            else None
        )
    num_workers = int(_config_get(config, "dataloader_num_workers", 0))
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": per_rank_batch_size,
        "shuffle": (split_name == "train" and sampler is None),
        "sampler": sampler,
        "drop_last": (split_name == "train"),
        "num_workers": num_workers,
        "pin_memory": bool(_config_get(config, "dataloader_pin_memory", False)),
        "collate_fn": ContrastiveOnlineBatchCollator(
            split,
            num_peaks=int(_config_get(config, "num_peaks", 60)),
            max_precursor_mz=float(
                _config_get(config, "max_precursor_mz", DEFAULT_MAX_PRECURSOR_MZ)
            ),
            min_peak_intensity=float(
                _config_get(config, "min_peak_intensity", DEFAULT_MIN_PEAK_INTENSITY)
            ),
            peak_drop_min_intensity=float(
                _config_get(
                    config,
                    "peak_drop_min_intensity",
                    _config_get(config, "min_peak_intensity", DEFAULT_MIN_PEAK_INTENSITY),
                )
            ),
            peak_ordering=str(_config_get(config, "peak_ordering", "mz")),
            precursor_peak_exclusion_window_da=float(
                _config_get(config, "precursor_peak_exclusion_window_da", 0.0)
            ),
            peak_filtering=str(
                _config_get(config, "peak_filtering", DEFAULT_PEAK_FILTERING)
            ),
            grouped_peak_shoulder_da=float(
                _config_get(
                    config,
                    "grouped_peak_shoulder_da",
                    DEFAULT_GROUPED_PEAK_SHOULDER_DA,
                )
            ),
            grouped_peak_isotope_charges=tuple(
                int(charge)
                for charge in _config_get(
                    config,
                    "grouped_peak_isotope_charges",
                    DEFAULT_GROUPED_PEAK_ISOTOPE_CHARGES,
                )
            ),
        ),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(
            _config_get(config, "dataloader_persistent_workers", False)
        )
        loader_kwargs["prefetch_factor"] = int(
            _config_get(config, "dataloader_prefetch_factor", 2)
        )
    return DataLoader(**loader_kwargs)


class OnlineProbeHead(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )
        torch.nn.init.xavier_uniform_(cast(torch.nn.Linear, self.net[0]).weight)
        torch.nn.init.zeros_(cast(torch.nn.Linear, self.net[0]).bias)
        torch.nn.init.normal_(cast(torch.nn.Linear, self.net[2]).weight, std=1e-3)
        torch.nn.init.zeros_(cast(torch.nn.Linear, self.net[2]).bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.net(features))


class ContrastiveTrainingModule(torch.nn.Module):
    def __init__(
        self,
        *,
        model: PeakSetJEPA,
        pooler: SinglePairCovariancePool,
        online_probe: OnlineProbeHead,
        teacher_model: PeakSetJEPA | None,
        temperature: float,
        loss_type: str,
        triplet_margin: float,
        triplet_negative_max_maccs_tanimoto: float | None,
        triplet_hard_fraction: float | None,
        info_nce_negatives: int | None,
        fingerprint_target_temperature: float,
        contrastive_loss_weight: float,
        online_probe_loss_weight: float,
        encoder_anchor_loss_weight: float,
        online_maccs_loss_type: str = "bce",
        online_maccs_loss_weight: float = 1.0,
        online_auc_loss_weight: float = 0.5,
        online_auc_hard_fraction: float | None = None,
        maccs_pos_weight: torch.Tensor | None = None,
        info_nce_negative_max_maccs_tanimoto: float | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.pooler = pooler
        self.online_probe = online_probe
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.loss_type = loss_type
        self.triplet_margin = triplet_margin
        self.triplet_negative_max_maccs_tanimoto = triplet_negative_max_maccs_tanimoto
        self.triplet_hard_fraction = triplet_hard_fraction
        self.info_nce_negatives = info_nce_negatives
        self.info_nce_negative_max_maccs_tanimoto = (
            info_nce_negative_max_maccs_tanimoto
        )
        self.fingerprint_target_temperature = fingerprint_target_temperature
        self.contrastive_loss_weight = contrastive_loss_weight
        self.online_probe_loss_weight = online_probe_loss_weight
        self.encoder_anchor_loss_weight = encoder_anchor_loss_weight
        self.online_maccs_loss_type = online_maccs_loss_type
        self.online_maccs_loss_weight = online_maccs_loss_weight
        self.online_auc_loss_weight = online_auc_loss_weight
        self.online_auc_hard_fraction = online_auc_hard_fraction
        self.register_buffer(
            "maccs_pos_weight",
            (
                torch.ones(MACCS_FINGERPRINT_BITS, dtype=torch.float32)
                if maccs_pos_weight is None
                else maccs_pos_weight.to(dtype=torch.float32)
            ),
        )
        if self.teacher_model is not None:
            self.teacher_model.eval()
            for parameter in self.teacher_model.parameters():
                parameter.requires_grad_(False)

    def _pooled_features(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        peak_embeddings, _, pair_embeddings = self.model.encoder.forward_with_block_outputs(
            batch["peak_mz"],
            batch["peak_intensity"],
            valid_mask=batch["peak_valid_mask"],
            precursor_mz=batch.get("precursor_mz", None),
        )
        return self.pooler(
            peak_embeddings.float(),
            batch["peak_valid_mask"].to(dtype=torch.bool),
            pair_embeddings.float(),
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        online_batch: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        pooled = self._pooled_features(batch)
        contrastive_features = F.normalize(pooled, dim=-1)
        contrastive_batch = gather_contrastive_batch(
            contrastive_features,
            positive_index=batch["positive_index"],
            compound_id=batch["compound_id"],
            maccs=batch["probe_maccs"],
        )
        if self.loss_type == "dreams_triplet":
            if "triplet_anchor_index" in batch:
                gather_offset = contrastive_batch.anchor_index[0]
                contrastive_loss, contrastive_accuracy = explicit_dreams_triplet_loss(
                    contrastive_batch.candidate_features,
                    anchor_index=batch["triplet_anchor_index"] + gather_offset,
                    positive_index=batch["triplet_positive_index"] + gather_offset,
                    negative_index=batch["triplet_negative_index"] + gather_offset,
                    margin=self.triplet_margin,
                    hard_fraction=self.triplet_hard_fraction,
                    candidate_maccs=contrastive_batch.candidate_maccs,
                    negative_max_maccs_tanimoto=(
                        self.triplet_negative_max_maccs_tanimoto
                    ),
                )
            else:
                contrastive_loss, contrastive_accuracy = dreams_triplet_loss(
                    contrastive_batch.anchor_features,
                    positive_index=contrastive_batch.positive_index,
                    compound_id=contrastive_batch.anchor_compound_id,
                    margin=self.triplet_margin,
                    hard_fraction=self.triplet_hard_fraction,
                    candidate_features=contrastive_batch.candidate_features,
                    candidate_compound_id=contrastive_batch.candidate_compound_id,
                    anchor_maccs=contrastive_batch.anchor_maccs,
                    candidate_maccs=contrastive_batch.candidate_maccs,
                    negative_max_maccs_tanimoto=(
                        self.triplet_negative_max_maccs_tanimoto
                    ),
                )
        elif self.loss_type in ("maccs_soft", "fingerprint_soft"):
            contrastive_loss, contrastive_accuracy = maccs_soft_contrastive_loss(
                contrastive_batch.anchor_features,
                candidate_features=contrastive_batch.candidate_features,
                anchor_index=contrastive_batch.anchor_index,
                anchor_maccs=contrastive_batch.anchor_maccs,
                candidate_maccs=contrastive_batch.candidate_maccs,
                temperature=self.temperature,
                target_temperature=self.fingerprint_target_temperature,
            )
        elif self.loss_type in ("maccs_similarity", "fingerprint_similarity"):
            contrastive_loss, contrastive_accuracy = maccs_similarity_loss(
                contrastive_batch.anchor_features,
                candidate_features=contrastive_batch.candidate_features,
                anchor_index=contrastive_batch.anchor_index,
                anchor_maccs=contrastive_batch.anchor_maccs,
                candidate_maccs=contrastive_batch.candidate_maccs,
            )
        elif self.loss_type in ("maccs_centroid_auc", "fingerprint_centroid_auc"):
            contrastive_loss, contrastive_accuracy = maccs_centroid_auc_loss(
                contrastive_batch.anchor_features,
                candidate_features=contrastive_batch.candidate_features,
                anchor_maccs=contrastive_batch.anchor_maccs,
                candidate_maccs=contrastive_batch.candidate_maccs,
                temperature=self.temperature,
            )
        else:
            contrastive_loss, contrastive_accuracy = info_nce_loss(
                contrastive_batch.anchor_features,
                positive_index=contrastive_batch.positive_index,
                compound_id=contrastive_batch.anchor_compound_id,
                temperature=self.temperature,
                candidate_features=contrastive_batch.candidate_features,
                candidate_compound_id=contrastive_batch.candidate_compound_id,
                anchor_index=contrastive_batch.anchor_index,
                max_negatives=self.info_nce_negatives,
                anchor_maccs=contrastive_batch.anchor_maccs,
                candidate_maccs=contrastive_batch.candidate_maccs,
                negative_max_maccs_tanimoto=(
                    self.info_nce_negative_max_maccs_tanimoto
                ),
            )
        probe_batch = batch if online_batch is None else online_batch
        probe_pooled = pooled if online_batch is None else self._pooled_features(online_batch)
        probe_loss, probe_maccs_bce, probe_bit_accuracy = (
            online_probe_loss(
                self.online_probe(probe_pooled),
                probe_batch,
                maccs_loss_type=self.online_maccs_loss_type,
                maccs_pos_weight=self.maccs_pos_weight,
                maccs_loss_weight=self.online_maccs_loss_weight,
                auc_loss_weight=self.online_auc_loss_weight,
                auc_hard_fraction=self.online_auc_hard_fraction,
            )
        )
        anchor_loss = pooled.new_zeros(())
        if self.teacher_model is not None and self.encoder_anchor_loss_weight > 0:
            with torch.no_grad():
                teacher_peak_embeddings, _, teacher_pair_embeddings = (
                    self.teacher_model.encoder.forward_with_block_outputs(
                        batch["peak_mz"],
                        batch["peak_intensity"],
                        valid_mask=batch["peak_valid_mask"],
                        precursor_mz=batch.get("precursor_mz", None),
                    )
                )
                teacher_pooled = self.pooler(
                    teacher_peak_embeddings.float(),
                    batch["peak_valid_mask"].to(dtype=torch.bool),
                    teacher_pair_embeddings.float(),
                )
            anchor_loss = (
                1.0
                - F.cosine_similarity(
                    pooled.float(),
                    teacher_pooled.float(),
                    dim=-1,
                )
            ).mean()
        loss = (
            contrastive_loss * pooled.new_tensor(self.contrastive_loss_weight)
            + probe_loss * pooled.new_tensor(self.online_probe_loss_weight)
            + anchor_loss * pooled.new_tensor(self.encoder_anchor_loss_weight)
        )
        return {
            "loss": loss,
            "contrastive_loss": contrastive_loss,
            "contrastive_accuracy": contrastive_accuracy,
            "online_probe_loss": probe_loss,
            "online_probe_maccs_bce": probe_maccs_bce,
            "online_probe_maccs_bit_accuracy": probe_bit_accuracy,
            "encoder_anchor_loss": anchor_loss,
        }


def info_nce_loss(
    features: torch.Tensor,
    *,
    positive_index: torch.Tensor,
    compound_id: torch.Tensor,
    temperature: float,
    candidate_features: torch.Tensor | None = None,
    candidate_compound_id: torch.Tensor | None = None,
    anchor_index: torch.Tensor | None = None,
    max_negatives: int | None = None,
    anchor_maccs: torch.Tensor | None = None,
    candidate_maccs: torch.Tensor | None = None,
    negative_max_maccs_tanimoto: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if candidate_features is None:
        candidate_features = features
    if candidate_compound_id is None:
        candidate_compound_id = compound_id
    if anchor_index is None:
        anchor_index = torch.arange(features.shape[0], device=features.device)
    logits = features.float() @ candidate_features.float().T
    logits = logits / temperature
    candidate_positions = torch.arange(
        candidate_features.shape[0],
        device=features.device,
    )
    self_mask = candidate_positions[None, :] == anchor_index[:, None]
    positives = candidate_positions[None, :] == positive_index[:, None]
    same_compound = compound_id[:, None] == candidate_compound_id[None, :]
    allowed = (~self_mask) & ((~same_compound) | positives)
    if negative_max_maccs_tanimoto is not None:
        assert anchor_maccs is not None
        assert candidate_maccs is not None
        negative_mask = allowed & (~positives)
        negative_mask = negative_mask & (
            maccs_tanimoto(anchor_maccs, candidate_maccs)
            <= negative_max_maccs_tanimoto
        )
        allowed = positives | negative_mask
    if max_negatives is not None:
        negative_mask = allowed & (~positives)
        if max_negatives > 0:
            random_scores = torch.rand(
                negative_mask.shape,
                device=features.device,
                dtype=features.float().dtype,
            ).masked_fill(~negative_mask, -torch.finfo(features.float().dtype).max)
            sampled_indices = random_scores.topk(
                min(max_negatives, candidate_features.shape[0]),
                dim=1,
            ).indices
            sampled_negatives = torch.zeros_like(negative_mask).scatter(
                1,
                sampled_indices,
                True,
            )
            allowed = positives | (sampled_negatives & negative_mask)
        else:
            allowed = positives
    logits = logits.masked_fill(~allowed, -torch.finfo(logits.dtype).max)
    loss = F.cross_entropy(logits, positive_index)
    accuracy = (logits.argmax(dim=1) == positive_index).float().mean()
    return loss, accuracy


def gather_contrastive_batch(
    features: torch.Tensor,
    *,
    positive_index: torch.Tensor,
    compound_id: torch.Tensor,
    maccs: torch.Tensor,
) -> GatheredContrastiveBatch:
    local_size = features.shape[0]
    if not (dist.is_available() and dist.is_initialized()):
        anchor_index = torch.arange(local_size, device=features.device)
        return GatheredContrastiveBatch(
            anchor_features=features,
            candidate_features=features,
            positive_index=positive_index,
            anchor_index=anchor_index,
            anchor_compound_id=compound_id,
            candidate_compound_id=compound_id,
            anchor_maccs=maccs,
            candidate_maccs=maccs,
        )
    size = torch.tensor([local_size], device=features.device, dtype=torch.long)
    sizes = [torch.empty_like(size) for _ in range(dist.get_world_size())]
    dist.all_gather(sizes, size)
    sizes_tensor = torch.cat(sizes)
    offset = sizes_tensor[: dist.get_rank()].sum()
    anchor_index = torch.arange(local_size, device=features.device) + offset
    return GatheredContrastiveBatch(
        anchor_features=features,
        candidate_features=torch.cat(dist_nn.all_gather(features), dim=0),
        positive_index=positive_index + offset,
        anchor_index=anchor_index,
        anchor_compound_id=compound_id,
        candidate_compound_id=_all_gather_no_grad(compound_id),
        anchor_maccs=maccs,
        candidate_maccs=_all_gather_no_grad(maccs),
    )


def _all_gather_no_grad(tensor: torch.Tensor) -> torch.Tensor:
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor.contiguous())
    return torch.cat(gathered, dim=0)


def maccs_soft_contrastive_loss(
    features: torch.Tensor,
    *,
    candidate_features: torch.Tensor,
    anchor_index: torch.Tensor,
    anchor_maccs: torch.Tensor,
    candidate_maccs: torch.Tensor,
    temperature: float,
    target_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = features.float() @ candidate_features.float().T
    logits = logits / temperature
    candidate_positions = torch.arange(
        candidate_features.shape[0],
        device=features.device,
    )
    self_mask = candidate_positions[None, :] == anchor_index[:, None]
    logits = logits.masked_fill(self_mask, -torch.finfo(logits.dtype).max)
    target_similarity = maccs_tanimoto(anchor_maccs, candidate_maccs)
    target_logits = target_similarity / target_temperature
    target_logits = target_logits.masked_fill(self_mask, -torch.finfo(target_logits.dtype).max)
    target = F.softmax(target_logits, dim=1)
    loss = -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    accuracy = (logits.argmax(dim=1) == target_similarity.masked_fill(self_mask, -1).argmax(dim=1)).float().mean()
    return loss, accuracy


def maccs_tanimoto(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    left = left.to(dtype=torch.float32)
    right = right.to(dtype=torch.float32)
    intersection = left @ right.T
    union = left.sum(dim=1, keepdim=True) + right.sum(dim=1)[None, :] - intersection
    return intersection / union.clamp_min(1.0)


def maccs_similarity_loss(
    features: torch.Tensor,
    *,
    candidate_features: torch.Tensor,
    anchor_index: torch.Tensor,
    anchor_maccs: torch.Tensor,
    candidate_maccs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cosine_similarity = features.float() @ candidate_features.float().T
    target_similarity = maccs_tanimoto(anchor_maccs, candidate_maccs)
    candidate_positions = torch.arange(
        candidate_features.shape[0],
        device=features.device,
    )
    pair_mask = candidate_positions[None, :] != anchor_index[:, None]
    loss = F.mse_loss(
        cosine_similarity[pair_mask],
        target_similarity[pair_mask],
    )
    closest_target = target_similarity.masked_fill(~pair_mask, -1).argmax(dim=1)
    accuracy = (
        cosine_similarity.masked_fill(~pair_mask, -torch.finfo(cosine_similarity.dtype).max)
        .argmax(dim=1)
        .eq(closest_target)
        .float()
        .mean()
    )
    return loss, accuracy


def maccs_centroid_auc_loss(
    features: torch.Tensor,
    *,
    candidate_features: torch.Tensor,
    anchor_maccs: torch.Tensor,
    candidate_maccs: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    candidate_target = candidate_maccs.to(dtype=torch.float32)
    positive_counts = candidate_target.sum(dim=0)
    valid_bits = (positive_counts > 0) & (
        positive_counts < candidate_target.shape[0]
    )
    if not bool(valid_bits.any()):
        zero = features.sum() * 0.0
        return zero, zero

    candidate_target = candidate_target[:, valid_bits]
    anchor_target = anchor_maccs.to(dtype=torch.float32)[:, valid_bits]
    positive_counts = positive_counts[valid_bits]
    negative_counts = candidate_target.shape[0] - positive_counts

    candidate = candidate_features.float()
    positive_centroids = candidate_target.T @ candidate / positive_counts[:, None]
    negative_centroids = (1.0 - candidate_target).T @ candidate / negative_counts[:, None]
    bit_directions = F.normalize(positive_centroids, dim=-1) - F.normalize(
        negative_centroids,
        dim=-1,
    )
    logits = features.float() @ bit_directions.T / temperature
    bce = F.binary_cross_entropy_with_logits(logits, anchor_target)
    gathered_logits, gathered_target = gather_online_probe_logits(logits, anchor_target)
    auc_loss = pairwise_maccs_auc_loss(gathered_logits, gathered_target)
    loss = 0.5 * bce + 0.5 * auc_loss
    accuracy = ((logits > 0) == (anchor_target > 0.5)).float().mean()
    return loss, accuracy


def dreams_triplet_loss(
    features: torch.Tensor,
    *,
    positive_index: torch.Tensor,
    compound_id: torch.Tensor,
    margin: float,
    hard_fraction: float | None = None,
    candidate_features: torch.Tensor | None = None,
    candidate_compound_id: torch.Tensor | None = None,
    anchor_maccs: torch.Tensor | None = None,
    candidate_maccs: torch.Tensor | None = None,
    negative_max_maccs_tanimoto: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if candidate_features is None:
        candidate_features = features
    if candidate_compound_id is None:
        candidate_compound_id = compound_id
    cosine_similarity = features.float() @ candidate_features.float().T
    positive_similarity = cosine_similarity.gather(1, positive_index[:, None])
    negative_mask = compound_id[:, None] != candidate_compound_id[None, :]
    triplet_loss = torch.clamp_min(
        margin - positive_similarity + cosine_similarity,
        0,
    )
    if negative_max_maccs_tanimoto is not None:
        assert anchor_maccs is not None
        assert candidate_maccs is not None
        negative_mask = negative_mask & (
            maccs_tanimoto(anchor_maccs, candidate_maccs)
            <= negative_max_maccs_tanimoto
        )
    valid_count = negative_mask.sum().clamp_min(1)
    active_loss = triplet_loss[negative_mask]
    if active_loss.numel() == 0:
        loss = active_loss.sum()
    elif hard_fraction is not None and 0.0 < hard_fraction < 1.0:
        k = max(1, int(math.ceil(active_loss.numel() * hard_fraction)))
        loss = active_loss.topk(k).values.mean()
    else:
        loss = active_loss.mean()
    accuracy = (
        (positive_similarity > cosine_similarity)[negative_mask].float().sum()
        / valid_count
    )
    return loss, accuracy


def explicit_dreams_triplet_loss(
    features: torch.Tensor,
    *,
    anchor_index: torch.Tensor,
    positive_index: torch.Tensor,
    negative_index: torch.Tensor,
    margin: float,
    hard_fraction: float | None = None,
    candidate_maccs: torch.Tensor | None = None,
    negative_max_maccs_tanimoto: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    cosine_similarity = features.float() @ features.float().T
    positive_similarity = cosine_similarity[anchor_index, positive_index]
    negative_similarity = cosine_similarity[anchor_index, negative_index]
    per_triplet_loss = torch.clamp_min(
        margin - positive_similarity + negative_similarity,
        0,
    )
    valid = torch.ones_like(per_triplet_loss, dtype=torch.bool)
    if negative_max_maccs_tanimoto is not None:
        assert candidate_maccs is not None
        intersection = (
            candidate_maccs[anchor_index].to(dtype=torch.float32)
            * candidate_maccs[negative_index].to(dtype=torch.float32)
        ).sum(dim=1)
        union = (
            candidate_maccs[anchor_index].to(dtype=torch.float32).sum(dim=1)
            + candidate_maccs[negative_index].to(dtype=torch.float32).sum(dim=1)
            - intersection
        )
        valid = intersection / union.clamp_min(1.0) <= negative_max_maccs_tanimoto
    valid_count = valid.sum().clamp_min(1)
    active_loss = per_triplet_loss[valid]
    if active_loss.numel() == 0:
        loss = active_loss.sum()
    elif hard_fraction is not None and 0.0 < hard_fraction < 1.0:
        k = max(1, int(math.ceil(active_loss.numel() * hard_fraction)))
        loss = active_loss.topk(k).values.mean()
    else:
        loss = active_loss.mean()
    accuracy = (positive_similarity > negative_similarity)[valid].float().sum() / valid_count
    return loss, accuracy


def online_probe_loss(
    logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    maccs_loss_type: str = "bce",
    maccs_pos_weight: torch.Tensor | None = None,
    maccs_loss_weight: float = 1.0,
    auc_loss_weight: float = 0.5,
    auc_hard_fraction: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    maccs_logits = logits
    maccs_target = batch["probe_maccs"].to(dtype=torch.float32)
    maccs_bce = F.binary_cross_entropy_with_logits(maccs_logits, maccs_target)
    if maccs_loss_type == "bce":
        maccs_loss = maccs_bce
    elif maccs_loss_type == "balanced_bce":
        assert maccs_pos_weight is not None
        maccs_loss = F.binary_cross_entropy_with_logits(
            maccs_logits,
            maccs_target,
            pos_weight=maccs_pos_weight,
        )
    elif maccs_loss_type in ("pairwise_auc", "auc"):
        gathered_logits, gathered_target = gather_online_probe_logits(
            maccs_logits,
            maccs_target,
        )
        maccs_loss = pairwise_maccs_auc_loss(
            gathered_logits,
            gathered_target,
            hard_fraction=auc_hard_fraction,
        )
    elif maccs_loss_type in ("pairwise_auc_bce", "auc_bce"):
        gathered_logits, gathered_target = gather_online_probe_logits(
            maccs_logits,
            maccs_target,
        )
        auc_loss = pairwise_maccs_auc_loss(
            gathered_logits,
            gathered_target,
            hard_fraction=auc_hard_fraction,
        )
        maccs_loss = (1.0 - auc_loss_weight) * maccs_bce + auc_loss_weight * auc_loss
    elif maccs_loss_type in ("balanced_auc_bce", "auc_balanced_bce"):
        assert maccs_pos_weight is not None
        balanced_bce = F.binary_cross_entropy_with_logits(
            maccs_logits,
            maccs_target,
            pos_weight=maccs_pos_weight,
        )
        gathered_logits, gathered_target = gather_online_probe_logits(
            maccs_logits,
            maccs_target,
        )
        auc_loss = pairwise_maccs_auc_loss(
            gathered_logits,
            gathered_target,
            hard_fraction=auc_hard_fraction,
        )
        maccs_loss = (1.0 - auc_loss_weight) * balanced_bce + auc_loss_weight * auc_loss
    else:
        raise ValueError(f"Unsupported online MACCS loss: {maccs_loss_type!r}")
    bit_accuracy = ((maccs_logits > 0) == (maccs_target > 0.5)).float().mean()
    probe_loss = maccs_loss * maccs_loss.new_tensor(maccs_loss_weight)
    return probe_loss, maccs_bce, bit_accuracy


def gather_online_probe_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not (dist.is_available() and dist.is_initialized()):
        return logits, target
    return torch.cat(dist_nn.all_gather(logits), dim=0), _all_gather_no_grad(target)


def pairwise_maccs_auc_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    hard_fraction: float | None = None,
) -> torch.Tensor:
    target = target > 0.5
    losses = []
    for bit_idx in range(target.shape[1]):
        bit_target = target[:, bit_idx]
        if bit_target.any() and (~bit_target).any():
            positive_logits = logits[bit_target, bit_idx]
            negative_logits = logits[~bit_target, bit_idx]
            pair_losses = F.softplus(
                -(positive_logits[:, None] - negative_logits[None, :])
            ).flatten()
            if hard_fraction is not None and 0.0 < hard_fraction < 1.0:
                k = max(1, int(math.ceil(pair_losses.numel() * hard_fraction)))
                pair_losses = pair_losses.topk(k).values
            losses.append(pair_losses.mean())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def build_contrastive_module(
    config: config_dict.ConfigDict,
    *,
    maccs_pos_weight: torch.Tensor | None = None,
) -> ContrastiveTrainingModule:
    model = build_model_from_config(config)
    compressed_dim = int(_config_get(config, "contrastive_covariance_dim", _config_get(config, "covariance_pooling_dim", 32)))
    pooler = SinglePairCovariancePool(
        single_dim=int(config.model_dim),
        pair_dim=int(_config_get(config, "pairmixer_pair_dim", config.model_dim)),
        compressed_dim=compressed_dim,
        include_diagonal=bool(
            _config_get(config, "contrastive_single_pair_include_diagonal", False)
        ),
    )
    pooled_dim = compressed_dim * compressed_dim
    online_probe = OnlineProbeHead(
        input_dim=pooled_dim,
        hidden_dim=int(_config_get(config, "contrastive_online_probe_hidden_dim", config.model_dim)),
        output_dim=MACCS_FINGERPRINT_BITS,
    )
    teacher_model = (
        build_model_from_config(config)
        if float(_config_get(config, "contrastive_encoder_anchor_loss_weight", 0.0)) > 0
        else None
    )
    return ContrastiveTrainingModule(
        model=model,
        pooler=pooler,
        online_probe=online_probe,
        teacher_model=teacher_model,
        temperature=float(_config_get(config, "contrastive_temperature", 0.1)),
        loss_type=str(_config_get(config, "contrastive_loss_type", "info_nce")),
        triplet_margin=float(_config_get(config, "contrastive_triplet_margin", 0.2)),
        triplet_negative_max_maccs_tanimoto=_optional_float(
            _config_get(config, "contrastive_triplet_negative_max_maccs_tanimoto", None)
        ),
        triplet_hard_fraction=_optional_float(
            _config_get(config, "contrastive_triplet_hard_fraction", None)
        ),
        info_nce_negatives=_optional_int(
            _config_get(config, "contrastive_info_nce_negatives", None)
        ),
        info_nce_negative_max_maccs_tanimoto=_optional_float(
            _config_get(
                config,
                "contrastive_info_nce_negative_max_maccs_tanimoto",
                None,
            )
        ),
        fingerprint_target_temperature=float(
            _config_get(config, "contrastive_fingerprint_target_temperature", 0.1)
        ),
        contrastive_loss_weight=float(_config_get(config, "contrastive_loss_weight", 1.0)),
        online_probe_loss_weight=float(_config_get(config, "online_probe_loss_weight", 1.0)),
        encoder_anchor_loss_weight=float(
            _config_get(config, "contrastive_encoder_anchor_loss_weight", 0.0)
        ),
        online_maccs_loss_type=str(
            _config_get(config, "contrastive_online_maccs_loss_type", "bce")
        ),
        online_maccs_loss_weight=float(
            _config_get(config, "contrastive_online_maccs_loss_weight", 1.0)
        ),
        online_auc_loss_weight=float(
            _config_get(config, "contrastive_online_auc_loss_weight", 0.5)
        ),
        online_auc_hard_fraction=_optional_float(
            _config_get(config, "contrastive_online_auc_hard_fraction", None)
        ),
        maccs_pos_weight=maccs_pos_weight,
    )


def build_contrastive_optimizers(
    config: config_dict.ConfigDict,
    module: ContrastiveTrainingModule,
    total_steps: int,
    device: torch.device,
) -> tuple[list[torch.optim.Optimizer], list[LRSchedulerLike]]:
    model_lr = _optional_float(
        _config_get(config, "contrastive_model_learning_rate", None)
    )
    pooler_lr = _optional_float(
        _config_get(config, "contrastive_pooler_learning_rate", None)
    )
    online_probe_lr = _optional_float(
        _config_get(config, "contrastive_online_probe_learning_rate", None)
    )
    if model_lr is None and pooler_lr is None and online_probe_lr is None:
        return build_optimizers(config, module, total_steps, device)

    base_lr = float(config.learning_rate)
    fused_cfg = _config_get(config, "optimizer_fused", None)
    fused = (
        device.type == "cuda"
        if fused_cfg is None
        else bool(fused_cfg) and device.type == "cuda"
    )
    param_groups = []
    for submodule, lr in (
        (module.model, base_lr if model_lr is None else model_lr),
        (module.pooler, base_lr if pooler_lr is None else pooler_lr),
        (module.online_probe, base_lr if online_probe_lr is None else online_probe_lr),
    ):
        decay_params = []
        no_decay_params = []
        for name, param in submodule.named_parameters():
            if param.requires_grad and is_weight_decay_target(name, param):
                decay_params.append(param)
            elif param.requires_grad:
                no_decay_params.append(param)
        if no_decay_params:
            param_groups.append(
                {"params": no_decay_params, "weight_decay": 0.0, "lr": lr}
            )
        if decay_params:
            param_groups.append(
                {
                    "params": decay_params,
                    "weight_decay": float(config.weight_decay),
                    "lr": lr,
                }
            )

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=base_lr,
        betas=(0.9, float(_config_get(config, "b2", 0.999))),
        fused=fused,
    )
    scheduler = make_cosine_schedule(
        optimizer,
        total_steps,
        int(_config_get(config, "warmup_steps", 0)),
        _config_get(config, "min_learning_rate", None),
    )
    return [optimizer], [scheduler]


def train_contrastive(
    config: config_dict.ConfigDict,
    workdir: str | Path,
) -> dict[str, object]:
    distributed = init_distributed_from_env()
    workdir = normalize_storage_path(workdir)
    local_workdir = local_scratch_dir(workdir)
    if distributed.is_main:
        local_workdir.mkdir(parents=True, exist_ok=True)
        storage_mkdir(workdir)
    barrier(distributed)
    torch.manual_seed(int(config.seed))
    np.random.seed(int(config.seed))
    config.training_mode = "contrastive"

    probe_data = MassSpecProbeData.from_config(
        config,
        distributed_world_size=distributed.world_size,
        distributed_rank=distributed.rank,
    )
    train_split = _load_contrastive_split(
        probe_data.train_files,
        max_samples=_optional_int(
            _config_get(config, "contrastive_max_train_samples", None)
        ),
    )
    val_split = _load_contrastive_split(
        probe_data.val_files,
        max_samples=_optional_int(
            _config_get(config, "contrastive_max_val_samples", None)
        ),
    )
    maccs_pos_weight = _maccs_pos_weight(train_split)

    global_spectra_batch_size = int(_config_get(config, "contrastive_batch_size", config.batch_size))
    spectra_per_pair = (
        3
        if _config_get(config, "contrastive_triplet_negative_mass_tolerance_da", None)
        is not None
        else 2
    )
    global_pairs_per_batch = max(1, global_spectra_batch_size // spectra_per_pair)
    if distributed.world_size > 1:
        global_pairs_per_batch -= global_pairs_per_batch % distributed.world_size
        global_pairs_per_batch = max(distributed.world_size, global_pairs_per_batch)
    train_pairs_per_epoch = int(
        _config_get(
            config,
            "contrastive_pairs_per_epoch",
            len(train_split.smiles),
        )
    )
    val_pairs_per_epoch = int(
        _config_get(
            config,
            "contrastive_val_pairs_per_epoch",
            min(len(val_split.smiles), 8192),
        )
    )
    train_loader = build_contrastive_loader(
        config,
        train_split,
        split_name="train",
        pairs_per_epoch=train_pairs_per_epoch,
        global_pairs_per_batch=global_pairs_per_batch,
        seed=int(config.seed),
        distributed=distributed,
    )
    val_loader = build_contrastive_loader(
        config,
        val_split,
        split_name="val",
        pairs_per_epoch=val_pairs_per_epoch,
        global_pairs_per_batch=global_pairs_per_batch,
        seed=int(config.seed) + 10_000,
        distributed=distributed,
    )
    online_train_loader = (
        build_contrastive_online_loader(
            config,
            train_split,
            split_name="train",
            global_batch_size=int(
                _config_get(config, "contrastive_online_batch_size", global_spectra_batch_size)
            ),
            seed=int(config.seed) + 20_000,
            distributed=distributed,
        )
        if bool(_config_get(config, "contrastive_online_full_train", False))
        else None
    )
    total_steps = total_contrastive_steps(config, train_loader)
    module = build_contrastive_module(
        config,
        maccs_pos_weight=maccs_pos_weight,
    )
    init_checkpoint = _config_get(config, "contrastive_init_checkpoint_path", "")
    if init_checkpoint:
        load_pretrained_weights(
            module.model,
            normalize_storage_path(init_checkpoint),
        )
        if module.teacher_model is not None:
            load_pretrained_weights(
                module.teacher_model,
                normalize_storage_path(init_checkpoint),
            )
    full_init_checkpoint = _config_get(config, "contrastive_init_full_checkpoint_path", "")
    if full_init_checkpoint:
        full_init_path = normalize_storage_path(full_init_checkpoint)
        full_init = load_torch_checkpoint(
            full_init_path,
            map_location="cpu",
            weights_only=True,
        )
        load_resume_model_state(module.model, full_init["model"])
        load_resume_covariance_pooler_state(module.pooler, full_init_path, full_init)
        module.online_probe.load_state_dict(full_init["online_probe"])
        if module.teacher_model is not None:
            load_resume_model_state(module.teacher_model, full_init["model"])
    module.to(distributed.device).train()
    if module.teacher_model is not None:
        module.teacher_model.eval()
    param_metrics = collect_and_log_param_metrics(module) if distributed.is_main else {}
    flops_per_optimizer_step = estimate_training_flops_per_optimizer_step(
        config,
        module,
        global_spectra_batch_size,
    )
    if distributed.is_main:
        param_metrics["model/flops_per_optimizer_step_estimate"] = (
            flops_per_optimizer_step
        )
        param_metrics["model/flops_per_sample_estimate"] = (
            flops_per_optimizer_step / float(global_spectra_batch_size)
        )
    autocast_dtype = parse_autocast_dtype(_config_get(config, "autocast_dtype", "bf16"))
    grad_scaler = build_grad_scaler(autocast_dtype, distributed.device)
    optimizers, schedulers = build_contrastive_optimizers(
        config,
        module,
        total_steps,
        distributed.device,
    )
    checkpoint_dir = storage_join(workdir, "checkpoints")
    if distributed.is_main:
        storage_mkdir(checkpoint_dir)
    logger = build_logger(config, local_workdir) if distributed.is_main else None
    start_epoch, global_step, resume_offset = restore_contrastive_state(
        checkpoint_dir=checkpoint_dir,
        module=module,
        optimizers=optimizers,
        schedulers=schedulers,
        grad_scaler=grad_scaler,
        steps_per_epoch=len(train_loader),
        device=distributed.device,
    )
    if distributed.is_main and logger is not None:
        logger.log_metrics(param_metrics, step=global_step)
    compile_mode = str(_config_get(config, "contrastive_compile_mode", "none"))
    if compile_mode.lower() != "none":
        module.compile(mode=compile_mode, fullgraph=False)
    train_model = wrap_distributed_model(
        module,
        distributed,
        static_graph=bool(_config_get(config, "ddp_static_graph", True)),
        find_unused_parameters=bool(_config_get(config, "ddp_find_unused_parameters", False)),
    )
    results = run_contrastive_loop(
        config=config,
        model=train_model,
        train_loader=train_loader,
        val_loader=val_loader,
        online_train_loader=online_train_loader,
        optimizers=optimizers,
        schedulers=schedulers,
        grad_scaler=grad_scaler,
        autocast_dtype=autocast_dtype,
        logger=logger,
        checkpoint_dir=checkpoint_dir,
        start_epoch=start_epoch,
        resume_offset=resume_offset,
        global_step=global_step,
        total_steps=total_steps,
        distributed=distributed,
        flops_per_optimizer_step=flops_per_optimizer_step,
    )
    cleanup_distributed(distributed)
    return {**results, **param_metrics}


def total_contrastive_steps(
    config: config_dict.ConfigDict,
    train_loader: DataLoader,
) -> int:
    steps = max(1, int(math.ceil(float(config.num_epochs) * len(train_loader))))
    max_steps = _config_get(config, "training_max_steps", None)
    if max_steps is None:
        return steps
    return min(steps, max(1, int(max_steps)))


def run_contrastive_loop(
    *,
    config: config_dict.ConfigDict,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    online_train_loader: DataLoader | None,
    optimizers: list[torch.optim.Optimizer],
    schedulers: list[LRSchedulerLike],
    grad_scaler: torch.amp.GradScaler | None,
    autocast_dtype: torch.dtype | None,
    logger: Any,
    checkpoint_dir: StoragePath,
    start_epoch: int,
    resume_offset: int,
    global_step: int,
    total_steps: int,
    distributed: DistributedContext,
    flops_per_optimizer_step: float | None = None,
) -> dict[str, object]:
    log_every_n_steps = int(_config_get(config, "log_every_n_steps", 50))
    val_every_n_steps = int(_config_get(config, "contrastive_val_every_n_steps", 0))
    checkpoint_every_steps = int(config.checkpoint_every_steps)
    grad_clip_norm = _optional_float(_config_get(config, "grad_clip_norm", None))
    if flops_per_optimizer_step is None:
        flops_per_optimizer_step = estimate_training_flops_per_optimizer_step(
            config,
            unwrap_model(model),
            int(_config_get(config, "contrastive_batch_size", config.batch_size)),
        )
    loop_epochs = max(1, math.ceil(float(config.num_epochs)))
    start_time = time.perf_counter()
    last_val_metrics: dict[str, torch.Tensor] = {}
    for epoch in range(start_epoch, loop_epochs):
        dataset = cast(NistMurckoContrastivePairs, train_loader.dataset)
        dataset.set_epoch(epoch)
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        if online_train_loader is not None and hasattr(
            online_train_loader.sampler,
            "set_epoch",
        ):
            online_train_loader.sampler.set_epoch(epoch)
        iterator = iter(train_loader)
        online_iterator = iter(online_train_loader) if online_train_loader is not None else None
        for _ in range(resume_offset if epoch == start_epoch else 0):
            next(iterator)
        pbar = tqdm(
            total=min(len(train_loader), total_steps - global_step),
            desc=f"Contrastive epoch {epoch}",
            unit="step",
            disable=not distributed.is_main,
        )
        if epoch == start_epoch and resume_offset:
            pbar.update(resume_offset)
        for batch in iterator:
            batch = _move_batch(batch, distributed.device)
            online_batch = None
            if online_train_loader is not None and online_iterator is not None:
                try:
                    online_batch = next(online_iterator)
                except StopIteration:
                    online_iterator = iter(online_train_loader)
                    online_batch = next(online_iterator)
                online_batch = _move_batch(online_batch, distributed.device)
            metrics = contrastive_train_step(
                model,
                batch,
                online_batch,
                optimizers,
                schedulers,
                grad_scaler=grad_scaler,
                autocast_dtype=autocast_dtype,
                grad_clip_norm=grad_clip_norm,
            )
            global_step += 1
            pbar.update(1)
            should_log = log_every_n_steps > 0 and global_step % log_every_n_steps == 0
            log_metrics = (
                reduce_metric_tensors(metrics, distributed) if should_log else metrics
            )
            if distributed.is_main and logger is not None and should_log:
                payload = {
                    f"train/{key}": float(value.detach())
                    for key, value in log_metrics.items()
                }
                payload["global_step"] = global_step
                payload["epoch"] = epoch
                cumulative_flops = cumulative_training_flops(
                    global_step,
                    flops_per_optimizer_step,
                )
                payload["train/cumulative_flops"] = cumulative_flops
                payload["train/cumulative_peta_flops"] = cumulative_flops / 1e15
                payload["train/flops_per_optimizer_step"] = float(
                    flops_per_optimizer_step
                )
                logger.log_metrics(payload, step=global_step)
                pbar.set_postfix(
                    loss=f"{float(log_metrics['loss'].detach()):.4f}",
                    step=global_step,
                )
            if val_every_n_steps > 0 and global_step % val_every_n_steps == 0:
                last_val_metrics = evaluate_contrastive(
                    model,
                    val_loader,
                    distributed=distributed,
                    autocast_dtype=autocast_dtype,
                )
                if distributed.is_main and logger is not None:
                    logger.log_metrics(
                        {
                            f"val/{key}": float(value.detach())
                            for key, value in last_val_metrics.items()
                        },
                        step=global_step,
                    )
            if global_step % checkpoint_every_steps == 0:
                save_contrastive_checkpoint_if_main(
                    checkpoint_dir=checkpoint_dir,
                    model=model,
                    optimizers=optimizers,
                    schedulers=schedulers,
                    grad_scaler=grad_scaler,
                    global_step=global_step,
                    epoch=epoch,
                    loss=float(metrics["loss"].detach()),
                    distributed=distributed,
                )
                barrier(distributed)
            if global_step >= total_steps:
                break
        pbar.close()
        resume_offset = 0
        if global_step >= total_steps:
            break
    last_val_metrics = evaluate_contrastive(
        model,
        val_loader,
        distributed=distributed,
        autocast_dtype=autocast_dtype,
    )
    save_contrastive_checkpoint_if_main(
        checkpoint_dir=checkpoint_dir,
        model=model,
        optimizers=optimizers,
        schedulers=schedulers,
        grad_scaler=grad_scaler,
        global_step=global_step,
        epoch=global_step // max(1, len(train_loader)),
        loss=float(last_val_metrics["loss"].detach()),
        distributed=distributed,
        name="last.pt",
    )
    barrier(distributed)
    elapsed = time.perf_counter() - start_time
    return {
        "run/final_global_step": float(global_step),
        "run/train_elapsed_seconds": elapsed,
        "run/steps_per_second": float(global_step) / elapsed if elapsed > 0 else 0.0,
        **{f"val/{key}": float(value.detach()) for key, value in last_val_metrics.items()},
    }


def contrastive_train_step(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    online_batch: dict[str, torch.Tensor] | None,
    optimizers: list[torch.optim.Optimizer],
    schedulers: list[LRSchedulerLike],
    *,
    grad_scaler: torch.amp.GradScaler | None,
    autocast_dtype: torch.dtype | None,
    grad_clip_norm: float | None,
) -> dict[str, torch.Tensor]:
    device_type = next(model.parameters()).device.type
    autocast_ctx = (
        torch.autocast(device_type=device_type, dtype=autocast_dtype)
        if autocast_dtype is not None
        else torch.no_grad()
    )
    if autocast_dtype is None:
        autocast_ctx = torch.enable_grad()
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    with autocast_ctx:
        metrics = cast(dict[str, torch.Tensor], model(batch, online_batch=online_batch))
    loss = metrics["loss"]
    if grad_scaler is not None and grad_scaler.is_enabled():
        grad_scaler.scale(loss).backward()
        if grad_clip_norm is not None and grad_clip_norm > 0:
            for optimizer in optimizers:
                grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm, foreach=True)
        for optimizer in optimizers:
            grad_scaler.step(optimizer)
        grad_scaler.update()
    else:
        loss.backward()
        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm, foreach=True)
        for optimizer in optimizers:
            optimizer.step()
    for scheduler in schedulers:
        scheduler.step()
    return metrics


@torch.no_grad()
def evaluate_contrastive(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    distributed: DistributedContext,
    autocast_dtype: torch.dtype | None,
) -> dict[str, torch.Tensor]:
    model.eval()
    device_type = distributed.device.type
    totals: dict[str, torch.Tensor] = {}
    count = 0
    for batch in loader:
        batch = _move_batch(batch, distributed.device)
        with (
            torch.autocast(device_type=device_type, dtype=autocast_dtype)
            if autocast_dtype is not None
            else torch.no_grad()
        ):
            metrics = cast(dict[str, torch.Tensor], model(batch))
        batch_size = batch["peak_mz"].shape[0]
        count += batch_size
        for key, value in metrics.items():
            totals[key] = totals.get(key, value.detach().new_zeros(())) + value.detach() * batch_size
    averaged = {key: value / max(count, 1) for key, value in totals.items()}
    averaged = reduce_metric_tensors(averaged, distributed)
    model.train()
    return averaged


def _move_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


def save_contrastive_checkpoint_if_main(
    *,
    checkpoint_dir: StoragePath,
    model: torch.nn.Module,
    optimizers: list[torch.optim.Optimizer],
    schedulers: list[LRSchedulerLike],
    grad_scaler: torch.amp.GradScaler | None,
    global_step: int,
    epoch: int,
    loss: float,
    distributed: DistributedContext,
    name: str | None = None,
) -> None:
    if not distributed.is_main:
        return
    base = cast(ContrastiveTrainingModule, unwrap_model(model))
    ckpt_name = name or f"step-{global_step:08d}.pt"
    path = storage_join(checkpoint_dir, ckpt_name)
    pooler_path = covariance_pooler_checkpoint_path(path)
    save_torch_checkpoint(
        {
            "pooler": base.pooler.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
        },
        pooler_path,
    )
    save_torch_checkpoint(
        {
            "model": base.model.state_dict(),
            "online_probe": base.online_probe.state_dict(),
            "optimizers": [optimizer.state_dict() for optimizer in optimizers],
            "schedulers": [scheduler.state_dict() for scheduler in schedulers],
            "grad_scaler": (
                grad_scaler.state_dict()
                if grad_scaler is not None and grad_scaler.is_enabled()
                else None
            ),
            "global_step": global_step,
            "epoch": epoch,
            "loss": loss,
            "training_mode": "contrastive",
            "covariance_pooler_checkpoint": storage_name(pooler_path),
        },
        path,
    )


def restore_contrastive_state(
    *,
    checkpoint_dir: StoragePath,
    module: ContrastiveTrainingModule,
    optimizers: list[torch.optim.Optimizer],
    schedulers: list[LRSchedulerLike],
    grad_scaler: torch.amp.GradScaler | None,
    steps_per_epoch: int,
    device: torch.device,
) -> tuple[int, int, int]:
    checkpoints = training_checkpoint_paths(checkpoint_dir)
    if not checkpoints:
        return 0, 0, 0
    ckpt_path = checkpoints[-1]
    log.info("Resuming contrastive training from %s", ckpt_path)
    ckpt = load_torch_checkpoint(ckpt_path, map_location=device, weights_only=True)
    load_resume_model_state(module.model, ckpt["model"])
    load_resume_covariance_pooler_state(module.pooler, ckpt_path, ckpt)
    module.online_probe.load_state_dict(ckpt["online_probe"])
    for optimizer, state in zip(optimizers, ckpt["optimizers"], strict=True):
        load_optimizer_state(optimizer, state)
    for scheduler, state in zip(schedulers, ckpt["schedulers"], strict=True):
        scheduler.load_state_dict(state)
    load_grad_scaler_state(grad_scaler, ckpt.get("grad_scaler"))
    global_step = int(ckpt["global_step"])
    start_epoch = int(ckpt["epoch"])
    resume_offset = global_step - start_epoch * steps_per_epoch
    start_epoch += resume_offset // steps_per_epoch
    resume_offset %= steps_per_epoch
    return start_epoch, global_step, resume_offset


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
