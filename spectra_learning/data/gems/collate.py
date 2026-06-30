from typing import Any

import numpy as np
import torch

from spectra_learning.data.gems.conversion import format_batch
from spectra_learning.data.gems.masking import (
    DEFAULT_JEPA_MASK_LENGTHS,
    DEFAULT_JEPA_MASK_STRATEGY,
    JEPA_MASK_STRATEGIES,
    _normalize_mask_strategy_name,
    _normalize_mask_strategy_names,
    _sample_all_mask_strategies_torch,
    _sample_block_masks_torch,
)
from spectra_learning.data.gems.intensity_aware import (
    AWARE_MIXED_MASK_CONFIG,
    INTENSITY_AWARE_MASK_STRATEGY,
    sample_intensity_aware_masks_torch,
)
from spectra_learning.data.spectra import (
    COLLISION_ENERGY_MAX,
    DEFAULT_GROUPED_PEAK_ISOTOPE_CHARGES,
    DEFAULT_GROUPED_PEAK_SHOULDER_DA,
    DEFAULT_PEAK_FILTERING,
    PEAK_MZ_MAX,
    preprocess_peak_batch_numpy,
    preprocess_peak_batch_torch,
)


class GemsBatchCollator:
    def __init__(
        self,
        *,
        augment: bool,
        num_target_blocks: int,
        context_fraction: float,
        target_fraction: float,
        block_min_len: int,
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
        mask_strategy: str | tuple[str, ...] | list[str] = DEFAULT_JEPA_MASK_STRATEGY,
        mask_lengths: tuple[int, ...] = DEFAULT_JEPA_MASK_LENGTHS,
        mask_round_from: int = len(DEFAULT_JEPA_MASK_LENGTHS),
        intensity_aware_mask_config: dict[str, float] | None = None,
        allow_target_overlap: bool = False,
        output_format: str = "torch",
    ) -> None:
        self.augment = augment
        self.num_target_blocks = num_target_blocks
        self.context_fraction = context_fraction
        self.target_fraction = target_fraction
        self.block_min_len = block_min_len
        self.mask_strategy = mask_strategy
        self.mask_lengths = tuple(length for length in mask_lengths)
        self.mask_round_from = mask_round_from
        self.intensity_aware_mask_config = dict(
            AWARE_MIXED_MASK_CONFIG
            if intensity_aware_mask_config is None
            else intensity_aware_mask_config
        )
        self.allow_target_overlap = allow_target_overlap
        self.num_peaks = num_peaks
        self.max_precursor_mz = max_precursor_mz
        self.min_peak_intensity = min_peak_intensity
        self.peak_drop_min_intensity = peak_drop_min_intensity
        self.peak_filtering = peak_filtering
        self.grouped_peak_shoulder_da = grouped_peak_shoulder_da
        self.grouped_peak_isotope_charges = grouped_peak_isotope_charges
        self.peak_ordering = peak_ordering
        self.precursor_peak_exclusion_window_da = precursor_peak_exclusion_window_da
        self.output_format = output_format

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        batch = self._preprocess(samples)
        self._ensure_nonempty(batch)
        if self.augment:
            batch["context_mask"], batch["target_masks"] = self._sample_masks(batch)
        return format_batch(batch, self.output_format)

    def _preprocess(self, samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if isinstance(samples[0]["spectra"], np.ndarray):
            return self._preprocess_numpy(samples)
        return self._preprocess_torch(samples)

    def _preprocess_numpy(
        self,
        samples: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        spectra = np.stack([sample["spectra"] for sample in samples], axis=0)
        precursor_mz_raw = np.asarray(
            [sample["precursor_mz_raw"] for sample in samples],
            dtype=np.float32,
        )
        batch = preprocess_peak_batch_numpy(
            spectra,
            precursor_mz_raw,
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
        self._add_numpy_spectrum_metadata(batch, samples)
        return {key: torch.from_numpy(value) for key, value in batch.items()}

    def _preprocess_torch(
        self,
        samples: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        spectra = torch.stack([sample["spectra"] for sample in samples], dim=0)
        precursor_mz_raw = torch.stack(
            [sample["precursor_mz_raw"] for sample in samples],
            dim=0,
        )
        batch = preprocess_peak_batch_torch(
            spectra[:, 0, :],
            spectra[:, 1, :],
            precursor_mz_raw,
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
        self._add_torch_spectrum_metadata(batch, samples)
        return batch

    def _add_numpy_spectrum_metadata(
        self,
        batch: dict[str, np.ndarray],
        samples: list[dict[str, Any]],
    ) -> None:
        collision_energy = np.asarray(
            [sample["collision_energy"] for sample in samples],
            dtype=np.float32,
        )
        batch["collision_energy"] = (
            np.clip(collision_energy, 0.0, COLLISION_ENERGY_MAX)
            / COLLISION_ENERGY_MAX
        ).astype(np.float32)
        batch["charge"] = np.asarray(
            [sample["charge"] for sample in samples],
            dtype=np.float32,
        )

    def _add_torch_spectrum_metadata(
        self,
        batch: dict[str, torch.Tensor],
        samples: list[dict[str, Any]],
    ) -> None:
        collision_energy = torch.stack(
            [torch.as_tensor(sample["collision_energy"]) for sample in samples],
        ).to(dtype=torch.float32)
        batch["collision_energy"] = collision_energy.clamp(
            0.0,
            COLLISION_ENERGY_MAX,
        ) / COLLISION_ENERGY_MAX
        batch["charge"] = torch.stack(
            [torch.as_tensor(sample["charge"]) for sample in samples],
        ).to(dtype=torch.float32)

    def _ensure_nonempty(self, batch: dict[str, torch.Tensor]) -> None:
        no_valid = ~batch["peak_valid_mask"].any(dim=1)
        if bool(no_valid.any()):
            batch["peak_valid_mask"] = batch["peak_valid_mask"].clone()
            batch["peak_valid_mask"][no_valid, 0] = True

    def _sample_masks(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sampling_valid_mask = batch["peak_valid_mask"]

        strategies = self._mask_strategy_pool()
        if len(strategies) == 1:
            context_mask, target_masks = self._sample_masks_for_strategy(
                batch,
                sampling_valid_mask,
                strategies[0],
            )
        elif INTENSITY_AWARE_MASK_STRATEGY in strategies:
            context_mask, target_masks = self._sample_mixed_strategy_masks(
                batch,
                sampling_valid_mask,
                strategies,
            )
        else:
            context_mask, target_masks = _sample_block_masks_torch(
                sampling_valid_mask,
                num_target_blocks=self.num_target_blocks,
                context_fraction=self.context_fraction,
                target_fraction=self.target_fraction,
                block_min_len=self.block_min_len,
                mask_strategy=self.mask_strategy,
                mask_lengths=self.mask_lengths,
                mask_round_from=self.mask_round_from,
                allow_target_overlap=self.allow_target_overlap,
            )

        return context_mask, target_masks

    def _mask_strategy_pool(self) -> tuple[str, ...]:
        pool: list[str] = []
        for strategy in _normalize_mask_strategy_names(self.mask_strategy):
            if strategy == "all":
                pool.extend(JEPA_MASK_STRATEGIES)
            else:
                pool.append(strategy)
        return tuple(dict.fromkeys(pool))

    def _sample_masks_for_strategy(
        self,
        batch: dict[str, torch.Tensor],
        sampling_valid_mask: torch.Tensor,
        strategy: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if _normalize_mask_strategy_name(strategy) == INTENSITY_AWARE_MASK_STRATEGY:
            return sample_intensity_aware_masks_torch(
                sampling_valid_mask,
                batch["peak_intensity"],
                batch["peak_mz"] * PEAK_MZ_MAX,
                num_target_blocks=self.num_target_blocks,
                allow_target_overlap=self.allow_target_overlap,
                **self.intensity_aware_mask_config,
            )
        return _sample_block_masks_torch(
            sampling_valid_mask,
            num_target_blocks=self.num_target_blocks,
            context_fraction=self.context_fraction,
            target_fraction=self.target_fraction,
            block_min_len=self.block_min_len,
            mask_strategy=strategy,
            mask_lengths=self.mask_lengths,
            mask_round_from=self.mask_round_from,
            allow_target_overlap=self.allow_target_overlap,
        )

    def _sample_mixed_strategy_masks(
        self,
        batch: dict[str, torch.Tensor],
        sampling_valid_mask: torch.Tensor,
        strategies: tuple[str, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row_strategies = _sample_all_mask_strategies_torch(
            sampling_valid_mask.shape[0],
            device=sampling_valid_mask.device,
            mask_strategies=strategies,
        )
        context_mask = torch.zeros_like(sampling_valid_mask)
        target_masks = torch.zeros(
            sampling_valid_mask.shape[0],
            self.num_target_blocks,
            sampling_valid_mask.shape[1],
            dtype=torch.bool,
            device=sampling_valid_mask.device,
        )
        for strategy in dict.fromkeys(row_strategies):
            strategy_context, strategy_targets = self._sample_masks_for_strategy(
                batch,
                sampling_valid_mask,
                strategy,
            )
            rows = torch.tensor(
                [row_strategy == strategy for row_strategy in row_strategies],
                dtype=torch.bool,
                device=sampling_valid_mask.device,
            )
            context_mask[rows] = strategy_context[rows]
            target_masks[rows] = strategy_targets[rows]
        return context_mask, target_masks
