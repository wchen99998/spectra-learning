import torch

from spectra_learning.data.gems.conversion import _prepend_precursor_token_torch
from spectra_learning.data.gems.masking import (
    DEFAULT_JEPA_MASK_LENGTHS,
    DEFAULT_JEPA_MASK_STRATEGY,
    _normalize_mask_strategy_name,
    _sample_block_masks_torch,
)
from spectra_learning.data.gems.intensity_aware import (
    AWARE_MIXED_MASK_CONFIG,
    INTENSITY_AWARE_MASK_STRATEGY,
    sample_intensity_aware_masks_torch,
)
from spectra_learning.data.spectra import PEAK_MZ_MAX, preprocess_peak_batch_torch


class GemsBatchCollator:
    def __init__(
        self,
        *,
        augment: bool,
        num_target_blocks: int,
        context_fraction: float,
        target_fraction: float,
        block_min_len: int,
        use_precursor_token: bool,
        num_peaks: int,
        max_precursor_mz: float,
        min_peak_intensity: float,
        peak_drop_min_intensity: float,
        peak_ordering: str,
        precursor_peak_exclusion_window_da: float,
        mask_strategy: str = DEFAULT_JEPA_MASK_STRATEGY,
        mask_lengths: tuple[int, ...] = DEFAULT_JEPA_MASK_LENGTHS,
        mask_round_from: int = len(DEFAULT_JEPA_MASK_LENGTHS),
        intensity_aware_mask_config: dict[str, float] | None = None,
    ) -> None:
        self.augment = bool(augment)
        self.num_target_blocks = int(num_target_blocks)
        self.context_fraction = float(context_fraction)
        self.target_fraction = float(target_fraction)
        self.block_min_len = int(block_min_len)
        self.mask_strategy = str(mask_strategy)
        self.mask_lengths = tuple(int(length) for length in mask_lengths)
        self.mask_round_from = int(mask_round_from)
        self.intensity_aware_mask_config = dict(
            AWARE_MIXED_MASK_CONFIG
            if intensity_aware_mask_config is None
            else intensity_aware_mask_config
        )
        self.use_precursor_token = bool(use_precursor_token)
        self.num_peaks = int(num_peaks)
        self.max_precursor_mz = float(max_precursor_mz)
        self.min_peak_intensity = float(min_peak_intensity)
        self.peak_drop_min_intensity = float(peak_drop_min_intensity)
        self.peak_ordering = str(peak_ordering)
        self.precursor_peak_exclusion_window_da = float(precursor_peak_exclusion_window_da)

    def __call__(self, samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        batch = self._preprocess(samples)
        if self.use_precursor_token:
            batch = _prepend_precursor_token_torch(batch)
        self._ensure_nonempty_without_precursor(batch)
        if self.augment:
            batch["context_mask"], batch["target_masks"] = self._sample_masks(batch)
        return batch

    def _preprocess(self, samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        spectra = torch.stack([sample["spectra"] for sample in samples], dim=0)
        precursor_mz_raw = torch.stack(
            [sample["precursor_mz_raw"] for sample in samples],
            dim=0,
        )
        return preprocess_peak_batch_torch(
            spectra[:, 0, :],
            spectra[:, 1, :],
            precursor_mz_raw,
            num_peaks=self.num_peaks,
            peak_drop_min_intensity=self.peak_drop_min_intensity,
            peak_ordering=self.peak_ordering,
            max_precursor_mz=self.max_precursor_mz,
            precursor_peak_exclusion_window_da=self.precursor_peak_exclusion_window_da,
            min_peak_intensity=self.min_peak_intensity,
        )

    def _ensure_nonempty_without_precursor(self, batch: dict[str, torch.Tensor]) -> None:
        no_valid = ~batch["peak_valid_mask"].any(dim=1)
        if bool(no_valid.any()) and not self.use_precursor_token:
            batch["peak_valid_mask"] = batch["peak_valid_mask"].clone()
            batch["peak_valid_mask"][no_valid, 0] = True

    def _sample_masks(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if _normalize_mask_strategy_name(self.mask_strategy) == INTENSITY_AWARE_MASK_STRATEGY:
            return sample_intensity_aware_masks_torch(
                batch["peak_valid_mask"],
                batch["peak_intensity"],
                batch["peak_mz"] * PEAK_MZ_MAX,
                num_target_blocks=self.num_target_blocks,
                **self.intensity_aware_mask_config,
            )
        return _sample_block_masks_torch(
            batch["peak_valid_mask"],
            num_target_blocks=self.num_target_blocks,
            context_fraction=self.context_fraction,
            target_fraction=self.target_fraction,
            block_min_len=self.block_min_len,
            mask_strategy=self.mask_strategy,
            mask_lengths=self.mask_lengths,
            mask_round_from=self.mask_round_from,
        )
