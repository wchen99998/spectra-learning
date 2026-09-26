from spectra_learning.data.gems.randomness import data_seed, sample_rng
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
    _sample_random_mask_1d_torch,
)
from spectra_learning.data.gems.gap_aware import (
    DEFAULT_GAP_AWARE_BUCKET_EDGES_DA,
    DEFAULT_GAP_AWARE_BUCKET_WEIGHTS,
    GAP_AWARE_MASK_STRATEGY,
    sample_gap_aware_masks_torch,
)
from spectra_learning.data.gems.intensity_aware import (
    AWARE_MIXED_MASK_CONFIG,
    INTENSITY_AWARE_MASK_STRATEGY,
    sample_intensity_aware_masks_torch,
)
from spectra_learning.data.spectra import (
    COLLISION_ENERGY_MAX,
    PEAK_MZ_MAX,
    canonicalize_precursor_charge_numpy,
    canonicalize_precursor_charge_torch,
    preprocess_peak_batch_numpy,
    preprocess_peak_batch_torch,
)
from spectra_learning.models.spectrum_metadata import (
    MASSIVE_V2_ACQUISITION_SCHEMA,
    drop_massive_v2_metadata_torch,
    torch_massive_v2_condition_from_batch,
)


ACQUISITION_METADATA_KEYS = (
    "precursor_mz_present",
    "collision_energy_present",
    "charge_present",
    "polarity_id",
    "acquisition_type_id",
    "isolation_window_lower_offset",
    "isolation_window_upper_offset",
    "isolation_window_present",
    "instrument_family_id",
    "mass_accuracy",
    "mass_accuracy_present",
    "retention_time_fraction",
    "retention_time_present",
    "precursor_intensity_zscore",
    "precursor_intensity_present",
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
        mask_strategy: str | tuple[str, ...] | list[str] = DEFAULT_JEPA_MASK_STRATEGY,
        mask_lengths: tuple[int, ...] = DEFAULT_JEPA_MASK_LENGTHS,
        mask_round_from: int = len(DEFAULT_JEPA_MASK_LENGTHS),
        intensity_aware_mask_config: dict[str, float] | None = None,
        gap_aware_bucket_edges_da: tuple[float, ...] = (
            DEFAULT_GAP_AWARE_BUCKET_EDGES_DA
        ),
        gap_aware_bucket_weights: tuple[float, ...] = (
            DEFAULT_GAP_AWARE_BUCKET_WEIGHTS
        ),
        allow_target_overlap: bool = False,
        output_format: str = "torch",
        spectrum_metadata_schema: str | None = None,
        spectrum_metadata_dropout_probability: float = 0.0,
        augmentation_seed: int | None = None,
        augmentation_epoch: int = 0,
        augmentation_split: str = "train",
        mae_objective: str = "masked_reconstruction",
    ) -> None:
        if mae_objective not in {"masked_reconstruction", "absorbing_diffusion"}:
            raise ValueError(f"Unknown mae_objective: {mae_objective}")
        if mae_objective == "absorbing_diffusion" and num_target_blocks != 1:
            raise ValueError("absorbing_diffusion requires num_target_blocks=1")
        self.mae_objective = mae_objective
        self.augmentation_seed = augmentation_seed
        self.augmentation_epoch = augmentation_epoch
        self.augmentation_split = augmentation_split
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
        self.gap_aware_bucket_edges_da = tuple(
            float(edge) for edge in gap_aware_bucket_edges_da
        )
        self.gap_aware_bucket_weights = tuple(
            float(weight) for weight in gap_aware_bucket_weights
        )
        self.allow_target_overlap = allow_target_overlap
        self.num_peaks = num_peaks
        self.max_precursor_mz = max_precursor_mz
        self.min_peak_intensity = min_peak_intensity
        self.peak_drop_min_intensity = peak_drop_min_intensity
        self.peak_ordering = peak_ordering
        self.precursor_peak_exclusion_window_da = precursor_peak_exclusion_window_da
        self.output_format = output_format
        self.spectrum_metadata_schema = spectrum_metadata_schema
        self.spectrum_metadata_dropout_probability = (
            spectrum_metadata_dropout_probability
        )

    def __call__(
        self, samples: list[dict[str, Any]] | dict[str, np.ndarray]
    ) -> dict[str, Any]:
        batch = self._preprocess(samples)
        if self.augment:
            batch = self._replace_rows_without_context_and_targets(batch)
            batch["context_mask"], batch["target_masks"] = self._sample_masks(batch)
        batch.pop("_sample_id", None)
        return format_batch(batch, self.output_format)

    def _replace_rows_without_context_and_targets(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        minimum_peaks = self.num_target_blocks * self.block_min_len + 1
        if self.mae_objective == "absorbing_diffusion":
            minimum_peaks = 1
            # Preprocessing gives empty spectra a valid zero-intensity placeholder.
            eligible = (batch["peak_intensity"] > 0).any(dim=1)
        else:
            eligible = batch["peak_valid_mask"].sum(dim=1) >= minimum_peaks
        eligible_rows = torch.nonzero(eligible, as_tuple=False).squeeze(-1)
        missing_rows = torch.nonzero(~eligible, as_tuple=False).squeeze(-1)
        if eligible_rows.numel() == 0:
            raise ValueError(
                "Training batch has no spectra with at least "
                f"{minimum_peaks} usable peaks; the artifact eligibility "
                "contract is invalid"
            )
        row_indices = torch.arange(eligible.shape[0], device=eligible.device)
        row_indices[missing_rows] = eligible_rows[
            torch.arange(missing_rows.numel(), device=eligible.device)
            % eligible_rows.numel()
        ]
        return {key: value[row_indices] for key, value in batch.items()}

    def _preprocess(
        self,
        samples: list[dict[str, Any]] | dict[str, np.ndarray],
    ) -> dict[str, torch.Tensor]:
        if isinstance(samples, dict):
            return self._preprocess_numpy(samples)
        if isinstance(samples[0]["spectra"], np.ndarray):
            # Sample-based probes share preprocessing with the batched reader.
            return self._preprocess_numpy(
                {
                    key: np.stack([sample[key] for sample in samples])
                    for key in samples[0]
                }
            )
        return self._preprocess_torch(samples)

    def _preprocess_numpy(
        self,
        samples: dict[str, np.ndarray],
    ) -> dict[str, torch.Tensor]:
        spectra = samples["spectra"]
        precursor_mz_raw = np.asarray(samples["precursor_mz_raw"], dtype=np.float32)
        batch = preprocess_peak_batch_numpy(
            spectra,
            precursor_mz_raw,
            num_peaks=self.num_peaks,
            peak_drop_min_intensity=self.peak_drop_min_intensity,
            peak_ordering=self.peak_ordering,
            max_precursor_mz=self.max_precursor_mz,
            precursor_peak_exclusion_window_da=self.precursor_peak_exclusion_window_da,
            min_peak_intensity=self.min_peak_intensity,
        )
        self._add_numpy_spectrum_metadata(batch, samples)
        torch_batch = {key: torch.from_numpy(value) for key, value in batch.items()}
        if self.augmentation_seed is not None and "sample_id" in samples:
            torch_batch["_sample_id"] = torch.from_numpy(
                np.asarray(samples["sample_id"], dtype=np.uint64).view(np.int64)
            )
        self._add_condition_vector(torch_batch)
        return torch_batch

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
        )
        self._add_torch_spectrum_metadata(batch, samples)
        self._add_condition_vector(batch)
        return batch

    def _add_numpy_spectrum_metadata(
        self,
        batch: dict[str, np.ndarray],
        samples: dict[str, np.ndarray],
    ) -> None:
        batch["collision_energy"] = (
            np.clip(samples["collision_energy"], 0.0, COLLISION_ENERGY_MAX)
            / COLLISION_ENERGY_MAX
        ).astype(np.float32)
        batch["charge"] = canonicalize_precursor_charge_numpy(
            np.asarray(samples["charge"], dtype=np.float32)
        )
        if self.spectrum_metadata_schema == MASSIVE_V2_ACQUISITION_SCHEMA:
            for key in ACQUISITION_METADATA_KEYS:
                batch[key] = np.asarray(samples[key], dtype=np.float32)

    def _add_torch_spectrum_metadata(
        self,
        batch: dict[str, torch.Tensor],
        samples: list[dict[str, Any]],
    ) -> None:
        collision_energy = torch.stack(
            [torch.as_tensor(sample["collision_energy"]) for sample in samples],
        ).to(dtype=torch.float32)
        batch["collision_energy"] = (
            collision_energy.clamp(
                0.0,
                COLLISION_ENERGY_MAX,
            )
            / COLLISION_ENERGY_MAX
        )
        batch["charge"] = canonicalize_precursor_charge_torch(
            torch.stack(
                [torch.as_tensor(sample["charge"]) for sample in samples],
            )
        )
        self._add_torch_acquisition_metadata(batch, samples)

    def _add_torch_acquisition_metadata(
        self,
        batch: dict[str, torch.Tensor],
        samples: list[dict[str, Any]],
    ) -> None:
        if self.spectrum_metadata_schema != MASSIVE_V2_ACQUISITION_SCHEMA:
            return
        for key in ACQUISITION_METADATA_KEYS:
            batch[key] = torch.stack(
                [torch.as_tensor(sample[key]) for sample in samples]
            ).to(dtype=torch.float32)

    def _sample_seed(self, sample_id: int, operation: str, view: int = 0) -> int:
        return data_seed(
            self.augmentation_seed,
            self.augmentation_epoch,
            self.augmentation_split,
            sample_id,
            operation,
            view,
        )

    def _add_condition_vector(self, batch: dict[str, torch.Tensor]) -> None:
        if self.spectrum_metadata_schema != MASSIVE_V2_ACQUISITION_SCHEMA:
            return
        metadata = torch_massive_v2_condition_from_batch(batch)
        if self.spectrum_metadata_dropout_probability > 0.0:
            if "_sample_id" in batch:
                rows = []
                for i, sample_id in enumerate(batch["_sample_id"].tolist()):
                    with sample_rng(self._sample_seed(sample_id, "metadata")):
                        rows.append(
                            drop_massive_v2_metadata_torch(
                                metadata[i : i + 1],
                                self.spectrum_metadata_dropout_probability,
                            )
                        )
                metadata = torch.cat(rows)
            else:
                metadata = drop_massive_v2_metadata_torch(
                    metadata, self.spectrum_metadata_dropout_probability
                )
        batch["spectrum_metadata"] = metadata

    def _sample_masks(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if "_sample_id" in batch:
            contexts, targets = [], []
            for i, sample_id in enumerate(batch["_sample_id"].tolist()):
                row = {
                    key: value[i : i + 1]
                    for key, value in batch.items()
                    if key != "_sample_id"
                }
                with sample_rng(self._sample_seed(sample_id, "mask")):
                    context, target = self._sample_masks(row)
                contexts.append(context)
                targets.append(target)
            return torch.cat(contexts), torch.cat(targets)
        sampling_valid_mask = batch["peak_valid_mask"]
        if self.mae_objective == "absorbing_diffusion":
            return self._sample_absorbing_diffusion_masks(sampling_valid_mask)

        strategies = self._mask_strategy_pool()
        if len(strategies) == 1:
            context_mask, target_masks = self._sample_masks_for_strategy(
                batch,
                sampling_valid_mask,
                strategies[0],
            )
        elif (
            INTENSITY_AWARE_MASK_STRATEGY in strategies
            or GAP_AWARE_MASK_STRATEGY in strategies
        ):
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

    def _sample_absorbing_diffusion_masks(
        self,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a uniform nonempty mask count and a uniform subset at that count."""
        targets = torch.zeros_like(valid_mask)
        for row_index, row_valid in enumerate(valid_mask):
            valid_count = int(row_valid.sum())
            mask_count = int(
                torch.randint(1, valid_count + 1, (), device=valid_mask.device)
            )
            targets[row_index] = _sample_random_mask_1d_torch(
                row_valid, mask_count=mask_count
            )
        return valid_mask & ~targets, targets.unsqueeze(1)

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
        normalized = _normalize_mask_strategy_name(strategy)
        if normalized == GAP_AWARE_MASK_STRATEGY:
            return sample_gap_aware_masks_torch(
                sampling_valid_mask,
                batch["peak_mz"] * PEAK_MZ_MAX,
                num_target_blocks=self.num_target_blocks,
                context_fraction=self.context_fraction,
                target_fraction=self.target_fraction,
                block_min_len=self.block_min_len,
                allow_target_overlap=self.allow_target_overlap,
                bucket_edges_da=self.gap_aware_bucket_edges_da,
                bucket_weights=self.gap_aware_bucket_weights,
            )
        if normalized == INTENSITY_AWARE_MASK_STRATEGY:
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


class SJEPABatchCollator(GemsBatchCollator):
    def __init__(
        self,
        *,
        num_global_views: int,
        num_local_views: int,
        global_view_fraction: float,
        local_view_fraction: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_global_views = num_global_views
        self.num_local_views = num_local_views
        self.global_view_fraction = global_view_fraction
        self.local_view_fraction = local_view_fraction

    def __call__(
        self, samples: list[dict[str, Any]] | dict[str, np.ndarray]
    ) -> dict[str, Any]:
        batch = self._preprocess(samples)
        valid_mask = batch["peak_valid_mask"]
        batch["sjepa_global_view_masks"] = self._sample_views(
            valid_mask,
            num_views=self.num_global_views,
            fraction=self.global_view_fraction,
            sample_ids=batch.get("_sample_id"),
            stream="global-view",
        )
        batch["sjepa_local_view_masks"] = self._sample_views(
            valid_mask,
            num_views=self.num_local_views,
            fraction=self.local_view_fraction,
            sample_ids=batch.get("_sample_id"),
            stream="local-view",
        )
        batch.pop("_sample_id", None)
        return format_batch(batch, self.output_format)

    def _sample_views(
        self,
        valid_mask: torch.Tensor,
        *,
        num_views: int,
        fraction: float,
        sample_ids: torch.Tensor | None = None,
        stream: str = "view",
    ) -> torch.Tensor:
        masks = torch.zeros(
            valid_mask.shape[0],
            num_views,
            valid_mask.shape[1],
            dtype=torch.bool,
            device=valid_mask.device,
        )
        for row_idx, row_valid in enumerate(valid_mask):
            retain_count = max(round(int(row_valid.sum()) * fraction), 1)
            for view_idx in range(num_views):
                if sample_ids is not None:
                    with sample_rng(
                        self._sample_seed(int(sample_ids[row_idx]), stream, view_idx)
                    ):
                        masks[row_idx, view_idx] = _sample_random_mask_1d_torch(
                            row_valid, mask_count=retain_count
                        )
                else:
                    masks[row_idx, view_idx] = _sample_random_mask_1d_torch(
                        row_valid,
                        mask_count=retain_count,
                    )
        return masks
