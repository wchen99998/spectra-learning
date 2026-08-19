from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import torch
from ml_collections import config_dict

from spectra_learning.data.gems.collate import GemsBatchCollator
from spectra_learning.data.gems.conversion import format_batch
from spectra_learning.data.gems.datamodule import GemsDataModule
from spectra_learning.data.spectra import (
    COLLISION_ENERGY_MAX,
    DEFAULT_MAX_PRECURSOR_MZ,
    DEFAULT_MIN_PEAK_INTENSITY,
    DEFAULT_NUM_PEAKS,
    DEFAULT_PRECURSOR_PEAK_EXCLUSION_WINDOW_DA,
    PEAK_MZ_MAX,
    PRECURSOR_CHARGE_MAX,
)


class SpectraARTokenKind(IntEnum):
    PAD = 0
    BOS = 1
    EOS = 2
    PRECURSOR_MZ_LEVEL_0 = 3
    PRECURSOR_MZ_LEVEL_1 = 4
    PRECURSOR_MZ_LEVEL_2 = 5
    PRECURSOR_MZ_LEVEL_3 = 6
    PRECURSOR_MZ_RESIDUAL = 7
    COLLISION_ENERGY = 8
    CHARGE = 9
    FRAGMENT_MZ_LEVEL_0 = 10
    FRAGMENT_MZ_LEVEL_1 = 11
    FRAGMENT_MZ_LEVEL_2 = 12
    FRAGMENT_MZ_LEVEL_3 = 13
    FRAGMENT_MZ_RESIDUAL = 14
    INTENSITY = 15


SPECTRA_AR_TARGET_KINDS = (
    SpectraARTokenKind.EOS,
    SpectraARTokenKind.FRAGMENT_MZ_LEVEL_0,
    SpectraARTokenKind.FRAGMENT_MZ_LEVEL_1,
    SpectraARTokenKind.FRAGMENT_MZ_LEVEL_2,
    SpectraARTokenKind.FRAGMENT_MZ_LEVEL_3,
    SpectraARTokenKind.FRAGMENT_MZ_RESIDUAL,
    SpectraARTokenKind.INTENSITY,
)

_REMOVED_AR_PREPROCESSING_KEYS = (
    "ar_mz_max",
    "ar_precursor_mz_max",
)


@dataclass(frozen=True)
class SpectraARTokenizerConfig:
    max_num_peaks: int = 128
    mz_max: float = PEAK_MZ_MAX
    precursor_mz_max: float = DEFAULT_MAX_PRECURSOR_MZ
    mz_bin_widths: tuple[float, ...] = (50.0, 25.0, 5.0, 1.0)
    residual_bins: int = 100
    intensity_bins: int = 101
    collision_energy_bins: int = 101
    charge_bins: int = int(PRECURSOR_CHARGE_MAX) + 1

    @classmethod
    def from_config(
        cls,
        config: config_dict.ConfigDict,
    ) -> "SpectraARTokenizerConfig":
        for key in _REMOVED_AR_PREPROCESSING_KEYS:
            if key in config:
                raise ValueError(
                    f"{key} has been removed; AR tokenization uses the shared "
                    "peak preprocessing scales."
                )
        return cls(
            max_num_peaks=int(
                config.get(
                    "ar_max_num_peaks",
                    config.get("num_peaks", DEFAULT_NUM_PEAKS),
                )
            ),
            mz_max=PEAK_MZ_MAX,
            precursor_mz_max=float(
                config.get("max_precursor_mz", DEFAULT_MAX_PRECURSOR_MZ)
            ),
            mz_bin_widths=tuple(
                float(width)
                for width in config.get(
                    "ar_mz_bin_widths",
                    (50.0, 25.0, 5.0, 1.0),
                )
            ),
            residual_bins=int(config.get("ar_residual_bins", 100)),
            intensity_bins=int(config.get("ar_intensity_bins", 101)),
            collision_energy_bins=int(
                config.get("ar_collision_energy_bins", 101)
            ),
            charge_bins=int(
                config.get("ar_charge_bins", int(PRECURSOR_CHARGE_MAX) + 1)
            ),
        )


class SpectraARTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __init__(self, config: SpectraARTokenizerConfig) -> None:
        self.config = config
        self.precursor_mz_level_kinds = (
            SpectraARTokenKind.PRECURSOR_MZ_LEVEL_0,
            SpectraARTokenKind.PRECURSOR_MZ_LEVEL_1,
            SpectraARTokenKind.PRECURSOR_MZ_LEVEL_2,
            SpectraARTokenKind.PRECURSOR_MZ_LEVEL_3,
        )
        self.fragment_mz_level_kinds = (
            SpectraARTokenKind.FRAGMENT_MZ_LEVEL_0,
            SpectraARTokenKind.FRAGMENT_MZ_LEVEL_1,
            SpectraARTokenKind.FRAGMENT_MZ_LEVEL_2,
            SpectraARTokenKind.FRAGMENT_MZ_LEVEL_3,
        )
        assert len(config.mz_bin_widths) == len(self.fragment_mz_level_kinds)
        self.mz_level_bin_counts = self._level_bin_counts(config.mz_max)
        self.precursor_mz_level_bin_counts = self._level_bin_counts(
            config.precursor_mz_max
        )
        self.prefix_length = 1 + len(config.mz_bin_widths) + 1 + 2
        self.tokens_per_peak = len(config.mz_bin_widths) + 1 + 1
        self.sequence_length = (
            self.prefix_length + config.max_num_peaks * self.tokens_per_peak + 1
        )
        self.num_token_kinds = max(kind.value for kind in SpectraARTokenKind) + 1
        self._starts: dict[SpectraARTokenKind, int] = {}
        self._sizes: dict[SpectraARTokenKind, int] = {}
        offset = 3
        token_spaces: list[tuple[SpectraARTokenKind, int]] = []
        token_spaces.extend(
            zip(
                self.precursor_mz_level_kinds,
                self.precursor_mz_level_bin_counts,
                strict=True,
            )
        )
        token_spaces.extend(
            (
                (SpectraARTokenKind.PRECURSOR_MZ_RESIDUAL, config.residual_bins),
                (SpectraARTokenKind.COLLISION_ENERGY, config.collision_energy_bins),
                (SpectraARTokenKind.CHARGE, config.charge_bins),
            )
        )
        token_spaces.extend(
            zip(
                self.fragment_mz_level_kinds,
                self.mz_level_bin_counts,
                strict=True,
            )
        )
        token_spaces.extend(
            (
                (SpectraARTokenKind.FRAGMENT_MZ_RESIDUAL, config.residual_bins),
                (SpectraARTokenKind.INTENSITY, config.intensity_bins),
            )
        )
        for kind, size in token_spaces:
            self._starts[kind] = offset
            self._sizes[kind] = size
            offset += size
        self.vocab_size = offset

    def _level_bin_counts(self, max_mz: float) -> tuple[int, ...]:
        widths = self.config.mz_bin_widths
        bin_counts = [math.ceil(max_mz / widths[0])]
        for previous_width, width in zip(widths, widths[1:], strict=False):
            ratio = previous_width / width
            rounded = round(ratio)
            assert abs(ratio - rounded) < 1e-6
            bin_counts.append(rounded)
        return tuple(bin_counts)

    def token_id(self, kind: SpectraARTokenKind, value: int = 0) -> int:
        if kind == SpectraARTokenKind.PAD:
            return self.pad_token_id
        if kind == SpectraARTokenKind.BOS:
            return self.bos_token_id
        if kind == SpectraARTokenKind.EOS:
            return self.eos_token_id
        return self._starts[kind] + int(value)

    def token_value(self, token_id: int, kind: SpectraARTokenKind) -> int:
        if kind in self._starts:
            return int(token_id) - self._starts[kind]
        return 0

    def tokenize_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        peak_mz_da = batch["peak_mz"].to(dtype=torch.float32) * self.config.mz_max
        peak_intensity = batch["peak_intensity"].to(dtype=torch.float32)
        peak_valid_mask = batch["peak_valid_mask"].to(dtype=torch.bool)
        precursor_mz_da = (
            batch["precursor_mz"].to(dtype=torch.float32) * self.config.precursor_mz_max
        )
        collision_energy = batch["collision_energy"].to(dtype=torch.float32)
        charge = batch["charge"].to(dtype=torch.float32)

        batch_size = peak_mz_da.shape[0]
        device = peak_mz_da.device
        token_ids = torch.full(
            (batch_size, self.sequence_length),
            self.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        token_kinds = torch.full(
            (batch_size, self.sequence_length),
            int(SpectraARTokenKind.PAD),
            dtype=torch.long,
            device=device,
        )
        loss_mask = torch.zeros(
            (batch_size, self.sequence_length),
            dtype=torch.bool,
            device=device,
        )

        token_ids[:, 0] = self.bos_token_id
        token_kinds[:, 0] = int(SpectraARTokenKind.BOS)

        precursor_levels, precursor_residual = self._mz_code_tensors(
            precursor_mz_da,
            max_mz=self.config.precursor_mz_max,
            level_bin_counts=self.precursor_mz_level_bin_counts,
        )
        for offset, kind in enumerate(self.precursor_mz_level_kinds):
            position = 1 + offset
            token_ids[:, position] = self.token_id(kind) + precursor_levels[:, offset]
            token_kinds[:, position] = int(kind)
        precursor_residual_position = 1 + len(self.precursor_mz_level_kinds)
        token_ids[:, precursor_residual_position] = (
            self.token_id(SpectraARTokenKind.PRECURSOR_MZ_RESIDUAL)
            + precursor_residual
        )
        token_kinds[:, precursor_residual_position] = int(
            SpectraARTokenKind.PRECURSOR_MZ_RESIDUAL
        )

        collision_position = precursor_residual_position + 1
        token_ids[:, collision_position] = (
            self.token_id(SpectraARTokenKind.COLLISION_ENERGY)
            + self._unit_interval_codes(
                collision_energy,
                self.config.collision_energy_bins,
            )
        )
        token_kinds[:, collision_position] = int(SpectraARTokenKind.COLLISION_ENERGY)

        charge_position = collision_position + 1
        token_ids[:, charge_position] = (
            self.token_id(SpectraARTokenKind.CHARGE)
            + charge.to(dtype=torch.long).clamp(0, self.config.charge_bins - 1)
        )
        token_kinds[:, charge_position] = int(SpectraARTokenKind.CHARGE)

        peak_count = min(int(self.config.max_num_peaks), int(peak_mz_da.shape[1]))
        peak_order = torch.argsort(
            torch.where(
                peak_valid_mask,
                peak_mz_da,
                torch.full_like(peak_mz_da, float("-inf")),
            ),
            dim=1,
            descending=True,
            stable=True,
        )[:, :peak_count]
        ordered_mz = torch.gather(peak_mz_da, 1, peak_order)
        ordered_intensity = torch.gather(peak_intensity, 1, peak_order)
        ordered_valid = torch.gather(peak_valid_mask, 1, peak_order)
        fragment_levels, fragment_residual = self._mz_code_tensors(
            ordered_mz,
            max_mz=self.config.mz_max,
            level_bin_counts=self.mz_level_bin_counts,
        )
        peak_offsets = torch.arange(peak_count, device=device) * self.tokens_per_peak
        for offset, kind in enumerate(self.fragment_mz_level_kinds):
            positions = self.prefix_length + peak_offsets + offset
            token_ids[:, positions] = torch.where(
                ordered_valid,
                self.token_id(kind) + fragment_levels[:, :, offset],
                self.pad_token_id,
            )
            token_kinds[:, positions] = torch.where(
                ordered_valid,
                torch.full_like(fragment_levels[:, :, offset], int(kind)),
                torch.full_like(fragment_levels[:, :, offset], int(SpectraARTokenKind.PAD)),
            )
            loss_mask[:, positions] = ordered_valid

        residual_positions = (
            self.prefix_length + peak_offsets + len(self.fragment_mz_level_kinds)
        )
        token_ids[:, residual_positions] = torch.where(
            ordered_valid,
            self.token_id(SpectraARTokenKind.FRAGMENT_MZ_RESIDUAL) + fragment_residual,
            self.pad_token_id,
        )
        token_kinds[:, residual_positions] = torch.where(
            ordered_valid,
            torch.full_like(fragment_residual, int(SpectraARTokenKind.FRAGMENT_MZ_RESIDUAL)),
            torch.full_like(fragment_residual, int(SpectraARTokenKind.PAD)),
        )
        loss_mask[:, residual_positions] = ordered_valid

        intensity_positions = residual_positions + 1
        token_ids[:, intensity_positions] = torch.where(
            ordered_valid,
            self.token_id(SpectraARTokenKind.INTENSITY)
            + self._unit_interval_codes(ordered_intensity, self.config.intensity_bins),
            self.pad_token_id,
        )
        token_kinds[:, intensity_positions] = torch.where(
            ordered_valid,
            torch.full_like(fragment_residual, int(SpectraARTokenKind.INTENSITY)),
            torch.full_like(fragment_residual, int(SpectraARTokenKind.PAD)),
        )
        loss_mask[:, intensity_positions] = ordered_valid

        row_indices = torch.arange(batch_size, device=device)
        eos_positions = (
            self.prefix_length
            + ordered_valid.sum(dim=1).to(dtype=torch.long) * self.tokens_per_peak
        )
        token_ids[row_indices, eos_positions] = self.eos_token_id
        token_kinds[row_indices, eos_positions] = int(SpectraARTokenKind.EOS)
        loss_mask[row_indices, eos_positions] = True

        return {
            "input_token_ids": token_ids[:, :-1],
            "input_token_kinds": token_kinds[:, :-1],
            "target_token_ids": token_ids[:, 1:],
            "target_token_kinds": token_kinds[:, 1:],
            "target_loss_mask": loss_mask[:, 1:],
        }

    def decode_fragment_mz(
        self,
        level_0_value: int,
        level_1_value: int,
        level_2_value: int,
        level_3_value: int,
        residual_value: int,
    ) -> float:
        level_values = (
            level_0_value,
            level_1_value,
            level_2_value,
            level_3_value,
        )
        mz_da = sum(
            value * width
            for value, width in zip(level_values, self.config.mz_bin_widths, strict=True)
        )
        return mz_da + (
            residual_value * self.config.mz_bin_widths[-1] / self.config.residual_bins
        )

    def _mz_code_tensors(
        self,
        mz_da: torch.Tensor,
        *,
        max_mz: float,
        level_bin_counts: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mz_da = mz_da.clamp(0.0, max_mz - 1e-6)
        base_mz = torch.zeros_like(mz_da)
        levels = []
        for width, bins in zip(
            self.config.mz_bin_widths,
            level_bin_counts,
            strict=True,
        ):
            value = torch.floor((mz_da - base_mz) / width).to(dtype=torch.long)
            value = value.clamp(0, bins - 1)
            levels.append(value)
            base_mz = base_mz + value.to(dtype=mz_da.dtype) * width
        residual_fraction = (mz_da - base_mz) / self.config.mz_bin_widths[-1]
        residual = torch.floor(
            residual_fraction.clamp_min(0.0) * self.config.residual_bins
        ).to(dtype=torch.long)
        residual = residual.clamp(0, self.config.residual_bins - 1)
        return torch.stack(levels, dim=-1), residual

    def _unit_interval_codes(self, values: torch.Tensor, bins: int) -> torch.Tensor:
        return torch.floor(values * bins).to(dtype=torch.long).clamp(0, bins - 1)


class SpectraARGemsBatchCollator:
    def __init__(
        self,
        *,
        tokenizer: SpectraARTokenizer,
        num_peaks: int,
        max_precursor_mz: float,
        min_peak_intensity: float,
        peak_drop_min_intensity: float,
        peak_ordering: str,
        precursor_peak_exclusion_window_da: float,
        output_format: str = "torch",
    ) -> None:
        self.tokenizer = tokenizer
        self.output_format = output_format
        self.peak_collator = GemsBatchCollator(
            augment=False,
            num_target_blocks=1,
            context_fraction=1.0,
            target_fraction=0.0,
            block_min_len=1,
            num_peaks=num_peaks,
            max_precursor_mz=max_precursor_mz,
            min_peak_intensity=min_peak_intensity,
            peak_drop_min_intensity=peak_drop_min_intensity,
            peak_ordering=peak_ordering,
            precursor_peak_exclusion_window_da=precursor_peak_exclusion_window_da,
            output_format="torch",
        )

    def __call__(
        self,
        samples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        batch = self.tokenizer.tokenize_batch(self.peak_collator(samples))
        return format_batch(batch, self.output_format)


class SpectraARGemsDataModule(GemsDataModule):
    ar_tokenizer_config: SpectraARTokenizerConfig
    ar_tokenizer: SpectraARTokenizer

    def __init__(
        self,
        config: config_dict.ConfigDict,
        seed: int,
        *,
        distributed_world_size: int = 1,
        distributed_rank: int = 0,
        distributed_local_rank: int | None = None,
    ) -> None:
        self.ar_tokenizer_config = SpectraARTokenizerConfig.from_config(config)
        self.ar_tokenizer = SpectraARTokenizer(self.ar_tokenizer_config)
        super().__init__(
            config,
            seed,
            distributed_world_size=distributed_world_size,
            distributed_rank=distributed_rank,
            distributed_local_rank=distributed_local_rank,
        )

    def _info(self) -> dict[str, Any]:
        info = super()._info()
        info.update(
            {
                "ar_vocab_size": self.ar_tokenizer.vocab_size,
                "ar_sequence_length": self.ar_tokenizer.sequence_length,
                "ar_prefix_length": self.ar_tokenizer.prefix_length,
                "ar_tokens_per_peak": self.ar_tokenizer.tokens_per_peak,
                "ar_mz_axis_order": "descending",
                "ar_mz_bin_widths": list(self.ar_tokenizer_config.mz_bin_widths),
                "ar_residual_bins": self.ar_tokenizer_config.residual_bins,
            }
        )
        return info

    def _collator(self, *, augment: bool) -> SpectraARGemsBatchCollator:
        del augment
        return SpectraARGemsBatchCollator(
            tokenizer=self.ar_tokenizer,
            num_peaks=self.num_peaks_output,
            max_precursor_mz=self.max_precursor_mz,
            min_peak_intensity=self.min_peak_intensity,
            peak_drop_min_intensity=self.peak_drop_min_intensity,
            peak_ordering=self.peak_ordering,
            precursor_peak_exclusion_window_da=self.precursor_peak_exclusion_window_da,
            output_format=self.dataloader_output_format,
        )
