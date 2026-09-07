import hashlib
from pathlib import Path
from typing import Any

from ml_collections import config_dict
from torch.utils.data import DataLoader
from torch import Generator

from spectra_learning.data.gems.artifacts import (
    resolve_gems_hdf5_artifact,
)
from spectra_learning.data.gems.collate import (
    ACQUISITION_METADATA_KEYS,
    GemsBatchCollator,
    SJEPABatchCollator,
)
from spectra_learning.models.spectrum_metadata import MASSIVE_V2_ACQUISITION_SCHEMA
from spectra_learning.data.gems.eligibility import (
    massive_v2_eligibility_contract,
)
from spectra_learning.data.gems.hdf5 import MassiveV2Hdf5ShardDataset
from spectra_learning.data.gems.randomness import data_seed, INPUT_RANDOMNESS_VERSION
from spectra_learning.data.gems.sampling import (
    ChunkedDistributedBatchSampler,
    LimitBatchSampler,
    OffsetBatchSampler,
)
from spectra_learning.data.gems.settings import GemsDataConfig
from spectra_learning.data.spectra import (
    NUM_PEAKS_INPUT,
    PEAK_MZ_MAX,
    PEAK_MZ_MIN,
)


class GemsDataModule:
    config: GemsDataConfig
    seed: int
    output_dir: Path
    gems_base_dir: Path
    distributed_world_size: int
    distributed_rank: int
    distributed_local_rank: int
    gems_dir: Path
    gems_manifest: Path
    info: dict[str, Any]
    train_steps: int
    global_batch_size: int
    batch_size: int
    gradient_accumulation_steps: int
    drop_remainder: bool
    min_precursor_mz: float
    max_precursor_mz: float
    min_peak_intensity: float
    peak_drop_min_intensity: float
    peak_ordering: str
    precursor_peak_exclusion_window_da: float
    jepa_num_target_blocks: int
    jepa_context_fraction: float
    jepa_target_fraction: float
    jepa_block_min_len: int
    jepa_mask_strategy: str | tuple[str, ...]
    jepa_mask_lengths: tuple[int, ...]
    jepa_mask_round_from: int
    jepa_intensity_aware_mask_config: dict[str, float]
    jepa_gap_aware_bucket_edges_da: tuple[float, ...]
    jepa_gap_aware_bucket_weights: tuple[float, ...]
    jepa_allow_target_overlap: bool
    num_peaks_output: int
    dataloader_pin_memory: bool
    dataloader_num_workers: int
    dataloader_prefetch_factor: int
    dataloader_persistent_workers: bool
    dataloader_multiprocessing_context: str
    dataloader_output_format: str
    gems_train_shards: list[str]
    gems_validation_shards: list[str]
    gems_train_files: list[str]
    gems_validation_files: list[str]

    def __init__(
        self,
        config: config_dict.ConfigDict,
        seed: int,
        *,
        distributed_world_size: int = 1,
        distributed_rank: int = 0,
        distributed_local_rank: int | None = None,
    ) -> None:
        if not 0 <= distributed_rank < distributed_world_size:
            raise ValueError("distributed_rank must be between zero and world_size - 1")
        distributed_local_rank = (
            distributed_rank
            if distributed_local_rank is None
            else distributed_local_rank
        )
        self.config = GemsDataConfig.from_config(config)
        self.seed = seed
        self.output_dir = self.config.artifact_dir
        self.gems_base_dir = self.output_dir / "gems"
        self.distributed_world_size = distributed_world_size
        self.distributed_rank = distributed_rank
        self.distributed_local_rank = distributed_local_rank
        self._set_public_config_attrs()
        self._set_distributed_batch_attrs()
        self.artifact = resolve_gems_hdf5_artifact(
            gems_base_dir=self.gems_base_dir,
            repo_id=self.config.gems_hdf5_repo_id,
            revision=self.config.gems_hdf5_revision,
            manifest_filename=self.config.gems_hdf5_manifest,
            distributed_world_size=distributed_world_size,
            seed=self.seed,
            global_batch_size=self.global_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            rows_per_block=self.config.gems_hdf5_rows_per_block,
            drop_remainder=self.drop_remainder,
            training_max_steps=self.config.training_max_steps,
            val_num_steps=self.config.val_num_steps,
        )
        self.gems_manifest = self.artifact.manifest_path
        self.gems_dir = self.gems_manifest.parent
        self._validate_massive_v2_contract()
        self._datasets = {
            "train": self._build_massive_v2_dataset("train"),
            "validation": self._build_massive_v2_dataset("validation"),
        }
        self.gems_train_shards = list(self._datasets["train"].paths)
        self.gems_validation_shards = list(self._datasets["validation"].paths)
        self.gems_train_files = list(self.gems_train_shards)
        self.gems_validation_files = list(self.gems_validation_shards)
        self.info = self._info()
        self.train_steps = self._train_steps()
        self._val_loader: DataLoader | None = None

    def _set_public_config_attrs(self) -> None:
        for key, value in self.config.__dict__.items():
            if key == "num_peaks":
                self.num_peaks_output = value
            else:
                setattr(self, key, value)

    def _set_distributed_batch_attrs(self) -> None:
        self.gradient_accumulation_steps = self.config.gradient_accumulation_steps
        self.global_batch_size = self.batch_size
        denominator = self.distributed_world_size * self.gradient_accumulation_steps
        assert self.global_batch_size % denominator == 0
        self.batch_size = self.global_batch_size // denominator

    def _build_massive_v2_dataset(
        self,
        split: str,
    ) -> MassiveV2Hdf5ShardDataset:
        shards = (
            self.artifact.train_shards(self.distributed_rank)
            if split == "train"
            else self.artifact.validation_shards(self.distributed_rank)
        )
        return MassiveV2Hdf5ShardDataset(
            self.gems_manifest,
            shards,
            spectrum_dataset=self.config.gems_hdf5_spectrum_dataset,
            precursor_dataset=self.config.gems_hdf5_precursor_dataset,
            split=split,
            cache_blocks=self.config.gems_hdf5_shuffle_window_blocks,
            rows_per_block=self.config.gems_hdf5_rows_per_block or None,
            metadata_keys=("collision_energy", "charge")
            + (
                ACQUISITION_METADATA_KEYS
                if self.encoder_metadata_schema == MASSIVE_V2_ACQUISITION_SCHEMA
                else ()
            ),
        )

    def _validate_massive_v2_contract(self) -> None:
        manifest = self.artifact.manifest
        if self.encoder_metadata_schema is not None:
            if manifest["spectrum_metadata"]["schema"] != self.encoder_metadata_schema:
                raise ValueError("MassIVE v2 spectrum metadata schema mismatch")
        eligibility = manifest["eligibility"]
        expected_eligibility = massive_v2_eligibility_contract()
        if eligibility != expected_eligibility:
            raise ValueError(
                "MassIVE v2 eligibility contract mismatch: "
                f"expected {expected_eligibility}, got {eligibility}"
            )
        if (
            self.min_precursor_mz != eligibility["min_precursor_mz"]
            or self.max_precursor_mz != eligibility["max_precursor_mz"]
        ):
            raise ValueError(
                "MassIVE v2 precursor bounds do not match the artifact "
                "eligibility contract"
            )
        intensity_threshold = max(
            self.min_peak_intensity,
            self.peak_drop_min_intensity,
        )
        if intensity_threshold > eligibility["min_relative_peak_intensity"]:
            raise ValueError(
                "MassIVE v2 runtime peak intensity threshold is stricter "
                "than the artifact eligibility contract"
            )
        if (
            self.precursor_peak_exclusion_window_da
            > eligibility["precursor_peak_exclusion_window_da"]
        ):
            raise ValueError(
                "MassIVE v2 runtime precursor exclusion window is stricter "
                "than the artifact eligibility contract"
            )
        minimum_peaks = (
            self.jepa_num_target_blocks * self.jepa_block_min_len + 1
            if self.training_task == "pretrain"
            else 1
        )
        if minimum_peaks > eligibility["min_usable_peaks"]:
            raise ValueError(
                "MassIVE v2 masking requires more usable peaks than the "
                "artifact eligibility contract guarantees"
            )
        if self.num_peaks_output < minimum_peaks:
            raise ValueError(
                "Configured output peak count cannot provide all context "
                "and target masks"
            )
        datasets = manifest["datasets"]
        expected_datasets = {
            "spectrum": self.config.gems_hdf5_spectrum_dataset,
            "precursor_mz": self.config.gems_hdf5_precursor_dataset,
            "retention_time": self.config.gems_hdf5_retention_time_dataset,
            "ms_level": self.config.gems_hdf5_ms_level_dataset,
        }
        if datasets != expected_datasets:
            raise ValueError(
                "MassIVE v2 dataset contract mismatch: "
                f"expected {expected_datasets}, got {datasets}"
            )
        expected_split = {
            "version": "entity_hash_v1",
            "algorithm": "splitmix64",
            "seed": 42,
            "modulus": 20,
            "validation_remainder": 0,
            "assigned_entity_key": ["massive_id", "global_group_id"],
            "unassigned_entity_key": [
                "massive_id",
                "source_row_index",
            ],
        }
        if manifest["split"] != expected_split:
            raise ValueError(
                "MassIVE v2 split contract mismatch: "
                f"expected {expected_split}, got {manifest['split']}"
            )
        grouping = manifest["grouping"]
        expected_grouping = {
            "assigned_rule": "group_id >= 0",
            "corpus_entity_key": ["massive_id", "global_group_id"],
            "raw_global_group_id_formula": ("file_id * 2**32 + group_id"),
        }
        actual_grouping = {key: grouping[key] for key in expected_grouping}
        if actual_grouping != expected_grouping:
            raise ValueError(
                "MassIVE v2 grouping contract mismatch: "
                f"expected {expected_grouping}, got {actual_grouping}"
            )

    def _info(self) -> dict[str, Any]:
        manifest = self.artifact.manifest
        train_split = manifest["splits"]["train"]
        validation_split = manifest["splits"]["validation"]
        return {
            "gems_input_contract": {
                "version": INPUT_RANDOMNESS_VERSION,
                "seed": self.seed,
                "shuffle_seed": self.gems_hdf5_shuffle_seed,
                "window_blocks": self.gems_hdf5_shuffle_window_blocks,
                "rows_per_block": self.gems_hdf5_rows_per_block,
            },
            "artifact_dir": str(self.output_dir),
            "gems_dir": str(self.gems_dir),
            "gems_manifest": str(self.gems_manifest),
            "gems_manifest_sha256": hashlib.sha256(
                self.gems_manifest.read_bytes()
            ).hexdigest(),
            "gems_hdf5_format": self.artifact.format,
            "gems_hdf5_repo_id": self.config.gems_hdf5_repo_id,
            "gems_hdf5_revision": self.config.gems_hdf5_revision,
            "gems_shard_plan_sha256": self.artifact.plan_sha256,
            "gems_split": manifest["split"],
            "gems_eligibility": manifest["eligibility"],
            "source_size": int(train_split["rows"]) + int(validation_split["rows"]),
            "train_size": int(train_split["eligible_rows"]),
            "validation_size": int(validation_split["eligible_rows"]),
            "local_train_size": len(self._datasets["train"]),
            "local_validation_size": len(self._datasets["validation"]),
            "local_train_shards": len(self.gems_train_shards),
            "local_validation_shards": len(self.gems_validation_shards),
            "planned_download_bytes": sum(
                shard.bytes
                for shard in (
                    *self.artifact.train_shards(self.distributed_rank),
                    *self.artifact.validation_shards(self.distributed_rank),
                )
            ),
            "num_peaks_input": NUM_PEAKS_INPUT,
            "num_peaks": self.num_peaks_output,
            "min_precursor_mz": self.min_precursor_mz,
            "max_precursor_mz": self.max_precursor_mz,
            "peak_mz_min": PEAK_MZ_MIN,
            "peak_mz_max": PEAK_MZ_MAX,
        }

    def _train_steps(self) -> int:
        rows = min(
            sum(shard.eligible_rows for shard in self.artifact.train_shards(rank))
            for rank in range(self.distributed_world_size)
        )
        micro_batches = (
            rows // self.batch_size
            if self.drop_remainder
            else (rows + self.batch_size - 1) // self.batch_size
        )
        return micro_batches // self.gradient_accumulation_steps

    def _get_dataset(
        self,
        split: str,
    ) -> MassiveV2Hdf5ShardDataset:
        return self._datasets[split]

    def _massive_v2_assignment_segments(
        self,
        split: str,
        rank: int,
    ) -> list[tuple[int, int, int]]:
        shards = (
            self.artifact.train_shards(rank)
            if split == "train"
            else self.artifact.validation_shards(rank)
        )
        start = 0
        segments = []
        for shard in shards:
            segments.append((start, shard.eligible_rows, shard.chunk_rows))
            start += shard.eligible_rows
        return segments

    def _make_batch_sampler(
        self,
        *,
        shuffle: bool,
        seed: int,
        drop_last: bool,
        epoch: int,
        split: str,
        rank: int | None = None,
    ) -> ChunkedDistributedBatchSampler:
        rank = self.distributed_rank if rank is None else rank
        partition_batches = split == "validation"
        if partition_batches:
            sampler_world_size, sampler_rank = (
                self.artifact.validation_sampler_partition(rank)
            )
        else:
            sampler_world_size, sampler_rank = 1, 0
        sampler = ChunkedDistributedBatchSampler(
            self._massive_v2_assignment_segments(split, rank),
            batch_size=self.batch_size,
            rows_per_block=self.config.gems_hdf5_rows_per_block or None,
            shuffle=shuffle,
            seed=(
                self.gems_hdf5_shuffle_seed
                if split == "train" and self.gems_hdf5_shuffle_seed is not None
                else seed
            ),
            drop_last=drop_last,
            world_size=sampler_world_size,
            rank=sampler_rank,
            partition_batches=partition_batches,
            shuffle_window_blocks=self.gems_hdf5_shuffle_window_blocks,
        )
        if split == "train":
            sampler.seed = data_seed(sampler.seed, "training-partition", rank)
        sampler.set_epoch(epoch)
        return sampler

    def _make_loader(
        self,
        *,
        augment: bool,
        shuffle: bool,
        seed: int,
        drop_last: bool,
        start_batch: int = 0,
        epoch: int = 0,
        num_workers: int | None = None,
        max_batches: int | None = None,
        split: str,
    ) -> DataLoader:
        resolved_num_workers = (
            self.dataloader_num_workers if num_workers is None else num_workers
        )
        batch_sampler = self._make_batch_sampler(
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
            epoch=epoch,
            split=split,
        )
        start_index = start_batch * self.batch_size * self.gradient_accumulation_steps
        if start_index:
            batch_sampler = OffsetBatchSampler(
                batch_sampler,
                start_index=start_index,
                batch_size=self.batch_size,
                drop_last=drop_last,
            )
        if max_batches is not None:
            batch_sampler = LimitBatchSampler(
                batch_sampler,
                max_batches=max_batches,
            )
        return self._loader_for_sampler(
            split=split,
            batch_sampler=batch_sampler,
            collator=self._collator(
                augment=augment,
                metadata_dropout=split == "train",
                epoch=epoch,
                split=split,
            ),
            num_workers=resolved_num_workers,
            epoch=epoch,
        )

    def _loader_for_sampler(
        self,
        *,
        split: str,
        batch_sampler: Any,
        collator: Any,
        num_workers: int | None = None,
        epoch: int = 0,
    ) -> DataLoader:
        workers = self.dataloader_num_workers if num_workers is None else num_workers
        dataset = self._get_dataset(split)
        if workers > 0:
            dataset.close()
        kwargs: dict[str, Any] = {
            "dataset": dataset,
            # The sampler yields a complete batch request, not a single row.
            "sampler": batch_sampler,
            "batch_size": None,
            "num_workers": workers,
            "pin_memory": self.dataloader_pin_memory,
            "collate_fn": collator,
            "generator": Generator().manual_seed(
                data_seed(
                    self.seed, "worker-bootstrap", split, epoch, self.distributed_rank
                )
            ),
        }
        if workers > 0:
            kwargs["persistent_workers"] = self.dataloader_persistent_workers
            kwargs["prefetch_factor"] = self.dataloader_prefetch_factor
            if self.dataloader_multiprocessing_context:
                kwargs["multiprocessing_context"] = (
                    self.dataloader_multiprocessing_context
                )
        return DataLoader(**kwargs)

    def _collator(
        self,
        *,
        augment: bool,
        metadata_dropout: bool = False,
        epoch: int = 0,
        split: str = "train",
    ) -> GemsBatchCollator:
        collator_type = (
            SJEPABatchCollator if self.training_mode == "sjepa" else GemsBatchCollator
        )
        sjepa_kwargs = (
            {
                "num_global_views": self.sjepa_num_global_views,
                "num_local_views": self.sjepa_num_local_views,
                "global_view_fraction": self.sjepa_global_view_fraction,
                "local_view_fraction": self.sjepa_local_view_fraction,
            }
            if self.training_mode == "sjepa"
            else {}
        )
        return collator_type(
            augmentation_seed=self.seed if split == "train" else 42,
            augmentation_epoch=epoch,
            augmentation_split=split,
            augment=augment,
            num_target_blocks=self.jepa_num_target_blocks,
            context_fraction=self.jepa_context_fraction,
            target_fraction=self.jepa_target_fraction,
            block_min_len=self.jepa_block_min_len,
            mask_strategy=self.jepa_mask_strategy,
            mask_lengths=self.jepa_mask_lengths,
            mask_round_from=self.jepa_mask_round_from,
            intensity_aware_mask_config=self.jepa_intensity_aware_mask_config,
            gap_aware_bucket_edges_da=self.jepa_gap_aware_bucket_edges_da,
            gap_aware_bucket_weights=self.jepa_gap_aware_bucket_weights,
            allow_target_overlap=self.jepa_allow_target_overlap,
            num_peaks=self.num_peaks_output,
            max_precursor_mz=self.max_precursor_mz,
            min_peak_intensity=self.min_peak_intensity,
            peak_drop_min_intensity=self.peak_drop_min_intensity,
            peak_ordering=self.peak_ordering,
            precursor_peak_exclusion_window_da=self.precursor_peak_exclusion_window_da,
            output_format=self.dataloader_output_format,
            spectrum_metadata_schema=self.encoder_metadata_schema,
            spectrum_metadata_dropout_probability=(
                self.spectrum_metadata_dropout_probability if metadata_dropout else 0.0
            ),
            **sjepa_kwargs,
        )

    def set_mask_fractions(
        self,
        context_fraction: float,
        target_fraction: float,
    ) -> None:
        self.jepa_context_fraction = context_fraction
        self.jepa_target_fraction = target_fraction
        self._val_loader = None

    def set_gradient_accumulation_steps(self, steps: int) -> None:
        self.gradient_accumulation_steps = steps
        self.batch_size = self.global_batch_size // (
            self.distributed_world_size * steps
        )
        self.train_steps = self._train_steps()
        self._val_loader = None

    @property
    def val_loader(self) -> DataLoader:
        if self._val_loader is None:
            self._val_loader = self.val_loader_for_eval(augment=True)
        return self._val_loader

    def val_loader_for_eval(self, *, augment: bool) -> DataLoader:
        return self._make_loader(
            augment=augment,
            shuffle=True,
            seed=42,
            drop_last=False,
            split="validation",
        )

    def train_loader_for_epoch(self, epoch: int, start_batch: int = 0) -> DataLoader:
        max_batches = max(
            0,
            (self.train_steps - start_batch) * self.gradient_accumulation_steps,
        )
        return self._make_loader(
            augment=True,
            shuffle=True,
            seed=self.seed,
            drop_last=self.drop_remainder,
            start_batch=start_batch,
            epoch=epoch,
            max_batches=max_batches,
            split="train",
        )
