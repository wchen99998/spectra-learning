from pathlib import Path
from typing import Any, NamedTuple, cast
from urllib.parse import quote

import numpy as np
import torch
from ml_collections import config_dict
from torch.utils.data import DataLoader, Dataset, Subset

from spectra_learning.data.contracts import peak_preprocessing_contract
from spectra_learning.data.gems.conversion import batch_to_numpy, format_batch
from spectra_learning.data.loading import (
    loader_sampler,
    local_batch_size,
    subset_for_max_samples,
)
from spectra_learning.data.massspec_targets import REGRESSION_TARGET_KEYS
from spectra_learning.data.murcko import (
    NIST_MURCKO_HF_REPO,
    NIST_MURCKO_HF_REVISION,
    NIST_MURCKO_PREPARED_SUBDIR,
    ensure_nist_murcko_probe_downloaded,
    precursor_charge_from_metadata_json,
)
from spectra_learning.data.spectra import (
    COLLISION_ENERGY_MAX,
    DEFAULT_GROUPED_PEAK_ISOTOPE_CHARGES,
    DEFAULT_GROUPED_PEAK_SHOULDER_DA,
    DEFAULT_PEAK_FILTERING,
    canonicalize_precursor_charge_torch,
    preprocess_peak_batch_numpy,
    preprocess_peak_batch_torch,
    spectra_from_peak_lists,
)
from spectra_learning.config.msg_probe import validate_msg_probe_config

_DEFAULT_BATCH_SIZE = 512
_DEFAULT_SHUFFLE_BUFFER = 10_000
_DEFAULT_ARTIFACT_DIR = Path("data/gems_artifacts")
_FINGERPRINT_BITS = 1024


def massspec_source_cache_dir(
    artifact_root: Path,
    repo_id: str,
    revision: str,
) -> Path:
    return (
        artifact_root
        / "huggingface"
        / f"repo={quote(repo_id, safe='')}"
        / f"revision={quote(revision, safe='')}"
    )


class _ProbeParquetDataset(Dataset):
    def __init__(
        self,
        entries: list[dict[str, Any]],
        *,
        adduct_vocab: dict[str, int],
        instrument_type_vocab: dict[str, int],
        indices: list[int] | np.ndarray | None = None,
    ) -> None:
        self._entries = [
            {
                "path": Path(entry["path"]),
                "length": int(entry["length"]),
                "morgan_files": [Path(path) for path in entry.get("morgan_files", [])],
                "dreams_files": [Path(path) for path in entry.get("dreams_files", [])],
            }
            for entry in entries
        ]
        lengths = np.asarray([entry["length"] for entry in self._entries], dtype=np.int64)
        self._starts = np.zeros(len(lengths) + 1, dtype=np.int64)
        np.cumsum(lengths, out=self._starts[1:])
        self._adduct_vocab = adduct_vocab
        self._instrument_type_vocab = instrument_type_vocab
        self._indices = (
            np.asarray(indices, dtype=np.int64)
            if indices is not None
            else None
        )
        self._position_map = self._build_position_map()
        self._arrays: list[dict[str, np.ndarray]] | None = None

    def subset(self, indices: list[int]) -> "_ProbeParquetDataset":
        base_indices = np.arange(int(self._starts[-1]), dtype=np.int64)
        if self._indices is not None:
            base_indices = self._indices
        return _ProbeParquetDataset(
            self._entries,
            adduct_vocab=self._adduct_vocab,
            instrument_type_vocab=self._instrument_type_vocab,
            indices=base_indices[np.asarray(indices, dtype=np.int64)],
        )

    def _build_position_map(self) -> np.ndarray | None:
        if self._indices is None:
            return None
        counts = np.zeros(len(self._entries), dtype=np.int64)
        position_map = np.empty((len(self._indices), 2), dtype=np.int64)
        for position, index in enumerate(self._indices):
            entry_idx = int(np.searchsorted(self._starts, index, side="right") - 1)
            position_map[position, 0] = entry_idx
            position_map[position, 1] = counts[entry_idx]
            counts[entry_idx] += 1
        return position_map

    def __len__(self) -> int:
        if self._indices is not None:
            return int(len(self._indices))
        return int(self._starts[-1])

    def _entry_local_indices(self, entry_idx: int) -> np.ndarray | None:
        if self._indices is None:
            return None
        assert self._position_map is not None
        selected = self._indices[self._position_map[:, 0] == entry_idx]
        return selected - int(self._starts[entry_idx])

    def _load_entry(
        self,
        entry: dict[str, Any],
        *,
        local_indices: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        import pyarrow.parquet as pq

        path = Path(entry["path"])
        table = pq.read_table(path)
        if local_indices is not None:
            table = table.take(local_indices)
        rows = table.to_pydict()
        n = len(rows["precursor_mz"])
        arrays: dict[str, np.ndarray] = {
            "spectra": spectra_from_peak_lists(
                rows["spectrum_mz"],
                rows["spectrum_intensity"],
            ),
            "precursor_mz_raw": np.asarray(rows["precursor_mz"], dtype=np.float32),
            "fingerprint": np.zeros((n, _FINGERPRINT_BITS), dtype=np.int8),
            "smiles": np.asarray(rows["canonical_smiles"], dtype=str),
            "adduct_id": np.asarray(
                [self._adduct_vocab[value] for value in rows["adduct"]],
                dtype=np.int32,
            ),
            "instrument_type_id": np.asarray(
                [self._instrument_type_vocab[value] for value in rows["instrument_type"]],
                dtype=np.int32,
            ),
            "collision_energy": np.asarray(rows["collision_energy"], dtype=np.float32),
            "collision_energy_present": np.asarray(
                rows["collision_energy_present"],
                dtype=np.int32,
            ),
            "charge": np.asarray(
                [
                    precursor_charge_from_metadata_json(value)
                    for value in rows["metadata_json"]
                ],
                dtype=np.float32,
            ),
            "probe_valid_mol": np.ones(n, dtype=bool),
            "probe_maccs": np.asarray(rows["maccs_166"], dtype=np.int8),
            "probe_fluorine": np.asarray(rows["has_fluorine"], dtype=np.float32),
            "probe_sulfur": np.asarray(rows["has_sulfur"], dtype=np.float32),
        }
        for name in REGRESSION_TARGET_KEYS:
            arrays[f"probe_{name}"] = np.asarray(rows[name], dtype=np.float32)
        morgan_files = entry.get("morgan_files", [])
        if morgan_files:
            morgan = np.concatenate(
                [
                    np.load(Path(path), allow_pickle=False)["morgan"].astype(np.int8)
                    for path in morgan_files
                ],
                axis=0,
            )
            arrays["probe_morgan"] = (
                morgan[local_indices] if local_indices is not None else morgan
            )
        dreams_files = entry.get("dreams_files", [])
        if dreams_files:
            dreams_payloads = [
                np.load(Path(path), allow_pickle=False) for path in dreams_files
            ]
            dreams = np.concatenate(
                [
                    payload["dreams_embedding"].astype(np.float32)
                    for payload in dreams_payloads
                ],
                axis=0,
            )
            dreams_valid = np.concatenate(
                [
                    payload["dreams_embedding_valid"].astype(bool)
                    for payload in dreams_payloads
                ],
                axis=0,
            )
            spectrum_index = np.concatenate(
                [
                    payload["spectrum_index"].astype(np.int64)
                    for payload in dreams_payloads
                ],
                axis=0,
            )
            if local_indices is not None:
                dreams = dreams[local_indices]
                dreams_valid = dreams_valid[local_indices]
                spectrum_index = spectrum_index[local_indices]
            assert dreams.shape[0] == n
            assert np.array_equal(spectrum_index, np.asarray(rows["spectrum_index"], dtype=np.int64))
            arrays["dreams_embedding"] = dreams
            arrays["dreams_embedding_valid"] = dreams_valid
        return arrays

    def _ensure_arrays(self) -> list[dict[str, np.ndarray]]:
        if self._arrays is not None:
            return self._arrays
        self._arrays = [
            self._load_entry(entry, local_indices=self._entry_local_indices(entry_idx))
            for entry_idx, entry in enumerate(self._entries)
        ]
        return self._arrays

    def __getitem__(self, index: int) -> dict[str, Any]:
        arrays_by_entry = self._ensure_arrays()
        if self._indices is None:
            entry_idx = int(np.searchsorted(self._starts, index, side="right") - 1)
            local_idx = index - int(self._starts[entry_idx])
        else:
            assert self._position_map is not None
            entry_idx = int(self._position_map[index, 0])
            local_idx = int(self._position_map[index, 1])
        arrays = arrays_by_entry[entry_idx]
        sample: dict[str, Any] = {}
        for key, value in arrays.items():
            item = value[local_idx]
            if isinstance(item, np.ndarray):
                sample[key] = item.copy()
            elif np.isscalar(item):
                if value.dtype.kind in {"U", "S"}:
                    sample[key] = str(item)
                elif value.dtype == np.bool_:
                    sample[key] = bool(item)
                elif value.dtype.kind in {"i", "u"}:
                    sample[key] = int(cast(Any, item))
                else:
                    sample[key] = float(cast(Any, item))
            else:
                sample[key] = item
        return sample


class _ProbeBatchCollator:
    def __init__(
        self,
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
        output_format: str = "torch",
    ) -> None:
        self.num_peaks = num_peaks
        self.max_precursor_mz = max_precursor_mz
        self.min_peak_intensity = min_peak_intensity
        self.peak_drop_min_intensity = peak_drop_min_intensity
        self.peak_ordering = peak_ordering
        self.precursor_peak_exclusion_window_da = precursor_peak_exclusion_window_da
        self.peak_filtering = peak_filtering
        self.grouped_peak_shoulder_da = grouped_peak_shoulder_da
        self.grouped_peak_isotope_charges = grouped_peak_isotope_charges
        self.output_format = output_format

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        batch = self._preprocess(samples)
        batch["fingerprint"] = torch.stack(
            [
                torch.as_tensor(sample["fingerprint"], dtype=torch.int32)
                for sample in samples
            ],
            dim=0,
        ).to(torch.int32)
        batch["smiles"] = [str(sample["smiles"]) for sample in samples]
        batch["adduct_id"] = torch.tensor(
            [int(sample["adduct_id"]) for sample in samples], dtype=torch.int32
        )
        batch["instrument_type_id"] = torch.tensor(
            [int(sample["instrument_type_id"]) for sample in samples], dtype=torch.int32
        )
        batch["collision_energy"] = torch.tensor(
            [float(sample["collision_energy"]) for sample in samples],
            dtype=torch.float32,
        ).clamp(0.0, COLLISION_ENERGY_MAX) / COLLISION_ENERGY_MAX
        batch["charge"] = canonicalize_precursor_charge_torch(
            torch.tensor(
                [float(sample["charge"]) for sample in samples],
                dtype=torch.float32,
            )
        )
        batch["collision_energy_present"] = torch.tensor(
            [int(sample["collision_energy_present"]) for sample in samples],
            dtype=torch.int32,
        )
        batch["probe_valid_mol"] = torch.tensor(
            [bool(sample["probe_valid_mol"]) for sample in samples],
            dtype=torch.bool,
        )
        batch["probe_maccs"] = torch.stack(
            [
                torch.as_tensor(sample["probe_maccs"], dtype=torch.int32)
                for sample in samples
            ],
            dim=0,
        ).to(torch.int32)
        if "probe_fluorine" in samples[0]:
            batch["probe_fluorine"] = torch.tensor(
                [float(sample["probe_fluorine"]) for sample in samples],
                dtype=torch.float32,
            )
        if "probe_sulfur" in samples[0]:
            batch["probe_sulfur"] = torch.tensor(
                [float(sample["probe_sulfur"]) for sample in samples],
                dtype=torch.float32,
            )
        if "probe_morgan" in samples[0]:
            batch["probe_morgan"] = torch.stack(
                [
                    torch.as_tensor(sample["probe_morgan"], dtype=torch.int32)
                    for sample in samples
                ],
                dim=0,
            ).to(torch.int32)
        for name in REGRESSION_TARGET_KEYS:
            batch[f"probe_{name}"] = torch.tensor(
                [float(sample[f"probe_{name}"]) for sample in samples],
                dtype=torch.float32,
            )
        if "dreams_embedding" in samples[0]:
            batch["dreams_embedding"] = torch.stack(
                [
                    torch.as_tensor(sample["dreams_embedding"], dtype=torch.float32)
                    for sample in samples
                ],
                dim=0,
            ).to(torch.float32)
        if "dreams_embedding_valid" in samples[0]:
            batch["dreams_embedding_valid"] = torch.tensor(
                [bool(sample["dreams_embedding_valid"]) for sample in samples],
                dtype=torch.bool,
            )
        return format_batch(batch, self.output_format)

    def _preprocess(self, samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if isinstance(samples[0]["spectra"], np.ndarray):
            spectra = np.stack([sample["spectra"] for sample in samples], axis=0)
            precursor_raw = np.asarray(
                [sample["precursor_mz_raw"] for sample in samples],
                dtype=np.float32,
            )
            batch = preprocess_peak_batch_numpy(
                spectra,
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
            return {key: torch.from_numpy(value) for key, value in batch.items()}
        spectra = torch.stack([sample["spectra"] for sample in samples], dim=0)
        precursor_raw = torch.tensor(
            [float(sample["precursor_mz_raw"]) for sample in samples],
            dtype=torch.float32,
        )
        return preprocess_peak_batch_torch(
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


class _LoaderAdapter:
    def __init__(self, loader: DataLoader) -> None:
        self._loader = loader

    def __iter__(self):
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._loader)

    def as_numpy_iterator(self):
        for batch in self._loader:
            yield batch_to_numpy(batch)


class MassSpecProbeData(NamedTuple):
    info: dict[str, Any]
    train_files: list[str]
    train_lengths: list[int]
    train_morgan_files: list[str]
    train_dreams_files: list[str]
    val_files: list[str]
    val_lengths: list[int]
    val_morgan_files: list[str]
    val_dreams_files: list[str]
    test_files: list[str]
    test_lengths: list[int]
    test_morgan_files: list[str]
    test_dreams_files: list[str]
    batch_size: int
    shuffle_buffer: int
    max_precursor_mz: float
    min_peak_intensity: float
    peak_drop_min_intensity: float
    peak_filtering: str
    grouped_peak_shoulder_da: float
    grouped_peak_isotope_charges: tuple[int, ...]
    peak_ordering: str
    num_peaks: int
    dreams_dim: int
    precursor_peak_exclusion_window_da: float
    pairwise_alignment_path: str

    @classmethod
    def from_config(
        cls,
        config: config_dict.ConfigDict,
        *,
        distributed_world_size: int = 1,
        distributed_rank: int = 0,
        distributed_local_rank: int | None = None,
        maccs_only: bool = False,
    ) -> "MassSpecProbeData":
        validate_msg_probe_config(config)
        artifact_root = (
            Path(config.get("artifact_dir", str(_DEFAULT_ARTIFACT_DIR)))
            .expanduser()
            .resolve()
        )
        preprocessing = peak_preprocessing_contract(config)
        min_precursor_mz = float(preprocessing["min_precursor_mz"])
        max_precursor_mz = float(preprocessing["max_precursor_mz"])
        msg_probe_fingerprint = (
            "maccs"
            if maccs_only
            else str(config.get("msg_probe_fingerprint", "maccs")).lower()
        )
        include_morgan_probe = msg_probe_fingerprint == "morgan"
        include_morgan = (not maccs_only) and (
            include_morgan_probe
            or int(config.get("msg_probe_pairwise_alignment_num_pairs", 0)) > 0
        )
        include_dreams = (not maccs_only) and bool(
            config.get("nist_murcko_probe_include_dreams_auxiliary", False)
        )
        murcko_subdir = str(
            config.get(
                "nist_murcko_probe_hf_subdir",
                NIST_MURCKO_PREPARED_SUBDIR,
            )
        ).strip("/")
        nist_repo_id = str(
            config.get("nist_murcko_probe_repo_id", NIST_MURCKO_HF_REPO)
        )
        nist_revision = str(
            config.get(
                "nist_murcko_probe_revision",
                NIST_MURCKO_HF_REVISION,
            )
        )
        nist_cache_dir = massspec_source_cache_dir(
            artifact_root,
            nist_repo_id,
            nist_revision,
        )
        nist_dir = nist_cache_dir / murcko_subdir
        nist_metadata = ensure_nist_murcko_probe_downloaded(
            nist_dir,
            min_precursor_mz=min_precursor_mz,
            max_precursor_mz=max_precursor_mz,
            repo_id=nist_repo_id,
            revision=nist_revision,
            subdir=murcko_subdir,
            include_morgan=include_morgan,
            include_dreams=include_dreams,
            distributed_world_size=distributed_world_size,
            distributed_rank=distributed_rank,
            distributed_local_rank=distributed_local_rank,
        )
        adduct_vocab = dict(nist_metadata.get("adduct_vocab", {"unknown": 0}))
        instrument_type_vocab = dict(
            nist_metadata.get("instrument_type_vocab", {"unknown": 0})
        )
        morgan_files = (
            {
                split: nist_metadata.get("morgan_auxiliary_files", {}).get(split, [])
                for split in ("train", "val", "test")
            }
            if include_morgan
            else {}
        )
        dreams_files = (
            {
                split: nist_metadata.get("dreams_auxiliary_files", {}).get(split, [])
                for split in ("train", "val", "test")
            }
            if include_dreams
            else {}
        )
        train_files = [str(nist_dir / name) for name in nist_metadata["train_files"]]
        val_files = [str(nist_dir / name) for name in nist_metadata["val_files"]]
        test_files = [str(nist_dir / name) for name in nist_metadata["test_files"]]
        info = {
            "massspec_train_size": int(nist_metadata.get("train_size", 0)),
            "massspec_val_size": int(nist_metadata.get("val_size", 0)),
            "massspec_test_size": int(nist_metadata.get("test_size", 0)),
            "massspec_metadata_version": int(nist_metadata.get("metadata_version", 0)),
            "massspec_nist_repo_id": nist_repo_id,
            "massspec_nist_revision": nist_revision,
            "massspec_nist_subdir": murcko_subdir,
            "massspec_nist_cache_dir": str(nist_cache_dir),
            "massspec_nist_source_dir": str(nist_dir),
            "massspec_train_positive": int(
                nist_metadata.get("train_positive", 0)
            ),
            "massspec_val_positive": int(nist_metadata.get("val_positive", 0)),
            "massspec_test_positive": int(nist_metadata.get("test_positive", 0)),
            "massspec_peak_preprocessing": preprocessing,
            "massspec_adduct_vocab": adduct_vocab,
            "massspec_instrument_type_vocab": instrument_type_vocab,
            "massspec_adduct_vocab_size": len(adduct_vocab),
            "massspec_instrument_type_vocab_size": len(instrument_type_vocab),
            "fingerprint_bits": _FINGERPRINT_BITS,
            "probe_maccs_bits": int(nist_metadata.get("probe_maccs_bits", 0)),
            "probe_morgan_bits": int(nist_metadata.get("probe_morgan_bits", 0))
            if include_morgan
            else 0,
            "probe_morgan_radius": int(nist_metadata.get("probe_morgan_radius", 0))
            if include_morgan
            else 0,
            "pairwise_alignment_available": bool(
                include_morgan
                and nist_metadata.get("pairwise_alignment_available", False)
            ),
            "pairwise_alignment_num_pairs": (
                int(nist_metadata.get("pairwise_alignment_num_pairs", 0))
                if include_morgan
                else 0
            ),
            "pairwise_alignment_num_endpoints": (
                int(nist_metadata.get("pairwise_alignment_num_endpoints", 0))
                if include_morgan
                else 0
            ),
            "dreams_auxiliary_available": bool(
                include_dreams
                and nist_metadata.get("dreams_auxiliary_available", False)
            ),
            "dreams_valid_counts": nist_metadata.get("dreams_valid_counts", {}),
            "dreams_invalid_counts": nist_metadata.get("dreams_invalid_counts", {}),
        }
        pairwise_file = (
            str(nist_metadata.get("pairwise_alignment_file", ""))
            if include_morgan
            else ""
        )
        pairwise_alignment_path = str(nist_dir / pairwise_file) if pairwise_file else ""
        return cls(
            info=info,
            train_files=train_files,
            train_lengths=[int(v) for v in nist_metadata["train_lengths"]],
            train_morgan_files=[
                str(nist_dir / name) for name in morgan_files.get("train", [])
            ],
            train_dreams_files=[
                str(nist_dir / name) for name in dreams_files.get("train", [])
            ],
            val_files=val_files,
            val_lengths=[int(v) for v in nist_metadata["val_lengths"]],
            val_morgan_files=[
                str(nist_dir / name) for name in morgan_files.get("val", [])
            ],
            val_dreams_files=[
                str(nist_dir / name) for name in dreams_files.get("val", [])
            ],
            test_files=test_files,
            test_lengths=[int(v) for v in nist_metadata["test_lengths"]],
            test_morgan_files=[
                str(nist_dir / name) for name in morgan_files.get("test", [])
            ],
            test_dreams_files=[
                str(nist_dir / name) for name in dreams_files.get("test", [])
            ],
            batch_size=int(
                config.get(
                    "msg_probe_batch_size",
                    config.get("batch_size", _DEFAULT_BATCH_SIZE),
                )
            ),
            shuffle_buffer=int(
                config.get("shuffle_buffer", _DEFAULT_SHUFFLE_BUFFER)
            ),
            max_precursor_mz=max_precursor_mz,
            min_peak_intensity=float(preprocessing["min_peak_intensity"]),
            peak_drop_min_intensity=float(
                preprocessing["peak_drop_min_intensity"]
            ),
            peak_filtering=str(preprocessing["peak_filtering"]),
            grouped_peak_shoulder_da=float(
                preprocessing["grouped_peak_shoulder_da"]
            ),
            grouped_peak_isotope_charges=tuple(
                int(charge)
                for charge in preprocessing["grouped_peak_isotope_charges"]
            ),
            peak_ordering=str(preprocessing["peak_ordering"]),
            num_peaks=int(preprocessing["num_peaks"]),
            dreams_dim=int(nist_metadata.get("dreams_dim", 0)),
            precursor_peak_exclusion_window_da=float(
                preprocessing["precursor_peak_exclusion_window_da"]
            ),
            pairwise_alignment_path=pairwise_alignment_path,
        )

    def build_dataset(
        self,
        split: str,
        *,
        seed: int = 0,
        peak_ordering: str | None = None,
        shuffle: bool = False,
        drop_remainder: bool = True,
        num_parallel_reads: int | None = None,
        max_samples: int | None = None,
        distributed_world_size: int = 1,
        distributed_rank: int = 0,
        pad_distributed: bool = False,
        output_format: str = "torch",
    ):
        if peak_ordering not in (None, self.peak_ordering):
            raise ValueError(
                "MassSpec peak ordering must match the checkpoint preprocessing "
                f"contract: requested={peak_ordering}, expected={self.peak_ordering}"
            )
        split_files = {
            "massspec_train": self.train_files,
            "massspec_val": self.val_files,
            "massspec_test": self.test_files,
            "train": self.train_files,
            "val": self.val_files,
            "test": self.test_files,
            "all": self.train_files + self.val_files + self.test_files,
        }[split]
        split_lengths = {
            "massspec_train": self.train_lengths,
            "massspec_val": self.val_lengths,
            "massspec_test": self.test_lengths,
            "train": self.train_lengths,
            "val": self.val_lengths,
            "test": self.test_lengths,
            "all": self.train_lengths + self.val_lengths + self.test_lengths,
        }[split]
        split_morgan_files = {
            "massspec_train": [self.train_morgan_files] if self.train_files else [],
            "massspec_val": [self.val_morgan_files] if self.val_files else [],
            "massspec_test": [self.test_morgan_files] if self.test_files else [],
            "train": [self.train_morgan_files] if self.train_files else [],
            "val": [self.val_morgan_files] if self.val_files else [],
            "test": [self.test_morgan_files] if self.test_files else [],
            "all": (
                ([self.train_morgan_files] if self.train_files else [])
                + ([self.val_morgan_files] if self.val_files else [])
                + ([self.test_morgan_files] if self.test_files else [])
            ),
        }[split]
        split_dreams_files = {
            "massspec_train": [self.train_dreams_files] if self.train_files else [],
            "massspec_val": [self.val_dreams_files] if self.val_files else [],
            "massspec_test": [self.test_dreams_files] if self.test_files else [],
            "train": [self.train_dreams_files] if self.train_files else [],
            "val": [self.val_dreams_files] if self.val_files else [],
            "test": [self.test_dreams_files] if self.test_files else [],
            "all": (
                ([self.train_dreams_files] if self.train_files else [])
                + ([self.val_dreams_files] if self.val_files else [])
                + ([self.test_dreams_files] if self.test_files else [])
            ),
        }[split]
        dataset = _ProbeParquetDataset(
            [
                {
                    "path": path,
                    "length": length,
                    "morgan_files": morgan_files,
                    "dreams_files": dreams_files,
                }
                for path, length, morgan_files, dreams_files in zip(
                    split_files,
                    split_lengths,
                    split_morgan_files,
                    split_dreams_files,
                    strict=True,
                )
            ],
            adduct_vocab=self.info["massspec_adduct_vocab"],
            instrument_type_vocab=self.info["massspec_instrument_type_vocab"],
        )
        dataset, shuffle = subset_for_max_samples(
            dataset,
            max_samples=max_samples,
            shuffle=shuffle,
            seed=seed,
        )
        sampler_world_size = distributed_world_size
        sampler_rank = distributed_rank
        if distributed_world_size > 1 and not pad_distributed:
            indices = list(range(distributed_rank, len(dataset), distributed_world_size))
            if hasattr(dataset, "subset"):
                dataset = dataset.subset(indices)
            else:
                dataset = Subset(dataset, indices)
            sampler_world_size = 1
            sampler_rank = 0
        sampler, loader_shuffle = loader_sampler(
            dataset,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_remainder,
            distributed_world_size=sampler_world_size,
            distributed_rank=sampler_rank,
        )
        generator = torch.Generator()
        generator.manual_seed(seed)
        batch_size = local_batch_size(self.batch_size, distributed_world_size)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=loader_shuffle,
            sampler=sampler,
            drop_last=drop_remainder,
            num_workers=0,
            collate_fn=_ProbeBatchCollator(
                num_peaks=self.num_peaks,
                max_precursor_mz=self.max_precursor_mz,
                min_peak_intensity=self.min_peak_intensity,
                peak_drop_min_intensity=self.peak_drop_min_intensity,
                peak_ordering=self.peak_ordering,
                precursor_peak_exclusion_window_da=self.precursor_peak_exclusion_window_da,
                peak_filtering=self.peak_filtering,
                grouped_peak_shoulder_da=self.grouped_peak_shoulder_da,
                grouped_peak_isotope_charges=self.grouped_peak_isotope_charges,
                output_format=output_format,
            ),
            generator=generator,
        )
        return _LoaderAdapter(loader)

    def build_indexed_dataset(
        self,
        split: str,
        indices: np.ndarray,
        *,
        peak_ordering: str | None = None,
        drop_remainder: bool = False,
        distributed_world_size: int = 1,
        output_format: str = "torch",
    ):
        if peak_ordering not in (None, self.peak_ordering):
            raise ValueError(
                "MassSpec peak ordering must match the checkpoint preprocessing "
                f"contract: requested={peak_ordering}, expected={self.peak_ordering}"
            )
        split_files = {
            "massspec_train": self.train_files,
            "massspec_val": self.val_files,
            "massspec_test": self.test_files,
            "train": self.train_files,
            "val": self.val_files,
            "test": self.test_files,
            "all": self.train_files + self.val_files + self.test_files,
        }[split]
        split_lengths = {
            "massspec_train": self.train_lengths,
            "massspec_val": self.val_lengths,
            "massspec_test": self.test_lengths,
            "train": self.train_lengths,
            "val": self.val_lengths,
            "test": self.test_lengths,
            "all": self.train_lengths + self.val_lengths + self.test_lengths,
        }[split]
        split_morgan_files = {
            "massspec_train": [self.train_morgan_files] if self.train_files else [],
            "massspec_val": [self.val_morgan_files] if self.val_files else [],
            "massspec_test": [self.test_morgan_files] if self.test_files else [],
            "train": [self.train_morgan_files] if self.train_files else [],
            "val": [self.val_morgan_files] if self.val_files else [],
            "test": [self.test_morgan_files] if self.test_files else [],
            "all": (
                ([self.train_morgan_files] if self.train_files else [])
                + ([self.val_morgan_files] if self.val_files else [])
                + ([self.test_morgan_files] if self.test_files else [])
            ),
        }[split]
        split_dreams_files = {
            "massspec_train": [self.train_dreams_files] if self.train_files else [],
            "massspec_val": [self.val_dreams_files] if self.val_files else [],
            "massspec_test": [self.test_dreams_files] if self.test_files else [],
            "train": [self.train_dreams_files] if self.train_files else [],
            "val": [self.val_dreams_files] if self.val_files else [],
            "test": [self.test_dreams_files] if self.test_files else [],
            "all": (
                ([self.train_dreams_files] if self.train_files else [])
                + ([self.val_dreams_files] if self.val_files else [])
                + ([self.test_dreams_files] if self.test_files else [])
            ),
        }[split]
        dataset = _ProbeParquetDataset(
            [
                {
                    "path": path,
                    "length": length,
                    "morgan_files": morgan_files,
                    "dreams_files": dreams_files,
                }
                for path, length, morgan_files, dreams_files in zip(
                    split_files,
                    split_lengths,
                    split_morgan_files,
                    split_dreams_files,
                    strict=True,
                )
            ],
            adduct_vocab=self.info["massspec_adduct_vocab"],
            instrument_type_vocab=self.info["massspec_instrument_type_vocab"],
        )
        batch_size = local_batch_size(self.batch_size, distributed_world_size)
        loader = DataLoader(
            Subset(dataset, [int(idx) for idx in indices]),
            batch_size=batch_size,
            shuffle=False,
            drop_last=drop_remainder,
            num_workers=0,
            collate_fn=_ProbeBatchCollator(
                num_peaks=self.num_peaks,
                max_precursor_mz=self.max_precursor_mz,
                min_peak_intensity=self.min_peak_intensity,
                peak_drop_min_intensity=self.peak_drop_min_intensity,
                peak_ordering=self.peak_ordering,
                precursor_peak_exclusion_window_da=self.precursor_peak_exclusion_window_da,
                peak_filtering=self.peak_filtering,
                grouped_peak_shoulder_da=self.grouped_peak_shoulder_da,
                grouped_peak_isotope_charges=self.grouped_peak_isotope_charges,
                output_format=output_format,
            ),
        )
        return _LoaderAdapter(loader)
