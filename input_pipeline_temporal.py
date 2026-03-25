"""Native PyTorch temporal pipeline for experiment-grouped .npz data.

Loads per-experiment .npz files and samples consecutive spectrum pairs for
next-frame prediction. Each sample is (frame, next_frame) from the same
LC-MS experiment.

All data is pre-loaded and pre-processed at init time into flat contiguous
numpy arrays.  DataLoader workers share this memory via fork COW, so
``__getitem__`` is pure array indexing with zero file I/O.
"""

import json
import logging
import tarfile
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from ml_collections import config_dict
from torch.utils.data import DataLoader, Dataset, RandomSampler

logger = logging.getLogger(__name__)

_PEAK_MZ_MIN = 20.0
_PEAK_MZ_MAX = 1000.0
_DEFAULT_MIN_PEAK_INTENSITY = 1e-4
_DEFAULT_MAX_PRECURSOR_MZ = 1000.0
_NUM_PEAKS_OUTPUT = 60
_SECONDS_PER_MINUTE = 60.0


def _preprocess_chunk(
    mz: np.ndarray,
    intensity: np.ndarray,
    *,
    num_peaks: int,
    min_peak_intensity: float,
    peak_ordering: str,
    peak_mz_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized preprocessing for (S, P) arrays.

    Returns:
        mz_out: (S, num_peaks) float32
        intensity_out: (S, num_peaks) float32
        valid: (S, num_peaks) bool
    """
    S, P = mz.shape

    keep = (mz >= _PEAK_MZ_MIN) & (mz <= peak_mz_max) & (intensity >= min_peak_intensity)
    mz = np.where(keep, mz, 0.0)
    intensity = np.where(keep, intensity, 0.0)

    if P > num_peaks:
        topk_idx = np.argpartition(-intensity, num_peaks, axis=1)[:, :num_peaks]
        rows = np.arange(S)[:, None]
        intensity = intensity[rows, topk_idx]
        mz = mz[rows, topk_idx]

        sort_within = np.argsort(-intensity, axis=1, kind="stable")
        intensity = np.take_along_axis(intensity, sort_within, axis=1)
        mz = np.take_along_axis(mz, sort_within, axis=1)
    elif P < num_peaks:
        pad_width = num_peaks - P
        intensity = np.pad(intensity, ((0, 0), (0, pad_width)))
        mz = np.pad(mz, ((0, 0), (0, pad_width)))

    max_int = np.maximum(intensity.max(axis=1, keepdims=True), 1e-8)
    intensity = intensity / max_int
    valid = intensity > 0

    if peak_ordering == "mz":
        sort_key = np.where(valid, mz, np.inf)
        order = np.argsort(sort_key, axis=1, kind="stable")
    else:
        sort_key = np.where(valid, intensity, -np.inf)
        order = np.argsort(-sort_key, axis=1, kind="stable")

    mz = np.take_along_axis(mz, order, axis=1)
    intensity = np.take_along_axis(intensity, order, axis=1)
    valid = np.take_along_axis(valid, order, axis=1)

    mz = np.where(valid, mz, 0.0)
    intensity = np.where(valid, intensity, 0.0)

    return (mz / peak_mz_max).astype(np.float32), intensity.astype(np.float32), valid


def _load_one_npz(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a single .npz and return (mz, intensity, rt, precursor_mz)."""
    data = np.load(path)
    return data["mz"], data["intensity"], data["rt"], data["precursor_mz"]


def _preprocess_chunk_worker(
    args: tuple[np.ndarray, np.ndarray, int, float, str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Picklable wrapper for _preprocess_chunk (for ProcessPoolExecutor)."""
    raw_mz, raw_int, num_peaks, min_peak_intensity, peak_ordering, peak_mz_max = args
    return _preprocess_chunk(
        raw_mz, raw_int,
        num_peaks=num_peaks,
        min_peak_intensity=min_peak_intensity,
        peak_ordering=peak_ordering,
        peak_mz_max=peak_mz_max,
    )


def _load_and_preprocess_all(
    file_list: list[dict],
    data_dir: Path,
    *,
    num_peaks: int,
    min_peak_intensity: float,
    peak_ordering: str,
    peak_mz_max: float,
    max_precursor_mz: float,
    chunk_size: int = 50_000,
    load_workers: int = 16,
    preprocess_workers: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load all .npz files and preprocess into flat contiguous arrays.

    Uses a thread pool for I/O (GIL released during zlib decompression)
    and a process pool for CPU-bound preprocessing.

    Returns:
        mz: (total, num_peaks) float32 — preprocessed, normalized
        intensity: (total, num_peaks) float32 — preprocessed, normalized
        valid: (total, num_peaks) bool
        rt: (total,) float32 in minutes
        precursor_mz: (total,) float32 — normalized
        offsets: (num_experiments,) int64
        lengths: (num_experiments,) int64
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

    # Phase 1: parallel load (threads — GIL released during zlib decompress)
    paths = [str(data_dir / entry["filename"]) for entry in file_list]
    logger.info("Loading %d experiment files with %d threads...", len(paths), load_workers)

    with ThreadPoolExecutor(max_workers=load_workers) as pool:
        results = list(pool.map(_load_one_npz, paths))

    raw_mz_parts = [r[0] for r in results]
    raw_int_parts = [r[1] for r in results]
    rt_parts = [r[2] for r in results]
    prec_parts = [r[3] for r in results]
    lengths = [len(r[2]) for r in results]
    del results

    raw_mz = np.concatenate(raw_mz_parts)
    raw_int = np.concatenate(raw_int_parts)
    del raw_mz_parts, raw_int_parts

    rt = (np.concatenate(rt_parts).astype(np.float32) / _SECONDS_PER_MINUTE)
    precursor_raw = np.concatenate(prec_parts).astype(np.float32)
    del rt_parts, prec_parts

    total = len(rt)
    logger.info("Loaded %d spectra, preprocessing with %d processes...", total, preprocess_workers)

    # Phase 2: parallel preprocess (processes — CPU-bound numpy)
    # Build chunk boundaries
    chunk_bounds = list(range(0, total, chunk_size))
    chunk_args = [
        (raw_mz[s : min(s + chunk_size, total)].copy(),
         raw_int[s : min(s + chunk_size, total)].copy(),
         num_peaks, min_peak_intensity, peak_ordering, peak_mz_max)
        for s in chunk_bounds
    ]
    del raw_mz, raw_int

    out_mz = np.empty((total, num_peaks), dtype=np.float32)
    out_int = np.empty((total, num_peaks), dtype=np.float32)
    out_valid = np.empty((total, num_peaks), dtype=bool)

    if preprocess_workers <= 1:
        chunk_results = [_preprocess_chunk_worker(a) for a in chunk_args]
    else:
        ctx = mp.get_context("forkserver")
        with ProcessPoolExecutor(max_workers=preprocess_workers, mp_context=ctx) as pool:
            chunk_results = list(pool.map(_preprocess_chunk_worker, chunk_args))

    for idx, (m, i, v) in enumerate(chunk_results):
        s = chunk_bounds[idx]
        e = min(s + chunk_size, total)
        out_mz[s:e] = m
        out_int[s:e] = i
        out_valid[s:e] = v
    del chunk_results, chunk_args

    # Normalize precursor_mz
    precursor_norm = (
        np.clip(precursor_raw, 0.0, max_precursor_mz) / max_precursor_mz
    ).astype(np.float32)
    del precursor_raw

    # Build offset index
    lengths_arr = np.array(lengths, dtype=np.int64)
    offsets = np.zeros(len(lengths), dtype=np.int64)
    np.cumsum(lengths_arr[:-1], out=offsets[1:])

    logger.info("Preprocessing complete: %d spectra, %.1f MB", total,
                (out_mz.nbytes + out_int.nbytes + out_valid.nbytes + rt.nbytes + precursor_norm.nbytes) / 1e6)

    return out_mz, out_int, out_valid, rt, precursor_norm, offsets, lengths_arr


class FramePairDataset(Dataset):
    """Map-style dataset that yields consecutive spectrum pairs.

    All data is pre-loaded and pre-processed at init time into flat
    contiguous numpy arrays.  ``__getitem__`` is pure array indexing —
    no file I/O.  DataLoader workers share the arrays via fork COW.

    Each sample randomly picks one spectrum from an experiment together with
    its immediate successor in RT order. Experiments with fewer than 2 spectra
    are filtered out.
    """

    def __init__(
        self,
        file_list: list[dict],
        data_dir: Path,
        *,
        num_peaks: int = _NUM_PEAKS_OUTPUT,
        min_peak_intensity: float = _DEFAULT_MIN_PEAK_INTENSITY,
        max_precursor_mz: float = _DEFAULT_MAX_PRECURSOR_MZ,
        peak_ordering: str = "mz",
        peak_mz_max: float = _PEAK_MZ_MAX,
    ) -> None:
        # Filter to experiments with >= 2 spectra
        usable = [f for f in file_list if f["num_spectra"] >= 2]
        n_filtered = len(file_list) - len(usable)
        if n_filtered > 0:
            logger.info(
                "Filtered %d experiments with <2 spectra (%d remaining)",
                n_filtered, len(usable),
            )

        # Pre-load and pre-process everything
        (
            self._mz,           # (total, num_peaks) float32
            self._intensity,    # (total, num_peaks) float32
            self._valid,        # (total, num_peaks) bool
            self._rt,           # (total,) float32
            self._precursor_mz, # (total,) float32
            self._offsets,      # (num_experiments,) int64
            self._lengths,      # (num_experiments,) int64
        ) = _load_and_preprocess_all(
            usable,
            data_dir,
            num_peaks=num_peaks,
            min_peak_intensity=min_peak_intensity,
            peak_ordering=peak_ordering,
            peak_mz_max=peak_mz_max,
            max_precursor_mz=max_precursor_mz,
        )
        self._num_experiments = len(usable)

    def __len__(self) -> int:
        return self._num_experiments

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        offset = self._offsets[idx]
        length = self._lengths[idx]

        start = np.random.randint(length - 1)
        i0 = offset + start
        i1 = i0 + 1

        # .copy() is required: without it torch.from_numpy returns a tensor
        # backed by the shared pre-loaded array, which is unsafe with
        # pin_memory and would cause COW page faults across workers.
        return {
            "frame_peak_mz": torch.from_numpy(self._mz[i0].copy()),
            "frame_peak_intensity": torch.from_numpy(self._intensity[i0].copy()),
            "frame_peak_valid_mask": torch.from_numpy(self._valid[i0].copy()),
            "frame_rt": torch.tensor(self._rt[i0]),
            "frame_precursor_mz": torch.tensor(self._precursor_mz[i0]),
            "next_frame_peak_mz": torch.from_numpy(self._mz[i1].copy()),
            "next_frame_peak_intensity": torch.from_numpy(self._intensity[i1].copy()),
            "next_frame_peak_valid_mask": torch.from_numpy(self._valid[i1].copy()),
            "next_frame_rt": torch.tensor(self._rt[i1]),
            "next_frame_precursor_mz": torch.tensor(self._precursor_mz[i1]),
        }


class TemporalLightningDataModule:
    """Config-driven wrapper for temporal experiment-grouped datasets.

    Downloads from HuggingFace if ``temporal_repo_id`` is set, extracts tar
    archives, reads manifest, creates :class:`FramePairDataset` instances, and
    builds DataLoaders.
    """

    def __init__(self, config: config_dict.ConfigDict, seed: int) -> None:
        self.config = config
        self.seed = int(seed)

        # Resolve data directory — download from HuggingFace if repo_id is set
        temporal_repo_id = str(config.get("temporal_repo_id", "")).strip()
        temporal_revision = str(config.get("temporal_revision", "main"))
        local_dir = Path(
            config.get("temporal_data_dir", "data/gems_grouped")
        ).expanduser().resolve()

        if temporal_repo_id:
            data_dir = self._download_and_extract(
                temporal_repo_id, temporal_revision, local_dir
            )
        else:
            data_dir = local_dir

        manifest_path = data_dir / "manifest.json"
        with manifest_path.open() as f:
            self.manifest = json.load(f)

        self.batch_size = int(config.get("batch_size", 256))
        self.num_peaks = int(config.get("num_peaks", _NUM_PEAKS_OUTPUT))
        self.min_peak_intensity = float(
            config.get("min_peak_intensity", _DEFAULT_MIN_PEAK_INTENSITY)
        )
        self.max_precursor_mz = float(
            config.get("max_precursor_mz", _DEFAULT_MAX_PRECURSOR_MZ)
        )
        self.peak_ordering = str(config.get("peak_ordering", "mz"))

        # DataLoader settings
        default_pin = torch.cuda.is_available()
        self.pin_memory = bool(config.get("dataloader_pin_memory", default_pin))
        self.num_workers = int(config.get("dataloader_num_workers", 4))
        self.prefetch_factor = int(config.get("dataloader_prefetch_factor", 2))
        self.persistent_workers = bool(
            config.get("dataloader_persistent_workers", self.num_workers > 0)
        )

        # Build datasets (pre-loads all data into RAM)
        self.train_dir = data_dir / "train"
        self.val_dir = data_dir / "validation"
        self.train_files = self.manifest["train"]["files"]
        self.val_files = self.manifest["validation"]["files"]

        train_total = self.manifest["train"]["total_spectra"]
        val_total = self.manifest["validation"]["total_spectra"]

        train_usable = sum(1 for f in self.train_files if f["num_spectra"] >= 2)
        configured_train_steps = int(config.get("num_train_steps", 0))
        if configured_train_steps > 0:
            self.train_steps = configured_train_steps
        else:
            self.train_steps = int(float(config.get("num_epochs", 1.0)) * max(1, train_usable // self.batch_size))

        self.info = {
            "data_dir": str(data_dir),
            "train_size": train_total,
            "validation_size": val_total,
            "train_experiments": len(self.train_files),
            "train_usable_experiments": train_usable,
            "validation_experiments": len(self.val_files),
            "num_peaks": self.num_peaks,
            "num_train_steps": self.train_steps,
            "rt_unit": "minutes",
            "max_precursor_mz": self.max_precursor_mz,
            "peak_mz_min": _PEAK_MZ_MIN,
            "peak_mz_max": _PEAK_MZ_MAX,
        }

        self._train_loader: DataLoader | None = None
        self._val_loader: DataLoader | None = None

        logger.info(
            "TemporalLightningDataModule: %d train experiments (%d usable), "
            "%d val experiments, batch_size=%d",
            len(self.train_files),
            train_usable,
            len(self.val_files),
            self.batch_size,
        )

    @staticmethod
    def _download_and_extract(
        repo_id: str, revision: str, local_dir: Path
    ) -> Path:
        """Download dataset from HuggingFace and extract tar archives if needed."""
        logger.info(
            "Downloading temporal dataset from %s@%s", repo_id, revision
        )
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=local_dir,
            allow_patterns=["manifest.json", "*.tar.gz", "train/*", "validation/*"],
        )

        for split in ("train", "validation"):
            tar_path = local_dir / f"{split}.tar.gz"
            split_dir = local_dir / split
            if tar_path.exists() and not split_dir.exists():
                logger.info("Extracting %s ...", tar_path.name)
                with tarfile.open(tar_path, "r:gz") as tf:
                    tf.extractall(local_dir, filter="data")
                logger.info("Extracted %s -> %s/", tar_path.name, split)

        return local_dir

    def _make_dataset(self, file_list: list[dict], data_dir: Path) -> FramePairDataset:
        return FramePairDataset(
            file_list=file_list,
            data_dir=data_dir,
            num_peaks=self.num_peaks,
            min_peak_intensity=self.min_peak_intensity,
            max_precursor_mz=self.max_precursor_mz,
            peak_ordering=self.peak_ordering,
        )

    def _make_loader(
        self,
        dataset: FramePairDataset,
        *,
        shuffle: bool,
    ) -> DataLoader:
        kwargs: dict = dict(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )
        if shuffle:
            kwargs["sampler"] = RandomSampler(
                dataset,
                replacement=True,
                num_samples=self.batch_size * self.train_steps,
            )
        else:
            kwargs["shuffle"] = False
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
            kwargs["persistent_workers"] = self.persistent_workers
        return DataLoader(**kwargs)

    @property
    def train_loader(self) -> DataLoader:
        if self._train_loader is None:
            ds = self._make_dataset(self.train_files, self.train_dir)
            self._train_loader = self._make_loader(ds, shuffle=True)
        return self._train_loader

    @property
    def val_loader(self) -> DataLoader:
        if self._val_loader is None:
            ds = self._make_dataset(self.val_files, self.val_dir)
            self._val_loader = self._make_loader(ds, shuffle=False)
        return self._val_loader
