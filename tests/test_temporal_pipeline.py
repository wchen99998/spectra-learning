"""Tests for the temporal experiment-grouped pipeline (frame -> next frame)."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from ml_collections import config_dict

from input_pipeline_temporal import (
    FramePairDataset,
    TemporalLightningDataModule,
    _load_and_preprocess_all,
    _preprocess_chunk,
)


def _make_synthetic_experiment(
    num_spectra: int,
    num_peaks: int = 128,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng(42)
    mz = rng.uniform(20.0, 1000.0, (num_spectra, num_peaks)).astype(np.float32)
    intensity = rng.exponential(0.5, (num_spectra, num_peaks)).astype(np.float32)
    rt = np.sort(rng.uniform(60.0, 3600.0, num_spectra)).astype(np.float32)
    precursor_mz = rng.uniform(100.0, 900.0, num_spectra).astype(np.float32)
    return {"mz": mz, "intensity": intensity, "rt": rt, "precursor_mz": precursor_mz}


@pytest.fixture()
def synthetic_data_dir(tmp_path: Path) -> Path:
    rng = np.random.default_rng(123)
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "validation"
    train_dir.mkdir()
    val_dir.mkdir()

    train_files = []
    for i in range(10):
        n = rng.integers(20, 200)
        data = _make_synthetic_experiment(int(n), rng=rng)
        fname = f"exp_{i:04d}.npz"
        np.savez_compressed(train_dir / fname, **data)
        train_files.append({"filename": fname, "experiment_name": f"exp_{i}", "num_spectra": int(n)})

    # Include one single-spectrum experiment to test filtering
    data = _make_synthetic_experiment(1, rng=rng)
    fname = "exp_single.npz"
    np.savez_compressed(train_dir / fname, **data)
    train_files.append({"filename": fname, "experiment_name": "exp_single", "num_spectra": 1})

    val_files = []
    for i in range(3):
        n = rng.integers(20, 200)
        data = _make_synthetic_experiment(int(n), rng=rng)
        fname = f"val_{i:04d}.npz"
        np.savez_compressed(val_dir / fname, **data)
        val_files.append({"filename": fname, "experiment_name": f"val_{i}", "num_spectra": int(n)})

    def _stats(files):
        counts = [f["num_spectra"] for f in files]
        arr = np.array(counts)
        return {
            "num_files": len(counts),
            "total_spectra": int(arr.sum()),
            "min_spectra": int(arr.min()),
            "max_spectra": int(arr.max()),
            "mean_spectra": float(arr.mean()),
            "median_spectra": float(np.median(arr)),
        }

    manifest = {
        "version": 1,
        "source_hdf5": "synthetic",
        "max_precursor_mz": 1000.0,
        "val_fraction": 0.05,
        "split_seed": 42,
        "total_experiments_in_hdf5": 14,
        "filtered_experiments": 0,
        "num_peaks_input": 128,
        "train": {"files": train_files, **_stats(train_files)},
        "validation": {"files": val_files, **_stats(val_files)},
    }
    with (tmp_path / "manifest.json").open("w") as f:
        json.dump(manifest, f)

    return tmp_path


class TestPreprocessChunk:
    def test_output_shape(self):
        rng = np.random.default_rng(0)
        mz = rng.uniform(20.0, 1000.0, (8, 128)).astype(np.float32)
        intensity = rng.exponential(0.5, (8, 128)).astype(np.float32)
        mz_out, int_out, valid = _preprocess_chunk(
            mz, intensity, num_peaks=60, min_peak_intensity=1e-4,
            peak_ordering="mz", peak_mz_max=1000.0,
        )
        assert mz_out.shape == (8, 60)
        assert int_out.shape == (8, 60)
        assert valid.shape == (8, 60)

    def test_mz_normalized_range(self):
        rng = np.random.default_rng(1)
        mz = rng.uniform(20.0, 1000.0, (4, 128)).astype(np.float32)
        intensity = rng.exponential(0.5, (4, 128)).astype(np.float32)
        mz_out, _, valid = _preprocess_chunk(
            mz, intensity, num_peaks=60, min_peak_intensity=1e-4,
            peak_ordering="mz", peak_mz_max=1000.0,
        )
        assert mz_out[valid].min() >= 0.0
        assert mz_out[valid].max() <= 1.0

    def test_intensity_normalized(self):
        rng = np.random.default_rng(2)
        mz = rng.uniform(20.0, 1000.0, (4, 128)).astype(np.float32)
        intensity = rng.exponential(0.5, (4, 128)).astype(np.float32)
        _, int_out, valid = _preprocess_chunk(
            mz, intensity, num_peaks=60, min_peak_intensity=1e-4,
            peak_ordering="mz", peak_mz_max=1000.0,
        )
        assert int_out[valid].max() <= 1.0 + 1e-6
        assert int_out[valid].min() >= 0.0

    def test_mz_ordering_per_row(self):
        rng = np.random.default_rng(3)
        mz = rng.uniform(20.0, 1000.0, (4, 128)).astype(np.float32)
        intensity = rng.exponential(0.5, (4, 128)).astype(np.float32)
        mz_out, _, valid = _preprocess_chunk(
            mz, intensity, num_peaks=60, min_peak_intensity=1e-4,
            peak_ordering="mz", peak_mz_max=1000.0,
        )
        for row in range(4):
            v = valid[row]
            valid_mz = mz_out[row, v]
            assert np.all(valid_mz[:-1] <= valid_mz[1:])

    def test_fewer_peaks_than_num_peaks(self):
        mz = np.array([[500.0, 600.0, 0.0]], dtype=np.float32)
        intensity = np.array([[1.0, 0.5, 0.0]], dtype=np.float32)
        mz_out, _, valid = _preprocess_chunk(
            mz, intensity, num_peaks=60, min_peak_intensity=1e-4,
            peak_ordering="mz", peak_mz_max=1000.0,
        )
        assert valid[0].sum() == 2
        assert mz_out.shape == (1, 60)


class TestLoadAndPreprocess:
    def test_offsets_and_lengths_consistent(self, synthetic_data_dir: Path):
        """Offsets + lengths should tile the flat arrays exactly."""
        manifest_path = synthetic_data_dir / "manifest.json"
        with manifest_path.open() as f:
            manifest = json.load(f)
        usable = [f for f in manifest["train"]["files"] if f["num_spectra"] >= 2]
        mz, intensity, valid, rt, prec, offsets, lengths = _load_and_preprocess_all(
            usable,
            synthetic_data_dir / "train",
            num_peaks=60,
            min_peak_intensity=1e-4,
            peak_ordering="mz",
            peak_mz_max=1000.0,
            max_precursor_mz=1000.0,
            preprocess_workers=1,
        )
        total = mz.shape[0]
        assert offsets[0] == 0
        assert (offsets + lengths)[-1] == total
        for i in range(len(offsets) - 1):
            assert offsets[i] + lengths[i] == offsets[i + 1]

    def test_rt_scaled_to_minutes(self, synthetic_data_dir: Path):
        manifest_path = synthetic_data_dir / "manifest.json"
        with manifest_path.open() as f:
            manifest = json.load(f)
        usable = [f for f in manifest["train"]["files"] if f["num_spectra"] >= 2]
        _, _, _, rt, _, offsets, lengths = _load_and_preprocess_all(
            usable,
            synthetic_data_dir / "train",
            num_peaks=60,
            min_peak_intensity=1e-4,
            peak_ordering="mz",
            peak_mz_max=1000.0,
            max_precursor_mz=1000.0,
            preprocess_workers=1,
        )
        first_file = usable[0]["filename"]
        raw = np.load(synthetic_data_dir / "train" / first_file)["rt"].astype(np.float32)
        offset = int(offsets[0])
        length = int(lengths[0])
        assert np.allclose(rt[offset : offset + length], raw / 60.0)


class TestFramePairDataset:
    def test_getitem_shapes(self, synthetic_data_dir: Path):
        manifest_path = synthetic_data_dir / "manifest.json"
        with manifest_path.open() as f:
            manifest = json.load(f)
        ds = FramePairDataset(
            file_list=manifest["train"]["files"],
            data_dir=synthetic_data_dir / "train",
            num_peaks=60,
        )
        item = ds[0]
        N = 60
        for prefix in ("frame", "next_frame"):
            assert item[f"{prefix}_peak_mz"].shape == (N,)
            assert item[f"{prefix}_peak_intensity"].shape == (N,)
            assert item[f"{prefix}_peak_valid_mask"].shape == (N,)
            assert item[f"{prefix}_rt"].shape == ()
            assert item[f"{prefix}_precursor_mz"].shape == ()

    def test_filters_single_spectrum_experiments(self, synthetic_data_dir: Path):
        manifest_path = synthetic_data_dir / "manifest.json"
        with manifest_path.open() as f:
            manifest = json.load(f)
        all_files = manifest["train"]["files"]
        ds = FramePairDataset(
            file_list=all_files,
            data_dir=synthetic_data_dir / "train",
        )
        # We added one single-spectrum experiment, so len should be one less
        assert len(ds) == len(all_files) - 1

    def test_frame_and_next_frame_are_different(self, synthetic_data_dir: Path):
        """Current and next frame should generally differ."""
        manifest_path = synthetic_data_dir / "manifest.json"
        with manifest_path.open() as f:
            manifest = json.load(f)
        ds = FramePairDataset(
            file_list=manifest["train"]["files"],
            data_dir=synthetic_data_dir / "train",
            num_peaks=60,
        )
        any_different = False
        for idx in range(min(10, len(ds))):
            item = ds[idx]
            if not torch.equal(item["frame_peak_mz"], item["next_frame_peak_mz"]):
                any_different = True
                break
        assert any_different, "Frame and next frame should differ for most pairs"

    def test_next_frame_is_consecutive(self, synthetic_data_dir: Path):
        """Returned pair should be adjacent in RT order within the experiment."""
        manifest_path = synthetic_data_dir / "manifest.json"
        with manifest_path.open() as f:
            manifest = json.load(f)
        ds = FramePairDataset(
            file_list=manifest["train"]["files"],
            data_dir=synthetic_data_dir / "train",
        )
        for idx in range(min(20, len(ds))):
            item = ds[idx]
            offset = int(ds._offsets[idx])
            length = int(ds._lengths[idx])
            rt_values = ds._rt[offset : offset + length]
            frame_pos = int(np.searchsorted(rt_values, float(item["frame_rt"])))
            next_frame_pos = int(np.searchsorted(rt_values, float(item["next_frame_rt"])))
            assert next_frame_pos == frame_pos + 1
            assert item["next_frame_rt"] >= item["frame_rt"]

    def test_two_spectrum_experiment(self, tmp_path: Path):
        """Edge case: experiment with exactly 2 spectra should always work."""
        rng = np.random.default_rng(99)
        data = _make_synthetic_experiment(2, rng=rng)
        fname = "tiny.npz"
        np.savez_compressed(tmp_path / fname, **data)
        ds = FramePairDataset(
            file_list=[{"filename": fname, "experiment_name": "tiny", "num_spectra": 2}],
            data_dir=tmp_path,
            num_peaks=60,
        )
        assert len(ds) == 1
        for _ in range(20):
            item = ds[0]
            assert item["next_frame_rt"] >= item["frame_rt"]

    def test_mz_values_in_range(self, synthetic_data_dir: Path):
        manifest_path = synthetic_data_dir / "manifest.json"
        with manifest_path.open() as f:
            manifest = json.load(f)
        ds = FramePairDataset(
            file_list=manifest["train"]["files"],
            data_dir=synthetic_data_dir / "train",
            num_peaks=60,
        )
        item = ds[0]
        for prefix in ("frame", "next_frame"):
            valid = item[f"{prefix}_peak_valid_mask"]
            mz = item[f"{prefix}_peak_mz"]
            assert mz[valid].min() >= 0.0
            assert mz[valid].max() <= 1.0

    def test_precursor_mz_normalized(self, synthetic_data_dir: Path):
        manifest_path = synthetic_data_dir / "manifest.json"
        with manifest_path.open() as f:
            manifest = json.load(f)
        ds = FramePairDataset(
            file_list=manifest["train"]["files"],
            data_dir=synthetic_data_dir / "train",
        )
        for idx in range(min(10, len(ds))):
            item = ds[idx]
            for prefix in ("frame", "next_frame"):
                assert 0.0 <= item[f"{prefix}_precursor_mz"] <= 1.0


class TestCollation:
    def test_default_collate_shapes(self, synthetic_data_dir: Path):
        manifest_path = synthetic_data_dir / "manifest.json"
        with manifest_path.open() as f:
            manifest = json.load(f)
        ds = FramePairDataset(
            file_list=manifest["train"]["files"],
            data_dir=synthetic_data_dir / "train",
            num_peaks=60,
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        N = 60
        for prefix in ("frame", "next_frame"):
            assert batch[f"{prefix}_peak_mz"].shape == (4, N)
            assert batch[f"{prefix}_peak_intensity"].shape == (4, N)
            assert batch[f"{prefix}_peak_valid_mask"].shape == (4, N)
            assert batch[f"{prefix}_rt"].shape == (4,)
            assert batch[f"{prefix}_precursor_mz"].shape == (4,)


class TestTemporalLightningDataModule:
    def test_init_and_info(self, synthetic_data_dir: Path):
        cfg = config_dict.ConfigDict()
        cfg.temporal_data_dir = str(synthetic_data_dir)
        cfg.batch_size = 2
        cfg.num_peaks = 60
        cfg.peak_ordering = "mz"
        cfg.dataloader_num_workers = 0
        cfg.dataloader_pin_memory = False

        dm = TemporalLightningDataModule(cfg, seed=42)
        assert dm.info["train_experiments"] == 11  # includes single-spectrum
        assert dm.info["train_usable_experiments"] == 10  # excludes single-spectrum
        assert dm.info["validation_experiments"] == 3
        assert dm.info["num_train_steps"] == 5
        assert dm.info["rt_unit"] == "minutes"

    def test_train_loader_batch_shapes(self, synthetic_data_dir: Path):
        cfg = config_dict.ConfigDict()
        cfg.temporal_data_dir = str(synthetic_data_dir)
        cfg.batch_size = 3
        cfg.num_train_steps = 7
        cfg.num_peaks = 60
        cfg.peak_ordering = "mz"
        cfg.dataloader_num_workers = 0
        cfg.dataloader_pin_memory = False

        dm = TemporalLightningDataModule(cfg, seed=42)
        loader = dm.train_loader
        batch = next(iter(loader))
        assert len(loader) == 7
        for prefix in ("frame", "next_frame"):
            assert batch[f"{prefix}_peak_mz"].shape == (3, 60)
            assert batch[f"{prefix}_rt"].shape == (3,)

    def test_val_loader_batch_shapes(self, synthetic_data_dir: Path):
        cfg = config_dict.ConfigDict()
        cfg.temporal_data_dir = str(synthetic_data_dir)
        cfg.batch_size = 2
        cfg.num_train_steps = 5
        cfg.num_peaks = 60
        cfg.peak_ordering = "mz"
        cfg.dataloader_num_workers = 0
        cfg.dataloader_pin_memory = False

        dm = TemporalLightningDataModule(cfg, seed=42)
        batch = next(iter(dm.val_loader))
        for prefix in ("frame", "next_frame"):
            assert batch[f"{prefix}_peak_mz"].shape == (2, 60)
