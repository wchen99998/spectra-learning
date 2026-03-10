"""Extract real spectra data from the pipeline for interactive visualization."""

import json
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
project_root = "/home/wuhao/spectra-learning"
os.chdir(project_root)
sys.path.insert(0, project_root)

from configs.gems_a_50_mask import get_config
from input_pipeline import TfLightningDataModule, _NUM_PEAKS_OUTPUT, _build_dataset


cfg = get_config()
cfg.tfrecord_dir = os.path.join(project_root, "data/gems_peaklist_tfrecord")

dm = TfLightningDataModule(cfg, seed=42)
train_files = dm.gems_train_files
print(f"Found {len(train_files)} train files")

BATCH_SIZE = 16
NUM_BATCHES = 3

raw_ds = _build_dataset(
    train_files,
    batch_size=BATCH_SIZE,
    shuffle_buffer=1000,
    seed=42,
    drop_remainder=True,
    tfrecord_buffer_size=8 * 1024 * 1024,
    max_precursor_mz=cfg.max_precursor_mz,
    min_peak_intensity=cfg.min_peak_intensity,
    augment=False,
    peak_ordering=cfg.peak_ordering,
)

raw_batches = []
raw_iter = raw_ds.as_numpy_iterator()
for i in range(NUM_BATCHES):
    raw_batches.append(next(raw_iter))
    print(f"  Raw batch {i}: {raw_batches[-1]['peak_mz'].shape}")

aug_ds = _build_dataset(
    train_files,
    batch_size=BATCH_SIZE,
    shuffle_buffer=1000,
    seed=42,
    drop_remainder=True,
    tfrecord_buffer_size=8 * 1024 * 1024,
    max_precursor_mz=cfg.max_precursor_mz,
    min_peak_intensity=cfg.min_peak_intensity,
    augment=True,
    jepa_num_target_blocks=cfg.jepa_num_target_blocks,
    jepa_context_fraction=cfg.jepa_context_fraction,
    jepa_target_fraction=cfg.jepa_target_fraction,
    jepa_block_min_len=cfg.jepa_block_min_len,
    mz_jitter_std=cfg.sigreg_mz_jitter_std,
    intensity_jitter_std=cfg.sigreg_intensity_jitter_std,
    peak_ordering=cfg.peak_ordering,
)

aug_batches = []
aug_iter = aug_ds.as_numpy_iterator()
for i in range(NUM_BATCHES):
    aug_batches.append(next(aug_iter))
    print(
        "  Aug batch "
        f"{i}: peak_mz={aug_batches[-1]['peak_mz'].shape} "
        f"context_mask={aug_batches[-1]['context_mask'].shape} "
        f"target_masks={aug_batches[-1]['target_masks'].shape}"
    )

N_JITTER = 30
jitter_runs = []
jitter_iter = aug_ds.as_numpy_iterator()
for _ in range(N_JITTER):
    ab = next(jitter_iter)
    jitter_runs.append(
        {
            "peak_mz": ab["peak_mz"][0].tolist(),
            "peak_intensity": ab["peak_intensity"][0].tolist(),
            "context_mask": ab["context_mask"][0].tolist(),
            "target_masks": ab["target_masks"][0].tolist(),
        }
    )
print(f"  Collected {N_JITTER} jitter runs")


def make_spectrum(raw_batch, aug_batch, sample_idx):
    return {
        "raw": {
            "mz": raw_batch["peak_mz"][sample_idx].tolist(),
            "intensity": raw_batch["peak_intensity"][sample_idx].tolist(),
            "valid_mask": raw_batch["peak_valid_mask"][sample_idx].tolist(),
            "precursor_mz": float(raw_batch["precursor_mz"][sample_idx]),
            "mz_unnorm": raw_batch["mz"][sample_idx].tolist(),
            "intensity_unnorm": raw_batch["intensity"][sample_idx].tolist(),
        },
        "augmented": {
            "mz": aug_batch["peak_mz"][sample_idx].tolist(),
            "intensity": aug_batch["peak_intensity"][sample_idx].tolist(),
            "valid_mask": aug_batch["peak_valid_mask"][sample_idx].tolist(),
            "context_mask": aug_batch["context_mask"][sample_idx].tolist(),
            "target_masks": aug_batch["target_masks"][sample_idx].tolist(),
        },
    }


spectra = []
for batch_idx in range(NUM_BATCHES):
    for sample_idx in range(BATCH_SIZE):
        spectra.append(make_spectrum(raw_batches[batch_idx], aug_batches[batch_idx], sample_idx))
print(f"Extracted {len(spectra)} spectra")

output = {
    "config": {
        "num_peaks": _NUM_PEAKS_OUTPUT,
        "num_target_blocks": cfg.jepa_num_target_blocks,
        "context_fraction": float(cfg.jepa_context_fraction),
        "target_fraction": float(cfg.jepa_target_fraction),
        "block_min_len": int(cfg.jepa_block_min_len),
        "mz_jitter_std": float(cfg.sigreg_mz_jitter_std),
        "intensity_jitter_std": float(cfg.sigreg_intensity_jitter_std),
        "batch_size": BATCH_SIZE,
        "max_precursor_mz": float(cfg.max_precursor_mz),
        "peak_ordering": cfg.peak_ordering,
    },
    "spectra": spectra,
    "jitter_runs": jitter_runs,
}

out_path = os.path.join(project_root, "visuals/pipeline-explorer/data/spectra.json")
with open(out_path, "w") as f:
    json.dump(output, f)

file_size_mb = os.path.getsize(out_path) / 1024 / 1024
print(f"Wrote {out_path} ({file_size_mb:.1f} MB)")
