"""Extract real spectra data from the pipeline for interactive visualization."""
import os
import sys
import json
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
project_root = '/home/wuhao/spectra-learning'
os.chdir(project_root)
sys.path.insert(0, project_root)

from input_pipeline import _build_dataset, _NUM_PEAKS_OUTPUT
from configs.gems_a_50_mask import get_config

cfg = get_config()
cfg.tfrecord_dir = os.path.join(project_root, 'data/gems_peaklist_tfrecord')

# Use TfLightningDataModule to discover files properly
dm = __import__('input_pipeline', fromlist=['TfLightningDataModule']).TfLightningDataModule(cfg, seed=42)
train_files = dm.gems_train_files
print(f"Found {len(train_files)} train files")

BATCH_SIZE = 16
NUM_BATCHES = 3  # extract multiple batches for variety

# --- 1. Raw (pre-augmentation) data ---
raw_ds = _build_dataset(
    train_files,
    batch_size=BATCH_SIZE,
    shuffle_buffer=1000,
    seed=42,
    drop_remainder=True,
    tfrecord_buffer_size=8 * 1024 * 1024,
    max_precursor_mz=cfg.max_precursor_mz,
    include_fingerprint=False,
    min_peak_intensity=cfg.min_peak_intensity,
    augmentation_type='none',
    peak_ordering=cfg.peak_ordering,
)

raw_batches = []
raw_iter = raw_ds.as_numpy_iterator()
for i in range(NUM_BATCHES):
    raw_batches.append(next(raw_iter))
    print(f"  Raw batch {i}: {raw_batches[-1]['peak_mz'].shape}")

# --- 2. Augmented (multicrop) data with same seed ---
aug_ds = _build_dataset(
    train_files,
    batch_size=BATCH_SIZE,
    shuffle_buffer=1000,
    seed=42,
    drop_remainder=True,
    tfrecord_buffer_size=8 * 1024 * 1024,
    max_precursor_mz=cfg.max_precursor_mz,
    include_fingerprint=False,
    min_peak_intensity=cfg.min_peak_intensity,
    augmentation_type='multicrop',
    multicrop_num_local_views=cfg.multicrop_num_local_views,
    multicrop_local_keep_fraction=cfg.multicrop_local_keep_fraction,
    mz_jitter_std=cfg.sigreg_mz_jitter_std,
    intensity_jitter_std=cfg.sigreg_intensity_jitter_std,
    peak_ordering=cfg.peak_ordering,
)

aug_batches = []
aug_iter = aug_ds.as_numpy_iterator()
for i in range(NUM_BATCHES):
    aug_batches.append(next(aug_iter))
    print(f"  Aug batch {i}: fused_mz={aug_batches[-1]['fused_mz'].shape}")

# --- 3. Multiple jitter runs for one spectrum ---
N_JITTER = 30
jitter_runs = []
jitter_iter = aug_ds.as_numpy_iterator()
for _ in range(N_JITTER):
    ab = next(jitter_iter)
    jitter_runs.append({
        'fused_mz': ab['fused_mz'][0].tolist(),  # sample 0, global view 1
        'fused_intensity': ab['fused_intensity'][0].tolist(),
    })
print(f"  Collected {N_JITTER} jitter runs")

# --- Build output JSON ---
V = 1 + cfg.multicrop_num_local_views
B = BATCH_SIZE

def make_spectrum(raw_batch, aug_batch, sample_idx):
    """Extract one spectrum with all its views."""
    spec = {
        'raw': {
            'mz': raw_batch['peak_mz'][sample_idx].tolist(),
            'intensity': raw_batch['peak_intensity'][sample_idx].tolist(),
            'valid_mask': raw_batch['peak_valid_mask'][sample_idx].tolist(),
            'precursor_mz': float(raw_batch['precursor_mz'][sample_idx]),
            'mz_unnorm': raw_batch['mz'][sample_idx].tolist(),
            'intensity_unnorm': raw_batch['intensity'][sample_idx].tolist(),
        },
        'views': [],
    }
    for v in range(V):
        offset = v * B + sample_idx
        view_type = 'global' if v == 0 else 'local'
        view_idx = 0 if v == 0 else v - 1
        spec['views'].append({
            'type': view_type,
            'index': view_idx,
            'mz': aug_batch['fused_mz'][offset].tolist(),
            'intensity': aug_batch['fused_intensity'][offset].tolist(),
            'valid_mask': aug_batch['fused_valid_mask'][offset].tolist(),
            'masked_positions': aug_batch['fused_masked_positions'][offset].tolist(),
            'padding_mask': aug_batch['fused_padding_mask'][offset].tolist(),
        })
    return spec

spectra = []
for batch_idx in range(NUM_BATCHES):
    for sample_idx in range(BATCH_SIZE):
        spectra.append(make_spectrum(raw_batches[batch_idx], aug_batches[batch_idx], sample_idx))
print(f"Extracted {len(spectra)} spectra with {V} views each")

output = {
    'config': {
        'num_peaks': _NUM_PEAKS_OUTPUT,
        'num_global_views': 1,
        'num_local_views': cfg.multicrop_num_local_views,
        'global_mask_fraction_target': 0.0,
        'local_keep_fraction': float(cfg.multicrop_local_keep_fraction),
        'mz_jitter_std': float(cfg.sigreg_mz_jitter_std),
        'intensity_jitter_std': float(cfg.sigreg_intensity_jitter_std),
        'total_views': V,
        'batch_size': B,
        'max_precursor_mz': float(cfg.max_precursor_mz),
        'peak_ordering': cfg.peak_ordering,
    },
    'spectra': spectra,
    'jitter_runs': jitter_runs,
}

out_path = os.path.join(project_root, 'visuals/pipeline-explorer/data/spectra.json')
with open(out_path, 'w') as f:
    json.dump(output, f)

file_size_mb = os.path.getsize(out_path) / 1024 / 1024
print(f"Wrote {out_path} ({file_size_mb:.1f} MB)")
