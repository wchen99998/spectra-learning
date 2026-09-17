"""Existing frozen embeddings and chemistry, with canonical FLARE peak features."""
from pathlib import Path

import numpy as np
import torch

import dgt_data


RAW_FEATURE_ROOT = Path('/results/jepa_fps_ftjepa_1p8b_step530500_flare_raw/data')


def append_peak_features(
    tokens: torch.Tensor,
    valid: torch.Tensor,
    mz: np.ndarray,
    intensity: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append cached values unchanged; the final, always-valid CLS has zeros."""
    shape = (len(tokens), tokens.shape[1] - 1)
    if tokens.shape[-1] != 2048 or valid.shape != tokens.shape[:2]:
        raise ValueError('Expected frozen JEPA tokens and their aligned validity mask')
    if mz.shape != shape or intensity.shape != shape:
        raise ValueError('Raw peak features do not align with the physical peak slots')
    if not valid[:, -1].all():
        raise ValueError('The final CLS slot must always be valid')
    raw = np.zeros((*tokens.shape[:2], 2), dtype=np.float32)
    raw[:, :-1, 0] = mz
    raw[:, :-1, 1] = intensity
    return torch.cat((tokens, torch.from_numpy(raw).to(tokens.device)), dim=-1), valid


class Data:
    def __init__(self):
        # Keep dgt_data.ROOT unchanged: it owns the complete chemistry cache.
        self.base = dgt_data.load_data()
        self.mz = np.load(RAW_FEATURE_ROOT / 'peak_mz_f16.npy', mmap_mode='r')
        self.intensity = np.load(RAW_FEATURE_ROOT / 'peak_intensity_f16.npy', mmap_mode='r')
        mask = np.load(RAW_FEATURE_ROOT / 'token_valid_mask.npy', mmap_mode='r')
        expected = self.base.peaks.shape[:2]
        if self.mz.shape != expected or self.intensity.shape != expected:
            raise ValueError('Canonical m/z and intensity cache shapes do not match JEPA peaks')
        if self.mz.dtype != np.float16 or self.intensity.dtype != np.float16:
            raise ValueError('Expected the original FLARE float16 feature caches')
        if mask.shape != (expected[0], expected[1] + 1) or not mask[:, -1].all():
            raise ValueError('Canonical validity mask must include an always-valid final CLS')
        if not np.array_equal(mask[:, :-1], self.base.mask):
            raise ValueError('Canonical raw features and frozen embedding masks differ')
        if not np.isfinite(self.mz).all() or not np.isfinite(self.intensity).all():
            raise ValueError('Nonfinite canonical raw peak features')

    def __getattr__(self, name):
        return getattr(self.base, name)

    def spectrum(self, rows):
        tokens, valid = self.base.spectrum(rows)
        return append_peak_features(tokens, valid, self.mz[rows], self.intensity[rows])


def load_data():
    return Data()
