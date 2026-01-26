# Intensity binning notes (GeMS peaklist)

## Sampling and distribution
- Source: `data/gems_peaklist_tfrecord/train` (TFRecord shards)
- Subsample size: 2,000,000 intensity values
- Zero fraction (padding): 0.539472
- Non-zero stats (sample):
  - min: 2.5023e-05
  - median: 0.025236
  - 90%: 0.191711
  - 95%: 0.367087
  - 99%: 1.0
- log10(non-zero) range: ~[-4.60, 0]

## Scaling for discretization (B=128 tokens)
Padding is removed. Vocabulary includes special tokens plus m/z, precursor m/z, and intensity tokens with non-overlapping ranges.

Formulas:
```
EPS = 1e-4
B = 128
SPECIAL = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3}
MZ_BINS = int(PEAK_MZ_MAX) + 1       # bin width = 1 for mz
PRECURSOR_BINS = int(max_precursor_mz) + 1

mz_token = floor(mz) + len(SPECIAL)                                      # [4, 4+MZ_BINS-1]
precursor_token = floor(clip(precursor_mz)) + len(SPECIAL) + MZ_BINS      # [4+MZ_BINS, 4+MZ_BINS+PRECURSOR_BINS-1]

x = clip(intensity, EPS, 1.0)
s = (log10(x) - log10(EPS)) / (0 - log10(EPS))  # s in [0, 1]
intensity_token = floor(s * (B - 1)) + len(SPECIAL) + MZ_BINS + PRECURSOR_BINS  # [offset, offset+127]

vocab_size = len(SPECIAL) + MZ_BINS + PRECURSOR_BINS + B
```

Rationale:
- Intensity is strongly log-skewed (log10 non-zero spans ~4.6 orders).
- eps=1e-4 only clips ~0.009% of non-zero values (negligible).
- Log-uniform bins give better coverage across the tail than linear bins.

Alternative tested (not chosen): power-law s = x**0.188 (median maps to 0.5).

## Occupancy (5,000,000 samples)
- Zero token fraction: 0.538673
- Non-zero token fraction: 0.461327

## Plots
- Mapping curve: `experiments/intensity_bins_128.png`
- Occupancy (token 0 removed): `experiments/intensity_bins_128_occupancy_5M.png`
