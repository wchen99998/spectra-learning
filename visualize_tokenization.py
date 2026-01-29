from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from input_pipeline import (
    _DEFAULT_MAX_PRECURSOR_MZ,
    _DEFAULT_NUM_SHARDS,
    _DEFAULT_PAIR_SEQUENCE_LENGTH,
    _DEFAULT_SPLIT_SEED,
    _DEFAULT_TFRECORD_DIR,
    _DEFAULT_VALIDATION_FRACTION,
    _INTENSITY_BINS,
    _NUM_PEAKS_OUTPUT,
    _NUM_SPECIAL_TOKENS,
    _PEAK_MZ_MAX,
    _PEAK_MZ_MIN,
    _PRECURSOR_MZ_WINDOW,
    _SPECIAL_TOKENS,
    _build_single_spectrum_input,
    _compact_sort_peaks,
    _ensure_processed,
    _filter_max_precursor_mz,
    _filter_peak_mz_range,
    _parse_example,
    _strip_padding_and_tokenize,
    _topk_peaks,
    detokenize_spectrum,
)


def _load_config(path: str):
    spec = importlib.util.spec_from_file_location("dataset_config", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.get_config()


def _stash_raw_peaks(example: dict) -> dict:
    example["mz_raw"] = example["mz"]
    example["intensity_raw"] = example["intensity"]
    example["precursor_mz_raw"] = example["precursor_mz"]
    return example


def _build_visual_dataset(
    filenames: list[str],
    *,
    max_precursor_mz: float,
    max_len: int,
) -> tf.data.Dataset:
    ds = tf.data.TFRecordDataset(
        filenames,
        compression_type="GZIP",
        num_parallel_reads=tf.data.AUTOTUNE,
    )
    ds = ds.map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.filter(_filter_max_precursor_mz(max_precursor_mz))
    ds = ds.map(
        _filter_peak_mz_range(_PEAK_MZ_MIN, _PEAK_MZ_MAX, _PRECURSOR_MZ_WINDOW),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.map(_topk_peaks(_NUM_PEAKS_OUTPUT), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(_compact_sort_peaks(), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(_stash_raw_peaks, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(
        _strip_padding_and_tokenize(max_precursor_mz),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.map(
        _build_single_spectrum_input(max_len),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _extract_token_bins(token_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pad_id = _SPECIAL_TOKENS["[PAD]"]
    pad_positions = np.where(token_ids == pad_id)[0]
    end = int(pad_positions[0]) if pad_positions.size > 0 else int(token_ids.shape[0])
    content = token_ids[1:end]
    peaks = content[1:]
    mz_tokens = peaks[0::2]
    intensity_tokens = peaks[1::2]

    mz_bins = mz_tokens - _NUM_SPECIAL_TOKENS
    intensity_offset = _NUM_SPECIAL_TOKENS + int(_PEAK_MZ_MAX) + 1
    intensity_bins = intensity_tokens - intensity_offset
    return mz_bins, intensity_bins


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize tokenized spectra.")
    parser.add_argument("--config", required=True, help="Path to dataset config.")
    parser.add_argument(
        "--split",
        default="train",
        choices=("train", "validation", "massspec_test"),
        help="Dataset split to sample from.",
    )
    parser.add_argument(
        "--output",
        default="tokenization_plot.png",
        help="Output image path.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = _load_config(args.config)

    output_dir = (
        Path(cfg.get("tfrecord_dir", str(_DEFAULT_TFRECORD_DIR)))
        .expanduser()
        .resolve()
    )
    validation_fraction = float(
        cfg.get("validation_fraction", _DEFAULT_VALIDATION_FRACTION)
    )
    split_seed = int(cfg.get("split_seed", _DEFAULT_SPLIT_SEED))
    num_shards = int(cfg.get("num_shards", _DEFAULT_NUM_SHARDS))
    max_precursor_mz = float(cfg.get("max_precursor_mz", _DEFAULT_MAX_PRECURSOR_MZ))
    max_len = int(cfg.get("pair_sequence_length", _DEFAULT_PAIR_SEQUENCE_LENGTH))

    metadata = _ensure_processed(
        output_dir, validation_fraction, split_seed, num_shards
    )

    if args.split == "train":
        filenames = [
            str(output_dir / "train" / fn) for fn in metadata["train_files"]
        ]
    elif args.split == "validation":
        filenames = [
            str(output_dir / "validation" / fn)
            for fn in metadata["validation_files"]
        ]
    else:
        filenames = [
            str(output_dir / "massspec_test" / fn)
            for fn in metadata.get("massspec_test_files", [])
        ]

    ds = _build_visual_dataset(
        filenames,
        max_precursor_mz=max_precursor_mz,
        max_len=max_len,
    )
    sample = next(ds.as_numpy_iterator())

    token_ids = sample["token_ids"]
    mz_raw = sample["mz_raw"]
    intensity_raw = sample["intensity_raw"]
    precursor_raw = float(sample["precursor_mz_raw"])

    detok = detokenize_spectrum(token_ids, max_precursor_mz=max_precursor_mz)
    mz_bins, intensity_bins = _extract_token_bins(token_ids)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].scatter(mz_bins, intensity_bins, s=18, color="tab:blue")
    axes[0].set_ylabel("Intensity bin")
    axes[0].set_title("Tokenized spectrum (bins)")

    axes[1].scatter(mz_raw, intensity_raw, s=18, color="gray", label="filtered peaks")
    axes[1].scatter(
        detok["mz"],
        detok["intensity"],
        s=18,
        color="tab:red",
        label="detokenized",
    )
    axes[1].set_xlabel("m/z")
    axes[1].set_ylabel("Intensity")
    axes[1].set_title(f"Detokenized spectrum (precursor={precursor_raw:.2f})")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
