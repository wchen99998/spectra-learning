from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import tensorflow as tf

from input_pipeline import (
    _DEFAULT_MAX_PRECURSOR_MZ,
    _DEFAULT_NUM_SHARDS,
    _DEFAULT_PAIR_SEQUENCE_LENGTH,
    _DEFAULT_SPLIT_SEED,
    _DEFAULT_TFRECORD_DIR,
    _DEFAULT_VALIDATION_FRACTION,
    _NUM_PEAKS_OUTPUT,
    _PEAK_MZ_MAX,
    _PEAK_MZ_MIN,
    _PRECURSOR_MZ_WINDOW,
    _compact_sort_peaks,
    _ensure_processed,
    _filter_max_precursor_mz,
    _filter_peak_mz_range,
    _parse_example,
    _topk_peaks,
)


def _load_config(path: str):
    spec = importlib.util.spec_from_file_location("dataset_config", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.get_config()


def _length_from_peaks(max_len: int):
    max_len_t = tf.constant(max_len, tf.int32)
    max_peaks = tf.constant((max_len - 2) // 2, tf.int32)

    def apply(example: dict) -> tf.Tensor:
        count = tf.reduce_sum(tf.cast(example["mz"] > 0, tf.int32))
        count = tf.minimum(count, max_peaks)
        length = 2 + 2 * count
        return tf.minimum(length, max_len_t)

    return apply


def _build_length_dataset(
    filenames: list[str],
    *,
    max_precursor_mz: float,
    max_len: int,
    batch_size: int,
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
    ds = ds.map(_length_from_peaks(max_len), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _parse_percentiles(raw: str) -> list[float]:
    return [float(x) for x in raw.split(",") if x.strip()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure token length statistics for spectra.",
    )
    parser.add_argument("--config", required=True, help="Path to dataset config.")
    parser.add_argument(
        "--split",
        default="train",
        choices=("train", "validation", "massspec_test"),
        help="Dataset split to scan.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5_000_000,
        help="Maximum number of spectra to scan.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size for tf.data iteration.",
    )
    parser.add_argument(
        "--percentiles",
        default="0,1,5,10,25,50,75,90,95,99,99.5,99.9,100",
        help="Comma-separated list of percentiles to report.",
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

    ds = _build_length_dataset(
        filenames,
        max_precursor_mz=max_precursor_mz,
        max_len=max_len,
        batch_size=int(args.batch_size),
    )

    max_samples = int(args.max_samples)
    lengths = np.empty(max_samples, dtype=np.int32)
    seen = 0

    for batch in ds.as_numpy_iterator():
        remaining = max_samples - seen
        if remaining <= 0:
            break
        take = min(len(batch), remaining)
        lengths[seen : seen + take] = batch[:take]
        seen += take

    lengths = lengths[:seen]
    percentiles = _parse_percentiles(args.percentiles)
    values = np.percentile(lengths, percentiles)

    print(f"samples={seen}")
    print(f"mean={lengths.mean():.4f}")
    print(f"min={lengths.min()} max={lengths.max()}")
    for p, v in zip(percentiles, values):
        print(f"p{p:g}={v:.4f}")


if __name__ == "__main__":
    main()
