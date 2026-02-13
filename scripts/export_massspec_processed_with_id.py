from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from input_pipeline import (
    TfLightningDataModule,
    _MASSSPEC_HF_REPO,
    _MASSSPEC_TSV_PATH,
    _build_dataset,
    _download_hf_file,
)
from utils.training import load_config

_SPLIT_TO_FOLD = {
    "massspec_train": "train",
    "massspec_val": "val",
    "massspec_test": "test",
}
_FOLD_TO_SPLIT = {v: k for k, v in _SPLIT_TO_FOLD.items()}


def _build_massspec_probe_dataset_serial(
    datamodule: TfLightningDataModule,
    split: str,
    *,
    seed: int,
    peak_ordering: str,
):
    if split == "massspec_train":
        files = datamodule.massspec_train_files
    elif split == "massspec_val":
        files = datamodule.massspec_val_files
    else:
        files = datamodule.massspec_test_files

    return _build_dataset(
        files,
        datamodule.batch_size,
        shuffle_buffer=0,
        seed=seed,
        drop_remainder=False,
        tfrecord_buffer_size=datamodule.tfrecord_buffer_size,
        max_precursor_mz=datamodule.max_precursor_mz,
        include_fingerprint=True,
        intensity_scaling=datamodule.intensity_scaling,
        min_peak_intensity=datamodule.min_peak_intensity,
        mz_representation=datamodule.mz_representation,
        include_sigreg_augmentation=False,
        sigreg_contiguous_mask_fraction=datamodule.sigreg_contiguous_mask_fraction,
        sigreg_contiguous_mask_min_len=datamodule.sigreg_contiguous_mask_min_len,
        sigreg_mz_jitter_std=datamodule.sigreg_mz_jitter_std,
        sigreg_intensity_jitter_std=datamodule.sigreg_intensity_jitter_std,
        peak_ordering=peak_ordering,
        num_parallel_reads=1,
    )


def _load_split_ids(
    datamodule: TfLightningDataModule,
    tfrecord_dir: Path,
) -> dict[str, list[str]]:
    tsv_path = _download_hf_file(
        _MASSSPEC_HF_REPO,
        _MASSSPEC_TSV_PATH,
        tfrecord_dir.parent,
    )
    ids = {
        "massspec_train": [],
        "massspec_val": [],
        "massspec_test": [],
    }
    with Path(tsv_path).open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            precursor_mz = float(row["precursor_mz"])
            if precursor_mz > datamodule.max_precursor_mz:
                continue
            split = _FOLD_TO_SPLIT[row["fold"]]
            ids[split].append(row["identifier"])
    return ids


def _batch_to_table(
    batch: dict[str, object],
    *,
    split: str,
    ids: list[str],
    take: int,
) -> pa.Table:
    columns: dict[str, pa.Array] = {}
    for key, value in batch.items():
        columns[key] = pa.array(value[:take].tolist())
    columns["massspecgym_id"] = pa.array(ids[:take])
    columns["split"] = pa.array([split] * take)
    return pa.table(columns)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export processed MassSpecGym rows with MassSpecGym IDs to parquet."
    )
    parser.add_argument("--config", required=True, help="Path to config file.")
    parser.add_argument("--output", required=True, help="Output parquet path.")
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum number of rows to export.",
    )
    parser.add_argument(
        "--split",
        default="all",
        choices=("all", "massspec_train", "massspec_val", "massspec_test"),
        help="Which split to export.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Optional config batch size override.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed override.",
    )
    parser.add_argument(
        "--peak_ordering",
        default=None,
        help="Optional peak ordering override.",
    )
    parser.add_argument(
        "--compression",
        default="zstd",
        help="Parquet compression codec.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    if args.batch_size is not None:
        config.batch_size = int(args.batch_size)

    datamodule = TfLightningDataModule(config, seed=int(config.seed))
    seed = int(config.seed) if args.seed is None else int(args.seed)
    peak_ordering = (
        str(config.get("peak_ordering", "intensity"))
        if args.peak_ordering is None
        else str(args.peak_ordering)
    )

    if args.split == "all":
        splits = ["massspec_train", "massspec_val", "massspec_test"]
    else:
        splits = [args.split]

    tfrecord_dir = Path(config.get("tfrecord_dir", "data/gems_peaklist_tfrecord")).expanduser().resolve()
    split_ids = _load_split_ids(datamodule, tfrecord_dir)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer: pq.ParquetWriter | None = None
    total_rows = 0
    for split in splits:
        dataset = _build_massspec_probe_dataset_serial(
            datamodule,
            split,
            seed=seed,
            peak_ordering=peak_ordering,
        )
        id_offset = 0
        ids = split_ids[split]
        for batch in dataset.as_numpy_iterator():
            remaining = int(args.limit) - total_rows
            if remaining <= 0:
                break
            batch_size = int(batch["precursor_mz"].shape[0])
            take = min(batch_size, remaining)
            table = _batch_to_table(
                batch,
                split=split,
                ids=ids[id_offset : id_offset + batch_size],
                take=take,
            )
            if writer is None:
                schema_metadata = {
                    "config_path": str(Path(args.config).resolve()),
                    "seed": str(seed),
                    "peak_ordering": peak_ordering,
                    "limit": str(args.limit),
                }
                schema = table.schema.with_metadata(
                    {
                        key.encode("utf-8"): value.encode("utf-8")
                        for key, value in schema_metadata.items()
                    }
                )
                writer = pq.ParquetWriter(
                    output_path,
                    schema=schema,
                    compression=args.compression,
                )
            writer.write_table(table.cast(writer.schema))
            total_rows += take
            id_offset += batch_size
        if total_rows >= int(args.limit):
            break

    if writer is not None:
        writer.close()

    print(f"Wrote {total_rows} rows to {output_path}")


if __name__ == "__main__":
    main()
