from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from ml_collections import config_dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from input_pipeline import (
    GemsNativeDataModule,
)
from models.model import PeakSetSIGReg
from utils.massspec_probe_data import MassSpecProbeData, download_massspec_tsv
from utils.training import build_model_from_config, load_config, load_pretrained_weights


def _inverse_vocab(vocab: dict[str, int]) -> dict[int, str]:
    return {int(idx): str(token) for token, idx in vocab.items()}


def _tensor_to_arrow(value: torch.Tensor) -> pa.Array:
    value = value.detach().cpu().contiguous()
    if value.ndim == 1:
        return pa.array(value.numpy())
    if value.ndim == 2:
        return pa.FixedSizeListArray.from_arrays(
            pa.array(value.reshape(-1).numpy()), int(value.shape[1])
        )
    return pa.array(value.numpy().tolist())


def _batch_to_table(
    batch: dict[str, object],
    *,
    covariance_embedding: torch.Tensor,
    adduct_vocab: dict[int, str],
    instrument_type_vocab: dict[int, str],
) -> pa.Table:
    columns: dict[str, pa.Array] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            columns[key] = _tensor_to_arrow(value)
        else:
            columns[key] = pa.array(list(value))

    adduct_ids = batch["adduct_id"].detach().cpu().tolist()
    instrument_ids = batch["instrument_type_id"].detach().cpu().tolist()
    columns["adduct"] = pa.array([adduct_vocab[int(idx)] for idx in adduct_ids])
    columns["instrument_type"] = pa.array(
        [instrument_type_vocab[int(idx)] for idx in instrument_ids]
    )
    columns["covariance_embedding"] = _tensor_to_arrow(
        covariance_embedding.detach().cpu().to(torch.float32)
    )
    return pa.table(columns)


def _build_massspec_probe_dataset_serial(
    massspec_data: MassSpecProbeData,
    split: str,
    *,
    seed: int,
    peak_ordering: str,
):
    return massspec_data.build_dataset(
        split,
        seed=seed,
        peak_ordering=peak_ordering,
        shuffle=False,
        drop_remainder=False,
        num_parallel_reads=1,
    )


def _normalize_split(split: str) -> str:
    return {
        "massspec_train": "train",
        "massspec_val": "val",
        "massspec_test": "test",
        "train": "train",
        "val": "val",
        "test": "test",
    }[split]


def _iter_split_smiles(
    config: config_dict.ConfigDict,
    split: str,
):
    fold = _normalize_split(split)
    max_precursor_mz = float(config.get("max_precursor_mz", 1000.0))
    tsv_path = download_massspec_tsv(
        Path(config.get("artifact_dir", "data/gems_artifacts"))
        .expanduser()
        .resolve()
        / "massspec_probe",
    )
    with Path(tsv_path).open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["fold"] == fold and float(row["precursor_mz"]) <= max_precursor_mz:
                yield row["smiles"]


def _encode_splits(
    *,
    splits: list[str],
    model: PeakSetSIGReg,
    config: config_dict.ConfigDict,
    massspec_data: MassSpecProbeData,
    output_path: Path,
    seed: int,
    peak_ordering: str,
    device: torch.device,
    compression: str,
    config_path: str,
    checkpoint_path: str,
) -> int:
    probe_dataset = str(config.get("probe_dataset", "massspec"))
    adduct_vocab = _inverse_vocab(massspec_data.info["massspec_adduct_vocab"])
    instrument_type_vocab = _inverse_vocab(
        massspec_data.info["massspec_instrument_type_vocab"]
    )

    writer: pq.ParquetWriter | None = None
    total_rows = 0
    for split in splits:
        dataset = _build_massspec_probe_dataset_serial(
            massspec_data,
            split,
            seed=seed,
            peak_ordering=peak_ordering,
        )
        split_label = _normalize_split(split)
        smiles_iter = (
            _iter_split_smiles(config, split)
            if probe_dataset == "massspec"
            else None
        )
        for batch in dataset:
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }

            with torch.no_grad():
                peak_mz = batch["peak_mz"]
                peak_intensity = batch["peak_intensity"]
                peak_valid_mask = batch["peak_valid_mask"]
                embeddings = model.encoder(
                    peak_mz,
                    peak_intensity,
                    valid_mask=peak_valid_mask,
                )
                peak_embeddings, _ = model.encoder.split_peak_and_cls(embeddings)
                covariance_embedding = model.covariance_pooler(
                    peak_embeddings.float(),
                    peak_valid_mask,
                )

            table = _batch_to_table(
                batch,
                covariance_embedding=covariance_embedding,
                adduct_vocab=adduct_vocab,
                instrument_type_vocab=instrument_type_vocab,
            )
            if "smiles" not in table.column_names and smiles_iter is not None:
                table = table.append_column(
                    "smiles",
                    pa.array([next(smiles_iter) for _ in range(table.num_rows)]),
                )
            table = table.append_column(
                "split", pa.array([split_label] * table.num_rows)
            )
            if writer is None:
                schema_metadata = {
                    "config_path": config_path,
                    "checkpoint_path": checkpoint_path,
                    "probe_dataset": probe_dataset,
                    "splits": json.dumps([_normalize_split(name) for name in splits]),
                    "peak_ordering": peak_ordering,
                    "embedding_variant": "covariance",
                    "covariance_pooling_dim": str(
                        model.covariance_pooler.left_proj.out_features
                    ),
                    "seed": str(seed),
                    "massspec_adduct_vocab": json.dumps(
                        massspec_data.info["massspec_adduct_vocab"]
                    ),
                    "massspec_instrument_type_vocab": json.dumps(
                        massspec_data.info["massspec_instrument_type_vocab"]
                    ),
                }
                schema = table.schema.with_metadata(
                    {
                        key.encode("utf-8"): value.encode("utf-8")
                        for key, value in schema_metadata.items()
                    }
                )
                writer = pq.ParquetWriter(
                    output_path, schema=schema, compression=compression
                )
            writer.write_table(table.cast(writer.schema))
            total_rows += table.num_rows

    if writer is not None:
        writer.close()
    return total_rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode MassSpec spectra with SIGReg encoder and export parquet."
    )
    parser.add_argument("--config", required=True, help="Path to config file.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument(
        "--output_dir", help="Directory for per-split parquet outputs."
    )
    output_group.add_argument(
        "--output_path", help="Single parquet file for one or more splits."
    )
    parser.add_argument(
        "--split",
        default="all",
        choices=("all", "train", "val", "test", "massspec_train", "massspec_val", "massspec_test"),
        help="Probe split to export.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Optional override for config batch size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for probe dataset iteration. Defaults to config.seed.",
    )
    parser.add_argument(
        "--peak_ordering",
        default=None,
        help="Override peak ordering for probe dataset. Defaults to config.peak_ordering.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help='Torch device, e.g. "cuda" or "cpu".',
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

    datamodule = GemsNativeDataModule(config, seed=int(config.seed))
    massspec_data = MassSpecProbeData.from_config(config)
    config.num_peaks = int(datamodule.info["num_peaks"])

    device = torch.device(args.device)
    model = build_model_from_config(config)
    load_pretrained_weights(model, args.checkpoint)
    model.to(device)
    model.eval()

    seed = int(config.seed) if args.seed is None else int(args.seed)
    peak_ordering = (
        str(config.get("peak_ordering", "intensity"))
        if args.peak_ordering is None
        else str(args.peak_ordering)
    )

    probe_dataset = str(config.get("probe_dataset", "massspec"))
    if args.split == "all":
        splits = (
            ["massspec_train", "massspec_val", "massspec_test"]
            if probe_dataset == "massspec"
            else ["train", "val", "test"]
        )
    else:
        splits = [args.split]

    if args.output_path is not None:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        num_rows = _encode_splits(
            splits=splits,
            model=model,
            config=config,
            massspec_data=massspec_data,
            output_path=output_path,
            seed=seed,
            peak_ordering=peak_ordering,
            device=device,
            compression=args.compression,
            config_path=str(Path(args.config).resolve()),
            checkpoint_path=str(Path(args.checkpoint).resolve()),
        )
        logging.info("Wrote %d rows to %s", num_rows, output_path)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        output_path = output_dir / f"{split}.parquet"
        num_rows = _encode_splits(
            splits=[split],
            model=model,
            config=config,
            massspec_data=massspec_data,
            output_path=output_path,
            seed=seed,
            peak_ordering=peak_ordering,
            device=device,
            compression=args.compression,
            config_path=str(Path(args.config).resolve()),
            checkpoint_path=str(Path(args.checkpoint).resolve()),
        )
        logging.info("Wrote %d rows to %s", num_rows, output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
