from __future__ import annotations

import argparse
import csv
import importlib.util
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
    TfLightningDataModule,
    _MASSSPEC_HF_REPO,
    _MASSSPEC_TSV_PATH,
    _build_dataset,
    _download_hf_file,
    numpy_batch_to_torch,
)
from models.model import PeakSetSIGReg


def _load_config(path: str) -> config_dict.ConfigDict:
    spec = importlib.util.spec_from_file_location("export_embeddings_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()


def _build_model_from_config(config: config_dict.ConfigDict) -> PeakSetSIGReg:
    return PeakSetSIGReg(
        num_peaks=int(config.num_peaks),
        model_dim=int(config.model_dim),
        encoder_num_layers=int(config.num_layers),
        encoder_num_heads=int(config.num_heads),
        encoder_num_kv_heads=config.get("num_kv_heads", None),
        attention_mlp_multiple=float(config.attention_mlp_multiple),
        feature_mlp_hidden_dim=int(config.get("feature_mlp_hidden_dim", 128)),
        mz_fourier_num_frequencies=int(config.get("mz_fourier_num_frequencies", 32)),
        mz_fourier_min_freq=float(config.get("mz_fourier_min_freq", 1.0)),
        mz_fourier_max_freq=float(config.get("mz_fourier_max_freq", 100.0)),
        mz_fourier_learnable=bool(config.get("mz_fourier_learnable", False)),
        pooling_type=str(config.get("pooling_type", "pma")),
        pma_num_heads=config.get("pma_num_heads", int(config.num_heads)),
        pma_num_seeds=int(config.get("pma_num_seeds", 1)),
        sigreg_use_projector=bool(config.get("sigreg_use_projector", True)),
        sigreg_proj_hidden_dim=int(config.get("sigreg_proj_hidden_dim", 2048)),
        sigreg_proj_output_dim=int(config.get("sigreg_proj_output_dim", 128)),
        bcs_num_slices=int(config.get("sigreg_num_slices", 256)),
        sigreg_lambda=float(config.get("sigreg_lambda", 10.0)),
        sigreg_drop_prob=float(config.get("sigreg_drop_prob", 0.20)),
        sigreg_mz_jitter_std=float(config.get("sigreg_mz_jitter_std", 0.005)),
        sigreg_intensity_jitter_std=float(config.get("sigreg_intensity_jitter_std", 0.05)),
    )


def _load_checkpoint(model: PeakSetSIGReg, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model_state = {
        key.removeprefix("model."): value
        for key, value in state_dict.items()
        if key.startswith("model.")
    }
    if not model_state:
        model_state = state_dict
    model.load_state_dict(model_state, strict=True)


def _inverse_vocab(vocab: dict[str, int]) -> dict[int, str]:
    return {int(idx): str(token) for token, idx in vocab.items()}


def _batch_to_table(
    batch: dict[str, torch.Tensor],
    *,
    encoder_embedding: torch.Tensor,
    projector_embedding: torch.Tensor,
    adduct_vocab: dict[int, str],
    instrument_type_vocab: dict[int, str],
) -> pa.Table:
    columns: dict[str, pa.Array] = {}
    for key, value in batch.items():
        values = value.detach().cpu().numpy()
        columns[key] = pa.array(values.tolist())

    adduct_ids = batch["adduct_id"].detach().cpu().tolist()
    instrument_ids = batch["instrument_type_id"].detach().cpu().tolist()
    columns["adduct"] = pa.array([adduct_vocab[int(idx)] for idx in adduct_ids])
    columns["instrument_type"] = pa.array(
        [instrument_type_vocab[int(idx)] for idx in instrument_ids]
    )
    columns["encoder_embedding"] = pa.array(
        encoder_embedding.detach().cpu().to(torch.float32).tolist()
    )
    columns["projector_embedding"] = pa.array(
        projector_embedding.detach().cpu().to(torch.float32).tolist()
    )
    return pa.table(columns)


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
        mz_representation=datamodule.mz_representation,
        peak_ordering=peak_ordering,
        num_parallel_reads=1,
    )


def _iter_split_smiles(
    config: config_dict.ConfigDict,
    split: str,
):
    fold = {
        "massspec_train": "train",
        "massspec_val": "val",
        "massspec_test": "test",
    }[split]
    max_precursor_mz = float(config.get("max_precursor_mz", 1000.0))
    tsv_path = _download_hf_file(
        _MASSSPEC_HF_REPO,
        _MASSSPEC_TSV_PATH,
        Path(config.get("tfrecord_dir", "data/gems_peaklist_tfrecord")).expanduser().resolve().parent,
    )
    with Path(tsv_path).open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["fold"] == fold and float(row["precursor_mz"]) <= max_precursor_mz:
                yield row["smiles"]


def _encode_split(
    *,
    split: str,
    model: PeakSetSIGReg,
    config: config_dict.ConfigDict,
    datamodule: TfLightningDataModule,
    output_path: Path,
    seed: int,
    peak_ordering: str,
    device: torch.device,
    compression: str,
    config_path: str,
    checkpoint_path: str,
) -> int:
    dataset = _build_massspec_probe_dataset_serial(
        datamodule,
        split,
        seed=seed,
        peak_ordering=peak_ordering,
    )
    smiles_iter = _iter_split_smiles(config, split)
    adduct_vocab = _inverse_vocab(datamodule.info["massspec_adduct_vocab"])
    instrument_type_vocab = _inverse_vocab(datamodule.info["massspec_instrument_type_vocab"])

    writer: pq.ParquetWriter | None = None
    total_rows = 0
    for numpy_batch in dataset.as_numpy_iterator():
        batch = numpy_batch_to_torch(numpy_batch)
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

        with torch.no_grad():
            embeddings = model.encoder(
                batch["peak_mz"],
                batch["peak_intensity"],
                batch["precursor_mz"],
            )
            pooled_encoder = model.pool(embeddings, batch["peak_valid_mask"])
            pooled_projector = model.projector(pooled_encoder)

        table = _batch_to_table(
            batch,
            encoder_embedding=pooled_encoder,
            projector_embedding=pooled_projector,
            adduct_vocab=adduct_vocab,
            instrument_type_vocab=instrument_type_vocab,
        )
        table = table.append_column(
            "smiles",
            pa.array([next(smiles_iter) for _ in range(table.num_rows)]),
        )
        if writer is None:
            schema_metadata = {
                "config_path": config_path,
                "checkpoint_path": checkpoint_path,
                "split": split,
                "peak_ordering": peak_ordering,
                "seed": str(seed),
                "massspec_adduct_vocab": json.dumps(datamodule.info["massspec_adduct_vocab"]),
                "massspec_instrument_type_vocab": json.dumps(
                    datamodule.info["massspec_instrument_type_vocab"]
                ),
            }
            schema = table.schema.with_metadata(
                {key.encode("utf-8"): value.encode("utf-8") for key, value in schema_metadata.items()}
            )
            writer = pq.ParquetWriter(output_path, schema=schema, compression=compression)
        writer.write_table(table.cast(writer.schema))
        total_rows += table.num_rows

    if writer is not None:
        writer.close()
    return total_rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode MassSpec spectra with SIGReg encoder/projector and export parquet."
    )
    parser.add_argument("--config", required=True, help="Path to config file.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--output_dir", required=True, help="Directory for parquet outputs.")
    parser.add_argument(
        "--split",
        default="all",
        choices=("all", "massspec_train", "massspec_val", "massspec_test"),
        help="MassSpec split to export.",
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
        help="Override peak ordering for probe dataset. Defaults to config.probe_peak_ordering.",
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
    config = _load_config(args.config)
    if args.batch_size is not None:
        config.batch_size = int(args.batch_size)

    datamodule = TfLightningDataModule(config, seed=int(config.seed))
    config.num_peaks = int(datamodule.info["num_peaks"])

    device = torch.device(args.device)
    model = _build_model_from_config(config)
    _load_checkpoint(model, args.checkpoint)
    model.to(device)
    model.eval()

    seed = int(config.seed) if args.seed is None else int(args.seed)
    peak_ordering = (
        str(config.get("probe_peak_ordering", "intensity"))
        if args.peak_ordering is None
        else str(args.peak_ordering)
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.split == "all":
        splits = ["massspec_train", "massspec_val", "massspec_test"]
    else:
        splits = [args.split]

    for split in splits:
        output_path = output_dir / f"{split}.parquet"
        num_rows = _encode_split(
            split=split,
            model=model,
            config=config,
            datamodule=datamodule,
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
