from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from tqdm import tqdm

from spectra_learning.config import load_config
from spectra_learning.data.mgf import _to_float, iter_mgf
from spectra_learning.data.spectra import (
    DEFAULT_GROUPED_PEAK_ISOTOPE_CHARGES,
    DEFAULT_GROUPED_PEAK_SHOULDER_DA,
    DEFAULT_MAX_PRECURSOR_MZ,
    DEFAULT_MIN_PEAK_INTENSITY,
    DEFAULT_PEAK_FILTERING,
    NUM_PEAKS_INPUT,
    preprocess_peak_batch_torch,
)
from spectra_learning.models.factory import build_model_from_config
from spectra_learning.models.model import PeakSetJEPA
from spectra_learning.models.pooling import CovariancePool
from spectra_learning.data.massspec_probe import MassSpecProbeData
from spectra_learning.probes.massspec.msg_probe import iter_massspec_probe
from spectra_learning.training.checkpointing import load_torch_checkpoint
from spectra_learning.training.storage import StoragePath, normalize_storage_path


log = logging.getLogger(__name__)

PEAK_COLUMNS = {"peak_mz", "peak_intensity"}

METADATA_COLUMNS = (
    "name",
    "title",
    "pepmass",
    "charge",
    "precursortype",
    "collisionenergy",
    "instrument",
    "instrumenttype",
    "ionmode",
    "spectrumtype",
    "formula",
    "exactmass",
    "inchikey",
    "smiles",
    "rtinseconds",
    "db",
    "db_ref",
    "splash",
    "comment",
    "peakannotations",
)


def _load_checkpoint_for_encoder(
    model: PeakSetJEPA,
    checkpoint_path: StoragePath,
) -> dict[str, Any]:
    checkpoint = load_torch_checkpoint(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model"])
    return {
        "global_step": checkpoint["global_step"],
        "epoch": checkpoint["epoch"],
        "loss": checkpoint["loss"],
    }


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _peak_tokens_only(
    token_embeddings: torch.Tensor,
    peak_valid_mask: torch.Tensor,
) -> torch.Tensor:
    return token_embeddings[:, : peak_valid_mask.shape[1]]


def _build_maccs_head(
    *,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    num_layers: int,
) -> torch.nn.Module:
    if num_layers == 1:
        return torch.nn.Linear(input_dim, output_dim)
    layers: list[torch.nn.Module] = [
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.SiLU(),
    ]
    for _ in range(num_layers - 2):
        layers.extend([torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.SiLU()])
    layers.append(torch.nn.Linear(hidden_dim, output_dim))
    return torch.nn.Sequential(*layers)


@torch.no_grad()
def _encode_peak_tokens(
    model: PeakSetJEPA,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    use_autocast = device.type == "cuda"
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=use_autocast,
    ):
        encoded = model.encoder(
            batch["peak_mz"],
            batch["peak_intensity"],
            valid_mask=batch["peak_valid_mask"],
            precursor_mz=batch.get("precursor_mz", None),
        )
        return _peak_tokens_only(encoded, batch["peak_valid_mask"])


def train_covariance_pooler(
    *,
    model: PeakSetJEPA,
    config: Any,
    device: torch.device,
    epochs: int,
    max_train_samples: int | None,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> tuple[CovariancePool, torch.nn.Module, dict[str, Any]]:
    compressed_dim = int(config.get("covariance_pooling_dim", 64))
    probe_data = MassSpecProbeData.from_config(config)
    maccs_bits = int(probe_data.info["probe_maccs_bits"])
    pooler = CovariancePool(
        input_dim=int(config.get("model_dim", model.model_dim)),
        compressed_dim=compressed_dim,
    ).to(device)
    maccs_head = _build_maccs_head(
        input_dim=compressed_dim * compressed_dim,
        output_dim=maccs_bits,
        hidden_dim=int(config.get("msg_probe_mlp_hidden_dim", model.model_dim)),
        num_layers=int(config.get("msg_probe_mlp_num_layers", 2)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        [*pooler.parameters(), *maccs_head.parameters()],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    peak_ordering = str(config.get("peak_ordering", "mz"))
    losses: list[float] = []
    samples_seen = 0
    model.eval()
    model.requires_grad_(False)
    pooler.train()
    maccs_head.train()
    for epoch in range(epochs):
        progress = tqdm(
            iter_massspec_probe(
                probe_data=probe_data,
                split="massspec_train",
                seed=seed + epoch,
                peak_ordering=peak_ordering,
                drop_remainder=False,
                max_samples=max_train_samples,
            ),
            desc=f"training covariance pooler epoch {epoch + 1}/{epochs}",
            unit="batch",
        )
        for batch in progress:
            batch = _move_batch(batch, device)
            valid_mol = batch["probe_valid_mol"].to(device=device, dtype=torch.bool)
            if not bool(valid_mol.any()):
                continue
            peak_embeddings = _encode_peak_tokens(model, batch, device)
            pooled = pooler(
                peak_embeddings.float(),
                batch["peak_valid_mask"].to(dtype=torch.bool),
            )
            logits = maccs_head(pooled[valid_mol])
            target = batch["probe_maccs"][valid_mol].to(dtype=torch.float32)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            samples_seen += int(valid_mol.sum().detach().cpu())
            progress.set_postfix(loss=f"{loss_value:.4f}")
    pooler.eval()
    pooler.requires_grad_(False)
    maccs_head.eval()
    maccs_head.requires_grad_(False)
    return pooler, maccs_head, {
        "target_key": "probe_maccs",
        "loss": "binary_cross_entropy_with_logits",
        "maccs_bits": maccs_bits,
        "compressed_dim": compressed_dim,
        "epochs": epochs,
        "max_train_samples": max_train_samples,
        "samples_seen": samples_seen,
        "num_steps": len(losses),
        "final_loss": losses[-1] if losses else None,
        "mean_loss": float(np.mean(losses)) if losses else None,
    }


def _precursor_mz(record: dict[str, Any]) -> float:
    pepmass = str(record.get("pepmass", ""))
    precursor = _to_float(pepmass.split()[0] if pepmass else None)
    return precursor if math.isfinite(precursor) else 0.0


def _record_metadata(record: dict[str, Any]) -> dict[str, str]:
    return {
        key: str(value)
        for key, value in record.items()
        if key not in PEAK_COLUMNS
    }


def _pack_peak_batch(
    records: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mz = np.zeros((len(records), NUM_PEAKS_INPUT), dtype=np.float32)
    intensity = np.zeros((len(records), NUM_PEAKS_INPUT), dtype=np.float32)
    precursor = np.zeros(len(records), dtype=np.float32)
    for i, record in enumerate(records):
        peak_mz = record["peak_mz"].astype(np.float32, copy=False)
        peak_intensity = record["peak_intensity"].astype(np.float32, copy=False)
        if peak_mz.size > NUM_PEAKS_INPUT:
            idx = np.argpartition(-peak_intensity, kth=NUM_PEAKS_INPUT - 1)[
                :NUM_PEAKS_INPUT
            ]
            peak_mz = peak_mz[idx]
            peak_intensity = peak_intensity[idx]
        n = min(int(peak_mz.size), NUM_PEAKS_INPUT)
        mz[i, :n] = peak_mz[:n]
        intensity[i, :n] = peak_intensity[:n]
        precursor[i] = _precursor_mz(record)
    return (
        torch.from_numpy(mz),
        torch.from_numpy(intensity),
        torch.from_numpy(precursor),
    )


def _preprocess_mgf_batch(
    records: list[dict[str, Any]],
    config: Any,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    mz, intensity, precursor = _pack_peak_batch(records)
    batch = preprocess_peak_batch_torch(
        mz,
        intensity,
        precursor,
        num_peaks=int(config.get("num_peaks", 64)),
        peak_drop_min_intensity=float(
            config.get(
                "peak_drop_min_intensity",
                config.get("min_peak_intensity", DEFAULT_MIN_PEAK_INTENSITY),
            )
        ),
        peak_ordering=str(config.get("peak_ordering", "mz")),
        max_precursor_mz=float(
            config.get("max_precursor_mz", DEFAULT_MAX_PRECURSOR_MZ)
        ),
        precursor_peak_exclusion_window_da=float(
            config.get("precursor_peak_exclusion_window_da", 0.0)
        ),
        min_peak_intensity=float(
            config.get("min_peak_intensity", DEFAULT_MIN_PEAK_INTENSITY)
        ),
        peak_filtering=str(config.get("peak_filtering", DEFAULT_PEAK_FILTERING)),
        grouped_peak_shoulder_da=float(
            config.get(
                "grouped_peak_shoulder_da",
                DEFAULT_GROUPED_PEAK_SHOULDER_DA,
            )
        ),
        grouped_peak_isotope_charges=tuple(
            int(charge)
            for charge in config.get(
                "grouped_peak_isotope_charges",
                DEFAULT_GROUPED_PEAK_ISOTOPE_CHARGES,
            )
        ),
    )
    return {key: value.to(device) for key, value in batch.items()}


def _embedding_array(embeddings: np.ndarray) -> pa.FixedSizeListArray:
    flat = pa.array(embeddings.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, embeddings.shape[1])


def _string_array(values: list[str]) -> pa.Array:
    return pa.array(values, type=pa.string())


def _metadata_column_values(
    records: list[dict[str, Any]],
    column: str,
) -> list[str]:
    return [str(record.get(column, "")) for record in records]


def _parquet_table(
    *,
    start_index: int,
    records: list[dict[str, Any]],
    embeddings: np.ndarray,
    model_peak_counts: np.ndarray,
) -> pa.Table:
    metadata = [_record_metadata(record) for record in records]
    arrays: list[pa.Array] = [
        pa.array(
            np.arange(start_index, start_index + len(records), dtype=np.int64),
            type=pa.int64(),
        ),
        _embedding_array(embeddings.astype(np.float32, copy=False)),
        pa.array([_precursor_mz(record) for record in records], type=pa.float32()),
        pa.array([int(record["peak_mz"].size) for record in records], type=pa.int32()),
        pa.array(model_peak_counts.astype(np.int32, copy=False), type=pa.int32()),
    ]
    names = [
        "spectrum_index",
        "embedding",
        "precursor_mz",
        "num_raw_peaks",
        "num_model_peaks",
    ]
    for column in METADATA_COLUMNS:
        arrays.append(_string_array(_metadata_column_values(records, column)))
        names.append(column)
    arrays.extend(
        [
            _string_array([json.dumps(item, sort_keys=True) for item in metadata]),
            pa.array(
                [record["peak_mz"].astype(np.float32, copy=False) for record in records],
                type=pa.list_(pa.float32()),
            ),
            pa.array(
                [
                    record["peak_intensity"].astype(np.float32, copy=False)
                    for record in records
                ],
                type=pa.list_(pa.float32()),
            ),
        ]
    )
    names.extend(["metadata_json", "spectrum_mz", "spectrum_intensity"])
    return pa.Table.from_arrays(arrays, names=names)


@torch.no_grad()
def embed_mgf_to_parquet(
    *,
    model: PeakSetJEPA,
    pooler: CovariancePool,
    config: Any,
    mgf_path: Path,
    output_dir: Path,
    batch_size: int,
    max_spectra: int | None,
    device: torch.device,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    pooler.eval()
    rows_total = 0
    writer: pq.ParquetWriter | None = None
    output_path = output_dir / "embeddings.parquet"

    def close_writer() -> None:
        nonlocal writer
        if writer is not None:
            writer.close()
            writer = None

    def write_table(table: pa.Table) -> None:
        nonlocal writer
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
        writer.write_table(table)

    records: list[dict[str, Any]] = []
    progress = tqdm(iter_mgf(mgf_path), desc="embedding MGF", unit="spec")
    for record in progress:
        records.append(record)
        if len(records) < batch_size:
            if max_spectra is None or rows_total + len(records) < max_spectra:
                continue
        if max_spectra is not None:
            records = records[: max_spectra - rows_total]
        batch = _preprocess_mgf_batch(records, config, device)
        peak_embeddings = _encode_peak_tokens(model, batch, device)
        embeddings = pooler(
            peak_embeddings.float(),
            batch["peak_valid_mask"].to(dtype=torch.bool),
        )
        table = _parquet_table(
            start_index=rows_total,
            records=records,
            embeddings=embeddings.detach().cpu().numpy(),
            model_peak_counts=(
                batch["peak_valid_mask"].sum(dim=1).detach().cpu().numpy()
            ),
        )
        write_table(table)
        rows_total += len(records)
        records = []
        progress.set_postfix(rows=rows_total)
        if max_spectra is not None and rows_total >= max_spectra:
            break
    if records:
        batch = _preprocess_mgf_batch(records, config, device)
        peak_embeddings = _encode_peak_tokens(model, batch, device)
        embeddings = pooler(
            peak_embeddings.float(),
            batch["peak_valid_mask"].to(dtype=torch.bool),
        )
        table = _parquet_table(
            start_index=rows_total,
            records=records,
            embeddings=embeddings.detach().cpu().numpy(),
            model_peak_counts=(
                batch["peak_valid_mask"].sum(dim=1).detach().cpu().numpy()
            ),
        )
        write_table(table)
        rows_total += len(records)
    close_writer()
    return {
        "rows": rows_total,
        "parquet": output_path.name,
        "embedding_dim": pooler.compressed_dim * pooler.compressed_dim,
        "metadata_columns": list(METADATA_COLUMNS),
    }


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a covariance pooler on NIST-Murcko probe data, then embed a "
            "NIST MGF into one parquet file."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mgf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pooler-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default=_default_device())
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--pooler-epochs", type=int, default=None)
    parser.add_argument("--pooler-max-train-samples", type=int, default=None)
    parser.add_argument("--pooler-lr", type=float, default=1e-3)
    parser.add_argument("--pooler-weight-decay", type=float, default=0.0)
    parser.add_argument("--max-spectra", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    device = torch.device(args.device)
    config = load_config(args.config)
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else int(
            config.get(
                "msg_probe_batch_size",
                config.get("batch_size", 256),
            )
        )
    )
    pooler_epochs = (
        args.pooler_epochs
        if args.pooler_epochs is not None
        else int(config.get("msg_probe_num_epochs", 1))
    )
    pooler_checkpoint = args.pooler_checkpoint or (
        args.output_dir / "covariance_pooler.pt"
    )

    log.info("building model from %s", args.config)
    model = build_model_from_config(config)
    checkpoint_info = _load_checkpoint_for_encoder(
        model,
        normalize_storage_path(args.checkpoint),
    )
    model.to(device)
    model.eval()
    model.requires_grad_(False)

    log.info("training covariance pooler")
    pooler, maccs_head, pooler_info = train_covariance_pooler(
        model=model,
        config=config,
        device=device,
        epochs=pooler_epochs,
        max_train_samples=args.pooler_max_train_samples,
        learning_rate=args.pooler_lr,
        weight_decay=args.pooler_weight_decay,
        seed=args.seed,
    )
    pooler_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "pooler": _cpu_state_dict(pooler),
            "maccs_head": _cpu_state_dict(maccs_head),
            "pooler_info": pooler_info,
            "config_path": str(args.config),
            "checkpoint_path": str(args.checkpoint),
        },
        pooler_checkpoint,
    )
    log.info("saved covariance pooler to %s", pooler_checkpoint)

    log.info("embedding %s", args.mgf)
    output_info = embed_mgf_to_parquet(
        model=model,
        pooler=pooler,
        config=config,
        mgf_path=args.mgf,
        output_dir=args.output_dir,
        batch_size=batch_size,
        max_spectra=args.max_spectra,
        device=device,
    )
    manifest = {
        "config_path": str(args.config),
        "checkpoint_path": str(args.checkpoint),
        "mgf_path": str(args.mgf),
        "pooler_checkpoint": str(pooler_checkpoint),
        "batch_size": batch_size,
        "device": str(device),
        "checkpoint": checkpoint_info,
        "pooler": pooler_info,
        "output": output_info,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metadata.json").write_text(json.dumps(manifest, indent=2))
    log.info("wrote %d rows to %s", output_info["rows"], args.output_dir)


if __name__ == "__main__":
    main()
