"""Cache frozen MoleBERT atom states for the original step-530500 FLARE trial."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
from rdkit import Chem, rdBase
import torch
from torch.utils.data import DataLoader

from molebert_atom_encoder import (
    MOLEBERT_CHECKPOINT_SHA256,
    MOLEBERT_CHECKPOINT_URL,
    MoleBERTAtomEncoder,
    featurize_smiles,
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def required_smiles(
    root: Path, bank_paths: list[Path],
) -> tuple[list[str], dict[str, object]]:
    """Use training spectra for fitting; test structures only need frozen features."""
    tsv_path = root / "msg15/MassSpecGym1.5.tsv"
    smiles_values, fold_values = [], []
    with tsv_path.open(newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            smiles_values.append(row["smiles"])
            fold_values.append(row["fold"])
    smiles = np.asarray(smiles_values, dtype=object)
    folds = np.asarray(fold_values)
    train_rows = np.flatnonzero(folds == "train")
    val_rows = np.flatnonzero(folds == "val")
    test_rows = np.flatnonzero(folds == "test")
    val_rows = np.sort(
        np.random.default_rng(0).choice(val_rows, min(2000, len(val_rows)), replace=False)
    )
    train_smiles = set(smiles[train_rows])
    needed = set(train_smiles)
    bank_metadata = []
    for path in bank_paths:
        bank = json.loads(path.read_text())
        if not set(bank).issubset(train_smiles):
            raise ValueError(f"Negative bank contains non-training query molecules: {path}")
        needed.update(bank)
        for candidates in bank.values():
            needed.update(candidates)
        bank_metadata.append({"path": str(path), "sha256": sha256(path), "queries": len(bank)})
    pool_path = root / "msg15/MassSpecGym1.5_retrieval_candidates_formula.json"
    pool = json.loads(pool_path.read_text())
    for row in np.concatenate((val_rows, test_rows)):
        needed.update(pool[smiles[row]])
    metadata = {
        "root": str(root),
        "dataset_tsv_sha256": sha256(tsv_path),
        "candidate_pools_sha256": sha256(pool_path),
        "split_sizes": {fold: int((folds == fold).sum()) for fold in ("train", "val", "test")},
        "validation_rows": val_rows.tolist(),
        "validation_seed": 0,
        "training_query_molecules": len(train_smiles),
        "negative_banks": bank_metadata,
        "candidate_coverage": "All training truths and bank candidates; fixed validation subset and all test candidate pools",
        "test_usage": "Frozen structure feature extraction only; no test spectra or retrieval labels fit the encoder",
    }
    return sorted(needed), metadata


def write_npy_from_raw(raw_path: Path, out_path: Path, shape: tuple[int, int]) -> None:
    """Add a NumPy header while copying the flat atom store once, without loading it."""
    with out_path.open("wb") as output, raw_path.open("rb") as raw:
        np.lib.format.write_array_header_2_0(output, {
            "descr": np.lib.format.dtype_to_descr(np.dtype(np.float16)),
            "fortran_order": False,
            "shape": shape,
        })
        shutil.copyfileobj(raw, output, length=8 * 1024 * 1024)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bank32", type=Path, required=True)
    parser.add_argument("--bank128", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or (args.limit is not None and args.limit < 1):
        parser.error("batch-size and limit must be positive; num-workers must be nonnegative")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.out_dir / "metadata.json"
    if metadata_path.exists():
        raise FileExistsError(f"Completed atom cache already exists: {metadata_path}")
    checkpoint_hash = sha256(args.checkpoint)
    if checkpoint_hash != MOLEBERT_CHECKPOINT_SHA256:
        raise ValueError(f"Unexpected MoleBERT checkpoint SHA-256: {checkpoint_hash}")
    started = time.monotonic()
    smiles, metadata = required_smiles(args.root, [args.bank32, args.bank128])
    required_count = len(smiles)
    if args.limit is not None:
        smiles = smiles[:args.limit]
    if not smiles:
        raise ValueError("No required candidate molecules")
    encoder = MoleBERTAtomEncoder.from_checkpoint(args.checkpoint).to(args.device).eval()
    loader = DataLoader(
        smiles, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=featurize_smiles,
        multiprocessing_context="spawn" if args.num_workers else None,
    )
    print(f"Caching {len(smiles):,}/{required_count:,} molecules on {args.device}", flush=True)
    counts: list[int] = []
    atoms_seen = 0
    progress_path = args.out_dir / "progress.json"
    with tempfile.TemporaryDirectory(prefix="molebert-atoms-") as temporary:
        raw_path = Path(temporary) / "atoms.float16.bin"
        with raw_path.open("wb") as raw, torch.inference_mode():
            for step, graph in enumerate(loader):
                for local_row, count in enumerate(graph.counts):
                    if count < 1 or count > 128:
                        value = smiles[len(counts) + local_row]
                        raise ValueError(f"Molecule has {count} atoms; cap 128 forbids truncation: {value}")
                embeddings = encoder(graph.to(args.device))
                if embeddings.shape != (sum(graph.counts), encoder.output_dim):
                    raise ValueError(f"Unexpected atom embedding shape: {tuple(embeddings.shape)}")
                array = embeddings.to(device="cpu", dtype=torch.float16).numpy()
                if not np.isfinite(array).all():
                    raise ValueError(f"Nonfinite float16 atom embeddings in batch {step}")
                raw.write(memoryview(array).cast("B"))
                counts.extend(graph.counts)
                atoms_seen += len(array)
                if step % 20 == 0 or len(counts) == len(smiles):
                    progress = {
                        "phase": "encoding", "molecules": len(counts), "total_molecules": len(smiles),
                        "atoms": atoms_seen, "elapsed_seconds": time.monotonic() - started,
                    }
                    progress_path.write_text(json.dumps(progress, indent=2))
                    print(json.dumps(progress), flush=True)
        progress_path.write_text(json.dumps({
            "phase": "writing_cache", "molecules": len(counts), "atoms": atoms_seen,
            "elapsed_seconds": time.monotonic() - started,
        }, indent=2))
        write_npy_from_raw(
            raw_path, args.out_dir / "atom_embeddings.float16.npy", (atoms_seen, encoder.output_dim),
        )
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64)))
    np.save(args.out_dir / "offsets.npy", offsets)
    (args.out_dir / "smiles.json").write_text(json.dumps(smiles))
    audit = []
    for index in sorted({0, len(smiles) // 4, len(smiles) // 2, 3 * len(smiles) // 4, len(smiles) - 1}):
        mol = Chem.MolFromSmiles(smiles[index])
        atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        if len(atomic_numbers) != counts[index]:
            raise ValueError(f"Cached atoms do not match RDKit order/count for molecule {index}")
        audit.append({"row": index, "smiles": smiles[index], "atomic_numbers_in_order": atomic_numbers})
    metadata.update({
        "complete": True,
        "encoder": "MoleBERT pretrained GIN: final node states before graph pooling",
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_source": MOLEBERT_CHECKPOINT_URL,
        "encoder_source_sha256": sha256(Path(__file__).with_name("molebert_atom_encoder.py")),
        "cache_script_sha256": sha256(Path(__file__)),
        "molecules": len(smiles), "required_molecules": required_count,
        "atoms": atoms_seen, "embedding_dim": encoder.output_dim, "dtype": "float16",
        "max_atoms": max(counts), "atom_cap": 128, "truncated_molecules": 0,
        "smoke_limit": args.limit,
        "atom_order": "Exact Chem.MolFromSmiles input atom indices; no reordering, canonicalization, hydrogen changes, or pooling",
        "atom_order_audit": audit,
        "device": args.device,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "torch_version": torch.__version__, "numpy_version": np.__version__, "rdkit_version": rdBase.rdkitVersion,
        "elapsed_seconds": time.monotonic() - started,
    })
    progress_path.write_text(json.dumps({
        "phase": "complete", "molecules": len(smiles), "atoms": atoms_seen,
        "elapsed_seconds": metadata["elapsed_seconds"],
    }, indent=2))
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"ATOM CACHE COMPLETE: {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
