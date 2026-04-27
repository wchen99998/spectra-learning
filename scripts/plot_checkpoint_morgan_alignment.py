from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from input_pipeline import _prepend_precursor_token_torch
from utils.spectra_preprocessing import preprocess_peak_batch_torch
from utils.training import build_model_from_config, load_config, load_pretrained_weights


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"step-(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def _load_split_arrays(artifact_dir: Path, split: str):
    metadata = json.loads((artifact_dir / "metadata.json").read_text())
    split_names = ("train", "val", "test") if split == "all" else (split,)
    shard_dirs = [
        artifact_dir / split_name / name
        for split_name in split_names
        for name in metadata[f"{split_name}_files"]
    ]
    lengths = [
        int(v)
        for split_name in split_names
        for v in metadata[f"{split_name}_lengths"]
    ]
    return {
        "metadata": metadata,
        "shard_dirs": shard_dirs,
        "lengths": lengths,
        "smiles": np.concatenate(
            [np.load(shard_dir / "smiles.npy", mmap_mode="r") for shard_dir in shard_dirs]
        ),
        "probe_valid_mol": np.concatenate(
            [
                np.load(shard_dir / "probe_valid_mol.npy", mmap_mode="r")
                for shard_dir in shard_dirs
            ]
        ),
    }


def _take_from_shards(shard_dirs: list[Path], lengths: list[int], key: str, indices: np.ndarray):
    starts = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(np.asarray(lengths, dtype=np.int64), out=starts[1:])
    result = []
    for shard_idx, shard_dir in enumerate(shard_dirs):
        start, end = starts[shard_idx], starts[shard_idx + 1]
        mask = (indices >= start) & (indices < end)
        if np.any(mask):
            local = indices[mask] - start
            result.append(np.load(shard_dir / f"{key}.npy", mmap_mode="r")[local])
    return np.concatenate(result, axis=0)


def _representative_smiles(smiles: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, list[str]]:
    raw_reps: dict[str, int] = {}
    for idx, (smi, ok) in enumerate(zip(smiles, valid)):
        if not ok:
            continue
        raw = str(smi)
        if raw not in raw_reps:
            raw_reps[raw] = idx

    reps: dict[str, int] = {}
    for raw, idx in raw_reps.items():
        mol = Chem.MolFromSmiles(raw)
        canonical = Chem.MolToSmiles(mol)
        if canonical not in reps:
            reps[canonical] = idx
    canonical_smiles = list(reps.keys())
    return np.asarray([reps[smi] for smi in canonical_smiles], dtype=np.int64), canonical_smiles


def _morgan_fps(smiles: list[str]) -> list[DataStructs.ExplicitBitVect]:
    return [
        AllChem.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles(smi),
            radius=2,
            nBits=4096,
        )
        for smi in smiles
    ]


def _balanced_pairs(
    fps: list[DataStructs.ExplicitBitVect],
    *,
    num_pairs: int,
    seed: int,
    bin_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_bins = int(round(1.0 / bin_size))
    target_per_bin = num_pairs // n_bins
    pairs_by_bin: list[list[tuple[int, int, float]]] = [[] for _ in range(n_bins)]
    all_indices = np.arange(len(fps))

    for i in rng.permutation(len(fps)):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fps[int(i)], fps), dtype=np.float32)
        bin_ids = np.ceil(sims / bin_size).astype(np.int16) - 1
        for b in range(n_bins):
            need = target_per_bin - len(pairs_by_bin[b])
            if need <= 0:
                continue
            js = all_indices[(bin_ids == b) & (all_indices != i)]
            if js.size:
                take = rng.choice(js, size=min(need, js.size), replace=False)
                pairs_by_bin[b].extend((int(i), int(j), float(sims[j])) for j in take)
        if all(len(bucket) >= target_per_bin for bucket in pairs_by_bin):
            break

    pairs = [pair for bucket in pairs_by_bin for pair in bucket[:target_per_bin]]
    bin_counts = np.asarray([len(bucket[:target_per_bin]) for bucket in pairs_by_bin])
    assert len(pairs) == target_per_bin * n_bins
    return (
        np.asarray([p[0] for p in pairs], dtype=np.int64),
        np.asarray([p[1] for p in pairs], dtype=np.int64),
        np.asarray([p[2] for p in pairs], dtype=np.float32),
        bin_counts,
    )


def _embed_covariance(
    *,
    model: torch.nn.Module,
    spectra: np.ndarray,
    precursor_mz: np.ndarray,
    config,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    embeddings = []
    for start in range(0, len(spectra), batch_size):
        end = min(start + batch_size, len(spectra))
        spec = torch.from_numpy(spectra[start:end].copy()).to(device=device)
        precursor = torch.from_numpy(precursor_mz[start:end].copy()).to(device=device)
        batch = preprocess_peak_batch_torch(
            spec[:, 0, :],
            spec[:, 1, :],
            precursor,
            num_peaks=int(config.num_peaks),
            peak_drop_min_intensity=float(config.peak_drop_min_intensity),
            peak_ordering=str(config.peak_ordering),
            max_precursor_mz=float(config.max_precursor_mz),
            precursor_peak_exclusion_window_da=float(
                config.get("precursor_peak_exclusion_window_da", 0.0)
            ),
            min_peak_intensity=float(config.min_peak_intensity),
        )
        if bool(config.get("use_precursor_token", False)):
            batch = _prepend_precursor_token_torch(batch)
        with torch.no_grad():
            encoded = model.encoder(
                batch["peak_mz"],
                batch["peak_intensity"],
                valid_mask=batch["peak_valid_mask"],
            )
            peak_embeddings, _ = model.encoder.split_peak_and_cls(encoded)
            cov = model.covariance_pooler(peak_embeddings.float(), batch["peak_valid_mask"])
        embeddings.append(cov.detach().cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def _plot_scatter(path: Path, tanimoto: np.ndarray, cosine: np.ndarray, title: str) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    cmap = LinearSegmentedColormap.from_list(
        "nature_teal",
        ["#ebe9fb", "#9bd6cf", "#1f9d8a"],
    )
    pearson = float(np.corrcoef(tanimoto, cosine)[0, 1])
    fig, ax = plt.subplots(figsize=(1.9, 1.9), dpi=300)
    ax.hist2d(
        tanimoto,
        cosine,
        bins=120,
        range=[[0, 1], [-0.25, 1.02]],
        cmap=cmap,
        norm=LogNorm(),
        cmin=1,
    )
    coef = np.polyfit(tanimoto, cosine, deg=1)
    xs = np.linspace(0, 1, 200)
    ax.plot(xs, coef[0] * xs + coef[1], color="#168f86", lw=1.1)
    ax.axhline(0, color="0.82", lw=0.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.25, 1.02)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel("Morgan Tanimoto")
    ax.set_ylabel("Covariance cosine similarity")
    ax.set_title(f"{title}\nPearson = {pearson:.2f}", pad=3)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_curve(path: Path, rows: list[dict[str, float]]) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    steps = np.asarray([row["step"] for row in rows], dtype=np.float32)
    pearson = np.asarray([row["pearson"] for row in rows], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(2.6, 1.7), dpi=300)
    ax.plot(steps, pearson, color="#168f86", marker="o", markersize=3, lw=1.1)
    ax.axhline(0, color="0.82", lw=0.6)
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel("Pearson")
    ax.set_title("Covariance-Morgan alignment", pad=3)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument(
        "--artifact-dir",
        default="data/nist_full_probe_prepared_build/artifact",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-pairs", type=int, default=20_000)
    parser.add_argument("--bin-size", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.*")

    config = load_config(args.config)
    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    workdir = Path(args.workdir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else workdir / "morgan_alignment"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    split_arrays = _load_split_arrays(artifact_dir, args.split)
    rep_indices, rep_smiles = _representative_smiles(
        split_arrays["smiles"],
        split_arrays["probe_valid_mol"],
    )
    print(f"{args.split}: {len(split_arrays['smiles'])} rows, {len(rep_smiles)} representative molecules")
    fps = _morgan_fps(rep_smiles)
    pair_i_rep, pair_j_rep, tanimoto, bin_counts = _balanced_pairs(
        fps,
        num_pairs=args.num_pairs,
        seed=args.seed,
        bin_size=args.bin_size,
    )
    print(f"Balanced pairs: {len(tanimoto)} pairs; bin count range {bin_counts.min()}-{bin_counts.max()}")
    pair_i = rep_indices[pair_i_rep]
    pair_j = rep_indices[pair_j_rep]
    endpoint_indices, inverse = np.unique(
        np.concatenate([pair_i, pair_j]),
        return_inverse=True,
    )
    pair_i_endpoint = inverse[: len(pair_i)]
    pair_j_endpoint = inverse[len(pair_i) :]
    print(f"Unique endpoint spectra: {len(endpoint_indices)}")

    spectra = _take_from_shards(
        split_arrays["shard_dirs"],
        split_arrays["lengths"],
        "spectra",
        endpoint_indices,
    ).astype(np.float32)
    precursor = _take_from_shards(
        split_arrays["shard_dirs"],
        split_arrays["lengths"],
        "precursor_mz_raw",
        endpoint_indices,
    ).astype(np.float32)

    checkpoints = sorted((workdir / "checkpoints").glob("*.pt"), key=_checkpoint_step)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, float]] = []
    latest_payload = None

    for checkpoint in checkpoints:
        model = build_model_from_config(config).to(device)
        load_pretrained_weights(model, str(checkpoint))
        model.eval()
        endpoint_embeddings = _embed_covariance(
            model=model,
            spectra=spectra,
            precursor_mz=precursor,
            config=config,
            batch_size=args.batch_size,
            device=device,
        )
        endpoint_embeddings = endpoint_embeddings / np.linalg.norm(
            endpoint_embeddings,
            axis=1,
            keepdims=True,
        )
        cosine = np.sum(
            endpoint_embeddings[pair_i_endpoint] * endpoint_embeddings[pair_j_endpoint],
            axis=1,
        )
        pearson = float(np.corrcoef(tanimoto, cosine)[0, 1])
        row = {
            "step": float(_checkpoint_step(checkpoint)),
            "pearson": pearson,
            "mean_tanimoto": float(np.mean(tanimoto)),
            "mean_cosine": float(np.mean(cosine)),
            "num_pairs": float(len(tanimoto)),
            "num_endpoints": float(len(endpoint_indices)),
        }
        rows.append(row)
        latest_payload = (checkpoint, cosine)
        print(f"{checkpoint.name}: Pearson={pearson:.6f}")

    with (output_dir / "checkpoint_morgan_alignment.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "checkpoint_morgan_alignment.json").write_text(json.dumps(rows, indent=2))
    np.savez_compressed(
        output_dir / "balanced_pairs.npz",
        split_indices_i=pair_i,
        split_indices_j=pair_j,
        tanimoto=tanimoto,
        endpoint_indices=endpoint_indices,
        bin_counts=bin_counts,
    )
    _plot_curve(output_dir / "checkpoint_morgan_alignment_curve", rows)

    assert latest_payload is not None
    latest_checkpoint, latest_cosine = latest_payload
    np.savez_compressed(
        output_dir / f"{latest_checkpoint.stem}_pairs.npz",
        tanimoto=tanimoto,
        cosine_similarity=latest_cosine,
        cosine_distance=1.0 - latest_cosine,
    )
    _plot_scatter(
        output_dir / f"{latest_checkpoint.stem}_covariance_vs_morgan",
        tanimoto,
        latest_cosine,
        latest_checkpoint.stem,
    )


if __name__ == "__main__":
    main()
