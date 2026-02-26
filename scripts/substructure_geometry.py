"""Substructure-based embedding geometry analysis.

Evaluates how learned representations relate to molecular substructure using
RDKit fingerprints and functional group annotations from MassSpecGym SMILES.

Analyses:
1. Embedding similarity vs molecular (Tanimoto) similarity correlation
2. kNN retrieval: embedding space vs fingerprint space
3. UMAP colored by molecular properties and functional groups
4. Linear probes: fingerprint & molecular property prediction
5. MACCS key enrichment in embedding neighborhoods

Usage:
    python scripts/substructure_geometry.py \
        --config configs/gems_a_50_mask.py \
        --dir experiments/TEST_ISAB_LATEST_MULTIVIEWS \
        --workdir results/substructure_geometry
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch._inductor.config as inductor_config
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
from sklearn.preprocessing import normalize as l2_normalize
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import umap

from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, rdMolDescriptors
from rdkit import DataStructs

torch.set_float32_matmul_precision("high")
inductor_config.coordinate_descent_tuning = True
inductor_config.triton.unique_kernel_names = True
inductor_config.fx_graph_cache = True
inductor_config.epilogue_fusion = True
inductor_config.shape_padding = True

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from input_pipeline import (
    TfLightningDataModule,
    numpy_batch_to_torch,
)
from models.model import PeakSetSIGReg
from utils.probing import FinalAttentiveProbe, FinalLinearProbe
from utils.schedulers import learning_rate_at_step
from utils.training import (
    build_model_from_config,
    load_config,
    load_pretrained_weights,
    latest_ckpt_path,
)

log = logging.getLogger(__name__)


def _git_info() -> dict[str, str]:
    """Capture current git branch and commit hash."""
    info: dict[str, str] = {}
    try:
        info["git_branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return info


# Functional groups via SMARTS
FG_SMARTS: dict[str, str] = {
    "hydroxyl": "[OX2H]",
    "carboxyl": "[CX3](=O)[OX2H1]",
    "amine": "[NX3;H2,H1;!$(NC=O)]",
    "amide": "[NX3][CX3](=[OX1])",
    "ester": "[#6][CX3](=O)[OX2H0][#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "aromatic_ring": "c1ccccc1",
    "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "sulfonyl": "[#16X4](=[OX1])(=[OX1])",
    "phosphate": "[PX4](=[OX1])([OX2])",
    "halide": "[F,Cl,Br,I]",
    "ether": "[OD2]([#6])[#6]",
    "thiol": "[#16X2H]",
    "nitrile": "[NX1]#[CX2]",
}

RANDOM_SEED = 42
UMAP_MAX_SAMPLES = 30_000
UMAP_NEIGHBORS = 30
UMAP_MIN_DIST = 0.3
KNN_SUBSAMPLE = 20_000
PROBE_SIZE = 50_000
N_PAIRS = 500_000
K_VALUES = [1, 5, 10, 20, 50]

NEURAL_PROBE_EPOCHS = 10
NEURAL_PROBE_LR = 1e-4
NEURAL_PROBE_WEIGHT_DECAY = 1e-4
NEURAL_PROBE_WARMUP_STEPS = 50
NEURAL_PROBE_BATCH_SIZE = 256
NEURAL_PROBE_HIDDEN_DIM = 256
NEURAL_PROBE_ATTENTION_HEADS = 8


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Path to config .py file (required unless --external-embed).")
    parser.add_argument("--dir", default=None, help="Experiment directory containing trial_*/checkpoints/ (required unless --external-embed).")
    parser.add_argument(
        "--workdir",
        default="results/substructure_geometry",
        help="Output directory for figures and results (default: results/substructure_geometry).",
    )
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path (default: latest in --dir).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--probe-type",
        choices=["sklearn", "knn", "hgb", "neural-linear", "neural-attentive", "all"],
        default="sklearn",
        help="Probe type: sklearn (Ridge/LogReg), knn, hgb (HistGradientBoosting), neural-linear, neural-attentive, or all.",
    )
    parser.add_argument(
        "--pool", choices=["pma", "mean"], default="pma",
        help="Pooling method for model embeddings: pma (learned PMA) or mean (mean over valid tokens).",
    )
    parser.add_argument("--external-embed", default=None, help="Path to HDF5 file with pre-computed embeddings (bypasses model loading).")
    parser.add_argument("--embed-key", default="DreaMS_embedding", help="HDF5 dataset key for embeddings (default: DreaMS_embedding).")
    parser.add_argument("--fold", default="train", help="Which fold to filter to in the HDF5 file (default: train).")
    args = parser.parse_args()
    if args.external_embed is None and (args.config is None or args.dir is None):
        parser.error("--config and --dir are required unless --external-embed is provided.")
    return args


# ---------------------------------------------------------------------------
# 1. Load model & extract embeddings with SMILES
# ---------------------------------------------------------------------------


def _load_model_and_data(
    config_path: str,
    checkpoint_dir: str,
    checkpoint_override: str | None,
    device: torch.device,
    seed: int,
):
    config = load_config(config_path)
    datamodule = TfLightningDataModule(config, seed=int(config.seed))
    config.num_peaks = datamodule.info["num_peaks"]
    config.fingerprint_bits = int(datamodule.info["fingerprint_bits"])

    backbone = build_model_from_config(config)

    if checkpoint_override:
        ckpt_path = checkpoint_override
    else:
        ckpt_path = latest_ckpt_path(Path(checkpoint_dir))
    assert ckpt_path is not None, f"No checkpoint found in {checkpoint_dir}"
    log.info("Loading checkpoint: %s", ckpt_path)
    try:
        load_pretrained_weights(backbone, ckpt_path)
    except RuntimeError as e:
        if "Missing key" in str(e) or "Unexpected key" in str(e):
            log.warning("Strict load failed, retrying with strict=False (teacher_encoder key mismatch)")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
            model_state = {k.removeprefix("model."): v for k, v in state_dict.items() if k.startswith("model.")}
            if not model_state:
                model_state = state_dict
            backbone.load_state_dict(model_state, strict=False)
        else:
            raise

    backbone = backbone.to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    log.info("Model on %s, params=%s", device, f"{sum(p.numel() for p in backbone.parameters()):,}")
    return config, datamodule, backbone, ckpt_path


def _encode_batch_impl(
    model: PeakSetSIGReg,
    peak_mz: torch.Tensor,
    peak_intensity: torch.Tensor,
    peak_valid_mask: torch.Tensor,
    precursor_mz: torch.Tensor,
) -> torch.Tensor:
    embeddings = model.encoder(
        peak_mz, peak_intensity,
        valid_mask=peak_valid_mask,
    )
    return model.pool(embeddings, peak_valid_mask)


def _encode_batch_mean_pool_impl(
    model: PeakSetSIGReg,
    peak_mz: torch.Tensor,
    peak_intensity: torch.Tensor,
    peak_valid_mask: torch.Tensor,
    precursor_mz: torch.Tensor,
) -> torch.Tensor:
    embeddings = model.encoder(
        peak_mz, peak_intensity,
        valid_mask=peak_valid_mask,
    )
    # Mean pool over valid (non-padding) tokens.
    mask = peak_valid_mask.unsqueeze(-1).float()  # [B, N, 1]
    return (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def _extract_embeddings(
    config,
    datamodule: TfLightningDataModule,
    backbone: PeakSetSIGReg,
    device: torch.device,
    seed: int,
    pool: str = "pma",
):
    probe_peak_ordering = str(config.peak_ordering)
    dataset = datamodule.build_massspec_probe_dataset(
        "massspec_train",
        seed=seed,
        peak_ordering=probe_peak_ordering,
        shuffle=False,
        drop_remainder=True,
    )

    autocast_dtype = torch.bfloat16 if device.type == "cuda" else None
    encode_fn = _encode_batch_mean_pool_impl if pool == "mean" else _encode_batch_impl
    compiled_encode = torch.compile(encode_fn, mode="max-autotune", fullgraph=True)
    log.info("Pooling method: %s", pool)

    embed_list: list[torch.Tensor] = []
    fp_list: list[np.ndarray] = []
    smiles_list: list[str] = []
    meta: dict[str, list[torch.Tensor]] = {
        "adduct": [],
        "instrument": [],
        "precursor_mz": [],
        "n_valid_peaks": [],
    }
    raw_peak_mz_list: list[torch.Tensor] = []
    raw_peak_intensity_list: list[torch.Tensor] = []
    raw_peak_valid_mask_list: list[torch.Tensor] = []
    raw_precursor_mz_list: list[torch.Tensor] = []

    log.info("Extracting embeddings (torch.compile + autocast)...")
    with torch.no_grad():
        for numpy_batch in tqdm(dataset.as_numpy_iterator(), desc="Extracting embeddings"):
            batch = numpy_batch_to_torch(numpy_batch)
            peak_mz = batch["peak_mz"].to(device, non_blocking=True)
            peak_intensity = batch["peak_intensity"].to(device, non_blocking=True)
            peak_valid_mask = batch["peak_valid_mask"].to(device, non_blocking=True)
            precursor_mz = batch["precursor_mz"].to(device, non_blocking=True)

            torch.compiler.cudagraph_mark_step_begin()
            if autocast_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    pooled = compiled_encode(
                        backbone, peak_mz, peak_intensity,
                        peak_valid_mask, precursor_mz,
                    )
            else:
                pooled = compiled_encode(
                    backbone, peak_mz, peak_intensity,
                    peak_valid_mask, precursor_mz,
                )

            embed_list.append(pooled.cpu().float())
            fp_list.append(batch["fingerprint"].numpy())
            smiles_list.extend(batch["smiles"])
            meta["adduct"].append(batch["adduct_id"].to(torch.long))
            meta["instrument"].append(batch["instrument_type_id"].to(torch.long))
            meta["precursor_mz"].append(batch["precursor_mz"].to(torch.float32))
            meta["n_valid_peaks"].append(batch["peak_valid_mask"].sum(dim=1).to(torch.long))

            raw_peak_mz_list.append(batch["peak_mz"])
            raw_peak_intensity_list.append(batch["peak_intensity"])
            raw_peak_valid_mask_list.append(batch["peak_valid_mask"])
            raw_precursor_mz_list.append(batch["precursor_mz"])

    all_embeds = torch.cat(embed_list, dim=0).numpy()
    all_fps_morgan = np.concatenate(fp_list, axis=0)
    all_meta = {k: torch.cat(v).numpy() for k, v in meta.items()}
    raw_peaks = {
        "peak_mz": torch.cat(raw_peak_mz_list, dim=0),
        "peak_intensity": torch.cat(raw_peak_intensity_list, dim=0),
        "peak_valid_mask": torch.cat(raw_peak_valid_mask_list, dim=0),
        "precursor_mz": torch.cat(raw_precursor_mz_list, dim=0),
    }

    log.info("Embeddings: %s, Morgan FPs: %s, SMILES: %d", all_embeds.shape, all_fps_morgan.shape, len(smiles_list))
    return all_embeds, all_fps_morgan, all_meta, smiles_list, raw_peaks


def _resolve_hdf5_key(f, desired: str) -> str:
    """Find an HDF5 key case-insensitively."""
    keys_lower = {k.lower(): k for k in f.keys()}
    actual = keys_lower.get(desired.lower())
    assert actual is not None, f"Key {desired!r} not found in HDF5 (available: {list(f.keys())})"
    return actual


def _load_external_embeddings(
    hdf5_path: str,
    embed_key: str,
    fold: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], list[str]]:
    """Load pre-computed embeddings from an HDF5 file (e.g. DreaMS).

    Returns (embeddings, morgan_fps, meta_dict, smiles_list).
    """
    import h5py

    log.info("Loading external embeddings from %s (key=%s, fold=%s)", hdf5_path, embed_key, fold)
    with h5py.File(hdf5_path, "r") as f:
        fold_key = _resolve_hdf5_key(f, "fold")
        smiles_key = _resolve_hdf5_key(f, "smiles")
        pmz_key = _resolve_hdf5_key(f, "precursor_mz")
        embed_actual = _resolve_hdf5_key(f, embed_key)

        folds = np.asarray(f[fold_key]).astype(str)
        fold_mask = folds == fold
        n_total = len(folds)
        n_selected = int(fold_mask.sum())
        log.info("Fold filter: %d / %d spectra match fold=%r", n_selected, n_total, fold)

        indices = np.where(fold_mask)[0]
        embeddings = np.asarray(f[embed_actual][indices])
        smiles_raw = np.asarray(f[smiles_key][indices])
        precursor_mz = np.asarray(f[pmz_key][indices]).astype(np.float32)

    smiles_list = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in smiles_raw]
    log.info("Embeddings: %s, SMILES: %d", embeddings.shape, len(smiles_list))

    # Compute Morgan fingerprints from SMILES.
    from input_pipeline import _compute_morgan_fingerprints
    morgan_fps = _compute_morgan_fingerprints(np.array(smiles_list))

    all_meta = {
        "precursor_mz": precursor_mz,
        "adduct": np.zeros(n_selected, dtype=np.int64),
        "instrument": np.zeros(n_selected, dtype=np.int64),
        "n_valid_peaks": np.zeros(n_selected, dtype=np.int64),
    }

    return embeddings, morgan_fps, all_meta, smiles_list


# ---------------------------------------------------------------------------
# 2. Compute RDKit substructure fingerprints & functional groups
# ---------------------------------------------------------------------------


def _compute_rdkit_features(all_smiles: list[str]):
    n = len(all_smiles)
    maccs_fps = np.zeros((n, 167), dtype=np.int8)
    fg_patterns = {name: Chem.MolFromSmarts(sma) for name, sma in FG_SMARTS.items()}
    fg_counts = {name: np.zeros(n, dtype=np.int16) for name in FG_SMARTS}

    mol_weight = np.zeros(n, dtype=np.float32)
    num_rings = np.zeros(n, dtype=np.int16)
    num_aromatic_rings = np.zeros(n, dtype=np.int16)
    num_heavy_atoms = np.zeros(n, dtype=np.int16)
    logp = np.zeros(n, dtype=np.float32)
    valid_mol_mask = np.ones(n, dtype=bool)

    for i, smi in enumerate(tqdm(all_smiles, desc="Computing RDKit features")):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            valid_mol_mask[i] = False
            continue
        maccs = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros(167, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(maccs, arr)
        maccs_fps[i] = arr
        for name, pat in fg_patterns.items():
            fg_counts[name][i] = len(mol.GetSubstructMatches(pat))
        mol_weight[i] = Descriptors.ExactMolWt(mol)
        num_rings[i] = rdMolDescriptors.CalcNumRings(mol)
        num_aromatic_rings[i] = rdMolDescriptors.CalcNumAromaticRings(mol)
        num_heavy_atoms[i] = mol.GetNumHeavyAtoms()
        logp[i] = Descriptors.MolLogP(mol)

    log.info("Valid molecules: %s / %s", f"{valid_mol_mask.sum():,}", f"{n:,}")

    mol_props = {
        "mol_weight": mol_weight,
        "num_rings": num_rings,
        "num_aromatic_rings": num_aromatic_rings,
        "num_heavy_atoms": num_heavy_atoms,
        "logp": logp,
    }
    return maccs_fps, fg_counts, mol_props, valid_mol_mask


def _compute_rdkit_features_cached(
    all_smiles: list[str],
    cache_dir: Path,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Compute RDKit features with on-disk caching (keyed by SMILES hash)."""
    smiles_hash = hashlib.sha256("\n".join(all_smiles).encode()).hexdigest()[:16]
    cache_path = cache_dir / f"rdkit_features_{smiles_hash}.npz"

    if cache_path.exists():
        log.info("Loading cached RDKit features from %s", cache_path)
        data = np.load(cache_path)
        maccs_fps = data["maccs_fps"]
        valid_mol_mask = data["valid_mol_mask"]
        mol_props = {
            k: data[k]
            for k in ("mol_weight", "num_rings", "num_aromatic_rings",
                       "num_heavy_atoms", "logp")
        }
        fg_counts = {
            k.removeprefix("fg_"): data[k]
            for k in data.files
            if k.startswith("fg_")
        }
        log.info("Loaded cached features for %d samples", len(maccs_fps))
        return maccs_fps, fg_counts, mol_props, valid_mol_mask

    maccs_fps, fg_counts, mol_props, valid_mol_mask = _compute_rdkit_features(all_smiles)

    save_dict: dict[str, np.ndarray] = {
        "maccs_fps": maccs_fps,
        "valid_mol_mask": valid_mol_mask,
    }
    save_dict.update(mol_props)
    for name, arr in fg_counts.items():
        save_dict[f"fg_{name}"] = arr
    np.savez_compressed(cache_path, **save_dict)
    log.info("Cached RDKit features to %s", cache_path)
    return maccs_fps, fg_counts, mol_props, valid_mol_mask


# ---------------------------------------------------------------------------
# 3. Embedding similarity vs molecular (Tanimoto) similarity
# ---------------------------------------------------------------------------


def _tanimoto_analysis(
    all_embeds: np.ndarray,
    all_fps_morgan: np.ndarray,
    valid_mol_mask: np.ndarray,
    outdir: Path,
    seed: int,
) -> dict:
    rng = np.random.RandomState(seed)
    valid_idx = np.where(valid_mol_mask)[0]

    idx_a = rng.choice(valid_idx, size=N_PAIRS)
    idx_b = rng.choice(valid_idx, size=N_PAIRS)
    mask_diff = idx_a != idx_b
    idx_a, idx_b = idx_a[mask_diff], idx_b[mask_diff]

    embeds_normed = l2_normalize(all_embeds, axis=1)
    cos_sims = (embeds_normed[idx_a] * embeds_normed[idx_b]).sum(axis=1)

    fp_a = all_fps_morgan[idx_a].astype(np.float32)
    fp_b = all_fps_morgan[idx_b].astype(np.float32)
    intersection = (fp_a * fp_b).sum(axis=1)
    union = fp_a.sum(axis=1) + fp_b.sum(axis=1) - intersection
    tanimoto = intersection / np.maximum(union, 1e-8)

    corr = float(np.corrcoef(cos_sims, tanimoto)[0, 1])
    spearman_rho = float(spearmanr(cos_sims, tanimoto).statistic)
    log.info("Pearson(cosine_sim, tanimoto) = %.4f", corr)
    log.info("Spearman(cosine_sim, tanimoto) = %.4f", spearman_rho)

    bins = np.linspace(0, 1, 21)
    bin_idx = np.digitize(tanimoto, bins) - 1
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_means = [cos_sims[bin_idx == i].mean() if (bin_idx == i).any() else np.nan for i in range(len(bin_centers))]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    axes[0].hist2d(tanimoto, cos_sims, bins=100, cmap="viridis", cmin=1)
    axes[0].set_xlabel("Tanimoto similarity (Morgan FP)")
    axes[0].set_ylabel("Cosine similarity (embeddings)")
    axes[0].set_title(f"Embedding vs Molecular Similarity\nPearson r = {corr:.4f}  Spearman ρ = {spearman_rho:.4f}")
    plt.colorbar(axes[0].collections[0], ax=axes[0], label="count")

    axes[1].bar(bin_centers, bin_means, width=0.04, alpha=0.7)
    axes[1].set_xlabel("Tanimoto similarity bin")
    axes[1].set_ylabel("Mean cosine similarity")
    axes[1].set_title("Mean Embedding Similarity by Tanimoto Bin")

    axes[2].hist(tanimoto, bins=100, density=True, alpha=0.7, edgecolor="none")
    axes[2].set_xlabel("Tanimoto similarity")
    axes[2].set_ylabel("density")
    axes[2].set_title("Tanimoto Similarity Distribution")

    plt.tight_layout()
    fig.savefig(outdir / "tanimoto_vs_cosine.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved tanimoto_vs_cosine.png")

    return {
        "pearson_cosine_tanimoto": corr,
        "spearman_cosine_tanimoto": spearman_rho,
        "cosine_sim_mean": float(cos_sims.mean()),
        "cosine_sim_std": float(cos_sims.std()),
        "tanimoto_mean": float(tanimoto.mean()),
        "tanimoto_std": float(tanimoto.std()),
        "n_pairs": int(len(idx_a)),
    }


# ---------------------------------------------------------------------------
# 4. kNN retrieval: embedding space vs fingerprint space
# ---------------------------------------------------------------------------


def _knn_analysis(
    all_embeds: np.ndarray,
    all_fps_morgan: np.ndarray,
    valid_mol_mask: np.ndarray,
    seed: int,
) -> dict:
    rng = np.random.RandomState(seed)
    valid_idx = np.where(valid_mol_mask)[0]
    subsample_size = min(KNN_SUBSAMPLE, valid_idx.shape[0])
    sub_idx = rng.choice(valid_idx, size=subsample_size, replace=False)

    sub_embeds = all_embeds[sub_idx]
    sub_fps = all_fps_morgan[sub_idx].astype(np.float32)

    knn_embed = NearestNeighbors(n_neighbors=max(K_VALUES) + 1, metric="cosine", algorithm="brute", n_jobs=-1)
    knn_embed.fit(sub_embeds)
    _, knn_embed_idx = knn_embed.kneighbors(sub_embeds)
    knn_embed_idx = knn_embed_idx[:, 1:]

    knn_fp = NearestNeighbors(n_neighbors=max(K_VALUES) + 1, metric="jaccard", algorithm="brute", n_jobs=-1)
    knn_fp.fit(sub_fps)
    _, knn_fp_idx = knn_fp.kneighbors(sub_fps)
    knn_fp_idx = knn_fp_idx[:, 1:]

    # Random baseline
    rand_a = rng.choice(subsample_size, size=10000)
    rand_b = rng.choice(subsample_size, size=10000)
    rand_mask = rand_a != rand_b
    rand_inter = (sub_fps[rand_a[rand_mask]] * sub_fps[rand_b[rand_mask]]).sum(axis=1)
    rand_union = sub_fps[rand_a[rand_mask]].sum(axis=1) + sub_fps[rand_b[rand_mask]].sum(axis=1) - rand_inter
    random_tanimoto = float((rand_inter / np.maximum(rand_union, 1e-8)).mean())

    results: dict[str, dict[str, float]] = {}
    header = f"{'k':>5s}  {'embed_kNN_tanimoto':>20s}  {'fp_kNN_tanimoto':>20s}  {'random_baseline':>16s}  {'nbr_overlap':>12s}"
    log.info("kNN retrieval results:\n%s", header)

    for k in K_VALUES:
        nn_fps_emb = sub_fps[knn_embed_idx[:, :k]]
        query_fps_exp = np.expand_dims(sub_fps, 1)
        inter_emb = (query_fps_exp * nn_fps_emb).sum(axis=2)
        union_emb = query_fps_exp.sum(axis=2) + nn_fps_emb.sum(axis=2) - inter_emb
        tan_emb = float((inter_emb / np.maximum(union_emb, 1e-8)).mean())

        nn_fps_fp = sub_fps[knn_fp_idx[:, :k]]
        inter_fp = (query_fps_exp * nn_fps_fp).sum(axis=2)
        union_fp = query_fps_exp.sum(axis=2) + nn_fps_fp.sum(axis=2) - inter_fp
        tan_fp = float((inter_fp / np.maximum(union_fp, 1e-8)).mean())

        # Neighborhood overlap: fraction of embed kNN also in FP kNN
        embed_sets = [set(row) for row in knn_embed_idx[:, :k]]
        fp_sets = [set(row) for row in knn_fp_idx[:, :k]]
        overlap = float(np.mean([len(e & f) / k for e, f in zip(embed_sets, fp_sets)]))

        log.info("%5d  %20.4f  %20.4f  %16.4f  %12.4f", k, tan_emb, tan_fp, random_tanimoto, overlap)
        results[f"k{k}"] = {
            "embed_knn_tanimoto": tan_emb,
            "fp_knn_tanimoto": tan_fp,
            "random_baseline": random_tanimoto,
            "neighborhood_overlap": overlap,
        }

    return {"knn": results, "knn_embed_idx": knn_embed_idx, "sub_idx": sub_idx}


# ---------------------------------------------------------------------------
# 5. UMAP colored by molecular properties
# ---------------------------------------------------------------------------


def _umap_analysis(
    all_embeds: np.ndarray,
    all_meta: dict[str, np.ndarray],
    mol_props: dict[str, np.ndarray],
    fg_counts: dict[str, np.ndarray],
    valid_mol_mask: np.ndarray,
    outdir: Path,
    seed: int,
):
    rng = np.random.RandomState(seed)
    valid_idx = np.where(valid_mol_mask)[0]
    umap_n = min(UMAP_MAX_SAMPLES, valid_idx.shape[0])
    umap_idx = rng.choice(valid_idx, size=umap_n, replace=False)
    umap_embeds = all_embeds[umap_idx]

    log.info("Running UMAP on %s samples...", f"{umap_n:,}")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=seed,
    )
    coords = reducer.fit_transform(umap_embeds)
    log.info("UMAP done: %s", coords.shape)

    # Molecular properties plot
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    props = [
        ("Molecular Weight", mol_props["mol_weight"][umap_idx], "viridis"),
        ("LogP", mol_props["logp"][umap_idx], "coolwarm"),
        ("Num Rings", mol_props["num_rings"][umap_idx].astype(float), "turbo"),
        ("Num Aromatic Rings", mol_props["num_aromatic_rings"][umap_idx].astype(float), "turbo"),
        ("Heavy Atoms", mol_props["num_heavy_atoms"][umap_idx].astype(float), "plasma"),
        ("Precursor m/z", all_meta["precursor_mz"][umap_idx], "inferno"),
    ]
    for ax, (title, values, cmap) in zip(axes.flat, props):
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=values, cmap=cmap, s=1.5, alpha=0.4, rasterized=True)
        ax.set_title(title)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_aspect("equal")
        plt.colorbar(sc, ax=ax)
    plt.tight_layout()
    fig.savefig(outdir / "umap_mol_properties.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved umap_mol_properties.png")

    # Functional groups plot
    fg_names = list(FG_SMARTS.keys())
    n_fgs = len(fg_names)
    ncols = 4
    nrows = (n_fgs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes_flat = axes.flat
    for i, name in enumerate(fg_names):
        has_fg = (fg_counts[name][umap_idx] > 0).astype(float)
        prevalence = has_fg.mean()
        axes_flat[i].scatter(
            coords[:, 0], coords[:, 1],
            c=has_fg, cmap="RdYlBu_r", s=1.5, alpha=0.4, rasterized=True, vmin=0, vmax=1,
        )
        axes_flat[i].set_title(f"{name} ({prevalence:.1%})")
        axes_flat[i].set_aspect("equal")
    for j in range(i + 1, nrows * ncols):
        axes_flat[j].set_visible(False)
    plt.suptitle("UMAP colored by functional group presence", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(outdir / "umap_functional_groups.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved umap_functional_groups.png")


# ---------------------------------------------------------------------------
# 6. Linear probes: fingerprint & molecular property prediction
# ---------------------------------------------------------------------------


def _linear_probes(
    all_embeds: np.ndarray,
    mol_props: dict[str, np.ndarray],
    fg_counts: dict[str, np.ndarray],
    valid_mol_mask: np.ndarray,
    seed: int,
) -> dict:
    rng = np.random.RandomState(seed)
    valid_idx = np.where(valid_mol_mask)[0]
    probe_size = min(PROBE_SIZE, valid_idx.shape[0])
    probe_idx = rng.choice(valid_idx, size=probe_size, replace=False)
    X_probe = all_embeds[probe_idx]

    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    results: dict[str, dict] = {}

    # Ridge regression targets
    regression_targets = [
        ("mol_weight", "Molecular Weight"),
        ("logp", "LogP"),
        ("num_heavy_atoms", "Num Heavy Atoms"),
        ("num_rings", "Num Rings"),
    ]
    for key, label in regression_targets:
        y = mol_props[key][probe_idx].astype(float)
        ridge = Ridge(alpha=1.0)
        scores = cross_val_score(ridge, X_probe, y, cv=cv, scoring="r2", n_jobs=-1)
        r2_mean, r2_std = float(scores.mean()), float(scores.std())
        log.info("Ridge %s: R^2 = %.4f +/- %.4f", label, r2_mean, r2_std)
        results[f"ridge_{key}"] = {"r2_mean": r2_mean, "r2_std": r2_std}

    # Logistic regression for functional groups
    fg_names = list(FG_SMARTS.keys())
    fg_results: dict[str, dict] = {}
    for name in fg_names:
        y = (fg_counts[name][probe_idx] > 0).astype(int)
        pos_frac = float(y.mean())
        if pos_frac < 0.01 or pos_frac > 0.99:
            log.info("  %s: skipped (prevalence=%.3f)", name, pos_frac)
            fg_results[name] = {"skipped": True, "prevalence": pos_frac}
            continue
        clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", n_jobs=-1)
        scores = cross_val_score(clf, X_probe, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        auc_mean, auc_std = float(scores.mean()), float(scores.std())
        log.info("  %s: AUC=%.4f +/- %.4f (prevalence=%.3f)", name, auc_mean, auc_std, pos_frac)
        fg_results[name] = {"auc_mean": auc_mean, "auc_std": auc_std, "prevalence": pos_frac}
    results["functional_groups"] = fg_results

    return results


# ---------------------------------------------------------------------------
# 6a-ii. KNN probes
# ---------------------------------------------------------------------------

KNN_PROBE_K = 10


def _knn_probes(
    all_embeds: np.ndarray,
    mol_props: dict[str, np.ndarray],
    fg_counts: dict[str, np.ndarray],
    valid_mol_mask: np.ndarray,
    seed: int,
) -> dict:
    rng = np.random.RandomState(seed)
    valid_idx = np.where(valid_mol_mask)[0]
    probe_size = min(PROBE_SIZE, valid_idx.shape[0])
    probe_idx = rng.choice(valid_idx, size=probe_size, replace=False)
    X_probe = all_embeds[probe_idx]

    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    results: dict[str, dict] = {}

    regression_targets = [
        ("mol_weight", "Molecular Weight"),
        ("logp", "LogP"),
        ("num_heavy_atoms", "Num Heavy Atoms"),
        ("num_rings", "Num Rings"),
    ]
    for key, label in regression_targets:
        y = mol_props[key][probe_idx].astype(float)
        reg = KNeighborsRegressor(n_neighbors=KNN_PROBE_K, metric="cosine", n_jobs=-1)
        scores = cross_val_score(reg, X_probe, y, cv=cv, scoring="r2", n_jobs=-1)
        r2_mean, r2_std = float(scores.mean()), float(scores.std())
        log.info("KNN-%d %s: R^2 = %.4f +/- %.4f", KNN_PROBE_K, label, r2_mean, r2_std)
        results[f"ridge_{key}"] = {"r2_mean": r2_mean, "r2_std": r2_std}

    fg_names = list(FG_SMARTS.keys())
    fg_results: dict[str, dict] = {}
    for name in fg_names:
        y = (fg_counts[name][probe_idx] > 0).astype(int)
        pos_frac = float(y.mean())
        if pos_frac < 0.01 or pos_frac > 0.99:
            log.info("  %s: skipped (prevalence=%.3f)", name, pos_frac)
            fg_results[name] = {"skipped": True, "prevalence": pos_frac}
            continue
        clf = KNeighborsClassifier(n_neighbors=KNN_PROBE_K, metric="cosine", n_jobs=-1)
        scores = cross_val_score(clf, X_probe, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        auc_mean, auc_std = float(scores.mean()), float(scores.std())
        log.info("  %s: AUC=%.4f +/- %.4f (prevalence=%.3f)", name, auc_mean, auc_std, pos_frac)
        fg_results[name] = {"auc_mean": auc_mean, "auc_std": auc_std, "prevalence": pos_frac}
    results["functional_groups"] = fg_results

    return results


# ---------------------------------------------------------------------------
# 6a-iii. HistGradientBoosting probes
# ---------------------------------------------------------------------------


def _hgb_probes(
    all_embeds: np.ndarray,
    mol_props: dict[str, np.ndarray],
    fg_counts: dict[str, np.ndarray],
    valid_mol_mask: np.ndarray,
    seed: int,
) -> dict:
    rng = np.random.RandomState(seed)
    valid_idx = np.where(valid_mol_mask)[0]
    probe_size = min(PROBE_SIZE, valid_idx.shape[0])
    probe_idx = rng.choice(valid_idx, size=probe_size, replace=False)
    X_probe = all_embeds[probe_idx]

    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    results: dict[str, dict] = {}

    regression_targets = [
        ("mol_weight", "Molecular Weight"),
        ("logp", "LogP"),
        ("num_heavy_atoms", "Num Heavy Atoms"),
        ("num_rings", "Num Rings"),
    ]
    for key, label in regression_targets:
        y = mol_props[key][probe_idx].astype(float)
        reg = HistGradientBoostingRegressor(max_iter=200, max_depth=6, random_state=seed)
        scores = cross_val_score(reg, X_probe, y, cv=cv, scoring="r2", n_jobs=-1)
        r2_mean, r2_std = float(scores.mean()), float(scores.std())
        log.info("HGB %s: R^2 = %.4f +/- %.4f", label, r2_mean, r2_std)
        results[f"ridge_{key}"] = {"r2_mean": r2_mean, "r2_std": r2_std}

    fg_names = list(FG_SMARTS.keys())
    fg_results: dict[str, dict] = {}
    for name in fg_names:
        y = (fg_counts[name][probe_idx] > 0).astype(int)
        pos_frac = float(y.mean())
        if pos_frac < 0.01 or pos_frac > 0.99:
            log.info("  %s: skipped (prevalence=%.3f)", name, pos_frac)
            fg_results[name] = {"skipped": True, "prevalence": pos_frac}
            continue
        clf = HistGradientBoostingClassifier(max_iter=200, max_depth=6, random_state=seed)
        scores = cross_val_score(clf, X_probe, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        auc_mean, auc_std = float(scores.mean()), float(scores.std())
        log.info("  %s: AUC=%.4f +/- %.4f (prevalence=%.3f)", name, auc_mean, auc_std, pos_frac)
        fg_results[name] = {"auc_mean": auc_mean, "auc_std": auc_std, "prevalence": pos_frac}
    results["functional_groups"] = fg_results

    return results


# ---------------------------------------------------------------------------
# 6b. Neural probes (attentive / linear) on token-level encoder features
# ---------------------------------------------------------------------------


def _encode_tokens_impl(
    model: PeakSetSIGReg,
    peak_mz: torch.Tensor,
    peak_intensity: torch.Tensor,
    peak_valid_mask: torch.Tensor,
) -> torch.Tensor:
    return model.encoder(
        peak_mz, peak_intensity,
        valid_mask=peak_valid_mask,
    )


def _build_substructure_targets(
    mol_props: dict[str, np.ndarray],
    fg_counts: dict[str, np.ndarray],
    indices: np.ndarray,
) -> tuple[dict[str, torch.Tensor], list[str], list[str]]:
    targets: dict[str, torch.Tensor] = {}
    regression_tasks: list[str] = []
    for key in ("mol_weight", "logp", "num_heavy_atoms", "num_rings"):
        targets[key] = torch.from_numpy(mol_props[key][indices].astype(np.float32))
        regression_tasks.append(key)

    classification_tasks: list[str] = []
    for name in FG_SMARTS:
        y = (fg_counts[name][indices] > 0).astype(np.float32)
        pos_frac = float(y.mean())
        if pos_frac < 0.01 or pos_frac > 0.99:
            continue
        targets[f"fg_{name}"] = torch.from_numpy(y)
        classification_tasks.append(f"fg_{name}")

    return targets, regression_tasks, classification_tasks


def _neural_probes(
    backbone: PeakSetSIGReg,
    raw_peaks: dict[str, torch.Tensor],
    mol_props: dict[str, np.ndarray],
    fg_counts: dict[str, np.ndarray],
    valid_mol_mask: np.ndarray,
    device: torch.device,
    seed: int,
    probe_class: type[FinalAttentiveProbe] | type[FinalLinearProbe] = FinalAttentiveProbe,
) -> dict:
    rng = np.random.RandomState(seed)
    valid_idx = np.where(valid_mol_mask)[0]
    probe_size = min(PROBE_SIZE, len(valid_idx))
    probe_idx = rng.choice(valid_idx, size=probe_size, replace=False)

    targets, regression_tasks, classification_tasks = _build_substructure_targets(
        mol_props, fg_counts, probe_idx,
    )
    all_task_names = regression_tasks + classification_tasks
    log.info(
        "Neural probe tasks: %d regression, %d classification",
        len(regression_tasks), len(classification_tasks),
    )

    # Standardize regression targets (zero-mean, unit-variance).
    reg_means: dict[str, float] = {}
    reg_stds: dict[str, float] = {}
    for key in regression_tasks:
        m = float(targets[key].mean())
        s = float(targets[key].std())
        s = max(s, 1e-8)
        reg_means[key] = m
        reg_stds[key] = s
        targets[key] = (targets[key] - m) / s

    # 80/20 train/test split.
    n_train = int(0.8 * probe_size)
    shuffled = rng.permutation(probe_size)
    train_sel = shuffled[:n_train]
    test_sel = shuffled[n_train:]

    # Slice raw peaks to probe subset.
    probe_peak_mz = raw_peaks["peak_mz"][probe_idx]
    probe_peak_intensity = raw_peaks["peak_intensity"][probe_idx]
    probe_peak_valid_mask = raw_peaks["peak_valid_mask"][probe_idx]
    probe_precursor_mz = raw_peaks["precursor_mz"][probe_idx]

    target_matrix = torch.stack([targets[name] for name in all_task_names], dim=1)

    def _make_dataset(sel: np.ndarray) -> TensorDataset:
        idx = torch.from_numpy(sel.astype(np.int64))
        return TensorDataset(
            probe_peak_mz[idx],
            probe_peak_intensity[idx],
            probe_peak_valid_mask[idx],
            probe_precursor_mz[idx],
            target_matrix[idx],
        )

    train_ds = _make_dataset(train_sel)
    test_ds = _make_dataset(test_sel)

    train_loader = DataLoader(
        train_ds, batch_size=NEURAL_PROBE_BATCH_SIZE, shuffle=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=NEURAL_PROBE_BATCH_SIZE, shuffle=False, drop_last=False,
    )

    # Build probe.
    input_dim = int(backbone.encoder.model_dim)
    head_dims = {name: 1 for name in all_task_names}

    if probe_class is FinalAttentiveProbe:
        probe = FinalAttentiveProbe(
            input_dim=input_dim,
            hidden_dim=NEURAL_PROBE_HIDDEN_DIM,
            num_attention_heads=NEURAL_PROBE_ATTENTION_HEADS,
            head_dims=head_dims,
        ).to(device)
        probe_label = "attentive"
    else:
        probe = FinalLinearProbe(input_dim=input_dim, head_dims=head_dims).to(device)
        probe_label = "linear"

    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=NEURAL_PROBE_LR, weight_decay=NEURAL_PROBE_WEIGHT_DECAY,
    )
    total_steps = NEURAL_PROBE_EPOCHS * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step_idx: learning_rate_at_step(
            step_idx + 1,
            base_lr=NEURAL_PROBE_LR,
            total_steps=total_steps,
            warmup_steps=NEURAL_PROBE_WARMUP_STEPS,
            schedule_type="cosine",
            min_learning_rate=None,
        ) / NEURAL_PROBE_LR,
    )

    compiled_encoder = torch.compile(_encode_tokens_impl, mode="max-autotune", fullgraph=True)
    autocast_dtype = torch.bfloat16 if device.type == "cuda" else None

    log.info("Training %s probe for %d epochs (%d steps)...", probe_label, NEURAL_PROBE_EPOCHS, total_steps)

    for epoch in range(NEURAL_PROBE_EPOCHS):
        probe.train()
        epoch_loss = 0.0
        epoch_count = 0
        for peak_mz_b, peak_int_b, mask_b, _, targets_b in train_loader:
            peak_mz_b = peak_mz_b.to(device, non_blocking=True)
            peak_int_b = peak_int_b.to(device, non_blocking=True)
            mask_b = mask_b.to(device, non_blocking=True)
            targets_b = targets_b.to(device, non_blocking=True)

            with torch.no_grad():
                torch.compiler.cudagraph_mark_step_begin()
                if autocast_dtype is not None:
                    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                        tokens = compiled_encoder(backbone, peak_mz_b, peak_int_b, mask_b)
                else:
                    tokens = compiled_encoder(backbone, peak_mz_b, peak_int_b, mask_b)

            logits = probe(tokens.float(), mask_b)

            loss = torch.tensor(0.0, device=device)
            for i, name in enumerate(all_task_names):
                y = targets_b[:, i]
                pred = logits[name].squeeze(-1)
                if name in regression_tasks:
                    loss = loss + F.mse_loss(pred, y)
                else:
                    loss = loss + F.binary_cross_entropy_with_logits(pred, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item() * peak_mz_b.shape[0]
            epoch_count += peak_mz_b.shape[0]

        log.info(
            "  %s probe epoch %d/%d train_loss=%.4f",
            probe_label, epoch + 1, NEURAL_PROBE_EPOCHS, epoch_loss / epoch_count,
        )

    # Evaluate on test set.
    probe.eval()
    all_preds: dict[str, list[np.ndarray]] = {name: [] for name in all_task_names}
    all_targets: dict[str, list[np.ndarray]] = {name: [] for name in all_task_names}

    with torch.no_grad():
        for peak_mz_b, peak_int_b, mask_b, _, targets_b in test_loader:
            peak_mz_b = peak_mz_b.to(device, non_blocking=True)
            peak_int_b = peak_int_b.to(device, non_blocking=True)
            mask_b = mask_b.to(device, non_blocking=True)
            targets_b = targets_b.to(device, non_blocking=True)

            torch.compiler.cudagraph_mark_step_begin()
            if autocast_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    tokens = compiled_encoder(backbone, peak_mz_b, peak_int_b, mask_b)
            else:
                tokens = compiled_encoder(backbone, peak_mz_b, peak_int_b, mask_b)

            logits = probe(tokens.float(), mask_b)

            for i, name in enumerate(all_task_names):
                pred = logits[name].squeeze(-1).cpu().numpy()
                y = targets_b[:, i].cpu().numpy()
                all_preds[name].append(pred)
                all_targets[name].append(y)

    # Compute metrics.
    results: dict[str, dict] = {}

    for name in regression_tasks:
        preds = np.concatenate(all_preds[name])
        tgts = np.concatenate(all_targets[name])
        # Un-standardize for R² on original scale.
        preds_orig = preds * reg_stds[name] + reg_means[name]
        tgts_orig = tgts * reg_stds[name] + reg_means[name]
        r2 = float(r2_score(tgts_orig, preds_orig))
        log.info("  %s probe Ridge %s: R^2 = %.4f", probe_label, name, r2)
        results[f"ridge_{name}"] = {"r2": r2}

    fg_results: dict[str, dict] = {}
    for name in classification_tasks:
        preds = np.concatenate(all_preds[name])
        tgts = np.concatenate(all_targets[name])
        auc = float(roc_auc_score(tgts, preds))
        short_name = name.removeprefix("fg_")
        prevalence = float(tgts.mean())
        log.info("  %s probe %s: AUC=%.4f (prevalence=%.3f)", probe_label, short_name, auc, prevalence)
        fg_results[short_name] = {"auc": auc, "prevalence": prevalence}
    results["functional_groups"] = fg_results

    return results


# ---------------------------------------------------------------------------
# 7. MACCS key enrichment in embedding neighborhoods
# ---------------------------------------------------------------------------


def _maccs_enrichment(
    maccs_fps: np.ndarray,
    knn_embed_idx: np.ndarray,
    sub_idx: np.ndarray,
    outdir: Path,
    k: int = 10,
) -> dict:
    sub_maccs = maccs_fps[sub_idx]
    global_prev = sub_maccs.mean(axis=0)
    nn_maccs = sub_maccs[knn_embed_idx[:, :k]]

    enrichment = []
    for bit in range(167):
        if global_prev[bit] < 0.05 or global_prev[bit] > 0.95:
            continue
        has_bit = sub_maccs[:, bit] == 1
        if has_bit.sum() < 100:
            continue
        nn_has_bit = nn_maccs[has_bit, :, bit]
        observed_rate = float(nn_has_bit.mean())
        expected_rate = float(global_prev[bit])
        lift = observed_rate / expected_rate
        enrichment.append((bit, observed_rate, expected_rate, lift))

    enrichment.sort(key=lambda x: -x[3])
    lifts = [x[3] for x in enrichment]

    log.info("MACCS enrichment (top 10 by lift):")
    for bit, obs, exp, lift in enrichment[:10]:
        log.info("  bit=%3d  observed=%.4f  expected=%.4f  lift=%.2fx", bit, obs, exp, lift)

    mean_lift = float(np.mean(lifts))
    median_lift = float(np.median(lifts))
    lift_gt_1_5 = sum(1 for l in lifts if l > 1.5)
    lift_gt_2_0 = sum(1 for l in lifts if l > 2.0)
    log.info("Mean lift=%.3f  Median=%.3f  >1.5: %d/%d  >2.0: %d/%d",
             mean_lift, median_lift, lift_gt_1_5, len(lifts), lift_gt_2_0, len(lifts))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(range(len(lifts)), sorted(lifts, reverse=True), edgecolor="none", alpha=0.7)
    axes[0].axhline(1.0, color="red", linestyle="--", alpha=0.7, label="no enrichment")
    axes[0].set_xlabel("MACCS key (sorted by lift)")
    axes[0].set_ylabel("lift (observed / expected)")
    axes[0].set_title(f"MACCS Key Enrichment in Embedding k={k} NN")
    axes[0].legend()

    axes[1].hist(lifts, bins=30, edgecolor="none", alpha=0.7)
    axes[1].axvline(1.0, color="red", linestyle="--", alpha=0.7, label="no enrichment")
    axes[1].axvline(median_lift, color="blue", linestyle="--", alpha=0.7, label=f"median={median_lift:.2f}")
    axes[1].set_xlabel("lift")
    axes[1].set_ylabel("count")
    axes[1].set_title("Distribution of MACCS Key Lifts")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(outdir / "maccs_enrichment.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved maccs_enrichment.png")

    return {
        "mean_lift": mean_lift,
        "median_lift": median_lift,
        "lift_gt_1_5": lift_gt_1_5,
        "lift_gt_2_0": lift_gt_2_0,
        "n_maccs_keys": len(lifts),
        "top10": [
            {"bit": int(bit), "observed": obs, "expected": exp, "lift": lift}
            for bit, obs, exp, lift in enrichment[:10]
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    outdir = Path(args.workdir)
    outdir.mkdir(parents=True, exist_ok=True)
    seed = args.seed
    device = torch.device(args.device)

    np.random.seed(seed)
    torch.manual_seed(seed)

    # 1. Load embeddings: either from external HDF5 or from model+TFRecords
    if args.external_embed:
        all_embeds, all_fps_morgan, all_meta, all_smiles = _load_external_embeddings(
            args.external_embed, args.embed_key, args.fold,
        )
        backbone = None
        raw_peaks = None
        ckpt_path = args.external_embed
    else:
        config, datamodule, backbone, ckpt_path = _load_model_and_data(
            args.config, args.dir, args.checkpoint, device, seed,
        )
        all_embeds, all_fps_morgan, all_meta, all_smiles, raw_peaks = _extract_embeddings(config, datamodule, backbone, device, seed, pool=args.pool)
    assert len(all_smiles) == len(all_embeds), f"Mismatch: {len(all_smiles)} vs {len(all_embeds)}"

    # 2. Compute RDKit features (cached on disk)
    if args.external_embed:
        cache_dir = outdir
    else:
        cache_dir = Path(config.get("tfrecord_dir", "data/gems_peaklist_tfrecord")).expanduser().resolve()
    maccs_fps, fg_counts, mol_props, valid_mol_mask = _compute_rdkit_features_cached(all_smiles, cache_dir)

    # 3. Tanimoto analysis
    tanimoto_results = _tanimoto_analysis(all_embeds, all_fps_morgan, valid_mol_mask, outdir, seed)

    # 4. kNN analysis
    knn_out = _knn_analysis(all_embeds, all_fps_morgan, valid_mol_mask, seed)
    knn_results = knn_out["knn"]

    # 5. UMAP
    _umap_analysis(all_embeds, all_meta, mol_props, fg_counts, valid_mol_mask, outdir, seed)

    # 6. Probes
    probe_type = args.probe_type
    probe_results: dict[str, dict] = {}
    if probe_type in ("sklearn", "all"):
        probe_results["sklearn"] = _linear_probes(all_embeds, mol_props, fg_counts, valid_mol_mask, seed)
    if probe_type in ("knn", "all"):
        probe_results["knn"] = _knn_probes(all_embeds, mol_props, fg_counts, valid_mol_mask, seed)
    if probe_type in ("hgb", "all"):
        probe_results["hgb"] = _hgb_probes(all_embeds, mol_props, fg_counts, valid_mol_mask, seed)
    neural_requested = probe_type in ("neural-attentive", "neural-linear", "all")
    if neural_requested and backbone is None:
        log.warning("Skipping neural probes: not available with --external-embed (no backbone model).")
    else:
        if probe_type in ("neural-attentive", "all"):
            probe_results["neural_attentive"] = _neural_probes(
                backbone, raw_peaks, mol_props, fg_counts, valid_mol_mask, device, seed,
                probe_class=FinalAttentiveProbe,
            )
        if probe_type in ("neural-linear", "all"):
            probe_results["neural_linear"] = _neural_probes(
                backbone, raw_peaks, mol_props, fg_counts, valid_mol_mask, device, seed,
                probe_class=FinalLinearProbe,
            )

    # 7. MACCS enrichment
    maccs_results = _maccs_enrichment(maccs_fps, knn_out["knn_embed_idx"], knn_out["sub_idx"], outdir)

    # Write summary JSON
    summary: dict = {
        "checkpoint": ckpt_path,
        "seed": seed,
        **_git_info(),
        "n_embeddings": int(all_embeds.shape[0]),
        "embedding_dim": int(all_embeds.shape[1]),
        "tanimoto": tanimoto_results,
        "knn": knn_results,
        "probes": probe_results,
        "maccs_enrichment": maccs_results,
    }
    if args.external_embed:
        summary["external_embed"] = args.external_embed
        summary["embed_key"] = args.embed_key
        summary["fold"] = args.fold
    else:
        summary["config"] = args.config
        summary["experiment_dir"] = args.dir
        summary["pool"] = args.pool
    summary_path = outdir / "results.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Results written to %s", summary_path)

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Checkpoint: {ckpt_path}")
    git = _git_info()
    if git:
        print(f"Git: {git.get('git_branch', '?')} @ {git.get('git_commit', '?')[:10]}")
    print(f"Embeddings: {all_embeds.shape}")
    print(f"\nPearson(cosine, tanimoto): {tanimoto_results['pearson_cosine_tanimoto']:.4f}")
    print(f"Spearman(cosine, tanimoto): {tanimoto_results['spearman_cosine_tanimoto']:.4f}")
    print(f"\nkNN Retrieval (embedding vs FP oracle | nbr_overlap):")
    for k_str, v in knn_results.items():
        print(f"  {k_str}: embed={v['embed_knn_tanimoto']:.4f}  fp={v['fp_knn_tanimoto']:.4f}  rand={v['random_baseline']:.4f}  overlap={v['neighborhood_overlap']:.4f}")
    for label, results in probe_results.items():
        print(f"\nProbes [{label}] (Ridge R^2):")
        for key in ["ridge_mol_weight", "ridge_logp", "ridge_num_heavy_atoms", "ridge_num_rings"]:
            r = results[key]
            if "r2_mean" in r:
                print(f"  {key}: {r['r2_mean']:.4f} +/- {r['r2_std']:.4f}")
            else:
                print(f"  {key}: {r['r2']:.4f}")
    print(f"\nMACCS enrichment: mean_lift={maccs_results['mean_lift']:.3f}  median={maccs_results['median_lift']:.3f}")
    print(f"\nOutputs saved to: {outdir}")
    print("=" * 70)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
