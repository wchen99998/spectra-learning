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
import csv
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch._inductor.config as inductor_config
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize as l2_normalize
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
    _MASSSPEC_HF_REPO,
    _MASSSPEC_TSV_PATH,
    _download_hf_file,
    numpy_batch_to_torch,
)
from models.model import PeakSetSIGReg
from utils.training import (
    build_model_from_config,
    load_config,
    load_pretrained_weights,
    latest_ckpt_path,
)

log = logging.getLogger(__name__)

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to config .py file.")
    parser.add_argument("--dir", required=True, help="Experiment directory containing trial_*/checkpoints/.")
    parser.add_argument(
        "--workdir",
        default="results/substructure_geometry",
        help="Output directory for figures and results (default: results/substructure_geometry).",
    )
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path (default: latest in --dir).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


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
    load_pretrained_weights(backbone, ckpt_path)

    backbone = backbone.to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    log.info("Model on %s, params=%s", device, f"{sum(p.numel() for p in backbone.parameters()):,}")
    return config, datamodule, backbone, ckpt_path


def _read_smiles(config) -> list[str]:
    max_precursor_mz = float(config.get("max_precursor_mz", 1000.0))
    tfrecord_dir = Path(config.get("tfrecord_dir", "data/gems_peaklist_tfrecord")).expanduser().resolve()
    tsv_path = _download_hf_file(_MASSSPEC_HF_REPO, _MASSSPEC_TSV_PATH, tfrecord_dir.parent)

    smiles_train: list[str] = []
    with Path(tsv_path).open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["fold"] == "train" and float(row["precursor_mz"]) <= max_precursor_mz:
                smiles_train.append(row["smiles"])
    log.info("SMILES for massspec_train: %s", f"{len(smiles_train):,}")
    return smiles_train


def _encode_batch_impl(
    model: PeakSetSIGReg,
    peak_mz: torch.Tensor,
    peak_intensity: torch.Tensor,
    peak_valid_mask: torch.Tensor,
    precursor_mz: torch.Tensor,
) -> torch.Tensor:
    embeddings = model.encoder(
        peak_mz, peak_intensity,
        valid_mask=peak_valid_mask, precursor_mz=precursor_mz,
    )
    return model.pool(embeddings, peak_valid_mask)


def _extract_embeddings(
    config,
    datamodule: TfLightningDataModule,
    backbone: PeakSetSIGReg,
    device: torch.device,
    seed: int,
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
    compiled_encode = torch.compile(_encode_batch_impl, mode="max-autotune", fullgraph=True)

    embed_list: list[torch.Tensor] = []
    fp_list: list[np.ndarray] = []
    meta: dict[str, list[torch.Tensor]] = {
        "adduct": [],
        "instrument": [],
        "precursor_mz": [],
        "n_valid_peaks": [],
    }

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
            meta["adduct"].append(batch["adduct_id"].to(torch.long))
            meta["instrument"].append(batch["instrument_type_id"].to(torch.long))
            meta["precursor_mz"].append(batch["precursor_mz"].to(torch.float32))
            meta["n_valid_peaks"].append(batch["peak_valid_mask"].sum(dim=1).to(torch.long))

    all_embeds = torch.cat(embed_list, dim=0).numpy()
    all_fps_morgan = np.concatenate(fp_list, axis=0)
    all_meta = {k: torch.cat(v).numpy() for k, v in meta.items()}

    log.info("Embeddings: %s, Morgan FPs: %s", all_embeds.shape, all_fps_morgan.shape)
    return all_embeds, all_fps_morgan, all_meta


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
    log.info("Pearson(cosine_sim, tanimoto) = %.4f", corr)

    bins = np.linspace(0, 1, 21)
    bin_idx = np.digitize(tanimoto, bins) - 1
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_means = [cos_sims[bin_idx == i].mean() if (bin_idx == i).any() else np.nan for i in range(len(bin_centers))]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    axes[0].hist2d(tanimoto, cos_sims, bins=100, cmap="viridis", cmin=1)
    axes[0].set_xlabel("Tanimoto similarity (Morgan FP)")
    axes[0].set_ylabel("Cosine similarity (embeddings)")
    axes[0].set_title(f"Embedding vs Molecular Similarity\nPearson r = {corr:.4f}")
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
    header = f"{'k':>5s}  {'embed_kNN_tanimoto':>20s}  {'fp_kNN_tanimoto':>20s}  {'random_baseline':>16s}"
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

        log.info("%5d  %20.4f  %20.4f  %16.4f", k, tan_emb, tan_fp, random_tanimoto)
        results[f"k{k}"] = {
            "embed_knn_tanimoto": tan_emb,
            "fp_knn_tanimoto": tan_fp,
            "random_baseline": random_tanimoto,
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

    # 1. Load model & extract embeddings
    config, datamodule, backbone, ckpt_path = _load_model_and_data(
        args.config, args.dir, args.checkpoint, device, seed,
    )
    all_smiles = _read_smiles(config)
    all_embeds, all_fps_morgan, all_meta = _extract_embeddings(config, datamodule, backbone, device, seed)
    all_smiles = all_smiles[:len(all_embeds)]
    assert len(all_smiles) == len(all_embeds), f"Mismatch: {len(all_smiles)} vs {len(all_embeds)}"

    # 2. Compute RDKit features
    maccs_fps, fg_counts, mol_props, valid_mol_mask = _compute_rdkit_features(all_smiles)

    # 3. Tanimoto analysis
    tanimoto_results = _tanimoto_analysis(all_embeds, all_fps_morgan, valid_mol_mask, outdir, seed)

    # 4. kNN analysis
    knn_out = _knn_analysis(all_embeds, all_fps_morgan, valid_mol_mask, seed)
    knn_results = knn_out["knn"]

    # 5. UMAP
    _umap_analysis(all_embeds, all_meta, mol_props, fg_counts, valid_mol_mask, outdir, seed)

    # 6. Linear probes
    probe_results = _linear_probes(all_embeds, mol_props, fg_counts, valid_mol_mask, seed)

    # 7. MACCS enrichment
    maccs_results = _maccs_enrichment(maccs_fps, knn_out["knn_embed_idx"], knn_out["sub_idx"], outdir)

    # Write summary JSON
    summary = {
        "checkpoint": ckpt_path,
        "config": args.config,
        "experiment_dir": args.dir,
        "seed": seed,
        "n_embeddings": int(all_embeds.shape[0]),
        "embedding_dim": int(all_embeds.shape[1]),
        "tanimoto": tanimoto_results,
        "knn": knn_results,
        "linear_probes": probe_results,
        "maccs_enrichment": maccs_results,
    }
    summary_path = outdir / "results.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Results written to %s", summary_path)

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Checkpoint: {ckpt_path}")
    print(f"Embeddings: {all_embeds.shape}")
    print(f"\nPearson(cosine, tanimoto): {tanimoto_results['pearson_cosine_tanimoto']:.4f}")
    print(f"\nkNN Retrieval (embedding vs FP oracle):")
    for k_str, v in knn_results.items():
        print(f"  {k_str}: embed={v['embed_knn_tanimoto']:.4f}  fp={v['fp_knn_tanimoto']:.4f}  rand={v['random_baseline']:.4f}")
    print(f"\nLinear Probes (Ridge R^2):")
    for key in ["ridge_mol_weight", "ridge_logp", "ridge_num_heavy_atoms", "ridge_num_rings"]:
        r = probe_results[key]
        print(f"  {key}: {r['r2_mean']:.4f} +/- {r['r2_std']:.4f}")
    print(f"\nMACCS enrichment: mean_lift={maccs_results['mean_lift']:.3f}  median={maccs_results['median_lift']:.3f}")
    print(f"\nOutputs saved to: {outdir}")
    print("=" * 70)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
