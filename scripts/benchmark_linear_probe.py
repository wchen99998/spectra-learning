"""Frozen linear probe benchmark: Ours vs DreaMS.

Compares frozen encoder embeddings against DreaMS using the weakest possible
downstream learner (linear probe), proper train/val/test splits, and
dimensionality controls.

Usage:
    python scripts/benchmark_linear_probe.py \
        --mode compare \
        --config configs/gems_a_50_mask.py \
        --dir experiments/TEST_LATEST_MULTIVIEWS \
        --dreams-hdf5 data/data/auxiliary/MassSpecGym_DreaMS.hdf5 \
        --workdir results/benchmark_linear_probe
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from utils.massspec_probe_targets import (
    FG_SMARTS,
    REGRESSION_TARGET_KEYS,
    compute_probe_targets_for_smiles,
)
from models.model import PeakSetSIGReg
from utils.training import (
    build_model_from_config,
    load_config,
    load_pretrained_weights,
    latest_ckpt_path,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants matching utils/massspec_probe_data.py
# ---------------------------------------------------------------------------
_NUM_PEAKS_OUTPUT = 60
_PEAK_MZ_MIN = 20.0
_PEAK_MZ_MAX = 1000.0
_PRECURSOR_MZ_WINDOW = 2.5
_MIN_PEAK_INTENSITY = 0.001
_MAX_PRECURSOR_MZ = 1000.0
_TEST_FRACTION = 0.15  # For datasets without a dedicated test split

# Probe hyperparameter grids
_RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
_LOGISTIC_CS = [0.01, 0.1, 1.0, 10.0, 100.0]
_FG_PREVALENCE_MIN = 0.01
_FG_PREVALENCE_MAX = 0.99


# ===== Step 1: Load DreaMS data from HDF5 =====


def load_dreams_data(hdf5_path: str) -> dict[str, np.ndarray | list[str]]:
    """Load spectra, DreaMS embeddings, SMILES, fold labels from HDF5."""
    log.info("Loading DreaMS HDF5 from %s", hdf5_path)
    t0 = time.time()
    with h5py.File(hdf5_path, "r") as f:
        dreams_embedding = f["DreaMS_embedding"][:].astype(np.float32)
        spectrum = f["spectrum"][:]  # [N, 2, 128]
        precursor_mz = f["precursor_mz"][:].astype(np.float64)
        smiles = [s.decode("utf-8") if isinstance(s, bytes) else s for s in f["smiles"][:]]
        fold = [s.decode("utf-8") if isinstance(s, bytes) else s for s in f["FOLD"][:]]
        identifiers = [
            s.decode("utf-8") if isinstance(s, bytes) else s for s in f["IDENTIFIER"][:]
        ]
    log.info(
        "Loaded %d spectra in %.1fs (DreaMS dim=%d, spectrum shape=%s)",
        len(smiles),
        time.time() - t0,
        dreams_embedding.shape[1],
        spectrum.shape,
    )
    # Safety check: identifiers should be monotonically ordered
    for i in range(1, min(100, len(identifiers))):
        assert identifiers[i] >= identifiers[i - 1], (
            f"IDENTIFIER not monotonically ordered at index {i}: "
            f"{identifiers[i - 1]} > {identifiers[i]}"
        )
    return {
        "dreams_embedding": dreams_embedding,
        "spectrum": spectrum,
        "precursor_mz": precursor_mz,
        "smiles": smiles,
        "fold": fold,
        "identifiers": identifiers,
    }


def load_dreams_atlas_hdf5(hdf5_path: str) -> dict[str, np.ndarray | list[str]]:
    """Load DreaMS Atlas HDF5 (no FOLD column). Creates InChIKey-based splits.

    Filters out entries with invalid SMILES (e.g. "-1") before splitting.
    Uses first 14 chars of InChIKey (connectivity layer) so that all spectra
    of the same molecule end up in the same split.
    Split ratios: 70% train, 15% val, 15% test.
    """
    from rdkit import Chem
    from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey

    log.info("Loading DreaMS Atlas HDF5 from %s", hdf5_path)
    t0 = time.time()
    with h5py.File(hdf5_path, "r") as f:
        dreams_embedding = f["DreaMS_embedding"][:].astype(np.float32)
        spectrum = f["spectrum"][:]  # [N, 2, 128]
        precursor_mz = f["precursor_mz"][:].astype(np.float64)
        smiles_raw = f["smiles"][:]
        ids_raw = f["id"][:]

    smiles_all = [s.decode("utf-8") if isinstance(s, bytes) else s for s in smiles_raw]
    identifiers_all = [s.decode("utf-8") if isinstance(s, bytes) else s for s in ids_raw]
    n_total = len(smiles_all)
    log.info("Loaded %d spectra in %.1fs", n_total, time.time() - t0)

    # Filter out entries with invalid SMILES (e.g. "-1", empty, unparseable)
    log.info("Filtering entries with valid SMILES...")
    valid_indices = []
    inchikey_conn_valid = []
    for i, smi in enumerate(smiles_all):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        inchi = MolToInchi(mol)
        if inchi is None:
            continue
        ik = InchiToInchiKey(inchi)[:14]
        valid_indices.append(i)
        inchikey_conn_valid.append(ik)

    valid_indices = np.array(valid_indices)
    log.info("Valid SMILES: %d / %d (%.1f%%)", len(valid_indices), n_total,
             100 * len(valid_indices) / n_total)

    # Subset arrays to valid only
    dreams_embedding = dreams_embedding[valid_indices]
    spectrum = spectrum[valid_indices]
    precursor_mz = precursor_mz[valid_indices]
    smiles = [smiles_all[i] for i in valid_indices]
    identifiers = [identifiers_all[i] for i in valid_indices]

    # Group by connectivity key, then split keys 70/15/15
    unique_keys = sorted(set(inchikey_conn_valid))
    rng = np.random.RandomState(42)
    rng.shuffle(unique_keys)

    n_keys = len(unique_keys)
    n_train = int(n_keys * 0.70)
    n_val = int(n_keys * 0.15)
    train_keys = set(unique_keys[:n_train])
    val_keys = set(unique_keys[n_train : n_train + n_val])
    # rest → test

    fold = []
    for ik in inchikey_conn_valid:
        if ik in train_keys:
            fold.append("train")
        elif ik in val_keys:
            fold.append("val")
        else:
            fold.append("test")

    from collections import Counter
    counts = Counter(fold)
    log.info(
        "InChIKey splits (%d unique keys): train=%d, val=%d, test=%d",
        n_keys, counts["train"], counts["val"], counts["test"],
    )

    return {
        "dreams_embedding": dreams_embedding,
        "spectrum": spectrum,
        "precursor_mz": precursor_mz,
        "smiles": smiles,
        "fold": fold,
        "identifiers": identifiers,
    }


def load_mona_pkl_data(pkl_path: str) -> dict[str, np.ndarray | list[str]]:
    """Load spectra, SMILES, fold labels from MoNA pickle file.

    The pkl has columns: PARSED PEAKS (2,128), SMILES, PRECURSOR M/Z, val (bool).
    val=True → validation, val=False → train. We further split train into train/test.
    """
    import pickle

    class _SafeUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str):
            if module.startswith("msml"):
                return type(name, (), {})
            return super().find_class(module, name)

    log.info("Loading MoNA pickle from %s", pkl_path)
    t0 = time.time()
    with open(pkl_path, "rb") as f:
        df = _SafeUnpickler(f).load()

    n = len(df)
    log.info("Loaded %d spectra in %.1fs", n, time.time() - t0)

    # Stack spectra into array
    spectrum = np.stack(df["PARSED PEAKS"].values).astype(np.float64)  # [N, 2, 128]
    precursor_mz = df["PRECURSOR M/Z"].values.astype(np.float64)
    smiles = df["SMILES"].tolist()
    val_mask = df["val"].values.astype(bool)

    # Create train/val/test fold labels
    # val=True → "val", val=False → split into "train" and "test"
    rng = np.random.RandomState(42)
    fold = np.array(["train"] * n, dtype=object)
    fold[val_mask] = "val"
    train_mask = ~val_mask
    train_indices = np.where(train_mask)[0]
    n_test = int(len(train_indices) * _TEST_FRACTION)
    test_indices = rng.choice(train_indices, size=n_test, replace=False)
    fold[test_indices] = "test"

    log.info(
        "MoNA splits: train=%d, val=%d, test=%d",
        (fold == "train").sum(),
        (fold == "val").sum(),
        (fold == "test").sum(),
    )

    return {
        "dreams_embedding": None,  # No DreaMS embeddings for MoNA
        "spectrum": spectrum,
        "precursor_mz": precursor_mz,
        "smiles": smiles,
        "fold": fold.tolist(),
        "identifiers": df["ID"].tolist() if "ID" in df.columns else [f"MoNA_{i}" for i in range(n)],
    }


# ===== Step 2: Preprocess DreaMS spectra in NumPy =====


def preprocess_dreams_spectra(
    spectrum: np.ndarray,
    precursor_mz: np.ndarray,
    *,
    num_peaks: int = _NUM_PEAKS_OUTPUT,
    peak_mz_min: float = _PEAK_MZ_MIN,
    peak_mz_max: float = _PEAK_MZ_MAX,
    precursor_window: float = _PRECURSOR_MZ_WINDOW,
    min_intensity: float = _MIN_PEAK_INTENSITY,
    max_precursor_mz: float = _MAX_PRECURSOR_MZ,
) -> dict[str, np.ndarray]:
    """Replicate TF preprocessing pipeline in NumPy.

    Input spectrum: [N, 2, 128] where [:,0,:] = mz, [:,1,:] = intensity
    """
    n = spectrum.shape[0]
    n_input = spectrum.shape[2]  # 128
    mz = spectrum[:, 0, :].astype(np.float32)  # [N, 128]
    intensity = spectrum[:, 1, :].astype(np.float32)  # [N, 128]

    # 1. Filter by mz range: keep peaks where mz >= 20.0 and mz <= min(precursor_mz - 2.5, 1000.0)
    upper = np.where(
        precursor_mz > 0.0,
        precursor_mz - precursor_window,
        peak_mz_max,
    ).astype(np.float32)  # [N]
    keep = (mz >= peak_mz_min) & (mz <= upper[:, np.newaxis])
    mz = np.where(keep, mz, 0.0)
    intensity = np.where(keep, intensity, 0.0)

    # 2. Filter by minimum intensity
    keep2 = intensity >= min_intensity
    mz = np.where(keep2, mz, 0.0)
    intensity = np.where(keep2, intensity, 0.0)

    # 3. Top-k by intensity
    # Use argpartition for efficiency, then sort the top-k
    if n_input > num_peaks:
        # Get top-k indices
        topk_idx = np.argpartition(-intensity, num_peaks, axis=1)[:, :num_peaks]
        # Gather
        row_idx = np.arange(n)[:, np.newaxis]
        mz = mz[row_idx, topk_idx]
        intensity = intensity[row_idx, topk_idx]

    # 4. Sort by mz ascending (invalid peaks → inf so they sort last)
    valid = intensity > 0
    sort_key = np.where(valid, mz, np.inf)
    sorted_idx = np.argsort(sort_key, axis=1, kind="stable")
    row_idx = np.arange(n)[:, np.newaxis]
    mz = mz[row_idx, sorted_idx]
    intensity = intensity[row_idx, sorted_idx]
    valid = valid[row_idx, sorted_idx]

    # Zero out invalid positions after sort
    mz = np.where(valid, mz, 0.0)
    intensity = np.where(valid, intensity, 0.0)

    # 5. Normalize
    peak_mz_out = mz / peak_mz_max
    prec_mz_out = np.clip(precursor_mz, 0.0, max_precursor_mz).astype(np.float32) / max_precursor_mz

    log.info(
        "Preprocessed spectra: valid peaks per spectrum min=%d, median=%d, max=%d",
        valid.sum(axis=1).min(),
        int(np.median(valid.sum(axis=1))),
        valid.sum(axis=1).max(),
    )
    return {
        "peak_mz": peak_mz_out.astype(np.float32),
        "peak_intensity": intensity.astype(np.float32),
        "peak_valid_mask": valid,
        "precursor_mz": prec_mz_out.astype(np.float32),
    }


# ===== Step 3: Extract our model's embeddings =====


@torch.no_grad()
def extract_our_embeddings(
    config_path: str,
    model_dir: str,
    preprocessed: dict[str, np.ndarray],
    *,
    batch_size: int = 512,
    device: str = "cuda",
) -> np.ndarray:
    """Run frozen encoder + mean pool to get embeddings."""
    config = load_config(config_path)
    model = build_model_from_config(config)

    ckpt_path = latest_ckpt_path(Path(model_dir))
    assert ckpt_path is not None, f"No checkpoint found in {model_dir}"
    log.info("Loading checkpoint: %s", ckpt_path)
    load_pretrained_weights(model, ckpt_path)

    model = model.to(device).eval()
    encoder = model.encoder
    encoder = torch.compile(encoder, mode="max-autotune")

    peak_mz = preprocessed["peak_mz"]
    peak_intensity = preprocessed["peak_intensity"]
    peak_valid_mask = preprocessed["peak_valid_mask"]
    precursor_mz_np = preprocessed.get("precursor_mz")
    n = peak_mz.shape[0]
    model_dim = config.model_dim

    all_embeddings = np.zeros((n, model_dim), dtype=np.float32)
    t0 = time.time()

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_mz = torch.from_numpy(peak_mz[start:end]).to(device)
        batch_int = torch.from_numpy(peak_intensity[start:end]).to(device)
        batch_mask = torch.from_numpy(peak_valid_mask[start:end]).to(device)

        if model.use_precursor_token and precursor_mz_np is not None:
            batch_pmz = torch.from_numpy(precursor_mz_np[start:end]).to(device)
            expanded = PeakSetSIGReg.prepend_precursor_token(
                batch_mz, batch_int, batch_mask, batch_pmz,
            )
            batch_mz = expanded["peak_mz"]
            batch_int = expanded["peak_intensity"]
            batch_mask = expanded["peak_valid_mask"]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            token_emb = encoder(batch_mz, batch_int, valid_mask=batch_mask)

        # Mean pool over valid tokens
        mask_f = batch_mask.unsqueeze(-1).float()
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        pooled = (token_emb.float() * mask_f).sum(dim=1) / denom
        all_embeddings[start:end] = pooled.cpu().numpy()

        if start % (batch_size * 50) == 0 and start > 0:
            log.info("  Encoded %d / %d spectra", start, n)

    elapsed = time.time() - t0
    log.info("Encoded %d spectra in %.1fs (%.0f spectra/s)", n, elapsed, n / elapsed)

    # Sanity: check embeddings aren't collapsed
    var = np.var(all_embeddings, axis=0).mean()
    log.info("Embedding mean per-dim variance: %.6f", var)
    assert var > 1e-8, f"Embeddings appear collapsed (variance={var:.2e})"

    return all_embeddings


# ===== Step 4: Compute targets from SMILES =====


def compute_targets_cached(
    smiles: list[str],
    cache_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Compute probe targets with hash-based NPZ caching."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    smiles_hash = hashlib.sha256("\n".join(smiles).encode()).hexdigest()[:16]
    cache_path = cache_dir / f"probe_targets_{smiles_hash}.npz"

    if cache_path.exists():
        log.info("Loading cached targets from %s", cache_path)
        data = np.load(cache_path, allow_pickle=True)
        mol_props = {k: data[f"reg_{k}"] for k in REGRESSION_TARGET_KEYS}
        fg_counts = {k: data[f"fg_{k}"] for k in FG_SMARTS}
        valid_mol_mask = data["valid_mol_mask"]
        return mol_props, fg_counts, valid_mol_mask

    log.info("Computing probe targets for %d SMILES (this may take a while)...", len(smiles))
    t0 = time.time()
    mol_props, fg_counts, valid_mol_mask = compute_probe_targets_for_smiles(smiles)
    log.info("Computed targets in %.1fs", time.time() - t0)

    save_dict: dict[str, np.ndarray] = {"valid_mol_mask": valid_mol_mask}
    for k, v in mol_props.items():
        save_dict[f"reg_{k}"] = v
    for k, v in fg_counts.items():
        save_dict[f"fg_{k}"] = v
    np.savez(cache_path, **save_dict)
    log.info("Cached targets to %s", cache_path)
    return mol_props, fg_counts, valid_mol_mask


# ===== Step 5: Split, standardize, PCA =====


def build_split_indices(
    fold_labels: list[str],
    valid_mol_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train/val/test index arrays, filtered by valid_mol_mask."""
    fold_arr = np.array(fold_labels)
    valid = valid_mol_mask.astype(bool)
    train_idx = np.where((fold_arr == "train") & valid)[0]
    val_idx = np.where((fold_arr == "val") & valid)[0]
    test_idx = np.where((fold_arr == "test") & valid)[0]
    log.info(
        "Split sizes: train=%d, val=%d, test=%d (total valid=%d / %d)",
        len(train_idx),
        len(val_idx),
        len(test_idx),
        valid.sum(),
        len(fold_labels),
    )
    return train_idx, val_idx, test_idx


def prepare_embeddings(
    embeddings: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    pca_dim: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize and optionally PCA-reduce embeddings."""
    scaler = StandardScaler()
    x_train = scaler.fit_transform(embeddings[train_idx])
    x_val = scaler.transform(embeddings[val_idx])
    x_test = scaler.transform(embeddings[test_idx])

    if pca_dim is not None and pca_dim < x_train.shape[1]:
        log.info("Applying PCA: %d → %d dims", x_train.shape[1], pca_dim)
        pca = PCA(n_components=pca_dim, random_state=42)
        x_train = pca.fit_transform(x_train)
        x_val = pca.transform(x_val)
        x_test = pca.transform(x_test)
        log.info("PCA explained variance: %.4f", pca.explained_variance_ratio_.sum())

    return x_train, x_val, x_test


# ===== Step 6: Linear probes with val-based HP selection =====


def ridge_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    target_name: str,
) -> dict[str, float]:
    """Ridge regression with val-based alpha selection."""
    best_alpha = _RIDGE_ALPHAS[0]
    best_val_r2 = -np.inf

    for alpha in _RIDGE_ALPHAS:
        model = Ridge(alpha=alpha)
        model.fit(x_train, y_train)
        val_r2 = r2_score(y_val, model.predict(x_val))
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_alpha = alpha

    # Refit with best alpha
    model = Ridge(alpha=best_alpha)
    model.fit(x_train, y_train)
    test_r2 = r2_score(y_test, model.predict(x_test))

    log.info("  %s: best_alpha=%.2f, val_R²=%.4f, test_R²=%.4f", target_name, best_alpha, best_val_r2, test_r2)
    return {"best_alpha": best_alpha, "val_r2": best_val_r2, "test_r2": test_r2}


def logistic_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    target_name: str,
) -> dict[str, float]:
    """L2 logistic regression with val-based C selection."""
    best_c = _LOGISTIC_CS[0]
    best_val_auc = -np.inf

    for c in _LOGISTIC_CS:
        model = LogisticRegression(C=c, penalty="l2", solver="lbfgs", max_iter=1000)
        model.fit(x_train, y_train)
        proba = model.predict_proba(x_val)[:, 1]
        val_auc = roc_auc_score(y_val, proba)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_c = c

    # Refit with best C
    model = LogisticRegression(C=best_c, penalty="l2", solver="lbfgs", max_iter=1000)
    model.fit(x_train, y_train)
    proba = model.predict_proba(x_test)[:, 1]
    test_auc = roc_auc_score(y_test, proba)

    log.info("  %s: best_C=%.2f, val_AUC=%.4f, test_AUC=%.4f", target_name, best_c, best_val_auc, test_auc)
    return {"best_C": best_c, "val_auc": best_val_auc, "test_auc": test_auc}


def run_linear_probes(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    mol_props: dict[str, np.ndarray],
    fg_counts: dict[str, np.ndarray],
    label: str,
) -> dict:
    """Run all regression and classification probes."""
    log.info("Running linear probes for: %s (dim=%d)", label, x_train.shape[1])
    results: dict = {"dim": x_train.shape[1], "regression": {}, "classification": {}}

    # Regression probes
    for name in REGRESSION_TARGET_KEYS:
        y = mol_props[name]
        results["regression"][name] = ridge_probe(
            x_train, y[train_idx], x_val, y[val_idx], x_test, y[test_idx], name
        )

    # Classification probes: filter to 1-99% prevalence on train split
    qualifying_fgs = []
    for name in FG_SMARTS:
        y_all = (fg_counts[name] > 0).astype(np.int32)
        train_prev = y_all[train_idx].mean()
        if _FG_PREVALENCE_MIN <= train_prev <= _FG_PREVALENCE_MAX:
            qualifying_fgs.append(name)

    log.info("  Qualifying FGs (%d / %d): %s", len(qualifying_fgs), len(FG_SMARTS), qualifying_fgs)
    for name in qualifying_fgs:
        y = (fg_counts[name] > 0).astype(np.int32)
        results["classification"][name] = logistic_probe(
            x_train, y[train_idx], x_val, y[val_idx], x_test, y[test_idx], name
        )

    return results


# ===== Step 6b: HGB probes with val-based early stopping =====

_HGB_LR = [0.01, 0.05, 0.1]
_HGB_MAX_DEPTH = [4, 6, None]


def hgb_regression_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    name: str,
) -> dict:
    """HGB regression with val-based HP selection."""
    best_r2, best_params, best_model = -np.inf, {}, None
    for lr in _HGB_LR:
        for md in _HGB_MAX_DEPTH:
            m = HistGradientBoostingRegressor(
                learning_rate=lr,
                max_depth=md,
                max_iter=500,
                early_stopping=True,
                validation_fraction=None,
                n_iter_no_change=20,
                random_state=42,
            )
            m.fit(x_train, y_train)
            val_r2 = r2_score(y_val, m.predict(x_val))
            if val_r2 > best_r2:
                best_r2, best_params, best_model = val_r2, {"lr": lr, "max_depth": md}, m
    test_r2 = r2_score(y_test, best_model.predict(x_test))
    log.info("  %s: lr=%.2f, depth=%s, val_R²=%.4f, test_R²=%.4f",
             name, best_params["lr"], best_params["max_depth"], best_r2, test_r2)
    return {"best_params": best_params, "val_r2": best_r2, "test_r2": test_r2}


def hgb_classification_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    name: str,
) -> dict:
    """HGB classification with val-based HP selection."""
    best_auc, best_params, best_model = -np.inf, {}, None
    for lr in _HGB_LR:
        for md in _HGB_MAX_DEPTH:
            m = HistGradientBoostingClassifier(
                learning_rate=lr,
                max_depth=md,
                max_iter=500,
                early_stopping=True,
                validation_fraction=None,
                n_iter_no_change=20,
                random_state=42,
            )
            m.fit(x_train, y_train)
            proba = m.predict_proba(x_val)
            if proba.shape[1] == 2:
                val_auc = roc_auc_score(y_val, proba[:, 1])
            else:
                val_auc = float("nan")
            if val_auc > best_auc:
                best_auc, best_params, best_model = val_auc, {"lr": lr, "max_depth": md}, m
    proba = best_model.predict_proba(x_test)
    test_auc = roc_auc_score(y_test, proba[:, 1]) if proba.shape[1] == 2 else float("nan")
    log.info("  %s: lr=%.2f, depth=%s, val_AUC=%.4f, test_AUC=%.4f",
             name, best_params["lr"], best_params["max_depth"], best_auc, test_auc)
    return {"best_params": {k: str(v) for k, v in best_params.items()}, "val_auc": best_auc, "test_auc": test_auc}


def run_hgb_probes(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    mol_props: dict[str, np.ndarray],
    fg_counts: dict[str, np.ndarray],
    qualifying_fgs: list[str],
    label: str,
) -> dict:
    """Run HGB regression and classification probes."""
    log.info("Running HGB probes for: %s (dim=%d)", label, x_train.shape[1])
    results: dict = {"dim": x_train.shape[1], "regression": {}, "classification": {}}

    for name in REGRESSION_TARGET_KEYS:
        y = mol_props[name]
        results["regression"][name] = hgb_regression_probe(
            x_train, y[train_idx], x_val, y[val_idx], x_test, y[test_idx], name
        )

    for name in qualifying_fgs:
        y = (fg_counts[name] > 0).astype(np.int32)
        results["classification"][name] = hgb_classification_probe(
            x_train, y[train_idx], x_val, y[val_idx], x_test, y[test_idx], name
        )

    return results


# ===== Step 7: No-probe sanity checks =====


def run_knn_probes(
    x_train: np.ndarray,
    x_test: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    mol_props: dict[str, np.ndarray],
    fg_counts: dict[str, np.ndarray],
    qualifying_fgs: list[str],
    label: str,
    k: int = 10,
) -> dict:
    """kNN label transfer (k=10, cosine)."""
    log.info("Running kNN probes (k=%d) for: %s", k, label)
    results: dict = {"regression": {}, "classification": {}}

    for name in REGRESSION_TARGET_KEYS:
        y = mol_props[name]
        knn = KNeighborsRegressor(n_neighbors=k, metric="cosine")
        knn.fit(x_train, y[train_idx])
        pred = knn.predict(x_test)
        r2 = r2_score(y[test_idx], pred)
        results["regression"][name] = {"test_r2": r2}
        log.info("  kNN %s: R²=%.4f", name, r2)

    for name in qualifying_fgs:
        y = (fg_counts[name] > 0).astype(np.int32)
        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        knn.fit(x_train, y[train_idx])
        proba = knn.predict_proba(x_test)
        # Handle case where only one class predicted
        if proba.shape[1] == 2:
            auc = roc_auc_score(y[test_idx], proba[:, 1])
        else:
            auc = float("nan")
        results["classification"][name] = {"test_auc": auc}
        log.info("  kNN %s: AUC=%.4f", name, auc)

    return results


def run_tanimoto_correlation(
    embeddings: np.ndarray,
    smiles: list[str],
    test_idx: np.ndarray,
    label: str,
    n_pairs: int = 500_000,
    seed: int = 42,
) -> dict[str, float]:
    """Cosine sim vs Tanimoto sim correlation on random test pairs."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    from scipy import stats

    log.info("Computing Tanimoto correlation for: %s (%d pairs)", label, n_pairs)
    rng = np.random.RandomState(seed)

    # Compute Morgan fingerprints for test set
    test_smiles = [smiles[i] for i in test_idx]
    test_emb = embeddings[test_idx]

    fps = []
    valid = np.ones(len(test_smiles), dtype=bool)
    for i, smi in enumerate(test_smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            valid[i] = False
            fps.append(None)
        else:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))

    valid_idx = np.where(valid)[0]
    if len(valid_idx) < 2:
        return {"pearson": float("nan"), "spearman": float("nan")}

    # Sample random pairs
    pairs_a = rng.choice(valid_idx, size=n_pairs, replace=True)
    pairs_b = rng.choice(valid_idx, size=n_pairs, replace=True)
    # Remove self-pairs
    mask = pairs_a != pairs_b
    pairs_a = pairs_a[mask]
    pairs_b = pairs_b[mask]

    # Cosine similarities
    norms_a = np.linalg.norm(test_emb[pairs_a], axis=1, keepdims=True)
    norms_b = np.linalg.norm(test_emb[pairs_b], axis=1, keepdims=True)
    cos_sims = (test_emb[pairs_a] * test_emb[pairs_b]).sum(axis=1) / (
        (norms_a.squeeze() * norms_b.squeeze()) + 1e-8
    )

    # Tanimoto similarities
    tan_sims = np.array([
        DataStructs.TanimotoSimilarity(fps[a], fps[b]) for a, b in zip(pairs_a, pairs_b)
    ], dtype=np.float32)

    pearson_r = float(stats.pearsonr(cos_sims, tan_sims)[0])
    spearman_r = float(stats.spearmanr(cos_sims, tan_sims)[0])
    log.info("  Tanimoto correlation: Pearson=%.4f, Spearman=%.4f", pearson_r, spearman_r)
    return {"pearson": pearson_r, "spearman": spearman_r}


# ===== Step 8: Output =====


def format_results_table(all_results: dict, mode: str) -> str:
    """Format results as a readable comparison table."""
    lines = []
    configs = list(all_results.keys())

    if mode == "compare":
        # Side-by-side comparison
        header = f"{'Metric':<30}" + "".join(f"  {c:>20}" for c in configs)
        lines.append(header)
        lines.append("=" * len(header))

        # Regression (Linear)
        if any("linear_probes" in all_results[c] for c in configs):
            lines.append("\nREGRESSION (Ridge, test R²)")
            for target in REGRESSION_TARGET_KEYS:
                row = f"  {target:<28}"
                for cfg in configs:
                    r = all_results[cfg].get("linear_probes", {}).get("regression", {}).get(target, {})
                    val = r.get("test_r2", float("nan"))
                    row += f"  {val:>20.4f}"
                lines.append(row)

            # Classification (Linear)
            lines.append("\nCLASSIFICATION (Logistic, test AUC)")
            all_fgs: set[str] = set()
            for cfg in configs:
                all_fgs |= set(all_results[cfg].get("linear_probes", {}).get("classification", {}).keys())
            for fg in sorted(all_fgs):
                row = f"  {fg:<28}"
                for cfg in configs:
                    r = all_results[cfg].get("linear_probes", {}).get("classification", {}).get(fg, {})
                    val = r.get("test_auc", float("nan"))
                    row += f"  {val:>20.4f}"
                lines.append(row)

        # Regression (HGB)
        if any("hgb_probes" in all_results[c] for c in configs):
            lines.append("\nREGRESSION (HGB, test R²)")
            for target in REGRESSION_TARGET_KEYS:
                row = f"  {target:<28}"
                for cfg in configs:
                    r = all_results[cfg].get("hgb_probes", {}).get("regression", {}).get(target, {})
                    val = r.get("test_r2", float("nan"))
                    row += f"  {val:>20.4f}"
                lines.append(row)

            # Classification (HGB)
            lines.append("\nCLASSIFICATION (HGB, test AUC)")
            all_fgs_hgb: set[str] = set()
            for cfg in configs:
                all_fgs_hgb |= set(all_results[cfg].get("hgb_probes", {}).get("classification", {}).keys())
            for fg in sorted(all_fgs_hgb):
                row = f"  {fg:<28}"
                for cfg in configs:
                    r = all_results[cfg].get("hgb_probes", {}).get("classification", {}).get(fg, {})
                    val = r.get("test_auc", float("nan"))
                    row += f"  {val:>20.4f}"
                lines.append(row)

        # kNN
        if any("knn" in all_results[c] for c in configs):
            lines.append("\nkNN (k=10, cosine, test)")
            for target in REGRESSION_TARGET_KEYS:
                row = f"  {target:<28}"
                for cfg in configs:
                    r = all_results[cfg].get("knn", {}).get("regression", {}).get(target, {})
                    val = r.get("test_r2", float("nan"))
                    row += f"  {val:>20.4f}"
                lines.append(row)

        # Tanimoto
        if any("tanimoto" in all_results[c] for c in configs):
            lines.append("\nTANIMOTO CORRELATION")
            for metric in ["pearson", "spearman"]:
                row = f"  {metric:<28}"
                for cfg in configs:
                    val = all_results[cfg].get("tanimoto", {}).get(metric, float("nan"))
                    row += f"  {val:>20.4f}"
                lines.append(row)
    else:
        # Single-column display
        cfg = configs[0]
        lines.append(f"Results for: {cfg}")
        lines.append("=" * 60)

        if "linear_probes" in all_results[cfg]:
            lines.append("\nREGRESSION (Ridge, test R²)")
            for target in REGRESSION_TARGET_KEYS:
                r = all_results[cfg]["linear_probes"].get("regression", {}).get(target, {})
                lines.append(f"  {target:<28} {r.get('test_r2', float('nan')):>10.4f}")
            lines.append("\nCLASSIFICATION (Logistic, test AUC)")
            for fg, r in all_results[cfg]["linear_probes"].get("classification", {}).items():
                lines.append(f"  {fg:<28} {r.get('test_auc', float('nan')):>10.4f}")

        if "hgb_probes" in all_results[cfg]:
            lines.append("\nREGRESSION (HGB, test R²)")
            for target in REGRESSION_TARGET_KEYS:
                r = all_results[cfg]["hgb_probes"].get("regression", {}).get(target, {})
                lines.append(f"  {target:<28} {r.get('test_r2', float('nan')):>10.4f}")
            lines.append("\nCLASSIFICATION (HGB, test AUC)")
            for fg, r in all_results[cfg]["hgb_probes"].get("classification", {}).items():
                lines.append(f"  {fg:<28} {r.get('test_auc', float('nan')):>10.4f}")

        if "knn" in all_results[cfg]:
            lines.append("\nkNN (k=10, cosine, test)")
            for target in REGRESSION_TARGET_KEYS:
                r = all_results[cfg]["knn"].get("regression", {}).get(target, {})
                lines.append(f"  {target:<28} {r.get('test_r2', float('nan')):>10.4f}")

        if "tanimoto" in all_results[cfg]:
            lines.append("\nTANIMOTO CORRELATION")
            t = all_results[cfg]["tanimoto"]
            lines.append(f"  pearson                      {t.get('pearson', float('nan')):>10.4f}")
            lines.append(f"  spearman                     {t.get('spearman', float('nan')):>10.4f}")

    return "\n".join(lines)


# ===== Preprocessing validation =====


def validate_preprocessing(
    preprocessed: dict[str, np.ndarray],
    config_path: str,
    smiles_from_hdf5: list[str],
    n_check: int = 100,
) -> None:
    """Validate NumPy preprocessing against TF pipeline by loading TFRecord data."""
    from utils.massspec_probe_data import MassSpecProbeData, numpy_batch_to_torch

    log.info("Validating preprocessing against TFRecord pipeline...")
    config = load_config(config_path)
    probe_data = MassSpecProbeData.from_config(config)
    ds = probe_data.build_dataset("val")

    # Collect a few batches from TFRecord
    tf_mz_list = []
    tf_int_list = []
    tf_smiles_list = []
    count = 0
    for batch in ds:
        batch = numpy_batch_to_torch(batch)
        tf_mz_list.append(batch["peak_mz"].numpy())
        tf_int_list.append(batch["peak_intensity"].numpy())
        tf_smiles_list.extend(batch["smiles"])
        count += tf_mz_list[-1].shape[0]
        if count >= n_check:
            break

    tf_mz = np.concatenate(tf_mz_list, axis=0)[:n_check]
    tf_int = np.concatenate(tf_int_list, axis=0)[:n_check]
    tf_smiles = tf_smiles_list[:n_check]

    # Match by SMILES to HDF5
    hdf5_smiles_set = {s: i for i, s in enumerate(smiles_from_hdf5)}
    matched = 0
    mismatches = 0
    for j, smi in enumerate(tf_smiles):
        if smi not in hdf5_smiles_set:
            continue
        i = hdf5_smiles_set[smi]
        mz_close = np.allclose(preprocessed["peak_mz"][i], tf_mz[j], atol=1e-5)
        int_close = np.allclose(preprocessed["peak_intensity"][i], tf_int[j], atol=1e-5)
        if mz_close and int_close:
            matched += 1
        else:
            mismatches += 1
            if mismatches <= 3:
                log.warning(
                    "Mismatch for SMILES=%s: mz_close=%s, int_close=%s",
                    smi[:50],
                    mz_close,
                    int_close,
                )

    log.info("Preprocessing validation: %d matched, %d mismatched out of %d checked", matched, mismatches, n_check)
    if mismatches > matched * 0.1:
        log.warning("High mismatch rate! Check preprocessing logic.")


# ===== Main =====


def run_benchmark(args: argparse.Namespace) -> dict:
    """Run the full benchmark pipeline."""
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load data from either HDF5 or pickle
    if args.pkl:
        data = load_mona_pkl_data(args.pkl)
    else:
        # Auto-detect HDF5 format: MassSpecGym has FOLD+IDENTIFIER, DreaMS Atlas doesn't
        with h5py.File(args.dreams_hdf5, "r") as _f:
            has_fold = "FOLD" in _f
        if has_fold:
            data = load_dreams_data(args.dreams_hdf5)
        else:
            data = load_dreams_atlas_hdf5(args.dreams_hdf5)

    # Step 2: Preprocess spectra
    preprocessed = preprocess_dreams_spectra(data["spectrum"], data["precursor_mz"])

    # Optionally validate preprocessing
    if args.validate_preprocessing and args.config and not args.pkl:
        validate_preprocessing(preprocessed, args.config, data["smiles"])

    # Step 4: Compute targets
    mol_props, fg_counts, valid_mol_mask = compute_targets_cached(data["smiles"], workdir)

    # Step 5: Build splits
    train_idx, val_idx, test_idx = build_split_indices(data["fold"], valid_mol_mask)

    # Determine qualifying FGs (for kNN too)
    qualifying_fgs = []
    for name in FG_SMARTS:
        y = (fg_counts[name] > 0).astype(np.int32)
        prev = y[train_idx].mean()
        if _FG_PREVALENCE_MIN <= prev <= _FG_PREVALENCE_MAX:
            qualifying_fgs.append(name)

    # Collect embedding sources
    embedding_sources: dict[str, np.ndarray] = {}

    need_ours = args.mode in ("ours", "compare")
    need_dreams = args.mode in ("dreams", "compare")

    if need_ours:
        assert args.config is not None, "--config required for mode 'ours' or 'compare'"
        assert args.dir is not None, "--dir required for mode 'ours' or 'compare'"
        # Step 3: Extract our embeddings
        ours_emb = extract_our_embeddings(
            args.config,
            args.dir,
            preprocessed,
            batch_size=args.batch_size,
            device=args.device,
        )
        embedding_sources["ours_native"] = ours_emb

    if need_dreams:
        if data["dreams_embedding"] is None:
            log.warning("No DreaMS embeddings in this data source; skipping 'dreams' mode")
        else:
            embedding_sources["dreams_native"] = data["dreams_embedding"]

    # In compare mode, also add PCA-matched versions
    if args.mode == "compare" and args.matched_dim is not None:
        matched_dim = args.matched_dim
        log.info("Will also evaluate PCA-matched embeddings at dim=%d", matched_dim)

    # Run probes for each source
    all_results: dict = {}

    for source_name, emb in embedding_sources.items():
        log.info("=" * 60)
        log.info("Evaluating: %s (native dim=%d)", source_name, emb.shape[1])
        x_train, x_val, x_test = prepare_embeddings(emb, train_idx, val_idx, test_idx)

        source_results: dict = {}

        if args.probe in ("linear", "both"):
            source_results["linear_probes"] = run_linear_probes(
                x_train, x_val, x_test, train_idx, val_idx, test_idx, mol_props, fg_counts, source_name
            )
        if args.probe in ("hgb", "both"):
            source_results["hgb_probes"] = run_hgb_probes(
                x_train, x_val, x_test, train_idx, val_idx, test_idx, mol_props, fg_counts, qualifying_fgs, source_name
            )

        source_results["knn"] = run_knn_probes(
            x_train, x_test, train_idx, test_idx, mol_props, fg_counts, qualifying_fgs, source_name
        )
        source_results["tanimoto"] = run_tanimoto_correlation(emb, data["smiles"], test_idx, source_name)

        all_results[source_name] = source_results

    # PCA-matched versions in compare mode
    if args.mode == "compare" and args.matched_dim is not None:
        matched_dim = args.matched_dim
        for source_name, emb in embedding_sources.items():
            pca_name = f"{source_name.replace('_native', '')}_pca{matched_dim}"
            log.info("=" * 60)
            log.info("Evaluating: %s (PCA %d → %d)", pca_name, emb.shape[1], matched_dim)
            x_train, x_val, x_test = prepare_embeddings(
                emb, train_idx, val_idx, test_idx, pca_dim=matched_dim
            )
            pca_results: dict = {}

            if args.probe in ("linear", "both"):
                pca_results["linear_probes"] = run_linear_probes(
                    x_train, x_val, x_test, train_idx, val_idx, test_idx, mol_props, fg_counts, pca_name
                )
            if args.probe in ("hgb", "both"):
                pca_results["hgb_probes"] = run_hgb_probes(
                    x_train, x_val, x_test, train_idx, val_idx, test_idx, mol_props, fg_counts, qualifying_fgs, pca_name
                )

            pca_results["knn"] = run_knn_probes(
                x_train, x_test, train_idx, test_idx, mol_props, fg_counts, qualifying_fgs, pca_name
            )
            pca_results["tanimoto"] = run_tanimoto_correlation(emb, data["smiles"], test_idx, pca_name)

            all_results[pca_name] = pca_results

    # Step 8: Output
    results_path = workdir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    log.info("Saved results to %s", results_path)

    table = format_results_table(all_results, args.mode)
    print("\n" + table + "\n")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Frozen linear probe benchmark: Ours vs DreaMS",
    )
    parser.add_argument(
        "--mode",
        choices=["ours", "dreams", "compare"],
        required=True,
        help="Which embeddings to evaluate",
    )
    parser.add_argument(
        "--probe",
        choices=["linear", "hgb", "both"],
        default="linear",
        help="Probe type: linear (Ridge/Logistic), hgb (HistGradientBoosting), or both",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to model config (required for 'ours' and 'compare' modes)",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Path to experiment directory with checkpoints (required for 'ours' and 'compare' modes)",
    )
    parser.add_argument(
        "--dreams-hdf5",
        type=str,
        default=None,
        help="Path to MassSpecGym_DreaMS.hdf5",
    )
    parser.add_argument(
        "--pkl",
        type=str,
        default=None,
        help="Path to MoNA/GeMS pickle file (alternative to --dreams-hdf5)",
    )
    parser.add_argument(
        "--workdir",
        type=str,
        default="results/benchmark_linear_probe",
        help="Output directory for results",
    )
    parser.add_argument(
        "--matched-dim",
        type=int,
        default=None,
        help="PCA dimension for matched comparison (only in 'compare' mode)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for model inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for model inference",
    )
    parser.add_argument(
        "--validate-preprocessing",
        action="store_true",
        help="Validate NumPy preprocessing against TFRecord pipeline",
    )
    args = parser.parse_args()

    # Validate args
    if not args.dreams_hdf5 and not args.pkl:
        parser.error("Either --dreams-hdf5 or --pkl is required")
    if args.dreams_hdf5 and args.pkl:
        parser.error("Only one of --dreams-hdf5 or --pkl can be specified")
    if args.mode in ("ours", "compare"):
        if args.config is None:
            parser.error("--config is required for mode 'ours' or 'compare'")
        if args.dir is None:
            parser.error("--dir is required for mode 'ours' or 'compare'")
    if args.pkl and args.mode in ("dreams", "compare"):
        parser.error("--mode dreams/compare requires --dreams-hdf5 (no DreaMS embeddings in pickle data)")

    run_benchmark(args)


if __name__ == "__main__":
    main()
