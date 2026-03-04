from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ml_collections import config_dict
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from input_pipeline import (
    TfLightningDataModule,
    _MASSSPEC_HF_REPO,
    _MASSSPEC_TSV_PATH,
    _download_hf_file,
)
from models.model import PeakSetSIGReg
from utils.probing import iter_massspec_probe


log = logging.getLogger(__name__)


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

REGRESSION_TARGET_KEYS = (
    "mol_weight",
    "logp",
    "num_heavy_atoms",
    "num_rings",
)

_CACHE_VERSION = 1
_CACHE_FILENAME = f"MassSpecGym_probe_targets_v{_CACHE_VERSION}.npz"


@dataclass(slots=True)
class MsgProbeTargets:
    smiles: np.ndarray
    mol_props: dict[str, np.ndarray]
    fg_counts: dict[str, np.ndarray]
    valid_mol_mask: np.ndarray
    index_by_smiles: dict[str, int]


def should_run_msg_linear_probe(global_step: int, every_n_steps: int) -> bool:
    return every_n_steps > 0 and global_step % every_n_steps == 0


def compute_probe_targets_for_smiles(
    smiles: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    n = len(smiles)
    patterns = {name: Chem.MolFromSmarts(smarts) for name, smarts in FG_SMARTS.items()}
    fg_counts = {name: np.zeros(n, dtype=np.int16) for name in FG_SMARTS}
    valid_mol_mask = np.ones(n, dtype=bool)

    mol_props = {
        "mol_weight": np.zeros(n, dtype=np.float32),
        "logp": np.zeros(n, dtype=np.float32),
        "num_heavy_atoms": np.zeros(n, dtype=np.int16),
        "num_rings": np.zeros(n, dtype=np.int16),
    }

    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            valid_mol_mask[i] = False
            continue
        for name, pattern in patterns.items():
            fg_counts[name][i] = len(mol.GetSubstructMatches(pattern))
        mol_props["mol_weight"][i] = Descriptors.ExactMolWt(mol)
        mol_props["logp"][i] = Descriptors.MolLogP(mol)
        mol_props["num_heavy_atoms"][i] = mol.GetNumHeavyAtoms()
        mol_props["num_rings"][i] = rdMolDescriptors.CalcNumRings(mol)

    return mol_props, fg_counts, valid_mol_mask


def _load_unique_smiles_from_tsv(tsv_path: Path) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    with tsv_path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            smiles = row["smiles"]
            if smiles in seen:
                continue
            seen.add(smiles)
            unique.append(smiles)
    return unique


def _pack_targets(
    *,
    smiles: np.ndarray,
    mol_props: dict[str, np.ndarray],
    fg_counts: dict[str, np.ndarray],
    valid_mol_mask: np.ndarray,
) -> MsgProbeTargets:
    index_by_smiles = {s: i for i, s in enumerate(smiles.tolist())}
    return MsgProbeTargets(
        smiles=smiles,
        mol_props=mol_props,
        fg_counts=fg_counts,
        valid_mol_mask=valid_mol_mask,
        index_by_smiles=index_by_smiles,
    )


def _load_targets_from_npz(data: np.lib.npyio.NpzFile) -> MsgProbeTargets:
    smiles = data["smiles"]
    mol_props = {
        key: data[key]
        for key in REGRESSION_TARGET_KEYS
    }
    fg_counts = {
        name: data[f"fg_{name}"]
        for name in FG_SMARTS
    }
    valid_mol_mask = data["valid_mol_mask"]
    return _pack_targets(
        smiles=smiles,
        mol_props=mol_props,
        fg_counts=fg_counts,
        valid_mol_mask=valid_mol_mask,
    )


def load_or_build_msg_probe_targets(
    *,
    tsv_path: Path,
    cache_dir: Path,
) -> MsgProbeTargets:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / _CACHE_FILENAME
    tsv_stat = tsv_path.stat()

    if cache_path.exists():
        data = np.load(cache_path)
        cache_version = int(data["cache_version"])
        tsv_size = int(data["tsv_size"])
        tsv_mtime_ns = int(data["tsv_mtime_ns"])
        if (
            cache_version == _CACHE_VERSION
            and tsv_size == int(tsv_stat.st_size)
            and tsv_mtime_ns == int(tsv_stat.st_mtime_ns)
        ):
            log.info("Loading MSG probe target cache from %s", cache_path)
            return _load_targets_from_npz(data)

    log.info("Building MSG probe target cache from %s", tsv_path)
    unique_smiles = _load_unique_smiles_from_tsv(tsv_path)
    mol_props, fg_counts, valid_mol_mask = compute_probe_targets_for_smiles(unique_smiles)
    smiles = np.asarray(unique_smiles, dtype=np.str_)

    save_dict: dict[str, np.ndarray] = {
        "cache_version": np.asarray(_CACHE_VERSION, dtype=np.int64),
        "tsv_size": np.asarray(int(tsv_stat.st_size), dtype=np.int64),
        "tsv_mtime_ns": np.asarray(int(tsv_stat.st_mtime_ns), dtype=np.int64),
        "smiles": smiles,
        "valid_mol_mask": valid_mol_mask,
    }
    for key, values in mol_props.items():
        save_dict[key] = values
    for name, values in fg_counts.items():
        save_dict[f"fg_{name}"] = values
    np.savez_compressed(cache_path, **save_dict)
    log.info("Wrote MSG probe target cache to %s", cache_path)

    return _pack_targets(
        smiles=smiles,
        mol_props=mol_props,
        fg_counts=fg_counts,
        valid_mol_mask=valid_mol_mask,
    )


def _resolve_tsv_and_cache_dir(
    config: config_dict.ConfigDict,
    cache_dir_override: str | Path | None,
) -> tuple[Path, Path]:
    tfrecord_parent = (
        Path(config.get("tfrecord_dir", "data/gems_peaklist_tfrecord"))
        .expanduser()
        .resolve()
        .parent
    )
    tsv_path = _download_hf_file(
        repo_id=_MASSSPEC_HF_REPO,
        filename=_MASSSPEC_TSV_PATH,
        local_dir=tfrecord_parent,
    )
    if cache_dir_override is None:
        cache_dir = tsv_path.parent
    else:
        cache_dir = Path(cache_dir_override).expanduser().resolve()
    return tsv_path, cache_dir


def _collect_split_embeddings(
    *,
    config: config_dict.ConfigDict,
    datamodule: TfLightningDataModule,
    model: PeakSetSIGReg,
    device: torch.device,
    split: str,
    seed: int,
    targets: MsgProbeTargets,
) -> tuple[np.ndarray, np.ndarray]:
    peak_ordering = str(config.get("peak_ordering", "intensity"))
    batches: list[np.ndarray] = []
    target_indices: list[np.ndarray] = []

    for batch in iter_massspec_probe(
        datamodule=datamodule,
        split=split,
        seed=seed,
        peak_ordering=peak_ordering,
        drop_remainder=False,
    ):
        peak_mz = batch["peak_mz"].to(device, non_blocking=True)
        peak_intensity = batch["peak_intensity"].to(device, non_blocking=True)
        peak_valid_mask = batch["peak_valid_mask"].to(device, non_blocking=True)
        pooled = model.encode(
            {
                "peak_mz": peak_mz,
                "peak_intensity": peak_intensity,
                "peak_valid_mask": peak_valid_mask,
            },
            train=False,
        )
        batches.append(pooled.detach().cpu().to(torch.float32).numpy())
        smiles = batch["smiles"]
        idx = np.asarray([targets.index_by_smiles[s] for s in smiles], dtype=np.int64)
        target_indices.append(idx)

    return np.concatenate(batches, axis=0), np.concatenate(target_indices, axis=0)


def _fit_msg_linear_probe_metrics(
    *,
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    targets: MsgProbeTargets,
    ridge_alpha: float = 10.0,
    ridge_solver: str = "svd",
    logreg_max_iter: int = 2000,
    logreg_solver: str = "lbfgs",
    logreg_c: float = 1.0,
) -> dict[str, float]:
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_embeddings)
    test_features = scaler.transform(test_embeddings)

    metrics: dict[str, float] = {
        "msg_linear_probe/train_samples": float(train_features.shape[0]),
        "msg_linear_probe/test_samples": float(test_features.shape[0]),
    }

    for key in REGRESSION_TARGET_KEYS:
        y_train = targets.mol_props[key][train_idx].astype(np.float32)
        y_test = targets.mol_props[key][test_idx].astype(np.float32)
        reg = Ridge(alpha=ridge_alpha, solver=ridge_solver)
        reg.fit(train_features, y_train)
        train_pred = reg.predict(train_features)
        test_pred = reg.predict(test_features)
        metrics[f"msg_linear_probe/train/r2_{key}"] = float(r2_score(y_train, train_pred))
        metrics[f"msg_linear_probe/test/r2_{key}"] = float(r2_score(y_test, test_pred))

    train_auc_values: list[float] = []
    test_auc_values: list[float] = []
    for name in FG_SMARTS:
        y_train = (targets.fg_counts[name][train_idx] > 0).astype(np.int32)
        y_test = (targets.fg_counts[name][test_idx] > 0).astype(np.int32)
        train_prevalence = float(y_train.mean())
        test_prevalence = float(y_test.mean())
        if train_prevalence < 0.01 or train_prevalence > 0.99:
            continue
        if test_prevalence == 0.0 or test_prevalence == 1.0:
            continue

        clf = LogisticRegression(
            max_iter=logreg_max_iter,
            solver=logreg_solver,
            C=logreg_c,
        )
        clf.fit(train_features, y_train)
        train_prob = clf.predict_proba(train_features)[:, 1]
        test_prob = clf.predict_proba(test_features)[:, 1]
        train_auc = float(roc_auc_score(y_train, train_prob))
        test_auc = float(roc_auc_score(y_test, test_prob))
        metrics[f"msg_linear_probe/train/auc_fg_{name}"] = train_auc
        metrics[f"msg_linear_probe/test/auc_fg_{name}"] = test_auc
        train_auc_values.append(train_auc)
        test_auc_values.append(test_auc)

    metrics["msg_linear_probe/train/auc_fg_mean"] = float(np.mean(train_auc_values))
    metrics["msg_linear_probe/test/auc_fg_mean"] = float(np.mean(test_auc_values))
    metrics["msg_linear_probe/num_fg_tasks"] = float(len(train_auc_values))
    return metrics


def run_msg_linear_probe(
    *,
    config: config_dict.ConfigDict,
    datamodule: TfLightningDataModule,
    model: PeakSetSIGReg,
    device: torch.device,
    cache_dir_override: str | Path | None = None,
    ridge_alpha: float = 10.0,
    ridge_solver: str = "svd",
    logreg_max_iter: int = 2000,
    logreg_solver: str = "lbfgs",
    logreg_c: float = 1.0,
) -> dict[str, float]:
    tsv_path, cache_dir = _resolve_tsv_and_cache_dir(config, cache_dir_override)
    targets = load_or_build_msg_probe_targets(tsv_path=tsv_path, cache_dir=cache_dir)

    train_seed = int(config.seed) + 1_100_000
    test_seed = int(config.seed) + 1_200_000
    was_training = model.training
    model.eval()
    with torch.no_grad():
        train_embeddings, train_idx = _collect_split_embeddings(
            config=config,
            datamodule=datamodule,
            model=model,
            device=device,
            split="massspec_train",
            seed=train_seed,
            targets=targets,
        )
        test_embeddings, test_idx = _collect_split_embeddings(
            config=config,
            datamodule=datamodule,
            model=model,
            device=device,
            split="massspec_test",
            seed=test_seed,
            targets=targets,
        )
    if was_training:
        model.train()

    metrics = _fit_msg_linear_probe_metrics(
        train_embeddings=train_embeddings,
        test_embeddings=test_embeddings,
        train_idx=train_idx,
        test_idx=test_idx,
        targets=targets,
        ridge_alpha=ridge_alpha,
        ridge_solver=ridge_solver,
        logreg_max_iter=logreg_max_iter,
        logreg_solver=logreg_solver,
        logreg_c=logreg_c,
    )
    log.info(
        "MSG linear probe done: test r2(mol_weight)=%.4f test auc_fg_mean=%.4f tasks=%d",
        metrics["msg_linear_probe/test/r2_mol_weight"],
        metrics["msg_linear_probe/test/auc_fg_mean"],
        int(metrics["msg_linear_probe/num_fg_tasks"]),
    )
    return metrics
