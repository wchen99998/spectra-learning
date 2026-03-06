from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

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
        "num_heavy_atoms": np.zeros(n, dtype=np.float32),
        "num_rings": np.zeros(n, dtype=np.float32),
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
        mol_props["num_heavy_atoms"][i] = float(mol.GetNumHeavyAtoms())
        mol_props["num_rings"][i] = float(rdMolDescriptors.CalcNumRings(mol))

    return mol_props, fg_counts, valid_mol_mask


def build_probe_targets_for_rows(
    smiles: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    unique_smiles: list[str] = []
    index_by_smiles: dict[str, int] = {}
    row_indices = np.empty(len(smiles), dtype=np.int64)

    for row_idx, smi in enumerate(smiles.tolist()):
        smi = str(smi)
        target_idx = index_by_smiles.get(smi)
        if target_idx is None:
            target_idx = len(unique_smiles)
            index_by_smiles[smi] = target_idx
            unique_smiles.append(smi)
        row_indices[row_idx] = target_idx

    mol_props_unique, fg_counts_unique, valid_unique = compute_probe_targets_for_smiles(unique_smiles)
    mol_props = {
        name: values[row_indices].astype(np.float32, copy=False)
        for name, values in mol_props_unique.items()
    }
    fg_binary = {
        name: (values[row_indices] > 0).astype(np.int32, copy=False)
        for name, values in fg_counts_unique.items()
    }
    valid_mol_mask = valid_unique[row_indices]
    return mol_props, fg_binary, valid_mol_mask
