import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, MACCSkeys, rdMolDescriptors
from rdkit.Chem import AllChem

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
MACCS_FINGERPRINT_BITS = 166
MORGAN_PROBE_FINGERPRINT_BITS = 4096
MORGAN_PROBE_FINGERPRINT_RADIUS = 2
_MACCS_TOTAL_BITS = 167


def compute_probe_targets_for_smiles(
    smiles: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    n = len(smiles)
    patterns = {name: Chem.MolFromSmarts(smarts) for name, smarts in FG_SMARTS.items()}
    fg_counts = {name: np.zeros(n, dtype=np.int16) for name in FG_SMARTS}
    valid_mol_mask = np.ones(n, dtype=bool)
    mol_props = {name: np.zeros(n, dtype=np.float32) for name in REGRESSION_TARGET_KEYS}
    for i, smi in enumerate(smiles):
        if (mol := Chem.MolFromSmiles(smi)) is None:
            valid_mol_mask[i] = False
            continue
        for name, pattern in patterns.items():
            fg_counts[name][i] = len(mol.GetSubstructMatches(pattern))
        mol_props["mol_weight"][i] = Descriptors.ExactMolWt(mol)
        mol_props["logp"][i] = Descriptors.MolLogP(mol)
        mol_props["num_heavy_atoms"][i] = float(mol.GetNumHeavyAtoms())
        mol_props["num_rings"][i] = float(rdMolDescriptors.CalcNumRings(mol))
    return mol_props, fg_counts, valid_mol_mask


def compute_maccs_fingerprint_bits_for_smiles(
    smiles: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    n = len(smiles)
    maccs_bits = np.zeros((n, MACCS_FINGERPRINT_BITS), dtype=np.int32)
    valid_mol_mask = np.ones(n, dtype=bool)
    for i, smi in enumerate(smiles):
        if (mol := Chem.MolFromSmiles(smi)) is None:
            valid_mol_mask[i] = False
            continue
        fp = MACCSkeys.GenMACCSKeys(mol)
        full_bits = np.zeros(_MACCS_TOTAL_BITS, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, full_bits)
        maccs_bits[i] = full_bits[1:]
    return maccs_bits, valid_mol_mask


def compute_morgan_fingerprint_bits_for_smiles(
    smiles: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    n = len(smiles)
    morgan_bits = np.zeros((n, MORGAN_PROBE_FINGERPRINT_BITS), dtype=np.int32)
    valid_mol_mask = np.ones(n, dtype=bool)
    for i, smi in enumerate(smiles):
        if (mol := Chem.MolFromSmiles(smi)) is None:
            valid_mol_mask[i] = False
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            MORGAN_PROBE_FINGERPRINT_RADIUS,
            nBits=MORGAN_PROBE_FINGERPRINT_BITS,
        )
        DataStructs.ConvertToNumpyArray(fp, morgan_bits[i])
    return morgan_bits, valid_mol_mask


def _build_row_indices(smiles: np.ndarray) -> tuple[list[str], np.ndarray]:
    unique_smiles, index_by_smiles = [], {}
    row_indices = np.empty(len(smiles), dtype=np.int64)
    for row_idx, smi in enumerate(smiles.tolist()):
        smi = str(smi)
        target_idx = index_by_smiles.get(smi)
        if target_idx is None:
            target_idx = len(unique_smiles)
            index_by_smiles[smi] = target_idx
            unique_smiles.append(smi)
        row_indices[row_idx] = target_idx
    return unique_smiles, row_indices


def build_probe_targets_for_rows(
    smiles: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    unique_smiles, row_indices = _build_row_indices(smiles)
    mp, fg, valid = compute_probe_targets_for_smiles(unique_smiles)
    return (
        {n: v[row_indices].astype(np.float32, copy=False) for n, v in mp.items()},
        {n: (v[row_indices] > 0).astype(np.int32, copy=False) for n, v in fg.items()},
        valid[row_indices],
    )


def build_maccs_targets_for_rows(
    smiles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    unique_smiles, row_indices = _build_row_indices(smiles)
    maccs_bits, valid = compute_maccs_fingerprint_bits_for_smiles(unique_smiles)
    return (
        maccs_bits[row_indices].astype(np.int32, copy=False),
        valid[row_indices],
    )


def build_morgan_targets_for_rows(
    smiles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    unique_smiles, row_indices = _build_row_indices(smiles)
    morgan_bits, valid = compute_morgan_fingerprint_bits_for_smiles(unique_smiles)
    return (
        morgan_bits[row_indices].astype(np.int32, copy=False),
        valid[row_indices],
    )
