"""Hard-negative candidate pools for the three out-of-distribution MGF peptides.

Every pool member is a stereo-free RDKit SMILES with the truth's exact formula,
except the cross pool. Families: ring-site isomers (alternative hydrolysis or
dehydrative cyclization sites), residue-order permutations, residue-level
isomer substitutions, and the PubChem same-formula pool (MassSpecGym-style).
"""

import argparse
import itertools
import json
import urllib.request
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from atom_flare_data import build_graph_store

# Backbone templates written N->C; {r} is a fresh ring-closure label and
# [OH:1] marks the hydroxyl that receives the caveamide B lactone ester.
RESIDUES = {
    "Ser": "NC(CO)C(=O)", "Hse": "NC(CCO)C(=O)", "NMeSer": "N(C)C(CO)C(=O)",
    "IsoSer": "NCC(O)C(=O)", "NOHAla": "N(O)C(C)C(=O)",
    "bOHLeu": "NC(C([OH:1])C(C)C)C(=O)", "gOHLeu": "NC(CC([OH:1])(C)C)C(=O)",
    "dOHLeu": "NC(CC(C)C[OH:1])C(=O)", "bOHNle": "NC(C([OH:1])CCC)C(=O)",
    "Piz": "N{r}NCCCC{r}C(=O)",
    "NMeCha2": "N(C)C(CC{r}CCCC=C{r})C(=O)", "NMeCha1": "N(C)C(CC{r}=CCCCC{r})C(=O)",
    "NMeCha3": "N(C)C(CC{r}CC=CCC{r})C(=O)", "aMeCha2": "NC(C)(CC{r}CCCC=C{r})C(=O)",
    "Ile": "NC(C(C)CC)C(=O)", "Leu": "NC(CC(C)C)C(=O)", "Nle": "NC(CCCC)C(=O)",
    "Thr": "NC(C(C)O)C(=O)", "Val": "NC(C(C)C)C(=O)", "Nva": "NC(CCC)C(=O)",
}
CAPS = {
    "octadienoyl": "CCC=CC=CC(=O)C(C)C(=O)", "methylheptadienoyl": "CC=CC=C(C)C(=O)C(C)C(=O)",
    "octadienoyl_skip": "CC=CC=CCC(=O)C(C)C(=O)", "octadienoyl_terminal": "C=CC=CCCC(=O)C(C)C(=O)",
    "octadienoyl_methyl": "CCC=CC=C(C)C(=O)CC(=O)", "formyl": "O=C", "acetyl": "CC(=O)",
}
# Same-formula residue swaps (residue formulas are identical within each tuple).
SWAPS = {
    "Ser": ("IsoSer", "NOHAla"), "NOHAla": ("Ser", "IsoSer"),
    "bOHLeu": ("gOHLeu", "dOHLeu", "bOHNle"), "NMeCha2": ("NMeCha1", "NMeCha3", "aMeCha2"),
    "Ile": ("Leu", "Nle"), "Leu": ("Ile", "Nle"), "Val": ("Nva",), "Thr": ("Hse", "NMeSer"),
}
CAP_SWAPS = {"octadienoyl": ("methylheptadienoyl", "octadienoyl_skip", "octadienoyl_terminal",
                             "octadienoyl_methyl")}
SPECIES = {
    "caveamide_a": {"cap": "octadienoyl", "residues": ("bOHLeu", "Piz", "NOHAla", "NMeCha2", "Piz", "Ser"),
                    "lactone": False},
    "caveamide_B": {"cap": "octadienoyl", "residues": ("bOHLeu", "Piz", "NOHAla", "NMeCha2", "Piz", "Ser"),
                    "lactone": True},
    "depsidomycin": {"cap": "formyl", "residues": ("Ile", "Thr", "Leu", "Val", "Piz", "Leu", "Piz"),
                     "lactone": False},
}
# formyl-Ile and acetyl-Val are the same N-terminal formula (C7H12NO2).
N_TERMINAL_SWAPS = {("formyl", "Ile"): (("acetyl", "Val"),)}
PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastformula/{}/property/IsomericSMILES/JSON"


def canonical(mol: Chem.Mol | str) -> str:
    mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else Chem.Mol(mol)
    Chem.RemoveStereochemistry(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def assemble(cap: str, residues: tuple[str, ...]) -> Chem.Mol:
    """Linear acid; the C-terminal hydroxyl carries map number 2."""
    labels = iter(range(10, 99))
    parts = []
    for name in residues:
        label = f"%{next(labels)}"
        parts.append(RESIDUES[name].replace("{r}", label))
    mol = Chem.MolFromSmiles(CAPS[cap] + "".join(parts) + "[OH:2]")
    if mol is None:
        raise ValueError(f"Invalid assembly: {cap} {residues}")
    return mol


def condense(mol: Chem.Mol, acid_oxygen: int, nucleophile: int) -> Chem.Mol | None:
    """Dehydrative acylation of `nucleophile` by the carboxyl owning `acid_oxygen`."""
    carbon = mol.GetAtomWithIdx(acid_oxygen).GetNeighbors()[0].GetIdx()
    if mol.GetBondBetweenAtoms(carbon, nucleophile) is not None:
        return None
    edit = Chem.RWMol(mol)
    edit.AddBond(carbon, nucleophile, Chem.BondType.SINGLE)
    atom = edit.GetAtomWithIdx(nucleophile)
    atom.SetNoImplicit(False)
    atom.SetNumExplicitHs(0)
    edit.RemoveAtom(acid_oxygen)
    product = edit.GetMol()
    try:
        Chem.SanitizeMol(product)
    except Chem.rdchem.MolSanitizeException:
        return None
    if any(len(ring) < 5 for ring in Chem.GetSymmSSSR(product)):
        return None
    return product


def lactone(mol: Chem.Mol) -> Chem.Mol | None:
    """C-terminal ester to the marked hydroxyl; None when it would close a <5-membered ring."""
    atoms = {atom.GetAtomMapNum(): atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum()}
    return condense(mol, atoms[2], atoms[1])


def species(cap: str, residues: tuple[str, ...], lactonize: bool) -> str | None:
    mol = assemble(cap, residues)
    if lactonize:
        mol = lactone(mol)
    return None if mol is None else canonical(mol)


ACID = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
NUCLEOPHILE = Chem.MolFromSmarts("[O,N;!H0;!$([OX2H1][CX3]=O)]")


def hydrolyses(mol: Chem.Mol) -> set[str]:
    """Add water across each in-ring amide, hydrazide or ester bond."""
    out = set()
    for carbon, _, hetero in mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3;R](=O)@[N,O;R]")):
        edit = Chem.RWMol(mol)
        edit.RemoveBond(carbon, hetero)
        oxygen = edit.AddAtom(Chem.Atom(8))
        edit.AddBond(carbon, oxygen, Chem.BondType.SINGLE)
        for index in (carbon, hetero):
            edit.GetAtomWithIdx(index).SetNoImplicit(False)
            edit.GetAtomWithIdx(index).SetNumExplicitHs(0)
        product = edit.GetMol()
        Chem.SanitizeMol(product)
        out.add(canonical(product))
    return out


def cyclizations(mol: Chem.Mol) -> set[str]:
    """Every dehydrative ring closure of every carboxylic acid onto any O-H or N-H."""
    out = set()
    for carbon, _, oxygen in mol.GetSubstructMatches(ACID):
        for (nucleophile,) in mol.GetSubstructMatches(NUCLEOPHILE):
            product = condense(mol, oxygen, nucleophile)
            if product is not None and len(Chem.GetSymmSSSR(product)) > len(Chem.GetSymmSSSR(mol)):
                out.add(canonical(product))
    return out


def ring_site_isomers(truth: dict) -> set[str]:
    """H1: same-formula ring-opening / ring-closing alternatives of the shared macrocycle."""
    linear = assemble(truth["cap"], truth["residues"])
    if truth["cap"] == "formyl":  # depsidomycin macrocycle: Thr side-chain ester to the C-terminal Piz
        macro = Chem.MolFromSmiles("CCC(C)C(C(=O)NC1C(OC(=O)C2CCCNN2C(=O)C(NC(=O)C3CCCNN3C(=O)"
                                   "C(NC(=O)C(NC1=O)CC(C)C)C(C)C)CC(C)C)C)NC=O")
    else:
        macro = lactone(linear)
    opened = {canonical(macro)} | hydrolyses(macro)
    closed = set()
    for smiles in opened:
        closed |= cyclizations(Chem.MolFromSmiles(smiles))
    if truth["lactone"]:
        return closed
    reopened = set()
    for smiles in closed:
        reopened |= hydrolyses(Chem.MolFromSmiles(smiles))
    return reopened | {s for s in opened if s != canonical(macro)}


def permutations(truth: dict) -> set[str]:
    """H2: every distinct residue order, keeping the N-terminal acyl and lactone residue."""
    out = {species(truth["cap"], order, truth["lactone"])
           for order in set(itertools.permutations(truth["residues"]))}
    return out - {None}


def substitutions(truth: dict, max_swaps: int = 2) -> set[str]:
    """H3: one or two same-formula residue/acyl isomer swaps at fixed sequence."""
    sites = [(-1, truth["cap"], CAP_SWAPS.get(truth["cap"], ()))]
    sites += [(i, name, SWAPS.get(name, ())) for i, name in enumerate(truth["residues"])]
    sites = [site for site in sites if site[2]]
    out = set()
    for count in range(1, max_swaps + 1):
        for chosen in itertools.combinations(sites, count):
            for replacement in itertools.product(*(site[2] for site in chosen)):
                cap, residues = truth["cap"], list(truth["residues"])
                for (index, _, _), name in zip(chosen, replacement):
                    if index < 0:
                        cap = name
                    else:
                        residues[index] = name
                out.add(species(cap, tuple(residues), truth["lactone"]))
    for (cap, first), alternatives in N_TERMINAL_SWAPS.items():
        if (truth["cap"], truth["residues"][0]) == (cap, first):
            for new_cap, new_first in alternatives:
                residues = (new_first, *truth["residues"][1:])
                out.add(species(new_cap, residues, truth["lactone"]))
    return out - {None}


def pubchem_formula_pool(formula: str) -> list[str]:
    with urllib.request.urlopen(PUBCHEM.format(formula), timeout=120) as response:
        rows = json.load(response)["PropertyTable"]["Properties"]
    smiles = (row.get("IsomericSMILES") or row["SMILES"] for row in rows)
    return sorted({canonical(value) for value in smiles if "." not in value})  # single components only


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    truths = {name: species(spec["cap"], spec["residues"], spec["lactone"]) for name, spec in SPECIES.items()}
    expected = {"caveamide_a": "C42H66N8O12", "caveamide_B": "C42H64N8O11", "depsidomycin": "C38H67N9O10"}
    pools, formulas = {}, {}
    for name, spec in SPECIES.items():
        truth = truths[name]
        formula = CalcMolFormula(Chem.MolFromSmiles(truth))
        if formula != expected[name]:
            raise ValueError(f"{name}: assembled formula {formula} != {expected[name]}")
        families = {"H1_ring_site": ring_site_isomers(spec), "H2_sequence": permutations(spec),
                    "H3_residue_isomer": substitutions(spec), "H4_pubchem_formula": set(pubchem_formula_pool(formula))}
        for family, members in families.items():
            bad = [s for s in members if CalcMolFormula(Chem.MolFromSmiles(s)) != formula]
            if bad:
                raise ValueError(f"{name} {family}: {len(bad)} members change the formula, e.g. {bad[0]}")
            members.discard(truth)
        families["X_cross_species"] = {s for other, s in truths.items() if other != name}
        pools[name] = {family: sorted(members) for family, members in families.items()}
        formulas[name] = formula
    record = {"truth": truths, "formula": formulas, "pools": pools,
              "sizes": {name: {family: len(members) for family, members in families.items()}
                        for name, families in pools.items()}}
    (args.out / "candidates.json").write_text(json.dumps(record, indent=1))
    molecules = set(truths.values()) | {s for families in pools.values() for members in families.values() for s in members}
    build_graph_store(sorted(molecules), args.out / "graphs", max_atoms=160)
    print(json.dumps(record["sizes"], indent=1))


if __name__ == "__main__":
    main()
