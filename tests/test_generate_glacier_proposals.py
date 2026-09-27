"""Critical atom-index and union semantics of the frozen GLACIER export."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem


_PATH = Path(__file__).resolve().parents[1] / "scripts/generate_glacier_proposals.py"
_SPEC = importlib.util.spec_from_file_location("generate_glacier_proposals", _PATH)
export = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(export)


@pytest.mark.parametrize("smiles", [
    "OC(C)C(=O)O", "N[C@@H](C)C(=O)O", "[13CH3]OC", "c1ncccc1O",
    "F/C=C/F", "F/C=C\\F", "N[C@H](C)[C@@H](O)C(=O)O", "[Na+].O=C([O-])C",
    r"C#C.C/C=C\C(=C/C)C(/C=C(C(C)=O)/C(C)=C/c1ccc(C)cc1C)=N\C.CCCN1CCCC1",
])
def test_canonical_mapping_preserves_atom_and_bond_identity(smiles):
    canonical, mapping = export.canonical_atom_mapping(smiles)
    source, target = Chem.MolFromSmiles(smiles), Chem.MolFromSmiles(canonical)
    assert sorted(mapping.tolist()) == list(range(source.GetNumAtoms()))
    for i, j in enumerate(mapping):
        a, b = target.GetAtomWithIdx(i), source.GetAtomWithIdx(int(j))
        assert (a.GetAtomicNum(), a.GetIsotope(), a.GetFormalCharge()) == (
            b.GetAtomicNum(), b.GetIsotope(), b.GetFormalCharge(),
        )
    for bond in target.GetBonds():
        mapped = source.GetBondBetweenAtoms(int(mapping[bond.GetBeginAtomIdx()]),
                                          int(mapping[bond.GetEndAtomIdx()]))
        assert mapped.GetBondType() == bond.GetBondType()
    # Atom order must be the writer's actual serialization order, including
    # disconnected-component reordering and stereochemistry normalization.
    writer = Chem.RemoveHs(source)
    assert Chem.MolToSmiles(writer) == canonical
    assert mapping.tolist() == export.json.loads(writer.GetProp("_smilesAtomOutputOrder"))


def test_merge_shifts_and_energies_uses_maximum_and_original_indices():
    merged = {}
    mapping = np.array([2, 0, 1], np.int16)
    export.merge_predictions(merged, np.array([[1, 1, 0], [1, 1, 0], [0, 1, 1]]),
                             np.array([.2, .7, .4]), mapping)
    export.merge_predictions(merged, np.array([[1, 1, 0], [0, 1, 1]]),
                             np.array([.5, .8]), mapping)
    assert merged == pytest.approx({(0, 2): .7, (0, 1): .8})
    packed = export.pack_proposals([(9, merged), (12, {})])
    np.testing.assert_array_equal(packed["graph_ids"], [9, 12])
    np.testing.assert_array_equal(packed["frag_off"], [0, 2, 2])
    np.testing.assert_array_equal(packed["mem_off"], [0, 2, 4])
    np.testing.assert_array_equal(packed["atoms"], [0, 1, 0, 2])
    np.testing.assert_allclose(packed["inten"], [.8, .7])


def test_fragment_padding_cannot_become_a_real_atom():
    with pytest.raises(ValueError, match="outside"):
        export.merge_predictions({}, np.array([[1, 0, 1]]), np.array([.2]),
                                 np.array([1, 0], np.int16))


@pytest.mark.parametrize("smiles", [
    "C1=CC2=CC(=C1)C21OCCOCCOCCOCCCOCCOCCOCCOCC12c1cccc2c1",
    "CCC(C)CNC1c2ccc(-c3ccccc3)c1c2-c1ncn[nH]1",
    "NCC1=CC2=CC(=C1)C21OCCOCCOCCOCCCOCCOCCOCCOCC12c1cc(CN)cc2c1",
])
def test_aromatic_kekule_normalization_keeps_the_writer_atom_mapping(smiles):
    changes = {}
    canonical, mapping = export.canonical_atom_mapping(smiles, annotation_changes=changes)
    original, normalized = Chem.MolFromSmiles(smiles), Chem.MolFromSmiles(canonical)
    assert sorted(mapping.tolist()) == list(range(original.GetNumAtoms()))
    for bond in normalized.GetBonds():
        assert original.GetBondBetweenAtoms(int(mapping[bond.GetBeginAtomIdx()]),
                                           int(mapping[bond.GetEndAtomIdx()])) is not None
    writer = Chem.RemoveHs(original)
    Chem.MolToSmiles(writer)
    assert mapping.tolist() == export.json.loads(writer.GetProp("_smilesAtomOutputOrder"))
