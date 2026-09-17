import pytest
import torch
from rdkit import Chem

from scripts.molclr_atom_encoder import (
    MolCLRAtomEncoder,
    featurize_molecules,
    featurize_smiles,
)


@pytest.fixture(scope="module")
def encoder():
    with torch.random.fork_rng():
        torch.manual_seed(41)
        return MolCLRAtomEncoder()


def test_batch_matches_individual_and_preserves_all_atoms(encoder):
    smiles = ["CCO", "[Na+]", "CC.CC", "N[C@@H](C)C(=O)O", "F/C=C/F", "C" * 151]
    together = encoder.encode_smiles(smiles)
    assert [len(item) for item in together] == [3, 1, 4, 6, 4, 151]
    for value, actual in zip(smiles, together):
        expected = encoder.encode_smiles([value])[0]
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-5)
        assert actual.shape[1] == 300
        assert torch.isfinite(actual).all()
    assert encoder.encode_smiles([]) == []


def test_atom_permutation_equivariance(encoder):
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    order = list(reversed(range(mol.GetNumAtoms())))
    changed = Chem.RenumberAtoms(mol, order)
    expected = encoder(featurize_molecules([mol]))[order]
    actual = encoder(featurize_molecules([changed]))
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-5)


def test_frozen_backbone_preserves_bn_and_allows_compatibility_training(encoder):
    previous = {name: value.clone() for name, value in encoder.named_buffers()}
    head = torch.nn.Linear(300, 1)
    outer = torch.nn.ModuleList([encoder, head]).train()
    assert outer.training and head.training
    assert not encoder.training
    assert not any(layer.training for layer in encoder.modules())
    assert not any(parameter.requires_grad for parameter in encoder.parameters())
    atoms = encoder.encode_smiles(["CCO"])[0]
    assert not atoms.requires_grad
    head(atoms).square().sum().backward()
    assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in encoder.parameters())
    for name, value in encoder.named_buffers():
        torch.testing.assert_close(value, previous[name], atol=0, rtol=0)


def test_checkpoint_load_is_strict_except_known_graph_head(encoder, tmp_path):
    checkpoint = dict(encoder.state_dict())
    checkpoint["feat_lin.weight"] = torch.zeros(512, 300)
    checkpoint["out_lin.0.bias"] = torch.zeros(512)
    path = tmp_path / "encoder.pt"
    torch.save(checkpoint, path)
    loaded = MolCLRAtomEncoder.from_checkpoint(path)
    torch.testing.assert_close(
        loaded.encode_smiles(["CCO"])[0], encoder.encode_smiles(["CCO"])[0], atol=0, rtol=0,
    )
    checkpoint["unexpected.weight"] = torch.zeros(1)
    torch.save(checkpoint, path)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        MolCLRAtomEncoder.from_checkpoint(path)
    del checkpoint["unexpected.weight"]
    del checkpoint["gnns.0.mlp.0.weight"]
    torch.save(checkpoint, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        MolCLRAtomEncoder.from_checkpoint(path)


@pytest.mark.parametrize("smiles", ["", "not a molecule", "*", None])
def test_invalid_and_unsupported_smiles_are_rejected(smiles):
    with pytest.raises(ValueError):
        featurize_smiles([smiles])


def test_unsupported_stereochemistry_and_bonds_are_rejected():
    mol = Chem.MolFromSmiles("CC")
    mol.GetAtomWithIdx(0).SetChiralTag(Chem.ChiralType.CHI_OTHER)
    with pytest.raises(ValueError, match="unsupported MolCLR atomic number/chirality"):
        featurize_molecules([mol])
    mol.GetAtomWithIdx(0).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    mol.GetBondWithIdx(0).SetBondType(Chem.BondType.DATIVE)
    with pytest.raises(ValueError, match="unsupported MolCLR bond type/direction"):
        featurize_molecules([mol])
    mol.GetBondWithIdx(0).SetBondType(Chem.BondType.SINGLE)
    mol.GetBondWithIdx(0).SetBondDir(Chem.BondDir.BEGINWEDGE)
    with pytest.raises(ValueError, match="unsupported MolCLR bond type/direction"):
        featurize_molecules([mol])
