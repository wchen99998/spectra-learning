import pytest
import torch
from rdkit import Chem

from scripts.molebert_atom_encoder import (
    MoleBERTAtomEncoder,
    featurize_molecules,
    featurize_smiles,
)


@pytest.fixture(scope="module")
def encoder():
    with torch.random.fork_rng():
        torch.manual_seed(49)
        return MoleBERTAtomEncoder()


def test_atom_tokens_and_stereo_features_follow_upstream_order():
    graph = featurize_smiles(["F/C=C/F", "N[C@@H](C)C(=O)O", "[Na+]"])
    assert graph.counts == (4, 6, 1)
    assert graph.x[:, 0].tolist() == [8, 5, 5, 8, 6, 5, 5, 5, 7, 7, 10]
    assert graph.x[5, 1].item() == 1
    assert graph.edge_index[:, :6].tolist() == [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]
    assert graph.edge_attr[:6].tolist() == [[0, 1], [0, 1], [1, 0], [1, 0], [0, 1], [0, 1]]


def test_batch_single_disconnected_and_no_truncation(encoder):
    smiles = ["CCO", "[Na+]", "CC.CC", "N[C@@H](C)C(=O)O", "C" * 151]
    together = encoder.encode_smiles(smiles)
    assert [len(atoms) for atoms in together] == [3, 1, 4, 6, 151]
    for value, atoms in zip(smiles, together):
        torch.testing.assert_close(atoms, encoder.encode_smiles([value])[0], atol=2e-4, rtol=2e-5)
        assert atoms.shape[1] == 300 and torch.isfinite(atoms).all()
    assert encoder.encode_smiles([]) == []


def test_atom_permutation_equivariance(encoder):
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    order = list(reversed(range(mol.GetNumAtoms())))
    actual = encoder(featurize_molecules([Chem.RenumberAtoms(mol, order)]))
    expected = encoder(featurize_molecules([mol]))[order]
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-5)


def test_frozen_weights_and_bn_allow_matcher_training(encoder):
    buffers = {key: value.clone() for key, value in encoder.named_buffers()}
    head = torch.nn.Linear(300, 1)
    torch.nn.ModuleList([encoder, head]).train()
    assert head.training and not any(module.training for module in encoder.modules())
    assert not any(parameter.requires_grad for parameter in encoder.parameters())
    output = encoder.encode_smiles(["CCO"])[0]
    assert not output.requires_grad
    head(output).square().sum().backward()
    assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in encoder.parameters())
    for key, value in encoder.named_buffers():
        torch.testing.assert_close(value, buffers[key], atol=0, rtol=0)


def test_checkpoint_is_strict_and_has_molebert_mask_vocabularies(encoder, tmp_path):
    assert encoder.x_embedding1.weight.shape == (120, 300)
    assert all(conv.edge_embedding1.weight.shape == (6, 300) for conv in encoder.gnns)
    checkpoint = dict(encoder.state_dict())
    path = tmp_path / "Mole-BERT.pth"
    torch.save(checkpoint, path)
    loaded = MoleBERTAtomEncoder.from_checkpoint(path)
    torch.testing.assert_close(loaded.encode_smiles(["CCO"])[0], encoder.encode_smiles(["CCO"])[0], atol=0, rtol=0)
    checkpoint["feat_lin.weight"] = torch.zeros(512, 300)
    torch.save(checkpoint, path)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        MoleBERTAtomEncoder.from_checkpoint(path)
    del checkpoint["feat_lin.weight"]
    checkpoint["x_embedding1.weight"] = torch.zeros(119, 300)
    torch.save(checkpoint, path)
    with pytest.raises(RuntimeError, match="size mismatch"):
        MoleBERTAtomEncoder.from_checkpoint(path)


def test_sum_messages_match_independent_dense_reference(encoder):
    graph = featurize_smiles(["CCO", "[Na+]"])
    h = encoder.x_embedding1(graph.x[:, 0]) + encoder.x_embedding2(graph.x[:, 1])
    conv = encoder.gnns[0]
    adjacency = h.new_zeros((len(h), len(h)))
    edge_sum = h.new_zeros(h.shape)
    for edge, attr in zip(graph.edge_index.T, graph.edge_attr):
        src, dst = edge.tolist()
        adjacency[dst, src] += 1
        edge_sum[dst] += conv.edge_embedding1.weight[attr[0]] + conv.edge_embedding2.weight[attr[1]]
    adjacency += torch.eye(len(h))
    edge_sum += conv.edge_embedding1.weight[4] + conv.edge_embedding2.weight[0]
    expected = conv.mlp(adjacency @ h + edge_sum)
    actual = conv(h, graph.edge_index, graph.edge_attr)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("smiles", ["", "not a molecule", "*", None])
def test_invalid_external_molecules_fail(smiles):
    with pytest.raises(ValueError):
        featurize_smiles([smiles])


def test_unsupported_chirality_and_bond_direction_fail():
    mol = Chem.MolFromSmiles("CC")
    mol.GetAtomWithIdx(0).SetChiralTag(Chem.ChiralType.CHI_OTHER)
    with pytest.raises(ValueError, match="unsupported Mole-BERT"):
        featurize_molecules([mol])
    mol.GetAtomWithIdx(0).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    mol.GetBondWithIdx(0).SetBondDir(Chem.BondDir.BEGINWEDGE)
    with pytest.raises(ValueError, match="unsupported Mole-BERT"):
        featurize_molecules([mol])
