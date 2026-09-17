import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from scripts.frozen_flare_adapter import AtomCompatibility, FrozenAtomCache


def write_cache(root, *, smiles=None, offsets=None, values=None):
    (root / "smiles.json").write_text(json.dumps(["CC", "COC"] if smiles is None else smiles))
    np.save(root / "offsets.npy", np.array([0, 2, 5], dtype=np.int64) if offsets is None else offsets)
    np.save(root / "atom_embeddings.float16.npy", (
        np.arange(20, dtype=np.float16).reshape(5, 4) if values is None else values
    ))


def test_cache_preserves_atom_order_counts_and_graph_batch_masks(tmp_path):
    write_cache(tmp_path)
    cache = FrozenAtomCache(tmp_path)
    assert isinstance(cache.values, np.memmap)
    assert isinstance(cache.offsets, np.memmap)
    assert len(cache) == 2 and cache.in_dim == 4
    assert "CC" in cache and "N" not in cache
    with pytest.raises(KeyError):
        cache["N"]

    graphs = [cache[smiles] for smiles in ["COC", "CC", "COC"]]
    for atoms, edges, edge_attributes in graphs:
        assert atoms.dtype == torch.float32 and not atoms.requires_grad
        assert edges.shape == (2, 0) and edges.dtype == torch.long
        assert edge_attributes.shape == (0, 1) and edge_attributes.dtype == torch.float32
    atoms = torch.nn.utils.rnn.pad_sequence([graph[0] for graph in graphs], batch_first=True)
    counts = torch.tensor([len(graph[0]) for graph in graphs])
    mask = torch.arange(atoms.shape[1])[None] < counts[:, None]
    assert mask.tolist() == [[True, True, True], [True, True, False], [True, True, True]]
    torch.testing.assert_close(atoms[0], torch.arange(8, 20, dtype=torch.float32).reshape(3, 4))
    torch.testing.assert_close(atoms[1, :2], torch.arange(8, dtype=torch.float32).reshape(2, 4))
    torch.testing.assert_close(atoms[0], atoms[2])
    graphs[0][0].zero_()
    assert cache["COC"][0][0, 0] == 8  # Callers cannot mutate the frozen store.


@pytest.mark.parametrize("overrides, message", [
    ({"smiles": ["COC", "CC"]}, "sorted"),
    ({"smiles": ["CC", "CC"]}, "duplicates"),
    ({"offsets": np.array([0, 5])}, "one more entry"),
    ({"offsets": np.array([0, 2, 5], dtype=np.int32)}, "int64"),
    ({"offsets": np.array([0, 2, 4])}, "end at the number"),
    ({"offsets": np.array([0, 0, 5])}, "between 1 and 128"),
    ({"values": np.zeros((5, 0), dtype=np.float16)}, "positive width"),
    ({"values": np.zeros((5, 4), dtype=np.float32)}, "float16 matrix"),
    ({"offsets": np.array([0, 129, 130]), "values": np.zeros((130, 4), dtype=np.float16)}, "truncation is forbidden"),
])
def test_cache_rejects_inconsistent_external_metadata(tmp_path, overrides, message):
    write_cache(tmp_path, **overrides)
    with pytest.raises(ValueError, match=message):
        FrozenAtomCache(tmp_path)


def test_atom_map_is_pointwise_normalized_and_deterministic_while_training():
    torch.manual_seed(3)
    atom = AtomCompatibility(7, 5).train()
    x = torch.randn(4, 7)
    projected_atoms = atom(x)
    torch.testing.assert_close(atom(x), projected_atoms, rtol=0, atol=0)
    atom.eval()
    torch.testing.assert_close(atom(x), projected_atoms, rtol=0, atol=0)
    order = torch.tensor([3, 0, 2, 1])
    torch.testing.assert_close(atom(x[order]), projected_atoms[order])
    torch.testing.assert_close(projected_atoms.norm(dim=-1), torch.ones(4))
    # No learned bias creates a representation for an absent/padded input.
    assert torch.count_nonzero(atom(torch.zeros_like(x))) == 0


def test_retrieval_optimizer_updates_atom_map_without_changing_cached_vectors(tmp_path):
    torch.manual_seed(5)
    values = np.random.default_rng(5).standard_normal((5, 4)).astype(np.float16)
    write_cache(tmp_path, values=values)
    cache = FrozenAtomCache(tmp_path)
    atom = AtomCompatibility(cache.in_dim, 5)
    assert set(dict(atom.named_parameters())) == {"out.weight"}
    assert atom.out.out_features == 5
    frozen_atoms, edges, edge_attr = cache["CC"]
    original_atoms = frozen_atoms.clone()
    parameters = list(atom.parameters())
    before = atom.out.weight.detach().clone()
    optimizer = torch.optim.SGD(parameters, lr=.1)
    molecules = atom(frozen_atoms, edges, edge_attr)
    scores = torch.randn(2, 5) @ molecules.T
    F.cross_entropy(scores, torch.arange(2)).backward()
    assert all(parameter.grad is not None and parameter.grad.norm() > 0 for parameter in parameters)
    optimizer.step()
    assert not torch.equal(before, atom.out.weight)
    assert frozen_atoms.grad is None and not frozen_atoms.requires_grad
    torch.testing.assert_close(frozen_atoms, original_atoms, rtol=0, atol=0)
    np.testing.assert_array_equal(cache.values, values)
