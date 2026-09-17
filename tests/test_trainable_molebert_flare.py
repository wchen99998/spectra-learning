import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from molebert_atom_encoder import MoleBERTAtomEncoder, featurize_smiles
from trainable_molebert_flare import TrainableMoleBERT, MoleBERTGraphCache
from frozen_flare_adapter import AtomCompatibility
from cache_molebert_graphs import build_cache


def test_graph_cache_roundtrip_preserves_bonds_atom_order_and_disconnected_graphs(tmp_path):
    smiles = sorted(['CCO', '[Na+]', 'CC.CC', 'N[C@@H](C)C(=O)O', 'F/C=C/F'])
    metadata = build_cache(smiles, tmp_path, batch_size=2, workers=0)
    assert metadata['molecules'] == 5 and metadata['truncated_molecules'] == 0
    cache = MoleBERTGraphCache(tmp_path)
    assert isinstance(cache.nodes, np.memmap)
    assert cache.nodes.dtype == np.uint8
    for value in smiles:
        expected = featurize_smiles([value])
        x, edges, attrs = cache[value]
        for actual, reference in zip([x, edges, attrs], [expected.x, expected.edge_index, expected.edge_attr], strict=True):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        assert x.dtype == edges.dtype == attrs.dtype == torch.long
        x.zero_()
        torch.testing.assert_close(cache[value][0], expected.x)


def test_unfrozen_backbone_receives_gradients_updates_bn_and_recomputes_embeddings(tmp_path):
    torch.set_num_threads(2)
    checkpoint = tmp_path / 'pretrained.pt'
    frozen = MoleBERTAtomEncoder()
    torch.save(frozen.state_dict(), checkpoint)
    model = TrainableMoleBERT(checkpoint, out_dim=16)
    graph = featurize_smiles(['CCO', 'CC(=O)O', 'c1ccccc1'])
    model.eval()
    with torch.no_grad():
        expected = torch.nn.functional.normalize(model.out(model.norm(frozen(graph))), dim=-1)
        before = model(graph.x, graph.edge_index, graph.edge_attr)
    torch.testing.assert_close(before, expected, rtol=0, atol=0)
    assert all(p.requires_grad for p in model.parameters())
    means = [b.running_mean.clone() for b in model.backbone.batch_norms]
    model.train()
    out = model(graph.x, graph.edge_index, graph.edge_attr)
    target = torch.randn_like(out)
    loss = (out-target).square().mean()
    loss.backward()
    for layer in model.backbone.gnns:
        assert sum(p.grad.abs().sum() for p in layer.parameters() if p.grad is not None) > 0
    assert model.backbone.x_embedding1.weight.grad.abs().sum() > 0
    assert model.out.weight.grad.abs().sum() > 0
    assert all(not torch.equal(old,b.running_mean) for old,b in zip(means,model.backbone.batch_norms))
    old_weight = model.backbone.gnns[0].mlp[0].weight.detach().clone()
    torch.optim.SGD(model.parameters(), lr=.02).step()
    assert not torch.equal(old_weight, model.backbone.gnns[0].mlp[0].weight)
    model.eval()
    with torch.no_grad(): after = model(graph.x, graph.edge_index, graph.edge_attr)
    assert not torch.equal(after, before)
    restored = TrainableMoleBERT(checkpoint, 16).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored(graph.x,graph.edge_index,graph.edge_attr),after,rtol=0,atol=0)


def test_projection_initialization_and_rng_match_frozen_reference(tmp_path):
    checkpoint=tmp_path/'pretrained.pt'
    torch.save(MoleBERTAtomEncoder().state_dict(),checkpoint)
    torch.manual_seed(0)
    reference=AtomCompatibility(300,512)
    expected_rng=torch.random.get_rng_state()
    torch.manual_seed(0)
    model=TrainableMoleBERT(checkpoint)
    torch.testing.assert_close(model.out.weight,reference.out.weight,rtol=0,atol=0)
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    assert set(model.state_dict()) == {'out.weight'} | {'backbone.'+k for k in model.backbone.state_dict()}
