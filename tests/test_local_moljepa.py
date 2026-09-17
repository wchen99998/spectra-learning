import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'results/dgt_moljepa_atom_latest/runtime')]

from dgt_chemistry import features
from dgt_data import pack_graphs
from local_moljepa import Model, MolecularEncoder, SpectrumEncoder


def test_atom_permutation_padding_and_directed_bonds():
    torch.set_num_threads(1)
    torch.manual_seed(3)
    graphs = [features('CC(=O)Oc1ccccc1C(=O)O'), features('[Na+].[Cl-]')]
    memory = torch.randn(2, 13, 512)
    model = MolecularEncoder(width=32, context_width=16, heads=4, dropout=0).eval()
    with torch.no_grad():
        for name, value in model.named_parameters():
            if name.endswith('gate'):
                value.fill_(.2)
    batch = pack_graphs(graphs, memory, 'cpu')
    original = model(batch)
    x, ei, ea, af = graphs[0]
    permutation = np.random.default_rng(4).permutation(len(x))
    inverse = np.argsort(permutation)
    changed = [(x[permutation], inverse[ei], ea, af[permutation]), graphs[1]]
    permuted = model(pack_graphs(changed, memory, 'cpu'))
    torch.testing.assert_close(permuted[0, :len(x)], original[0, permutation], atol=2e-6, rtol=2e-5)
    alone = model(pack_graphs([graphs[1]], memory[1:], 'cpu'))
    torch.testing.assert_close(alone[0], original[1, :2], atol=2e-6, rtol=2e-5)
    assert not original[1, 2:].count_nonzero()
    torch.testing.assert_close(original[batch.atom_valid].norm(dim=-1), torch.ones(batch.atom_valid.sum()))
    reversed_graph = (x, ei[:, ::-1].copy(), ea.reshape(-1, 2, 57)[:, ::-1].reshape(-1, 57).copy(), af)
    reversed_output = model(pack_graphs([reversed_graph, graphs[1]], memory, 'cpu'))
    torch.testing.assert_close(reversed_output, original, atol=2e-6, rtol=2e-5)
    # Reversing descriptors without reversing their endpoints changes the messages.
    wrong_graph = (x, ei, ea.reshape(-1, 2, 57)[:, ::-1].reshape(-1, 57).copy(), af)
    wrong_output = model(pack_graphs([wrong_graph, graphs[1]], memory, 'cpu'))
    assert (wrong_output - original).abs().max() > 1e-5


def test_zero_gates_preserve_local_outputs_and_learn_to_open():
    torch.set_num_threads(1)
    torch.manual_seed(9)
    model = Model(width=32, context_width=16, heads=4, dropout=0).train()
    graph = pack_graphs([features('CC(=O)O'), features('c1ccccc1O')], torch.randn(2, 13, 512), 'cpu')
    x = torch.randn(2, 128, 2050)
    valid = torch.ones(2, 128, dtype=torch.bool)
    local_molecule = model.molecule.local_features(graph)
    local_spectrum = model.spectrum.local_features(x, valid)
    atoms, peaks = model.molecule(graph), model.spectrum(x, valid)
    torch.testing.assert_close(atoms, F.normalize(local_molecule, dim=-1), atol=0, rtol=0)
    torch.testing.assert_close(peaks, F.normalize(local_spectrum, dim=-1), atol=0, rtol=0)
    # Neither pretrained context nor extra chemistry can alter the base at initialization.
    perturbed = copy.deepcopy(graph)
    perturbed.moljepa = 100 * torch.randn_like(graph.moljepa)
    perturbed.atoms[..., 49:127] = torch.randn_like(perturbed.atoms[..., 49:127])
    perturbed.atoms[..., 127:] = 123
    perturbed.bonds[..., 8:] = torch.randn_like(perturbed.bonds[..., 8:])
    torch.testing.assert_close(model.molecule(perturbed), atoms, atol=0, rtol=0)
    atom_target, peak_target = torch.randn_like(atoms), torch.randn_like(peaks)
    ((atoms * atom_target).sum() + (peaks * peak_target).sum()).backward()
    gates = [value for name, value in model.named_parameters() if name.endswith('gate')]
    assert len(gates) == 8
    for gate in gates:
        assert gate.item() == 0 and torch.isfinite(gate.grad) and gate.grad.abs() > 0
    assert model.molecule.node_enc.weight.grad.abs().sum() > 0
    assert model.spectrum.token_projection[1].weight.grad.abs().sum() > 0
    branch_weights = [model.molecule.extra_node_enc.weight, model.molecule.extra_edge_enc.weight,
                      model.molecule.context[0].attention.in_proj_weight,
                      model.molecule.moljepa_context[0].attention.in_proj_weight,
                      model.spectrum.context[0].attention.in_proj_weight]
    for weight in branch_weights:
        assert weight.grad.count_nonzero() == 0
    with torch.no_grad():
        for gate in gates:
            gate.fill_(.1)
    model.zero_grad(set_to_none=True)
    ((model.molecule(graph) * atom_target).sum() + (model.spectrum(x, valid) * peak_target).sum()).backward()
    for weight in branch_weights:
        assert torch.isfinite(weight.grad).all() and weight.grad.abs().sum() > 0
    assert model.molecule.environments[0].weight.grad.abs().sum() > 0


def test_spectrum_raw_measurements_cls_and_padding():
    torch.set_num_threads(1)
    torch.manual_seed(12)
    model = SpectrumEncoder(width=32, context_width=16, heads=4, dropout=0).eval()
    x = torch.randn(2, 128, 2050)
    valid = torch.ones(2, 128, dtype=torch.bool)
    valid[:, 5:127] = False
    valid[:, 127] = False  # CLS remains on even if an external mask omits it.
    original = model(x, valid)
    assert original[:, 5:127].count_nonzero() == 0
    torch.testing.assert_close(original[:, -1].norm(dim=-1), torch.ones(2))
    for column in (2048, 2049):
        changed = x.clone()
        changed[:, 0, column] += 2
        assert (model(changed, valid)[:, 0] - original[:, 0]).abs().max() > 1e-3
    changed = x.clone()
    changed[:, -1, 2048:] = 1000
    torch.testing.assert_close(model(changed, valid), original, atol=0, rtol=0)
    with torch.no_grad():
        for block in model.context:
            block.gate.fill_(.3)
    active = model(x, valid)
    changed[:, 5:127] = 1e6
    torch.testing.assert_close(model(changed, valid), active, atol=0, rtol=0)
    permutation = torch.cat((torch.tensor([4, 2, 0, 1, 3]), torch.arange(5, 128)))
    permuted = model(x[:, permutation], valid[:, permutation])
    torch.testing.assert_close(permuted, active[:, permutation], atol=2e-6, rtol=2e-5)


def test_batchnorm_excludes_padding_and_global_head_ignores_raw_cls_features():
    torch.set_num_threads(1)
    torch.manual_seed(16)
    model = Model(width=32, context_width=16, heads=4, dropout=0)
    graph = pack_graphs([features('CCO')], torch.randn(1, 13, 512), 'cpu')
    padded = copy.deepcopy(graph)
    padded.atoms = F.pad(graph.atoms, (0, 0, 0, 5))
    padded.atom_valid = F.pad(graph.atom_valid, (0, 5))
    padded.fractions = F.pad(graph.fractions, (0, 0, 0, 5), value=-1)
    second = copy.deepcopy(model.molecule)
    original = model.molecule(graph)
    expanded = second(padded)
    torch.testing.assert_close(original, expanded[:, :3], atol=0, rtol=0)
    assert expanded[:, 3:].count_nonzero() == 0
    for a, b in zip(model.molecule.norms, second.norms):
        torch.testing.assert_close(a.running_mean, b.running_mean, atol=0, rtol=0)
        torch.testing.assert_close(a.running_var, b.running_var, atol=0, rtol=0)
    cls = torch.randn(3, 2050)
    changed = cls.clone()
    changed[:, -2:] += 1000
    torch.testing.assert_close(model.global_head(cls), model.global_head(changed), atol=0, rtol=0)


def test_context_norm_bound_even_with_saturated_gates_and_large_weights():
    torch.set_num_threads(1)
    torch.manual_seed(20)
    model = Model(width=32, context_width=16, heads=4, dropout=0).eval()
    graph = pack_graphs([features('CC(=O)O')], torch.randn(1, 13, 512), 'cpu')
    x = torch.randn(1, 128, 2050)
    valid = torch.ones(1, 128, dtype=torch.bool)
    branches = [*model.spectrum.context, *model.molecule.context, *model.molecule.moljepa_context]
    with torch.no_grad():
        for branch in branches:
            branch.gate.fill_(100)
            branch.output.weight.mul_(1000)
            branch.output.bias.mul_(1000)
    base = model.spectrum.local_features(x, valid)
    norm = base.norm(dim=-1, keepdim=True)
    mixed = base
    for branch in model.spectrum.context:
        mixed = branch(mixed, valid, norm)
    assert torch.all((mixed - base).norm(dim=-1) <= .500001 * norm.squeeze(-1))
    base = model.molecule.local_features(graph)
    norm = base.norm(dim=-1, keepdim=True)
    mixed = base
    for attention, cross in zip(model.molecule.context, model.molecule.moljepa_context):
        mixed = attention(mixed, graph.atom_valid, norm)
        mixed = cross(mixed, graph.atom_valid, graph.moljepa, norm)
    assert torch.all((mixed - base).norm(dim=-1) <= .500001 * norm.squeeze(-1))


@pytest.mark.parametrize('bf16', [False, True])
def test_initial_local_gine_matches_original_flare(bf16):
    from flare_mol import MolGNN

    torch.set_num_threads(1)
    torch.manual_seed(27)
    model = MolecularEncoder(width=32, context_width=16, heads=4, dropout=0).eval()
    reference = MolGNN(hidden=32, out_dim=32, n_layers=3, dropout=0, feature_set='basic').eval()
    state = model.state_dict()
    reference.load_state_dict({name: state[name] for name in reference.state_dict()})
    graph = features('CC(=O)Oc1ccccc1C(=O)O')
    x, endpoints, edges, _ = graph
    directed = np.stack((endpoints, endpoints[:, ::-1]), axis=1).reshape(-1, 2).T
    batch = pack_graphs([graph], torch.randn(1, 13, 512), 'cpu')
    captured = []
    handle = reference.out.register_forward_hook(lambda _module, _args, output: captured.append(output))
    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=bf16):
        # Compare the pre-normalized local path too, so FP32 normalization cannot
        # conceal a change in the original BF16 GINE residual arithmetic.
        expected = reference(torch.from_numpy(x[:, :49]), torch.from_numpy(directed.astype(np.int64)), torch.from_numpy(edges[:, :8]))
        actual_local = model.local_features(batch)[0]
        actual = model(batch)[0]
    handle.remove()
    torch.testing.assert_close(actual_local, captured[0], atol=0, rtol=0)
    torch.testing.assert_close(actual, expected.float(), atol=.0015 if bf16 else 0, rtol=.006 if bf16 else 0)
