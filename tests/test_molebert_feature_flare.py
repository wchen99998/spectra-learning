import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from molebert_feature_flare import widen_atom_input
from initialize_molebert_feature_checkpoint import extend_checkpoint


def test_extra_atom_inputs_preserve_chemical_weights_and_rng_and_receive_gradients():
    torch.manual_seed(0)
    original = nn.Linear(49, 512)
    rng = torch.random.get_rng_state()
    expanded = widen_atom_input(original, 300)
    assert torch.equal(rng, torch.random.get_rng_state())
    torch.testing.assert_close(expanded.weight[:, :49], original.weight, rtol=0, atol=0)
    torch.testing.assert_close(expanded.bias, original.bias, rtol=0, atol=0)
    chemical = torch.randn(8, 49)
    zeros = torch.zeros(8, 300)
    torch.testing.assert_close(expanded(torch.cat((chemical, zeros), -1)), original(chemical), rtol=1e-5, atol=1e-6)
    embedding = torch.nn.functional.normalize(torch.randn(8, 300), dim=-1)
    prediction = expanded(torch.cat((chemical, embedding), -1))
    prediction.square().mean().backward()
    assert expanded.weight.grad[:, :49].norm() > 0
    assert expanded.weight.grad[:, 49:].norm() > 0
    before = expanded.weight[:, 49:].detach().clone()
    torch.optim.AdamW(expanded.parameters(), lr=1e-4).step()
    assert not torch.equal(before, expanded.weight[:, 49:])


def test_trained_checkpoint_starts_with_original_function_and_learns_extra_inputs():
    torch.manual_seed(17)
    original = nn.Linear(49, 512)
    checkpoint = {'gnn': {'node_enc.weight': original.weight.detach().clone(),
                          'node_enc.bias': original.bias.detach().clone(),
                          'norms.0.running_mean': torch.randn(512)},
                  'spectrum_encoder': {'trained': torch.randn(8)}, 'epoch': 40}
    converted = extend_checkpoint(checkpoint)
    assert checkpoint['gnn']['node_enc.weight'].shape == (512, 49)
    assert set(checkpoint) == set(converted)
    for key, value in checkpoint['gnn'].items():
        actual = converted['gnn'][key]
        torch.testing.assert_close(actual[:, :49] if key == 'node_enc.weight' else actual,
                                   value, rtol=0, atol=0)
    assert converted['spectrum_encoder'] is checkpoint['spectrum_encoder']
    expanded = nn.Linear(349, 512)
    expanded.load_state_dict({'weight': converted['gnn']['node_enc.weight'],
                              'bias': converted['gnn']['node_enc.bias']})
    assert torch.count_nonzero(expanded.weight[:, 49:]) == 0
    chemical = torch.randn(32, 49)
    extra = torch.nn.functional.normalize(torch.randn(32, 300), dim=-1)
    inputs = torch.cat((chemical, extra), dim=-1)
    torch.testing.assert_close(expanded(inputs), original(chemical), rtol=1e-5, atol=1e-6)
    prediction = expanded(inputs)
    prediction.square().mean().backward()
    assert expanded.weight.grad[:, :49].norm() > 0
    assert expanded.weight.grad[:, 49:].norm() > 0
    torch.optim.AdamW(expanded.parameters(), lr=3e-5).step()
    assert torch.count_nonzero(expanded.weight[:, 49:]) > 0
