"""Numerical parity with the PyTorch atom FLARE recipe at jepa-fps/55dfbdf."""

from __future__ import annotations

import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from spectra_learning.probes.massspec.atom_flare_jax import (
    MolGNN,
    PeakEncoder,
    info_nce,
    normalize,
)


def _array(tensor):
    return jnp.asarray(tensor.detach().numpy())


def _copy_linear(params, layer):
    params["kernel"] = _array(layer.weight.T)
    params["bias"] = _array(layer.bias)


def _assert_close(actual, expected, atol=3e-6, rtol=3e-5):
    if isinstance(expected, torch.Tensor):
        expected = expected.detach().numpy()
    np.testing.assert_allclose(np.asarray(actual), expected, atol=atol, rtol=rtol)


def test_import_has_no_torch_runtime_dependency():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import spectra_learning.probes.massspec.atom_flare_jax; "
         "assert 'torch' not in sys.modules"],
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_peak_encoder_forward_and_gradient_match_torch():
    torch.manual_seed(13)
    reference = nn.Sequential(nn.LayerNorm(9), nn.Linear(9, 7), nn.GELU(),
                              nn.Dropout(0.0), nn.Linear(7, 7))
    x = torch.randn(2, 4, 9, requires_grad=True)
    weights = torch.randn(2, 4, 7)
    model = PeakEncoder(out_dim=7, dropout=0.0)
    variables = model.init(jax.random.key(1), _array(x))
    params = variables["params"]
    params["norm"]["scale"] = _array(reference[0].weight)
    params["norm"]["bias"] = _array(reference[0].bias)
    _copy_linear(params["linear1"], reference[1])
    _copy_linear(params["linear2"], reference[4])

    expected = F.normalize(reference(x), dim=-1)
    (expected * weights).sum().backward()

    def loss(p, inputs):
        output = model.apply({"params": p}, inputs)
        return (output * _array(weights)).sum(), output

    (_, actual), (param_grad, x_grad) = jax.value_and_grad(loss, (0, 1), has_aux=True)(params, _array(x))
    _assert_close(actual, expected)
    _assert_close(x_grad, x.grad)
    for name, layer in (("linear1", reference[1]), ("linear2", reference[4])):
        _assert_close(param_grad[name]["kernel"], layer.weight.grad.T)
        _assert_close(param_grad[name]["bias"], layer.bias.grad)
    _assert_close(param_grad["norm"]["scale"], reference[0].weight.grad)
    _assert_close(param_grad["norm"]["bias"], reference[0].bias.grad)


class _TorchGNN(nn.Module):
    """The upstream architecture using the actual PyG GINEConv operator."""

    def __init__(self):
        super().__init__()
        pyg = pytest.importorskip("torch_geometric.nn")
        self.node_enc = nn.Linear(49, 7)
        self.edge_enc = nn.Linear(8, 7)
        self.convs = nn.ModuleList([
            pyg.GINEConv(nn.Sequential(nn.Linear(7, 7), nn.ReLU(), nn.Linear(7, 7)), edge_dim=7)
            for _ in range(3)
        ])
        self.norms = nn.ModuleList([nn.BatchNorm1d(7) for _ in range(3)])
        self.out = nn.Linear(7, 5)

    def forward(self, x, edge_index, edge_attr):
        h, e = self.node_enc(x), self.edge_enc(edge_attr)
        for conv, norm in zip(self.convs, self.norms):
            h = h + F.relu(norm(conv(h, edge_index, e)))
        return F.normalize(self.out(h), dim=-1)


def test_gine_padding_forward_gradients_and_running_statistics_match_torch():
    torch.manual_seed(20)
    reference = _TorchGNN()
    nodes = torch.randn(7, 49, requires_grad=True)
    edges = torch.randn(8, 8, requires_grad=True)
    edge_index = torch.tensor([[0, 1, 1, 2, 3, 4, 4, 5], [1, 0, 2, 1, 4, 3, 5, 4]])
    weights = torch.randn(7, 5)
    node_mask = jnp.arange(11) < 7
    edge_mask = jnp.arange(13) < 8
    padded_nodes = jnp.pad(_array(nodes), ((0, 4), (0, 0)), constant_values=123)
    padded_edges = jnp.pad(_array(edges), ((0, 5), (0, 0)), constant_values=123)
    # Padding edges point into real nodes to detect any unmasked messages.
    senders = jnp.pad(_array(edge_index[0]), (0, 5), constant_values=0)
    receivers = jnp.pad(_array(edge_index[1]), (0, 5), constant_values=1)
    model = MolGNN(hidden=7, out_dim=5, dropout=0)
    args = (padded_nodes, padded_edges, senders, receivers, node_mask, edge_mask)
    variables = model.init(jax.random.key(2), *args)
    params = variables["params"]
    layer_pairs = [("node_enc", reference.node_enc), ("edge_enc", reference.edge_enc),
                   ("out", reference.out)]
    for i, conv in enumerate(reference.convs):
        layer_pairs.extend([(f"edge_{i}", conv.lin), (f"mlp_{i}_0", conv.nn[0]),
                            (f"mlp_{i}_2", conv.nn[2])])
    for name, layer in layer_pairs:
        _copy_linear(params[name], layer)

    expected = reference(nodes, edge_index, edges)
    (expected * weights).sum().backward()

    def loss(p, x, e):
        output, updates = model.apply(
            {"params": p, "batch_stats": variables["batch_stats"]},
            x, e, senders, receivers, node_mask, edge_mask, train=True, mutable=["batch_stats"],
        )
        return (output[:7] * _array(weights)).sum(), (output, updates)

    (_, (actual, updates)), grads = jax.value_and_grad(loss, (0, 1, 2), has_aux=True)(
        params, padded_nodes, padded_edges
    )
    _assert_close(actual[:7], expected, atol=8e-6)
    _assert_close(actual[7:], 0)
    _assert_close(grads[1][:7], nodes.grad, atol=2e-5, rtol=3e-4)
    _assert_close(grads[2][:8], edges.grad, atol=2e-5, rtol=3e-4)
    _assert_close(grads[1][7:], 0)
    _assert_close(grads[2][8:], 0)
    for name, layer in layer_pairs:
        _assert_close(grads[0][name]["kernel"], layer.weight.grad.T, atol=3e-5, rtol=3e-4)
        _assert_close(grads[0][name]["bias"], layer.bias.grad, atol=3e-5, rtol=3e-4)
    for i, bn in enumerate(reference.norms):
        _assert_close(grads[0][f"norm_{i}"]["scale"], bn.weight.grad, atol=1e-5)
        _assert_close(grads[0][f"norm_{i}"]["bias"], bn.bias.grad, atol=1e-5)
        _assert_close(updates["batch_stats"][f"norm_{i}"]["mean"], bn.running_mean)
        _assert_close(updates["batch_stats"][f"norm_{i}"]["var"], bn.running_var)

    # A second batch proves the running-state recurrence, then use it for eval.
    reference(nodes * 0.7, edge_index, edges * 0.9)
    _, second_updates = model.apply(
        {"params": params, **updates}, padded_nodes * 0.7, padded_edges * 0.9,
        senders, receivers, node_mask, edge_mask, train=True, mutable=["batch_stats"],
    )
    for i, bn in enumerate(reference.norms):
        _assert_close(second_updates["batch_stats"][f"norm_{i}"]["mean"], bn.running_mean)
        _assert_close(second_updates["batch_stats"][f"norm_{i}"]["var"], bn.running_var)
    reference.eval()
    _assert_close(model.apply({"params": params, **second_updates}, *args)[:7],
                  reference(nodes, edge_index, edges), atol=8e-6)


def _torch_bonus(fp, afrac, amask, log_sigma, log_beta):
    dmin = torch.full(torch.broadcast_shapes(fp.shape, afrac.shape[:-1]), 1e9)
    for slot in range(afrac.shape[-1]):
        frac = afrac[..., slot]
        delta = torch.where(frac >= 0, (fp - frac).abs(), 1e9)
        dmin = torch.minimum(dmin, delta)
    return torch.where(amask, log_beta.exp() * torch.exp(-(dmin / log_sigma.exp()) ** 2), 0)


def _torch_maxsim(sim, pmask, amask):
    peaks = sim.masked_fill(~amask[..., None, :], -1e4).amax(-1)
    s2m = peaks.masked_fill(~pmask, 0).sum(-1) / pmask.sum(-1).clamp(min=1)
    atoms = sim.masked_fill(~pmask[..., :, None], -1e4).amax(-2)
    m2s = atoms.masked_fill(~amask, 0).sum(-1) / amask.sum(-1).clamp(min=1)
    return 0.5 * (s2m + m2s)


