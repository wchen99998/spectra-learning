"""Numerical parity with upstream flare_frag.py's PyTorch fragment equations.

The reference below transcribes its modules using dense membership matrices in
place of index_add; both perform the same mean and boundary-mean pooling.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from spectra_learning.probes.massspec.glacier_fragments_jax import FragmentPooler


def _jax(x):
    return jnp.asarray(x.detach().numpy())


def _close(actual, expected, atol=3e-6, rtol=3e-5):
    if isinstance(expected, torch.Tensor):
        expected = expected.detach().numpy()
    np.testing.assert_allclose(np.asarray(actual), expected, atol=atol, rtol=rtol)


class _TorchPooler(nn.Module):
    def __init__(self, use_intensity):
        super().__init__()
        self.cuts_emb = nn.Embedding(8, 3)
        self.kind_emb = nn.Embedding(4, 3)
        self.use_intensity = use_intensity
        self.mlp = nn.Sequential(
            nn.LayerNorm(18), nn.Linear(18, 7), nn.GELU(), nn.Dropout(0.0), nn.Linear(7, 6),
        )
        self.gate = nn.Linear(18 + (2 if use_intensity else 0), 1)

    def forward(self, H, member, boundary, mask, cuts, kind, mass, logn, intensity):
        features = torch.cat([
            torch.einsum("gfa,gah->gfh", member, H),
            torch.einsum("gfa,gah->gfh", boundary, H),
            self.cuts_emb(cuts), self.kind_emb(kind), mass[..., None], logn[..., None],
        ], dim=-1)
        z = F.normalize(self.mlp(features), dim=-1)
        gate_features = features
        if self.use_intensity:
            intensity = (intensity * mask).clamp(min=0)
            gate_features = torch.cat([
                features, intensity[..., None], torch.log1p(intensity)[..., None],
            ], dim=-1)
        return z, self.gate(gate_features).squeeze(-1)


def test_fragment_pooling_embeddings_gate_and_gradients_match_torch():
    use_intensity = True
    torch.manual_seed(31)
    reference = _TorchPooler(use_intensity)
    H = torch.randn(2, 5, 5, requires_grad=True)
    member = torch.tensor([
        [[0.2] * 5, [0.5, 0.5, 0, 0, 0], [0, 0, 1 / 3, 1 / 3, 1 / 3]],
        [[0.2] * 5, [0, 0, 0, 0.5, 0.5], [0] * 5],
    ])
    boundary = torch.zeros_like(member)
    boundary[0, 1, 1] = boundary[0, 2, 2] = boundary[1, 1, 3] = 1
    mask = torch.tensor([[True, True, True], [True, True, False]])
    cuts = torch.tensor([[0, 1, 1], [0, 1, 0]])
    kind = torch.tensor([[0, 1, 2], [0, 1, 0]])
    mass = torch.tensor([[1.0, 0.3, 0.7], [1.0, 0.4, 0.0]])
    logn = torch.log(torch.tensor([[5, 2, 3], [5, 2, 1]], dtype=torch.float32))
    intensity = torch.tensor([[1.0, -2.0, 0.3], [0.1, 0.8, 0.0]])
    args = (member, boundary, mask, cuts, kind, mass, logn, intensity)
    model = FragmentPooler(out_dim=6, hidden=7, emb=3, dropout=0.0)
    jargs = tuple(map(_jax, args))
    variables = model.init(jax.random.key(1), _jax(H), *jargs)
    params = variables["params"]
    pairs = [("linear1", reference.mlp[1]), ("linear2", reference.mlp[4]), ("gate", reference.gate)]
    for name, layer in pairs:
        params[name]["kernel"] = _jax(layer.weight.T)
        params[name]["bias"] = _jax(layer.bias)
    for name in ("cuts_emb", "kind_emb"):
        params[name]["embedding"] = _jax(getattr(reference, name).weight)
    params["norm"]["scale"] = _jax(reference.mlp[0].weight)
    params["norm"]["bias"] = _jax(reference.mlp[0].bias)
    expected_z, expected_gate = reference(H, *args)
    zw = torch.randn_like(expected_z) * mask[..., None]
    gw = torch.randn_like(expected_gate) * mask
    ((expected_z * zw).sum() + (expected_gate * gw).sum()).backward()

    def loss(p, h):
        z, gate = model.apply({"params": p}, h, *jargs)
        result = (z * _jax(zw)).sum() + (jnp.where(_jax(mask), gate, 0.0) * _jax(gw)).sum()
        return result, (z, gate)

    (_, (z, gate)), (pg, hg) = jax.value_and_grad(loss, (0, 1), has_aux=True)(params, _jax(H))
    _close(z, expected_z * mask[..., None])
    _close(gate[_jax(mask)], expected_gate[mask])
    assert np.isneginf(np.asarray(gate)[~np.asarray(mask)]).all()
    _close(hg, H.grad)
    for name, layer in pairs:
        _close(pg[name]["kernel"], layer.weight.grad.T)
        _close(pg[name]["bias"], layer.bias.grad)
    for name in ("cuts_emb", "kind_emb"):
        _close(pg[name]["embedding"], getattr(reference, name).weight.grad)
    _close(pg["norm"]["scale"], reference.mlp[0].weight.grad)
    _close(pg["norm"]["bias"], reference.mlp[0].bias.grad)
    assert np.linalg.norm(np.asarray(hg)) > 1e-4
    if use_intensity:
        alternative = (*jargs[:-1], jnp.ones_like(jargs[-1]) * 3)
        different_z, different_gate = model.apply({"params": params}, _jax(H), *alternative)
        _close(different_z, z, atol=0, rtol=0)
        assert np.linalg.norm(np.asarray(different_gate[_jax(mask)] - gate[_jax(mask)])) > 1e-3


def test_pooler_empty_graphs_are_zero_and_have_finite_gradients():
    model = FragmentPooler(out_dim=6, hidden=7, emb=3, dropout=0.0)
    H = jnp.ones((2, 4, 5))
    means = jnp.zeros((2, 3, 4)).at[0, 0].set(0.25)
    mask = jnp.zeros((2, 3), dtype=bool).at[0, 0].set(True)
    zeros = jnp.zeros((2, 3))
    categories = zeros.astype(jnp.int32)
    args = (means, means * 0, mask, categories, categories, zeros, zeros, zeros)
    variables = model.init(jax.random.key(13), H, *args)

    def loss(params, h):
        z, gate = model.apply({"params": params}, h, *args)
        return z.sum() + jnp.where(mask, gate, 0).sum(), (z, gate)

    (_, (z, gate)), gradients = jax.value_and_grad(loss, (0, 1), has_aux=True)(variables["params"], H)
    _close(z[1], 0)
    assert np.isneginf(np.asarray(gate[1])).all()
    for grad in jax.tree.leaves(gradients):
        assert np.isfinite(np.asarray(grad)).all()
    _close(gradients[1][1], 0)
