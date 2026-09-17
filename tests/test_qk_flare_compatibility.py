"""Check the requested bilinear scoring contract, initialization, and gradients."""

import importlib.util
from pathlib import Path

import torch


SPEC = importlib.util.spec_from_file_location(
    "qk_flare", Path(__file__).resolve().parents[1] / "scripts/qk_flare_compatibility.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
QueryKeyCompatibility = MODULE.QueryKeyCompatibility


def test_identity_preserves_embeddings_and_random_state():
    torch.manual_seed(17)
    state = torch.get_rng_state().clone()
    model = QueryKeyCompatibility(5)
    assert torch.equal(state, torch.get_rng_state())
    peaks = torch.randn(2, 4, 5)
    atoms = torch.randn(2, 3, 6, 5)
    queries, keys = model(peaks, atoms)
    assert torch.equal(queries, peaks)
    assert torch.equal(keys, atoms)


def test_bilinear_orientation_asymmetry_and_no_renormalization():
    model = QueryKeyCompatibility(2)
    with torch.no_grad():
        model.query_weight.copy_(torch.tensor([[1., 2.], [0., 1.]]))
        model.key_weight.copy_(torch.tensor([[2., 0.], [0., 3.]]))
    vectors = torch.eye(2)
    queries, keys = model(vectors, vectors)
    scores = queries @ keys.T
    assert torch.equal(scores, torch.tensor([[2., 6.], [0., 3.]]))
    assert not torch.equal(scores, scores.T)
    assert scores.max() > 1  # No cosine renormalization after the projections.


def test_both_maps_train_and_survive_checkpoint_round_trip(tmp_path):
    model = QueryKeyCompatibility(2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    peaks = torch.tensor([[.2, .7], [-.6, .8]])
    atoms = torch.tensor([[.9, -.1], [.3, .5]])
    queries, keys = model(peaks, atoms)
    loss = torch.nn.functional.cross_entropy(queries @ keys.T / .03, torch.arange(2))
    loss.backward()
    for parameter in model.parameters():
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    optimizer.step()
    for parameter in model.parameters():
        assert not torch.equal(parameter, torch.eye(2))
    path = tmp_path / "compatibility.pt"
    torch.save(model.state_dict(), path)
    restored = QueryKeyCompatibility(2)
    restored.load_state_dict(torch.load(path, weights_only=True))
    for actual, expected in zip(restored(peaks, atoms), model(peaks, atoms), strict=True):
        assert torch.equal(actual, expected)
