import copy
import json

import numpy as np
import torch
import torch.nn.functional as F

from scripts.pair_flare_flat import (
    PairAgreement, RaggedPairTokens, chunked_alignment, chunked_query,
    load_pair_batch, packed_encoder,
)


class PointwiseReadout(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tokens = torch.nn.Sequential(torch.nn.LayerNorm(4), torch.nn.Linear(4, 6), torch.nn.GELU())
        self.coordinates = torch.nn.Linear(2, 6)
        self.global_projection = torch.nn.Linear(4, 6)

    def forward(self, tokens, mz, intensity, cls):
        local = F.normalize(self.tokens(tokens) + self.coordinates(torch.stack([mz, intensity], -1)), dim=-1)
        return local, self.global_projection(cls)


def dense_alignment(peaks, peak_mask, atoms, atom_mask, bonus=None):
    scores = torch.einsum("bpd,bcad->bcpa", peaks, atoms)
    if bonus is not None:
        scores = scores + bonus
    matches = scores.masked_fill(~atom_mask[:, :, None, :], -1e4).amax(-1)
    reverse = scores.masked_fill(~peak_mask[:, None, :, None], -1e4).amax(-2)
    return matches, (reverse * atom_mask).sum(-1) / atom_mask.sum(-1)


class Agreement(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.log_sigma = torch.nn.Parameter(torch.tensor(-1.2))
        self.log_beta = torch.nn.Parameter(torch.tensor(-.7))

    def bonus_batched(self, peaks, fractions, mask):
        distance = (peaks[:, None, :, None, None] - fractions[:, :, None]).abs().amin(-1)
        return self.log_beta.exp() * torch.exp(-(distance / self.log_sigma.exp()).square()) * mask[:, :, None]


def test_local_pair_values_keep_canonical_offsets_and_metadata(tmp_path):
    root = tmp_path / "canonical"
    root.mkdir()
    values = np.arange(20, dtype=np.float16).reshape(5, 4)
    local = tmp_path / "local.npy"
    np.save(local, values)
    np.save(root / "pair_offsets.npy", np.array([0, 4, 5]))
    (root / "embedding_metadata.json").write_text(json.dumps({"samples": 2, "maximum_pair_tokens": 4, "pair_dim": 4}))
    storage = RaggedPairTokens(root, local)
    assert storage.shape == (2, 4, 4)
    np.testing.assert_array_equal(storage.offsets, [0, 4, 5])
    np.testing.assert_array_equal(storage.values, values)


def test_packed_readout_matches_dense_values_and_gradients(tmp_path):
    torch.manual_seed(0)
    values = torch.randn(5, 4).numpy().astype(np.float16)
    np.save(tmp_path / "pair_values.float16.npy", values)
    np.save(tmp_path / "pair_offsets.npy", np.array([0, 4, 5]))
    (tmp_path / "embedding_metadata.json").write_text(json.dumps({"samples": 2, "maximum_pair_tokens": 4, "pair_dim": 4}))
    storage = RaggedPairTokens(tmp_path)
    cls = torch.randn(2, 4).numpy()
    mz = np.array([[.1, .1, .2, .2], [.3, 0, 0, 0]], np.float32)
    intensity = np.array([[.4, .4, .8, .8], [1, 0, 0, 0]], np.float32)
    rows = np.array([1, 0, 1])
    packed, x_mz, x_intensity, x_cls = load_pair_batch(storage, cls, mz, intensity, rows, 4, "cpu")
    assert packed.values.shape == (6, 4)
    torch.testing.assert_close(x_mz, torch.tensor([.3, .1, .1, .2, .2, .3]))
    torch.testing.assert_close(packed.values, torch.from_numpy(values[[4, 0, 1, 2, 3, 4]].astype(np.float32)))
    dense = torch.randn(3, 4, 4)
    dense[packed.owners, packed.positions] = packed.values
    mask = torch.zeros(3, 4, dtype=torch.bool)
    mask[packed.owners, packed.positions] = True
    reference = PointwiseReadout()
    actual = packed_encoder(PointwiseReadout)()
    actual.load_state_dict(reference.state_dict())
    expected_local, expected_cls = reference(dense, torch.from_numpy(mz[rows]), torch.from_numpy(intensity[rows]), x_cls)
    local, global_embedding = actual(packed, x_mz, x_intensity, x_cls)
    torch.testing.assert_close(local[mask], expected_local[mask])
    torch.testing.assert_close(global_embedding, expected_cls)
    assert torch.count_nonzero(local[~mask]) == 0
    objective = torch.randn_like(local)
    (local.mul(objective).sum() + global_embedding.square().sum()).backward()
    (expected_local[mask].mul(objective[mask]).sum() + expected_cls.square().sum()).backward()
    for first, second in zip(actual.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(first.grad, second.grad)


def test_candidate_chunks_preserve_global_evidence_scores_and_agreement_gradients():
    torch.manual_seed(4)
    peaks = torch.randn(2, 9, 4, requires_grad=True)
    atoms = torch.randn(2, 7, 5, 4, requires_grad=True)
    pm = torch.rand(2, 9) > .2
    am = torch.rand(2, 7, 5) > .2
    pm[:, 0] = True
    am[:, :, 0] = True
    fp = torch.rand(2, 9)
    af = torch.rand(2, 7, 5, 3)
    agreement = Agreement()
    direct_bonus = agreement.bonus_batched(fp, af, am)
    matches, means = dense_alignment(peaks, pm, atoms, am, direct_bonus)

    def loss(matches, means):
        probabilities = (matches / .1).softmax(dim=1)
        entropy = -(probabilities * probabilities.clamp(min=1e-20).log()).sum(1)
        information = (torch.log(torch.tensor(7.)) - entropy).clamp(min=0) + 1e-4
        weights = information * pm
        weights = weights / weights.sum(-1, keepdim=True)
        scores = .5 * ((matches * weights[:, None]).sum(-1) + means)
        return F.cross_entropy(scores / .03, torch.tensor([0, 3]))

    expected = loss(matches, means)
    expected_grad = torch.autograd.grad(expected, [peaks, atoms, *agreement.parameters()])
    chunk_agreement = copy.deepcopy(agreement)
    p = peaks.detach().clone().requires_grad_()
    a = atoms.detach().clone().requires_grad_()
    actual_matches, actual_means = chunked_alignment(dense_alignment, 2)(
        p, pm, a, am, PairAgreement(chunk_agreement, fp, af, am),
    )
    torch.testing.assert_close(actual_matches, matches)
    torch.testing.assert_close(actual_means, means)
    actual = loss(actual_matches, actual_means)
    actual_grad = torch.autograd.grad(actual, [p, a, *chunk_agreement.parameters()])
    torch.testing.assert_close(actual, expected)
    for first, second in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(first, second, atol=2e-5, rtol=2e-5)


def test_flat_token_order_and_candidate_chunks_leave_maxsim_unchanged():
    torch.manual_seed(3)
    peaks, atoms = torch.randn(9, 5), torch.randn(7, 4, 5)
    pm, am = torch.tensor([True] * 7 + [False] * 2), torch.ones(7, 4, dtype=torch.bool)
    am[2, -1] = False

    def query(p, mask, a, a_mask):
        matches, means = dense_alignment(p[None], mask[None], a[None], a_mask[None])
        return .5 * ((matches[0] * mask).sum(-1) / mask.sum() + means[0])

    expected = query(peaks, pm, atoms, am)
    perm = torch.randperm(len(peaks))
    actual = chunked_query(query, 2)(peaks[perm], pm[perm], atoms, am)
    torch.testing.assert_close(actual, expected)


def test_candidate_chunks_support_the_four_argument_pre_agreement_scorer():
    torch.manual_seed(7)
    peaks, atoms = torch.randn(2, 8, 5), torch.randn(2, 7, 4, 5)
    pm, am = torch.ones(2, 8, dtype=torch.bool), torch.ones(2, 7, 4, dtype=torch.bool)

    def original(p, p_mask, a, a_mask):
        return dense_alignment(p, p_mask, a, a_mask)

    actual = chunked_alignment(original, 3)(peaks, pm, atoms, am)
    expected = original(peaks, pm, atoms, am)
    for first, second in zip(actual, expected, strict=True):
        torch.testing.assert_close(first, second)
