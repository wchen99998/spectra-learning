import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'scripts'), str(REPO / 'results/dgt_moljepa_atom_latest/runtime')]

from local_moljepa_data import append_peak_features
from run_local_moljepa_flare import within_set_cosines


def test_raw_features_keep_frozen_slots_and_mask_aligned():
    tokens = torch.arange(2 * 128 * 2048, dtype=torch.float32).reshape(2, 128, 2048)
    valid = torch.ones(2, 128, dtype=torch.bool)
    valid[0, 5:12] = False
    mz = np.linspace(0, 1, 254, dtype=np.float16).reshape(2, 127)
    intensity = mz[:, ::-1].copy()
    combined, returned_mask = append_peak_features(tokens, valid, mz, intensity)
    torch.testing.assert_close(combined[..., :2048], tokens, rtol=0, atol=0)
    torch.testing.assert_close(combined[:, :-1, -2], torch.from_numpy(mz.astype(np.float32)), rtol=0, atol=0)
    torch.testing.assert_close(combined[:, :-1, -1], torch.from_numpy(intensity.astype(np.float32)), rtol=0, atol=0)
    assert combined[:, -1, -2:].count_nonzero() == 0
    assert returned_mask is valid and returned_mask[:, -1].all()


def test_geometry_monitor_matches_pairwise_reference_and_ignores_padding():
    values = torch.tensor([[[1., 0.], [1., 1.], [0., 1.], [1e5, -1e5]],
                           [[1., 0.], [1e5, 0.], [0., 1e5], [1e5, 1e5]]])
    valid = torch.tensor([[True, True, True, False], [True, False, False, False]])
    normalized = torch.nn.functional.normalize(values[0, :3], dim=-1)
    expected = (normalized @ normalized.T)[~torch.eye(3, dtype=torch.bool)].mean()
    measured = within_set_cosines(values, valid)
    assert len(measured) == 1
    torch.testing.assert_close(torch.tensor(measured[0]), expected)


def test_candidate_padding_does_not_enter_molecule_batch(monkeypatch):
    import run_local_moljepa_flare as entry
    encoded_ids = []

    def encode(model, data, ids):
        encoded_ids.append(ids.tolist())
        atoms = torch.eye(3).reshape(3, 1, 3)
        return atoms, torch.ones(3, 1, dtype=torch.bool), torch.zeros(3, 1, 6)

    monkeypatch.setattr(entry, 'encode_molecules', encode)
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda tensor: tensor)
    monkeypatch.setattr(torch, 'autocast', lambda *args, **kwargs: nullcontext())
    data = SimpleNamespace(spectrum=lambda rows: (torch.ones(2, 2, 3), torch.ones(2, 2, dtype=torch.bool)))
    model = SimpleNamespace(spectrum=lambda x, mask: x)
    selected = np.array([[2, 3, 0], [4, 0, 0]])
    valid = torch.tensor([[True, True, False], [True, False, False]])
    score = entry.candidate_forward(model, data, [1, 2], selected, valid, 'evidence', False)
    selected[~valid.numpy()] = 9876
    changed = entry.candidate_forward(model, data, [1, 2], selected, valid, 'evidence', False)
    assert encoded_ids == [[2, 3, 4], [2, 3, 4]]
    torch.testing.assert_close(score, changed)
