from unittest import mock

import torch
import torch._dynamo

from spectra_learning.models.losses import SIGReg, SlotwiseSIGReg


def _normalize_directions(directions: torch.Tensor) -> torch.Tensor:
    return directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)


def test_sigreg_returns_zero_for_empty_mask():
    torch.manual_seed(0)
    sigreg = SIGReg(num_slices=8)
    proj = torch.randn(2, 3, 5)
    valid_mask = torch.zeros(2, 3)

    result = sigreg(proj, valid_mask=valid_mask)

    torch.testing.assert_close(result, torch.zeros_like(result))


def test_slotwise_sigreg_matches_sum_of_slot_losses():
    torch.manual_seed(0)
    sigreg = SlotwiseSIGReg(num_slices=8)
    proj = torch.randn(2, 3, 4, 5)
    valid_mask = torch.ones(2, 3, 4)
    valid_mask[:, :, 2] = 0.0
    valid_mask[0, 0, 0] = 2.5
    valid_mask[1, 2, 3] = 0.0
    directions = _normalize_directions(
        torch.randn(proj.shape[-1], sigreg.num_slices, dtype=proj.dtype)
    )

    flat_proj = proj.reshape(-1, proj.shape[2], proj.shape[-1])
    flat_mask = valid_mask.reshape(-1, proj.shape[2])
    expected = proj.new_zeros(())
    for slot_idx in range(proj.shape[2]):
        expected = expected + sigreg._loss_from_flat(
            flat_proj[:, slot_idx, :],
            flat_mask[:, slot_idx],
            directions=directions,
        )

    with mock.patch.object(sigreg, "_sample_directions", return_value=directions):
        actual = sigreg(proj, valid_mask=valid_mask)

    torch.testing.assert_close(actual, expected)


def test_slotwise_sigreg_uses_provided_directions_without_sampling():
    torch.manual_seed(0)
    sigreg = SlotwiseSIGReg(num_slices=8)
    proj = torch.randn(2, 3, 4, 5)
    valid_mask = torch.ones(2, 3, 4)
    valid_mask[:, :, 1] = 0.0
    directions = _normalize_directions(
        torch.randn(proj.shape[-1], sigreg.num_slices, dtype=proj.dtype)
    )

    flat_proj = proj.reshape(-1, proj.shape[2], proj.shape[-1])
    flat_mask = valid_mask.reshape(-1, proj.shape[2])
    expected = proj.new_zeros(())
    for slot_idx in range(proj.shape[2]):
        expected = expected + sigreg._loss_from_flat(
            flat_proj[:, slot_idx, :],
            flat_mask[:, slot_idx],
            directions=directions,
        )

    with mock.patch.object(sigreg, "_sample_directions", side_effect=AssertionError):
        actual = sigreg(proj, valid_mask=valid_mask, directions=directions)

    torch.testing.assert_close(actual, expected)


def test_slotwise_sigreg_supports_encoder_layout():
    torch.manual_seed(0)
    sigreg = SlotwiseSIGReg(num_slices=8)
    proj = torch.randn(4, 6, 5)
    valid_mask = torch.tensor(
        [
            [1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        ]
    )

    result = sigreg(proj, valid_mask=valid_mask)

    assert result.ndim == 0
    assert torch.isfinite(result)


def test_slotwise_sigreg_has_no_dynamo_graph_breaks():
    torch.manual_seed(0)
    sigreg = SlotwiseSIGReg(num_slices=8)
    proj = torch.randn(2, 3, 4, 5)
    valid_mask = torch.zeros(2, 3, 4)
    valid_mask[:, :, :2] = 1.0

    explain = torch._dynamo.explain(sigreg)(proj, valid_mask=valid_mask)

    assert explain.graph_break_count == 0


def test_slotwise_sigreg_has_no_dynamo_graph_breaks_with_directions_input():
    torch.manual_seed(0)
    sigreg = SlotwiseSIGReg(num_slices=8)
    proj = torch.randn(2, 3, 4, 5)
    valid_mask = torch.zeros(2, 3, 4)
    valid_mask[:, :, :2] = 1.0
    directions = _normalize_directions(
        torch.randn(proj.shape[-1], sigreg.num_slices, dtype=proj.dtype)
    )

    explain = torch._dynamo.explain(sigreg)(
        proj,
        valid_mask=valid_mask,
        directions=directions,
    )

    assert explain.graph_break_count == 0
