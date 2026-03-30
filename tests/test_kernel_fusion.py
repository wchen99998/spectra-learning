from __future__ import annotations

import torch

from models.losses import SIGReg
from models.model import PeakSetEncoder, PeakSetSIGReg
from networks.transformer_torch import Attention, create_visible_attention_mask

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
B, N, D = 16, 60, 128


def _make_batch(
    batch_size: int = B, num_peaks: int = N, num_targets: int = 3, device: str = DEVICE
) -> dict[str, torch.Tensor]:
    peak_valid_mask = torch.ones(batch_size, num_peaks, dtype=torch.bool, device=device)
    peak_mz = torch.rand(batch_size, num_peaks, device=device)
    peak_intensity = torch.rand(batch_size, num_peaks, device=device)
    context_mask = torch.zeros(batch_size, num_peaks, dtype=torch.bool, device=device)
    context_mask[:, :12] = True
    target_masks = torch.zeros(
        batch_size, num_targets, num_peaks, dtype=torch.bool, device=device
    )
    for target_idx in range(num_targets):
        target_masks[:, target_idx, 12 + target_idx] = True
    return {
        "peak_mz": peak_mz,
        "peak_intensity": peak_intensity,
        "peak_valid_mask": peak_valid_mask,
        "context_mask": context_mask,
        "target_masks": target_masks,
    }


def test_attention_mask_plumbing():
    if DEVICE != "cuda":
        return

    torch.manual_seed(42)
    attn = Attention(D, n_heads=4).to(DEVICE).eval()
    x = torch.randn(2, N, D, device=DEVICE)
    visible_mask = torch.ones(2, N, dtype=torch.bool, device=DEVICE)
    visible_mask[:, N // 2 :] = False
    attn_mask = create_visible_attention_mask(visible_mask)

    with torch.no_grad():
        out_masked = attn(x, attn_mask=attn_mask)
        out_no_mask = attn(x, attn_mask=None)

    assert out_masked.shape == (2, N, D)
    assert torch.isfinite(out_masked).all()
    assert torch.isfinite(out_no_mask).all()
    assert not torch.allclose(out_masked, out_no_mask, atol=1e-5)


def test_sigreg_forward():
    torch.manual_seed(42)
    sigreg = SIGReg(num_slices=256).to(DEVICE)
    proj = torch.randn(4, B, D, device=DEVICE)
    result = sigreg(proj)
    assert result.ndim == 0
    assert torch.isfinite(result)


def test_sigreg_is_per_token_position():
    sigreg = SIGReg(num_slices=128)
    proj_a = torch.tensor(
        [
            [[2.0, 0.0], [-2.0, 0.0]],
            [[2.0, 0.0], [-2.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    proj_b = torch.tensor(
        [
            [[2.0, 0.0], [2.0, 0.0]],
            [[-2.0, 0.0], [-2.0, 0.0]],
        ],
        dtype=torch.float32,
    )

    torch.manual_seed(0)
    loss_a = sigreg(proj_a)
    torch.manual_seed(0)
    loss_b = sigreg(proj_b)

    assert not torch.allclose(loss_a, loss_b, atol=1e-6)


def test_encoder_with_visible_mask():
    torch.manual_seed(42)
    encoder = (
        PeakSetEncoder(
            model_dim=D,
            num_layers=2,
            num_heads=4,
            feature_mlp_hidden_dim=64,
        )
        .to(DEVICE)
        .eval()
    )
    batch = _make_batch()

    with torch.no_grad():
        out_mask = encoder(
            batch["peak_mz"],
            batch["peak_intensity"],
            valid_mask=batch["peak_valid_mask"],
            visible_mask=batch["context_mask"],
        )
        out_none = encoder(
            batch["peak_mz"],
            batch["peak_intensity"],
            valid_mask=batch["peak_valid_mask"],
        )

    assert out_mask.shape == (B, N + 1, D)
    assert out_none.shape == (B, N + 1, D)
    assert torch.isfinite(out_mask).all()
    assert torch.isfinite(out_none).all()
    assert not torch.allclose(out_mask, out_none, atol=1e-3)


def test_full_model_forward():
    torch.manual_seed(42)
    model = (
        PeakSetSIGReg(
            model_dim=D,
            encoder_num_layers=2,
            encoder_num_heads=4,
            feature_mlp_hidden_dim=64,
            sigreg_num_slices=128,
            jepa_num_target_blocks=3,
        )
        .to(DEVICE)
        .eval()
    )
    batch = _make_batch(num_targets=model.jepa_num_target_blocks)

    with torch.no_grad():
        result = model.forward_augmented(batch)

    assert "loss" in result
    assert "token_sigreg_loss" in result
    assert "local_global_loss" in result
    assert "context_fraction" in result
    assert "masked_fraction" in result
    for value in result.values():
        assert torch.isfinite(value)
