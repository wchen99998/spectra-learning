"""Correctness tests for ISAB (Set Transformer) encoder.

Tests:
1. Shape sanity: encoder output is [B,N,D] and pooled output [B,D'] is finite.
2. Padding isolation: randomized padding must not affect valid-position outputs.
3. Permutation equivariance: permuting peaks permutes encoder output identically.
4. All-padding robustness: all-False valid_mask produces finite encoder output.
"""

import torch

from models.model import PeakSetSIGReg


def _build_isab_model(**overrides):
    defaults = dict(
        num_peaks=60,
        model_dim=64,
        encoder_num_layers=2,
        encoder_num_heads=4,
        feature_mlp_hidden_dim=32,
        sigreg_use_projector=False,
        pooling_type="pma",
        pma_num_seeds=1,
        encoder_block_type="isab",
        isab_num_inducing_points=32,
    )
    defaults.update(overrides)
    return PeakSetSIGReg(**defaults)


@torch.no_grad()
def test_isab_shape_sanity():
    """Encoder output is [B,N,D] and pooled output is finite."""
    print("Test 1: ISAB shape sanity ... ", end="", flush=True)
    torch.manual_seed(42)
    model = _build_isab_model()
    model.eval()

    B, N = 4, 60
    batch = {
        "peak_mz": torch.rand(B, N).sort(dim=-1).values,
        "peak_intensity": torch.rand(B, N),
        "peak_valid_mask": torch.ones(B, N, dtype=torch.bool),
        "precursor_mz": torch.rand(B) * 500 + 100,
    }

    enc_out = model.encoder(batch["peak_mz"], batch["peak_intensity"], batch["peak_valid_mask"])
    assert enc_out.shape == (B, N, 64), f"Wrong encoder shape: {enc_out.shape}"
    assert torch.isfinite(enc_out).all(), "Non-finite encoder output"

    pooled = model.encode(batch)
    assert pooled.shape == (B, 64), f"Wrong pooled shape: {pooled.shape}"
    assert torch.isfinite(pooled).all(), "Non-finite pooled output"
    print("PASSED")


@torch.no_grad()
def test_isab_padding_isolation():
    """Randomizing padding peaks must not change valid-position outputs."""
    print("Test 2: ISAB padding isolation ... ", end="", flush=True)
    torch.manual_seed(42)
    model = _build_isab_model()
    model.eval()

    B, N = 4, 60
    batch = {
        "peak_mz": torch.rand(B, N).sort(dim=-1).values,
        "peak_intensity": torch.rand(B, N),
        "peak_valid_mask": torch.zeros(B, N, dtype=torch.bool),
        "precursor_mz": torch.rand(B) * 500 + 100,
    }
    batch["peak_valid_mask"][:, :30] = True
    batch["peak_mz"][:, 30:] = 0.0
    batch["peak_intensity"][:, 30:] = 0.0

    z1 = model.encode(batch)

    batch2 = {k: v.clone() for k, v in batch.items()}
    batch2["peak_mz"][:, 30:] = torch.rand(B, 30) * 1000
    batch2["peak_intensity"][:, 30:] = torch.rand(B, 30)

    z2 = model.encode(batch2)

    max_diff = (z1 - z2).abs().max().item()
    assert max_diff < 1e-4, f"ISAB padding leak: max_diff={max_diff}"
    print(f"PASSED (max_diff={max_diff:.2e})")


@torch.no_grad()
def test_isab_permutation_equivariance():
    """Permuting input peaks should permute encoder output identically."""
    print("Test 3: ISAB permutation equivariance ... ", end="", flush=True)
    torch.manual_seed(42)
    model = _build_isab_model()
    model.eval()

    B, N = 2, 60
    peak_mz = torch.rand(B, N).sort(dim=-1).values
    peak_intensity = torch.rand(B, N)
    valid_mask = torch.ones(B, N, dtype=torch.bool)

    out1 = model.encoder(peak_mz, peak_intensity, valid_mask)

    perm = torch.randperm(N)
    peak_mz_perm = peak_mz[:, perm]
    peak_intensity_perm = peak_intensity[:, perm]
    valid_mask_perm = valid_mask[:, perm]

    out2 = model.encoder(peak_mz_perm, peak_intensity_perm, valid_mask_perm)

    max_diff = (out1[:, perm] - out2).abs().max().item()
    assert max_diff < 1e-4, f"Not equivariant: max_diff={max_diff}"
    print(f"PASSED (max_diff={max_diff:.2e})")


@torch.no_grad()
def test_isab_all_padding_finite():
    """All-False valid_mask should produce finite encoder outputs."""
    print("Test 4: ISAB all-padding robustness ... ", end="", flush=True)
    torch.manual_seed(42)
    model = _build_isab_model()
    model.eval()

    B, N = 4, 60
    peak_mz = torch.rand(B, N).sort(dim=-1).values
    peak_intensity = torch.rand(B, N)
    valid_mask = torch.zeros(B, N, dtype=torch.bool)

    out = model.encoder(peak_mz, peak_intensity, valid_mask)
    assert torch.isfinite(out).all(), f"Non-finite outputs with all-padding: nan={torch.isnan(out).any()}, inf={torch.isinf(out).any()}"
    print("PASSED")


if __name__ == "__main__":
    test_isab_shape_sanity()
    test_isab_padding_isolation()
    test_isab_permutation_equivariance()
    test_isab_all_padding_finite()
    print("\nAll ISAB correctness tests PASSED!")
