"""Correctness tests for PMA pooling with padding mask.

Concerns:
1. key_padding_mask polarity: ~valid_mask → True=ignore correct?
2. Padding isolation: randomize padding embeddings, PMA output unchanged
3. All-padding row: does nn.MHA produce NaN?
4. Gradient flow: do pool_query and pool_mha get gradients?
5. Full pipeline: encoder (with block_mask) → PMA → projector
"""

import torch

from models.model import PeakSetSIGReg


def _build_model(**overrides):
    defaults = dict(
        num_peaks=60,
        model_dim=64,
        encoder_num_layers=2,
        encoder_num_heads=4,
        feature_mlp_hidden_dim=32,
        sigreg_use_projector=False,
        pooling_type="pma",
        pma_num_seeds=1,
    )
    defaults.update(overrides)
    return PeakSetSIGReg(**defaults)


@torch.no_grad()
def test_pma_padding_isolation():
    """PMA output at valid positions must not change when padding embeddings change."""
    print("Test 1: PMA padding isolation ... ", end="", flush=True)
    torch.manual_seed(42)
    model = _build_model()
    model.eval()

    B, N = 4, 60
    embeddings = torch.randn(B, N, 64)
    valid_mask = torch.zeros(B, N, dtype=torch.bool)
    valid_mask[:, :30] = True

    pooled1 = model.pool(embeddings, valid_mask)

    # Randomize padding embeddings
    embeddings2 = embeddings.clone()
    embeddings2[:, 30:, :] = torch.randn(B, 30, 64) * 100
    pooled2 = model.pool(embeddings2, valid_mask)

    max_diff = (pooled1 - pooled2).abs().max().item()
    assert max_diff < 1e-5, f"PMA leaks padding: max_diff={max_diff}"
    print(f"PASSED (max_diff={max_diff:.2e})")


@torch.no_grad()
def test_pma_polarity_empirical():
    """Verify polarity: PMA should attend to valid keys, not padding keys.

    Set valid embeddings to a known value and padding to a very different value.
    If polarity is correct, output should reflect valid values only.
    """
    print("Test 2: PMA polarity empirical ... ", end="", flush=True)
    torch.manual_seed(42)
    model = _build_model()
    model.eval()

    B, N, D = 4, 60, 64
    embeddings = torch.zeros(B, N, D)
    embeddings[:, :30, :] = 1.0   # valid positions: value 1
    embeddings[:, 30:, :] = 999.0  # padding positions: value 999

    valid_mask = torch.zeros(B, N, dtype=torch.bool)
    valid_mask[:, :30] = True

    pooled = model.pool(embeddings, valid_mask)

    # If polarity is correct, pooled should be close to some transform of 1.0,
    # NOT influenced by 999.0
    # Compare to pooling with padding set to 0
    embeddings_zero_pad = embeddings.clone()
    embeddings_zero_pad[:, 30:, :] = 0.0
    pooled_zero_pad = model.pool(embeddings_zero_pad, valid_mask)

    max_diff = (pooled - pooled_zero_pad).abs().max().item()
    assert max_diff < 1e-5, (
        f"PMA output changed with different padding values: max_diff={max_diff} "
        f"(polarity may be wrong)"
    )
    print(f"PASSED (max_diff={max_diff:.2e})")


@torch.no_grad()
def test_pma_all_padding_row():
    """When ALL keys are padding for a batch element, check for NaN."""
    print("Test 3: PMA all-padding row ... ", end="", flush=True)
    torch.manual_seed(42)
    model = _build_model()
    model.eval()

    B, N, D = 4, 60, 64
    embeddings = torch.randn(B, N, D)

    valid_mask = torch.zeros(B, N, dtype=torch.bool)
    valid_mask[0, :30] = True  # batch 0: 30 valid
    valid_mask[1, :50] = True  # batch 1: 50 valid
    # batch 2: ALL PADDING (0 valid) — edge case
    # batch 3: ALL PADDING (0 valid) — edge case

    pooled = model.pool(embeddings, valid_mask)

    has_nan = torch.isnan(pooled).any().item()
    has_inf = torch.isinf(pooled).any().item()

    if has_nan or has_inf:
        print(f"WARNING: NaN={has_nan}, Inf={has_inf} in all-padding rows")
        print(f"  Row 0 (30 valid): nan={torch.isnan(pooled[0]).any()}")
        print(f"  Row 2 (0 valid):  nan={torch.isnan(pooled[2]).any()}")
        # This is a real concern — all-masked softmax produces NaN
        # Check if valid rows are at least ok
        assert torch.isfinite(pooled[0]).all(), "Valid row 0 has NaN/Inf"
        assert torch.isfinite(pooled[1]).all(), "Valid row 1 has NaN/Inf"
        print("  (Valid rows are finite, all-padding rows have NaN — expected edge case)")
    else:
        print("PASSED (all outputs finite)")


@torch.no_grad()
def test_pma_variable_valid_lengths():
    """Each batch element has a different valid length."""
    print("Test 4: PMA variable lengths ... ", end="", flush=True)
    torch.manual_seed(42)
    model = _build_model()
    model.eval()

    B, N, D = 8, 60, 64
    embeddings = torch.randn(B, N, D)

    valid_lengths = [5, 15, 30, 45, 55, 60, 1, 20]
    valid_mask = torch.zeros(B, N, dtype=torch.bool)
    for i, vl in enumerate(valid_lengths):
        valid_mask[i, :vl] = True

    pooled = model.pool(embeddings, valid_mask)
    assert pooled.shape == (B, D), f"Wrong shape: {pooled.shape}"
    assert torch.isfinite(pooled).all(), "Non-finite values in PMA output"

    # Verify isolation per row: changing padding in row i shouldn't affect row i's output
    for i in range(B):
        vl = valid_lengths[i]
        if vl >= N:
            continue
        embeddings2 = embeddings.clone()
        embeddings2[i, vl:, :] = torch.randn(N - vl, D) * 100
        pooled2 = model.pool(embeddings2, valid_mask)
        diff = (pooled[i] - pooled2[i]).abs().max().item()
        assert diff < 1e-5, f"Row {i} (valid_len={vl}): padding leaked, diff={diff}"

    print("PASSED")


def test_pma_gradient_flow():
    """Gradients must flow through PMA to pool_query and pool_mha parameters."""
    print("Test 5: PMA gradient flow ... ", end="", flush=True)
    torch.manual_seed(42)
    model = _build_model(sigreg_use_projector=True, sigreg_proj_hidden_dim=64, sigreg_proj_output_dim=32)

    B, N, D = 4, 60, 64
    embeddings = torch.randn(B, N, D, requires_grad=True)
    valid_mask = torch.zeros(B, N, dtype=torch.bool)
    valid_mask[:, :30] = True

    pooled = model.pool(embeddings, valid_mask)
    projected = model.projector(pooled)
    loss = projected.sum()
    loss.backward()

    assert model.pool_query.grad is not None, "pool_query has no gradient"
    assert model.pool_query.grad.abs().sum() > 0, "pool_query gradient is zero"

    mha_grads = [(n, p.grad) for n, p in model.pool_mha.named_parameters() if p.grad is not None]
    assert len(mha_grads) > 0, "No MHA parameters have gradients"
    for name, grad in mha_grads:
        assert grad.abs().sum() > 0, f"MHA param {name} has zero gradient"

    assert embeddings.grad is not None, "Input embeddings have no gradient"
    # Gradient should be zero at padding positions
    pad_grad_norm = embeddings.grad[:, 30:, :].abs().sum().item()
    assert pad_grad_norm < 1e-5, f"Padding positions have non-zero gradient: {pad_grad_norm}"

    print("PASSED")


@torch.no_grad()
def test_full_pipeline_encoder_to_pma():
    """Full pipeline: raw batch → encoder (with block_mask) → PMA → output.

    Verify padding isolation end-to-end.
    """
    print("Test 6: Full encoder→PMA pipeline ... ", end="", flush=True)
    torch.manual_seed(42)
    model = _build_model()
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

    # Randomize padding peaks
    batch2 = {k: v.clone() for k, v in batch.items()}
    batch2["peak_mz"][:, 30:] = torch.rand(B, 30) * 1000
    batch2["peak_intensity"][:, 30:] = torch.rand(B, 30)

    z2 = model.encode(batch2)

    max_diff = (z1 - z2).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(z1, z2, dim=-1).mean().item()

    assert max_diff < 1e-5, f"Full pipeline padding leak: max_diff={max_diff}"
    assert cos_sim > 0.9999, f"Cosine similarity: {cos_sim}"
    print(f"PASSED (max_diff={max_diff:.2e}, cos_sim={cos_sim:.6f})")


@torch.no_grad()
def test_pma_num_seeds_greater_than_one():
    """With pma_num_seeds > 1, verify output shape and padding isolation."""
    print("Test 7: PMA num_seeds > 1 ... ", end="", flush=True)
    torch.manual_seed(42)
    model = _build_model(pma_num_seeds=4)
    model.eval()

    B, N, D = 4, 60, 64
    embeddings = torch.randn(B, N, D)
    valid_mask = torch.zeros(B, N, dtype=torch.bool)
    valid_mask[:, :30] = True

    pooled = model.pool(embeddings, valid_mask)
    # pool_query is [4, D], expanded to [B, 4, D], output [B, 4, D], mean(dim=1) → [B, D]
    assert pooled.shape == (B, D), f"Wrong shape: {pooled.shape}"

    embeddings2 = embeddings.clone()
    embeddings2[:, 30:, :] = torch.randn(B, 30, D) * 100
    pooled2 = model.pool(embeddings2, valid_mask)

    max_diff = (pooled - pooled2).abs().max().item()
    assert max_diff < 1e-5, f"PMA (seeds=4) leaks padding: max_diff={max_diff}"
    print("PASSED")


if __name__ == "__main__":
    test_pma_padding_isolation()
    test_pma_polarity_empirical()
    test_pma_all_padding_row()
    test_pma_variable_valid_lengths()
    test_pma_gradient_flow()
    test_full_pipeline_encoder_to_pma()
    test_pma_num_seeds_greater_than_one()
    print("\nAll PMA correctness tests PASSED!")
