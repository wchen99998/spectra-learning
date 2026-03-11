"""Thorough correctness tests for create_visible_block_mask + flex_attention.

Concerns to verify:
1. BLOCK_SIZE=128 vs seq_len=60: OOB index access in mask_mod?
2. mask_mod polarity: True=attend, False=mask
3. Actual attention output correctness vs naive reference
4. Variable valid lengths per batch element
5. Edge cases: all valid, all invalid, single valid position
6. Fused 2B batch (training scenario)
"""

import torch
import torch.nn.functional as F

from networks.transformer_torch import (
    create_visible_block_mask,
    Attention,
    flex_attention,
)


def _naive_masked_attention(q, k, v, valid_mask):
    """Reference: manual softmax attention with explicit -inf masking.

    Args:
        q, k, v: [B, H, N, D]
        valid_mask: [B, N] bool, True=valid
    Returns:
        [B, H, N, D]
    """
    B, H, N, D = q.shape
    scale = D**-0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, N, N]

    # Build 2D mask: position (i,j) is valid only if both i and j are valid
    # valid_mask: [B, N] -> mask: [B, 1, N, N]
    row_mask = valid_mask.unsqueeze(-1)  # [B, N, 1]
    col_mask = valid_mask.unsqueeze(-2)  # [B, 1, N]
    attn_mask = (row_mask & col_mask).unsqueeze(1)  # [B, 1, N, N]

    scores = scores.masked_fill(~attn_mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    # NaN from all-masked rows -> replace with 0
    weights = weights.nan_to_num(0.0)
    return torch.matmul(weights, v)


@torch.no_grad()
def test_block_mask_creation_no_oob():
    """Test 1: create_block_mask doesn't crash with seq_len < BLOCK_SIZE.

    Default BLOCK_SIZE=128, our seq_len=60. If mask_mod is called with
    indices >= 60, valid_mask[b, idx] would OOB.
    """
    print("Test 1: BlockMask creation (no OOB) ... ", end="", flush=True)
    for seq_len in [1, 10, 30, 60, 127, 128, 129, 256]:
        valid_mask = torch.ones(4, seq_len, dtype=torch.bool)
        valid_mask[:, seq_len // 2 :] = False
        bm = create_visible_block_mask(valid_mask)
        assert bm is not None, f"Failed for seq_len={seq_len}"
    print("PASSED")


@torch.no_grad()
def test_flex_attention_vs_naive_reference():
    """Test 2: flex_attention with block_mask matches naive masked attention.

    This is the critical correctness test. We compute attention both ways
    and check the VALID positions match.
    """
    print("Test 2: flex_attention vs naive reference ... ", end="", flush=True)
    torch.manual_seed(123)
    B, H, N, D = 4, 4, 60, 16

    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)

    # Variable valid lengths per batch element
    valid_lengths = [50, 30, 60, 10]
    valid_mask = torch.zeros(B, N, dtype=torch.bool)
    for i, vl in enumerate(valid_lengths):
        valid_mask[i, :vl] = True

    block_mask = create_visible_block_mask(valid_mask)

    out_flex = flex_attention(q, k, v, block_mask=block_mask)
    out_naive = _naive_masked_attention(q, k, v, valid_mask)

    # Compare only at VALID positions (padding outputs don't matter)
    for i in range(B):
        vl = valid_lengths[i]
        flex_valid = out_flex[i, :, :vl, :]
        naive_valid = out_naive[i, :, :vl, :]
        max_diff = (flex_valid - naive_valid).abs().max().item()
        assert max_diff < 1e-5, f"Batch {i}: valid_len={vl}, max_diff={max_diff}"
    print("PASSED")


@torch.no_grad()
def test_padding_positions_dont_attend():
    """Test 3: Verify that changing padding values doesn't affect valid outputs."""
    print("Test 3: Padding isolation check ... ", end="", flush=True)
    torch.manual_seed(42)
    B, H, N, D = 4, 4, 60, 16

    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)

    valid_mask = torch.zeros(B, N, dtype=torch.bool)
    valid_mask[:, :30] = True
    block_mask = create_visible_block_mask(valid_mask)

    out1 = flex_attention(q, k, v, block_mask=block_mask)

    # Randomize padding positions in k and v
    k2 = k.clone()
    v2 = v.clone()
    k2[:, :, 30:, :] = torch.randn_like(k[:, :, 30:, :]) * 100
    v2[:, :, 30:, :] = torch.randn_like(v[:, :, 30:, :]) * 100
    out2 = flex_attention(q, k2, v2, block_mask=block_mask)

    # Valid positions should be identical
    valid_out1 = out1[:, :, :30, :]
    valid_out2 = out2[:, :, :30, :]
    max_diff = (valid_out1 - valid_out2).abs().max().item()
    assert max_diff < 1e-6, f"Padding leaked into valid positions: max_diff={max_diff}"
    print("PASSED")


@torch.no_grad()
def test_mask_polarity():
    """Test 4: Verify True=attend, False=ignore polarity.

    If polarity is reversed, masking out half the keys would still let the
    model attend to them (or vice versa).
    """
    print("Test 4: Mask polarity check ... ", end="", flush=True)
    torch.manual_seed(7)
    B, H, N, D = 2, 2, 16, 8

    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    # Make v have clearly different values in first vs second half
    v = torch.zeros(B, H, N, D)
    v[:, :, :8, :] = 1.0  # valid positions have value 1
    v[:, :, 8:, :] = 100.0  # padding positions have value 100

    # Only first 8 positions are valid
    valid_mask = torch.zeros(B, N, dtype=torch.bool)
    valid_mask[:, :8] = True
    block_mask = create_visible_block_mask(valid_mask)

    out = flex_attention(q, k, v, block_mask=block_mask)

    # Valid query positions should only see v=1.0 keys, so output values
    # should be close to 1.0, NOT close to 100.0
    valid_out = out[:, :, :8, :]
    mean_val = valid_out.mean().item()
    assert abs(mean_val - 1.0) < 0.1, (
        f"Mean output at valid positions: {mean_val:.4f} "
        f"(expected ~1.0, got closer to 100 means polarity is reversed)"
    )
    print(f"PASSED (mean valid output: {mean_val:.4f}, expected ~1.0)")


@torch.no_grad()
def test_all_valid():
    """Test 5: All positions valid should match unmasked attention."""
    print("Test 5: All-valid mask ... ", end="", flush=True)
    torch.manual_seed(99)
    B, H, N, D = 4, 4, 60, 16

    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)

    valid_mask = torch.ones(B, N, dtype=torch.bool)
    block_mask = create_visible_block_mask(valid_mask)

    out_masked = flex_attention(q, k, v, block_mask=block_mask)
    out_unmasked = flex_attention(q, k, v)

    max_diff = (out_masked - out_unmasked).abs().max().item()
    assert max_diff < 1e-5, f"All-valid mask differs from no mask: max_diff={max_diff}"
    print("PASSED")


@torch.no_grad()
def test_single_valid_position():
    """Test 6: Only 1 valid position per batch element."""
    print("Test 6: Single valid position ... ", end="", flush=True)
    torch.manual_seed(42)
    B, H, N, D = 4, 4, 60, 16

    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)

    valid_mask = torch.zeros(B, N, dtype=torch.bool)
    valid_mask[:, 0] = True  # only first position valid
    block_mask = create_visible_block_mask(valid_mask)

    out = flex_attention(q, k, v, block_mask=block_mask)

    # Position 0 attends only to itself, so output should be v[:,:,0,:]
    expected = v[:, :, 0:1, :]
    actual = out[:, :, 0:1, :]
    max_diff = (expected - actual).abs().max().item()
    assert max_diff < 1e-5, (
        f"Single-position self-attention should output v directly: max_diff={max_diff}"
    )
    print("PASSED")


@torch.no_grad()
def test_variable_lengths_per_batch():
    """Test 7: Each batch element has different valid length."""
    print("Test 7: Variable lengths per batch ... ", end="", flush=True)
    torch.manual_seed(42)
    B, H, N, D = 8, 4, 60, 16

    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)

    valid_lengths = [5, 15, 30, 45, 55, 60, 1, 20]
    valid_mask = torch.zeros(B, N, dtype=torch.bool)
    for i, vl in enumerate(valid_lengths):
        valid_mask[i, :vl] = True

    block_mask = create_visible_block_mask(valid_mask)
    out_flex = flex_attention(q, k, v, block_mask=block_mask)
    out_naive = _naive_masked_attention(q, k, v, valid_mask)

    for i in range(B):
        vl = valid_lengths[i]
        diff = (out_flex[i, :, :vl, :] - out_naive[i, :, :vl, :]).abs().max().item()
        assert diff < 1e-5, f"Batch {i} (valid_len={vl}): max_diff={diff}"
    print("PASSED")


@torch.no_grad()
def test_fused_2b_batch():
    """Test 8: Fused batch with 2B elements (training scenario).

    In training, view1 and view2 are stacked along batch dim.
    The valid_mask for view1 may differ from view2 (masked positions
    affect valid_mask differently).
    """
    print("Test 8: Fused 2B training batch ... ", end="", flush=True)
    torch.manual_seed(42)
    B_per_view = 4
    B_fused = 2 * B_per_view
    H, N, D = 4, 60, 16

    q = torch.randn(B_fused, H, N, D)
    k = torch.randn(B_fused, H, N, D)
    v = torch.randn(B_fused, H, N, D)

    # View1 (first B): 40 valid peaks each
    # View2 (second B): 50 valid peaks each
    valid_mask = torch.zeros(B_fused, N, dtype=torch.bool)
    valid_mask[:B_per_view, :40] = True
    valid_mask[B_per_view:, :50] = True

    block_mask = create_visible_block_mask(valid_mask)
    out_flex = flex_attention(q, k, v, block_mask=block_mask)
    out_naive = _naive_masked_attention(q, k, v, valid_mask)

    # Check view1
    for i in range(B_per_view):
        diff = (out_flex[i, :, :40, :] - out_naive[i, :, :40, :]).abs().max().item()
        assert diff < 1e-5, f"View1 batch {i}: max_diff={diff}"

    # Check view2
    for i in range(B_per_view, B_fused):
        diff = (out_flex[i, :, :50, :] - out_naive[i, :, :50, :]).abs().max().item()
        assert diff < 1e-5, f"View2 batch {i}: max_diff={diff}"

    print("PASSED")


@torch.no_grad()
def test_through_attention_module():
    """Test 9: End-to-end through the Attention module (not just raw flex_attention)."""
    print("Test 9: Through Attention module ... ", end="", flush=True)
    torch.manual_seed(42)
    B, N, D = 4, 60, 64

    attn = Attention(D, n_heads=4)
    attn.eval()

    x = torch.randn(B, N, D)
    valid_mask = torch.zeros(B, N, dtype=torch.bool)
    valid_mask[:, :30] = True
    block_mask = create_visible_block_mask(valid_mask)

    out1 = attn(x, block_mask=block_mask)

    # Perturb padding positions in input
    x2 = x.clone()
    x2[:, 30:, :] = torch.randn_like(x[:, 30:, :]) * 100
    out2 = attn(x2, block_mask=block_mask)

    # Valid position outputs should be identical
    max_diff = (out1[:, :30, :] - out2[:, :30, :]).abs().max().item()
    assert max_diff < 1e-5, (
        f"Padding leaked through Attention module: max_diff={max_diff}"
    )
    print("PASSED")


if __name__ == "__main__":
    test_block_mask_creation_no_oob()
    test_flex_attention_vs_naive_reference()
    test_padding_positions_dont_attend()
    test_mask_polarity()
    test_all_valid()
    test_single_valid_position()
    test_variable_lengths_per_batch()
    test_fused_2b_batch()
    test_through_attention_module()
    print("\nAll block mask correctness tests PASSED!")
