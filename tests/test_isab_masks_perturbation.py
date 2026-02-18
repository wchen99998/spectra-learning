"""Rigorous verification of ISAB mask correctness and perturbation isolation.

Strategy: Since BlockMask cannot be easily materialized on CPU, we verify mask
correctness via independent reference computations (manual masked SDPA) and
surgical perturbation at each ISAB pass.

Tests:
 1. kv_block_mask: CrossAttention reference — manual masked SDPA matches flex_attention
 2. q_block_mask: CrossAttention reference — manual masked SDPA matches flex_attention
 3. Pass-1 isolation: perturbing invalid KV positions doesn't change H (inducing summary)
 4. Pass-2 isolation: perturbing H (from changed padding) doesn't reach valid set outputs
 5. Per-pass perturbation with variable valid lengths per batch element
 6. Single valid peak edge case
 7. Alternating valid/invalid pattern
 8. Gradient does not flow through masked positions
 9. Mask polarity verification: swapping mask produces different output
10. Two-layer ISAB stack: padding isolation across multiple layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.set_transformer_torch import (
    ISAB,
    MAB,
    CrossAttention,
    create_kv_padding_block_mask,
    create_q_padding_block_mask,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _manual_cross_attention(
    ca: CrossAttention,
    q_in: torch.Tensor,
    kv_in: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference cross-attention using F.scaled_dot_product_attention.

    attn_mask: [B, Q, KV] boolean, True=attend, False=ignore.
    """
    bsz, q_len, _ = q_in.shape
    _, kv_len, _ = kv_in.shape

    xq = ca.wq(q_in).view(bsz, q_len, ca.n_heads, ca.head_dim).transpose(1, 2)
    kv = ca.wkv(kv_in)
    xk, xv = kv.split(ca.n_kv_heads * ca.head_dim, dim=-1)
    xk = xk.view(bsz, kv_len, ca.n_kv_heads, ca.head_dim).transpose(1, 2)
    xv = xv.view(bsz, kv_len, ca.n_kv_heads, ca.head_dim).transpose(1, 2)

    if ca.n_kv_heads != ca.n_heads:
        rep = ca.n_heads // ca.n_kv_heads
        xk = xk.unsqueeze(2).expand(-1, -1, rep, -1, -1).reshape(bsz, ca.n_heads, kv_len, ca.head_dim)
        xv = xv.unsqueeze(2).expand(-1, -1, rep, -1, -1).reshape(bsz, ca.n_heads, kv_len, ca.head_dim)

    # Build float attention mask for SDPA: 0.0 for attend, -inf for ignore
    sdpa_mask = None
    if attn_mask is not None:
        # attn_mask: [B, Q, KV] -> [B, 1, Q, KV] for broadcast over heads
        sdpa_mask = torch.zeros(bsz, 1, q_len, kv_len, dtype=q_in.dtype)
        sdpa_mask.masked_fill_(~attn_mask.unsqueeze(1), float("-inf"))

    out = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=sdpa_mask, is_causal=False)
    out = out.transpose(1, 2).reshape(bsz, q_len, -1)
    return ca.wo(out)


def _manual_mab(
    mab: MAB,
    x: torch.Tensor,
    y: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference MAB using manual cross-attention."""
    h = x + _manual_cross_attention(mab.cross_attn, mab.q_norm(x), mab.kv_norm(y), attn_mask)
    return h + mab.feed_forward(mab.ffn_norm(h))


# ---------------------------------------------------------------------------
# Test 1: kv_block_mask — CrossAttention reference
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_kv_mask_cross_attention_reference():
    """Verify flex_attention with kv_block_mask matches manual masked SDPA."""
    print("Test 1: kv_block_mask CrossAttention reference ... ", end="", flush=True)
    torch.manual_seed(42)

    dim, n_heads = 16, 2
    B, q_len, kv_len = 2, 4, 10  # inducing=4, set=10
    ca = CrossAttention(dim, n_heads)
    ca.eval()

    q_in = torch.randn(B, q_len, dim)
    kv_in = torch.randn(B, kv_len, dim)

    # Variable valid lengths: batch 0 has 7 valid, batch 1 has 3 valid
    valid_kv = torch.zeros(B, kv_len, dtype=torch.bool)
    valid_kv[0, :7] = True
    valid_kv[1, :3] = True

    # BlockMask version
    block_mask = create_kv_padding_block_mask(valid_kv, q_len=q_len)
    out_flex = ca(q_in, kv_in, block_mask=block_mask)

    # Manual reference: mask[b, q, kv] = valid_kv[b, kv] (broadcast over all q)
    attn_mask = valid_kv.unsqueeze(1).expand(B, q_len, kv_len)
    out_ref = _manual_cross_attention(ca, q_in, kv_in, attn_mask)

    max_diff = (out_flex - out_ref).abs().max().item()
    assert max_diff < 1e-5, f"kv_block_mask mismatch: {max_diff}"
    print(f"PASSED (max_diff={max_diff:.2e})")


# ---------------------------------------------------------------------------
# Test 2: q_block_mask — CrossAttention reference
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_q_mask_cross_attention_reference():
    """Verify flex_attention with q_block_mask matches manual masked SDPA."""
    print("Test 2: q_block_mask CrossAttention reference ... ", end="", flush=True)
    torch.manual_seed(42)

    dim, n_heads = 16, 2
    B, q_len, kv_len = 2, 10, 4  # set=10, inducing=4
    ca = CrossAttention(dim, n_heads)
    ca.eval()

    q_in = torch.randn(B, q_len, dim)
    kv_in = torch.randn(B, kv_len, dim)

    # Variable valid lengths: batch 0 has 6 valid queries, batch 1 has 9
    valid_q = torch.zeros(B, q_len, dtype=torch.bool)
    valid_q[0, :6] = True
    valid_q[1, :9] = True

    block_mask = create_q_padding_block_mask(valid_q, kv_len=kv_len)
    out_flex = ca(q_in, kv_in, block_mask=block_mask)

    # Manual reference: mask[b, q, kv] = valid_q[b, q] (broadcast over all kv)
    attn_mask = valid_q.unsqueeze(2).expand(B, q_len, kv_len)
    out_ref = _manual_cross_attention(ca, q_in, kv_in, attn_mask)

    # Compare only valid query positions (invalid queries may differ due to
    # different -inf handling between flex_attention and SDPA)
    for b in range(B):
        n_valid = valid_q[b].sum().item()
        diff = (out_flex[b, :n_valid] - out_ref[b, :n_valid]).abs().max().item()
        assert diff < 1e-5, f"q_block_mask mismatch at batch {b}: {diff}"

    print("PASSED")


# ---------------------------------------------------------------------------
# Test 3: Pass-1 isolation — perturbing invalid KV doesn't change H
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_pass1_kv_perturbation_isolation():
    """In pass 1 (inducing←set), invalid set positions must not affect H."""
    print("Test 3: Pass-1 KV perturbation isolation ... ", end="", flush=True)
    torch.manual_seed(42)

    dim, m, n_heads = 16, 4, 2
    B, n = 3, 12
    mab = MAB(dim, n_heads)
    mab.eval()

    # Inducing points (queries) — always valid
    I = torch.randn(B, m, dim)

    # Set (keys/values) with partial validity
    valid_lengths = [5, 8, 3]
    X1 = torch.randn(B, n, dim)
    valid_mask = torch.zeros(B, n, dtype=torch.bool)
    for b, vl in enumerate(valid_lengths):
        valid_mask[b, :vl] = True

    # Build kv mask and run
    kv_block_mask = create_kv_padding_block_mask(valid_mask, q_len=m)
    H1 = mab(I, X1, block_mask=kv_block_mask)

    # Perturb invalid positions with extreme values
    X2 = X1.clone()
    for b, vl in enumerate(valid_lengths):
        X2[b, vl:] = torch.randn(n - vl, dim) * 10_000

    H2 = mab(I, X2, block_mask=kv_block_mask)

    # All inducing outputs must be identical (they're all valid queries)
    max_diff = (H1 - H2).abs().max().item()
    assert max_diff < 1e-4, f"Pass-1 padding leaked into H: max_diff={max_diff}"

    # Also verify via manual reference to be sure
    for b, vl in enumerate(valid_lengths):
        attn_mask = valid_mask.unsqueeze(1).expand(B, m, n)
        H_ref = _manual_mab(mab, I, X1, attn_mask)
        ref_diff = (H1[b] - H_ref[b]).abs().max().item()
        assert ref_diff < 1e-5, f"Pass-1 doesn't match reference at batch {b}: {ref_diff}"

    print(f"PASSED (max_diff={max_diff:.2e})")


# ---------------------------------------------------------------------------
# Test 4: Pass-2 isolation — valid set outputs unaffected by padding
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_pass2_q_perturbation_isolation():
    """In pass 2 (set←inducing), valid set positions must be unaffected by padding positions."""
    print("Test 4: Pass-2 Q perturbation isolation ... ", end="", flush=True)
    torch.manual_seed(42)

    dim, m, n_heads = 16, 4, 2
    B, n = 3, 12
    mab = MAB(dim, n_heads)
    mab.eval()

    valid_lengths = [5, 8, 3]
    valid_mask = torch.zeros(B, n, dtype=torch.bool)
    for b, vl in enumerate(valid_lengths):
        valid_mask[b, :vl] = True

    # H (keys/values from inducing) — all valid, same for both runs
    H = torch.randn(B, m, dim)

    # X1: set embeddings
    X1 = torch.randn(B, n, dim)

    q_block_mask = create_q_padding_block_mask(valid_mask, kv_len=m)
    out1 = mab(X1, H, block_mask=q_block_mask)

    # Perturb padding positions in X
    X2 = X1.clone()
    for b, vl in enumerate(valid_lengths):
        X2[b, vl:] = torch.randn(n - vl, dim) * 10_000

    out2 = mab(X2, H, block_mask=q_block_mask)

    # Valid positions must be identical
    for b, vl in enumerate(valid_lengths):
        diff = (out1[b, :vl] - out2[b, :vl]).abs().max().item()
        assert diff < 1e-4, f"Pass-2 padding leaked at batch {b}: diff={diff}"

    print("PASSED")


# ---------------------------------------------------------------------------
# Test 5: Full ISAB with variable valid lengths per batch element
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_isab_variable_lengths_per_batch():
    """Each batch element has different valid length; verify isolation per element."""
    print("Test 5: Variable valid lengths per batch ... ", end="", flush=True)
    torch.manual_seed(42)

    dim, m, n_heads = 16, 8, 2
    B, n = 6, 20
    isab = ISAB(dim, num_inducing_points=m, n_heads=n_heads)
    isab.eval()

    valid_lengths = [1, 5, 10, 15, 19, 20]
    X = torch.randn(B, n, dim)
    valid_mask = torch.zeros(B, n, dtype=torch.bool)
    for b, vl in enumerate(valid_lengths):
        valid_mask[b, :vl] = True

    kv_block_mask = create_kv_padding_block_mask(valid_mask, q_len=m)
    q_block_mask = create_q_padding_block_mask(valid_mask, kv_len=m)

    out1 = isab(X, kv_block_mask=kv_block_mask, q_block_mask=q_block_mask)
    assert torch.isfinite(out1).all(), "Non-finite in baseline"

    # Per-element: perturb that element's padding and check its valid positions
    for b, vl in enumerate(valid_lengths):
        if vl >= n:
            continue
        X2 = X.clone()
        X2[b, vl:] = torch.randn(n - vl, dim) * 10_000
        out2 = isab(X2, kv_block_mask=kv_block_mask, q_block_mask=q_block_mask)

        diff = (out1[b, :vl] - out2[b, :vl]).abs().max().item()
        assert diff < 1e-4, f"Batch {b} (valid_len={vl}): padding leaked, diff={diff}"

        # Other batch elements should also be unaffected
        for b2 in range(B):
            if b2 == b:
                continue
            vl2 = valid_lengths[b2]
            cross_diff = (out1[b2, :vl2] - out2[b2, :vl2]).abs().max().item()
            assert cross_diff < 1e-6, (
                f"Perturbing batch {b} affected batch {b2}: diff={cross_diff}"
            )

    print("PASSED")


# ---------------------------------------------------------------------------
# Test 6: Single valid peak edge case
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_single_valid_peak():
    """Only 1 peak is valid per batch element. Verify finite output and isolation."""
    print("Test 6: Single valid peak ... ", end="", flush=True)
    torch.manual_seed(42)

    dim, m, n_heads = 16, 4, 2
    B, n = 4, 10
    isab = ISAB(dim, num_inducing_points=m, n_heads=n_heads)
    isab.eval()

    X = torch.randn(B, n, dim)
    valid_mask = torch.zeros(B, n, dtype=torch.bool)
    # Each batch element: only position 0 is valid
    valid_mask[:, 0] = True

    kv_block_mask = create_kv_padding_block_mask(valid_mask, q_len=m)
    q_block_mask = create_q_padding_block_mask(valid_mask, kv_len=m)

    out1 = isab(X, kv_block_mask=kv_block_mask, q_block_mask=q_block_mask)
    assert torch.isfinite(out1[:, 0]).all(), "Non-finite at valid position"

    # Perturb all padding
    X2 = X.clone()
    X2[:, 1:] = torch.randn(B, n - 1, dim) * 10_000
    out2 = isab(X2, kv_block_mask=kv_block_mask, q_block_mask=q_block_mask)

    diff = (out1[:, 0] - out2[:, 0]).abs().max().item()
    assert diff < 1e-4, f"Single valid peak leaks: diff={diff}"
    print(f"PASSED (diff={diff:.2e})")


# ---------------------------------------------------------------------------
# Test 7: Alternating valid/invalid pattern
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_alternating_valid_pattern():
    """Non-contiguous valid mask (alternating True/False). Verify isolation."""
    print("Test 7: Alternating valid/invalid pattern ... ", end="", flush=True)
    torch.manual_seed(42)

    dim, m, n_heads = 16, 4, 2
    B, n = 2, 12
    isab = ISAB(dim, num_inducing_points=m, n_heads=n_heads)
    isab.eval()

    X = torch.randn(B, n, dim)
    valid_mask = torch.zeros(B, n, dtype=torch.bool)
    # Even indices valid, odd indices padding
    valid_mask[:, 0::2] = True  # positions 0, 2, 4, 6, 8, 10

    kv_block_mask = create_kv_padding_block_mask(valid_mask, q_len=m)
    q_block_mask = create_q_padding_block_mask(valid_mask, kv_len=m)

    out1 = isab(X, kv_block_mask=kv_block_mask, q_block_mask=q_block_mask)

    # Perturb odd (invalid) positions
    X2 = X.clone()
    X2[:, 1::2] = torch.randn(B, n // 2, dim) * 10_000
    out2 = isab(X2, kv_block_mask=kv_block_mask, q_block_mask=q_block_mask)

    # Check all valid (even) positions
    valid_indices = torch.arange(0, n, 2)
    diff = (out1[:, valid_indices] - out2[:, valid_indices]).abs().max().item()
    assert diff < 1e-4, f"Alternating pattern leaks: diff={diff}"
    print(f"PASSED (diff={diff:.2e})")


# ---------------------------------------------------------------------------
# Test 8: Gradient does not flow through masked positions
# ---------------------------------------------------------------------------

def test_gradient_blocked_at_masked_positions():
    """Verify that gradients are zero at padding positions in the input."""
    print("Test 8: Gradient blocked at masked positions ... ", end="", flush=True)
    torch.manual_seed(42)

    dim, m, n_heads = 16, 4, 2
    B, n = 2, 10
    n_valid = 6
    isab = ISAB(dim, num_inducing_points=m, n_heads=n_heads)

    X = torch.randn(B, n, dim, requires_grad=True)
    valid_mask = torch.zeros(B, n, dtype=torch.bool)
    valid_mask[:, :n_valid] = True

    kv_block_mask = create_kv_padding_block_mask(valid_mask, q_len=m)
    q_block_mask = create_q_padding_block_mask(valid_mask, kv_len=m)

    out = isab(X, kv_block_mask=kv_block_mask, q_block_mask=q_block_mask)

    # Only use valid positions for the loss
    loss = out[:, :n_valid].sum()
    loss.backward()

    assert X.grad is not None, "No gradient on input"

    # Gradient at valid positions should be nonzero
    valid_grad_norm = X.grad[:, :n_valid].abs().sum().item()
    assert valid_grad_norm > 1e-3, f"Valid gradient is zero: {valid_grad_norm}"

    # Gradient at padding positions in pass 1 (KV) should be zero because
    # kv_block_mask prevents attention to those positions.
    # In pass 2, padding queries are masked by q_block_mask, so their
    # contribution is also zeroed.
    pad_grad_norm = X.grad[:, n_valid:].abs().sum().item()
    assert pad_grad_norm < 1e-5, f"Padding gradient nonzero: {pad_grad_norm}"

    print(f"PASSED (valid_grad={valid_grad_norm:.2e}, pad_grad={pad_grad_norm:.2e})")


# ---------------------------------------------------------------------------
# Test 9: Mask polarity — swapping masks produces different output
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_mask_polarity_swap():
    """Using inverted mask should produce observably different valid-position outputs."""
    print("Test 9: Mask polarity (swap test) ... ", end="", flush=True)
    torch.manual_seed(42)

    dim, m, n_heads = 16, 4, 2
    B, n = 2, 10
    isab = ISAB(dim, num_inducing_points=m, n_heads=n_heads)
    isab.eval()

    X = torch.randn(B, n, dim) * 5  # larger values to amplify differences
    valid_mask = torch.zeros(B, n, dtype=torch.bool)
    valid_mask[:, :5] = True

    kv_bm = create_kv_padding_block_mask(valid_mask, q_len=m)
    q_bm = create_q_padding_block_mask(valid_mask, kv_len=m)

    out_correct = isab(X, kv_block_mask=kv_bm, q_block_mask=q_bm)

    # Now with inverted mask: only padding positions are "valid"
    inv_mask = ~valid_mask
    kv_bm_inv = create_kv_padding_block_mask(inv_mask, q_len=m)
    q_bm_inv = create_q_padding_block_mask(inv_mask, kv_len=m)

    out_inverted = isab(X, kv_block_mask=kv_bm_inv, q_block_mask=q_bm_inv)

    # With correct mask, positions 0-4 attend to data in positions 0-4
    # With inverted mask, positions 5-9 attend to data in positions 5-9
    # Output at positions 0-4 should differ substantially
    diff = (out_correct[:, :5] - out_inverted[:, :5]).abs().max().item()
    assert diff > 0.01, (
        f"Swapping mask polarity had no effect (diff={diff:.6f}). "
        f"Masks may be ignored."
    )
    print(f"PASSED (polarity diff={diff:.4f})")


# ---------------------------------------------------------------------------
# Test 10: Multi-layer ISAB stack isolation
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_multi_layer_isab_stack_isolation():
    """Padding isolation holds across a stack of multiple ISAB layers."""
    print("Test 10: Multi-layer ISAB stack isolation ... ", end="", flush=True)
    torch.manual_seed(42)

    dim, m, n_heads = 16, 4, 2
    B, n = 3, 12
    num_layers = 3
    valid_lengths = [4, 8, 12]

    layers = nn.ModuleList([
        ISAB(dim, num_inducing_points=m, n_heads=n_heads)
        for _ in range(num_layers)
    ])
    layers.eval()

    X = torch.randn(B, n, dim)
    valid_mask = torch.zeros(B, n, dtype=torch.bool)
    for b, vl in enumerate(valid_lengths):
        valid_mask[b, :vl] = True

    kv_bm = create_kv_padding_block_mask(valid_mask, q_len=m)
    q_bm = create_q_padding_block_mask(valid_mask, kv_len=m)

    # Forward through stack
    x1 = X.clone()
    for layer in layers:
        x1 = layer(x1, kv_block_mask=kv_bm, q_block_mask=q_bm)

    # Perturb padding and forward again
    X2 = X.clone()
    for b, vl in enumerate(valid_lengths):
        if vl < n:
            X2[b, vl:] = torch.randn(n - vl, dim) * 10_000

    x2 = X2.clone()
    for layer in layers:
        x2 = layer(x2, kv_block_mask=kv_bm, q_block_mask=q_bm)

    # Verify valid-position isolation after 3 layers
    for b, vl in enumerate(valid_lengths):
        diff = (x1[b, :vl] - x2[b, :vl]).abs().max().item()
        assert diff < 1e-3, (
            f"Multi-layer stack: batch {b} (valid_len={vl}) leaked after "
            f"{num_layers} layers, diff={diff}"
        )

    assert torch.isfinite(x1).all(), "Non-finite in multi-layer output"
    print("PASSED")


# ---------------------------------------------------------------------------
# Test 11: kv_mask vs q_mask are NOT interchangeable
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_kv_and_q_masks_not_interchangeable():
    """Swapping kv_block_mask and q_block_mask should break isolation."""
    print("Test 11: kv/q masks not interchangeable ... ", end="", flush=True)
    torch.manual_seed(42)

    dim, m, n_heads = 16, 4, 2
    B, n = 2, 10
    n_valid = 5
    isab = ISAB(dim, num_inducing_points=m, n_heads=n_heads)
    isab.eval()

    X = torch.randn(B, n, dim)
    valid_mask = torch.zeros(B, n, dtype=torch.bool)
    valid_mask[:, :n_valid] = True

    kv_bm = create_kv_padding_block_mask(valid_mask, q_len=m)
    q_bm = create_q_padding_block_mask(valid_mask, kv_len=m)

    # Correct mask assignment
    out_correct = isab(X, kv_block_mask=kv_bm, q_block_mask=q_bm)

    # Perturb padding
    X2 = X.clone()
    X2[:, n_valid:] = torch.randn(B, n - n_valid, dim) * 10_000
    out_correct_perturbed = isab(X2, kv_block_mask=kv_bm, q_block_mask=q_bm)

    correct_isolation = (out_correct[:, :n_valid] - out_correct_perturbed[:, :n_valid]).abs().max().item()
    assert correct_isolation < 1e-4, f"Correct masks should isolate: {correct_isolation}"

    # Swapped masks: kv_block_mask has shape (B,1,m,N) but q_block_mask has (B,1,N,m)
    # They can't even be swapped because the dimensions won't match.
    # Instead, test: what if we use kv_block_mask for BOTH passes?
    # Pass 2 expects (Q=N, KV=m) but kv_bm is (Q=m, KV=N) — dimension mismatch.
    # So test: what if we use NO mask for pass 1 (the critical isolation point)?
    out_no_kv_mask = isab(X, kv_block_mask=None, q_block_mask=q_bm)
    out_no_kv_mask_perturbed = isab(X2, kv_block_mask=None, q_block_mask=q_bm)

    no_mask_diff = (out_no_kv_mask[:, :n_valid] - out_no_kv_mask_perturbed[:, :n_valid]).abs().max().item()
    assert no_mask_diff > 0.01, (
        f"Without kv_block_mask, padding should leak (diff={no_mask_diff}). "
        f"kv_block_mask is doing nothing?"
    )

    print(f"PASSED (with_mask={correct_isolation:.2e}, without_kv_mask={no_mask_diff:.4f})")


# ---------------------------------------------------------------------------
# Test 12: Reference computation — full ISAB vs manual two-MAB computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_isab_matches_manual_two_pass():
    """ISAB output matches a manual two-pass MAB computation with the same masks."""
    print("Test 12: ISAB vs manual two-pass reference ... ", end="", flush=True)
    torch.manual_seed(42)

    dim, m, n_heads = 16, 4, 2
    B, n = 2, 8
    n_valid = 5
    isab = ISAB(dim, num_inducing_points=m, n_heads=n_heads)
    isab.eval()

    X = torch.randn(B, n, dim)
    valid_mask = torch.zeros(B, n, dtype=torch.bool)
    valid_mask[:, :n_valid] = True

    # ISAB with block masks
    kv_bm = create_kv_padding_block_mask(valid_mask, q_len=m)
    q_bm = create_q_padding_block_mask(valid_mask, kv_len=m)
    out_isab = isab(X, kv_block_mask=kv_bm, q_block_mask=q_bm)

    # Manual reference
    I = isab.inducing_points.unsqueeze(0).expand(B, -1, -1)

    # Pass 1: inducing (Q=m) attends to set (KV=n)
    # attn_mask: [B, m, n] where mask[b, q, kv] = valid_mask[b, kv]
    mask1 = valid_mask.unsqueeze(1).expand(B, m, n)
    H_ref = _manual_mab(isab.mab1, I, X, mask1)

    # Pass 2: set (Q=n) attends to inducing (KV=m)
    # attn_mask: [B, n, m] where mask[b, q, kv] = valid_mask[b, q]
    mask2 = valid_mask.unsqueeze(2).expand(B, n, m)
    out_ref = _manual_mab(isab.mab2, X, H_ref, mask2)

    # Compare valid positions
    diff = (out_isab[:, :n_valid] - out_ref[:, :n_valid]).abs().max().item()
    assert diff < 1e-5, f"ISAB vs manual reference mismatch: {diff}"
    print(f"PASSED (max_diff={diff:.2e})")


if __name__ == "__main__":
    test_kv_mask_cross_attention_reference()
    test_q_mask_cross_attention_reference()
    test_pass1_kv_perturbation_isolation()
    test_pass2_q_perturbation_isolation()
    test_isab_variable_lengths_per_batch()
    test_single_valid_peak()
    test_alternating_valid_pattern()
    test_gradient_blocked_at_masked_positions()
    test_mask_polarity_swap()
    test_multi_layer_isab_stack_isolation()
    test_kv_and_q_masks_not_interchangeable()
    test_isab_matches_manual_two_pass()
    print("\nAll ISAB mask & perturbation tests PASSED!")
