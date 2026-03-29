"""Correctness tests for dense SDPA masking."""

import torch
import torch.nn.functional as F

from networks.transformer_torch import Attention, create_visible_attention_mask


def _naive_masked_attention(q, k, v, valid_mask):
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    key_mask = valid_mask[:, None, None, :]
    scores = scores.masked_fill(~key_mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)


@torch.no_grad()
def test_attention_mask_matches_naive_reference():
    torch.manual_seed(123)
    B, H, N, D = 4, 4, 60, 16

    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)

    valid_lengths = [50, 30, 60, 10]
    valid_mask = torch.zeros(B, N, dtype=torch.bool)
    for i, vl in enumerate(valid_lengths):
        valid_mask[i, :vl] = True

    attn_mask = create_visible_attention_mask(valid_mask)
    out_sdpa = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    out_naive = _naive_masked_attention(q, k, v, valid_mask)

    max_diff = (out_sdpa - out_naive).abs().max().item()
    assert max_diff < 1e-5, f"max_diff={max_diff}"


@torch.no_grad()
def test_padding_keys_do_not_affect_outputs():
    torch.manual_seed(42)
    B, H, N, D = 4, 4, 60, 16

    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)

    valid_mask = torch.zeros(B, N, dtype=torch.bool)
    valid_mask[:, :30] = True
    attn_mask = create_visible_attention_mask(valid_mask)

    out1 = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

    k2 = k.clone()
    v2 = v.clone()
    k2[:, :, 30:, :] = torch.randn_like(k[:, :, 30:, :]) * 100
    v2[:, :, 30:, :] = torch.randn_like(v[:, :, 30:, :]) * 100
    out2 = F.scaled_dot_product_attention(q, k2, v2, attn_mask=attn_mask)

    max_diff = (out1 - out2).abs().max().item()
    assert max_diff < 1e-6, f"max_diff={max_diff}"


@torch.no_grad()
def test_mask_polarity():
    torch.manual_seed(7)
    B, H, N, D = 2, 2, 16, 8

    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    v = torch.zeros(B, H, N, D)
    v[:, :, :8, :] = 1.0
    v[:, :, 8:, :] = 100.0

    valid_mask = torch.zeros(B, N, dtype=torch.bool)
    valid_mask[:, :8] = True
    attn_mask = create_visible_attention_mask(valid_mask)

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    mean_val = out.mean().item()
    assert abs(mean_val - 1.0) < 0.1, f"mean_val={mean_val:.4f}"


@torch.no_grad()
def test_all_valid_matches_unmasked():
    torch.manual_seed(99)
    B, H, N, D = 4, 4, 60, 16

    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)

    valid_mask = torch.ones(B, N, dtype=torch.bool)
    attn_mask = create_visible_attention_mask(valid_mask)

    out_masked = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    out_unmasked = F.scaled_dot_product_attention(q, k, v)

    max_diff = (out_masked - out_unmasked).abs().max().item()
    assert max_diff < 1e-5, f"max_diff={max_diff}"


@torch.no_grad()
def test_single_valid_position():
    torch.manual_seed(42)
    B, H, N, D = 4, 4, 60, 16

    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)

    valid_mask = torch.zeros(B, N, dtype=torch.bool)
    valid_mask[:, 0] = True
    attn_mask = create_visible_attention_mask(valid_mask)

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    expected = v[:, :, 0:1, :].expand_as(out)

    max_diff = (expected - out).abs().max().item()
    assert max_diff < 1e-5, f"max_diff={max_diff}"


@torch.no_grad()
def test_attention_module_masks_hidden_keys():
    torch.manual_seed(42)
    B, N, D = 4, 60, 64

    attn = Attention(D, n_heads=4)
    attn.eval()

    x = torch.randn(B, N, D)
    valid_mask = torch.zeros(B, N, dtype=torch.bool)
    valid_mask[:, :30] = True
    attn_mask = create_visible_attention_mask(valid_mask)

    out1 = attn(x, attn_mask=attn_mask)

    x2 = x.clone()
    x2[:, 30:, :] = torch.randn_like(x[:, 30:, :]) * 100
    out2 = attn(x2, attn_mask=attn_mask)

    max_diff = (out1[:, :30, :] - out2[:, :30, :]).abs().max().item()
    assert max_diff < 1e-5, f"max_diff={max_diff}"


if __name__ == "__main__":
    test_attention_mask_matches_naive_reference()
    test_padding_keys_do_not_affect_outputs()
    test_mask_polarity()
    test_all_valid_matches_unmasked()
    test_single_valid_position()
    test_attention_module_masks_hidden_keys()
    print("\nAll SDPA mask correctness tests PASSED!")
