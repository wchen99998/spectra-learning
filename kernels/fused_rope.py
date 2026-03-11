"""Fused RoPE (Rotary Position Embedding) Triton kernel.

Replaces the multi-kernel interleaved RoPE computation with a single fused kernel
for both forward and backward passes.

Key optimizations vs naive:
1. Loads cos/sin from original [1, S, 1, D] tensor via index computation
   (no expand+contiguous copy, saving 76 kernel launches per step).
2. Handles non-contiguous Q/K inputs directly via stride parameters
   (no .contiguous() copy, saving 72 kernel launches per step).

The interleaved RoPE format: pairs (x[2i], x[2i+1]) are rotated by angle theta_i.
  out[2i]   = x[2i] * cos(theta_i) - x[2i+1] * sin(theta_i)
  out[2i+1] = x[2i+1] * cos(theta_i) + x[2i] * sin(theta_i)

Backward is the inverse rotation (negate sin).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_fwd_kernel(
    X_ptr, COS_ptr, SIN_ptr, OUT_ptr,
    N_pairs,
    stride_xb, stride_xs, stride_xh,  # Input strides (may be non-contiguous)
    cos_stride_s, cos_stride_d,
    HALF_D: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused RoPE forward. Reads from possibly non-contiguous input,
    writes to contiguous output. Loads cos/sin from [1, S, 1, D] via strides."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_pairs

    # Decompose flat pair index → (b, s, h, d_pair)
    d_pair = offsets % HALF_D
    remainder = offsets // HALF_D
    h = remainder % H
    remainder2 = remainder // H
    s = remainder2 % S
    b = remainder2 // S

    # Input: read from non-contiguous tensor using strides
    even_d = d_pair * 2
    odd_d = even_d + 1
    x_base = b * stride_xb + s * stride_xs + h * stride_xh
    x0 = tl.load(X_ptr + x_base + even_d, mask=mask, other=0.0)
    x1 = tl.load(X_ptr + x_base + odd_d, mask=mask, other=0.0)

    # cos/sin from original [1, S, 1, D]
    cos_off = s * cos_stride_s + even_d * cos_stride_d
    c = tl.load(COS_ptr + cos_off, mask=mask, other=1.0)
    sn = tl.load(SIN_ptr + cos_off, mask=mask, other=0.0)

    out0 = x0 * c - x1 * sn
    out1 = x1 * c + x0 * sn

    # Output: write to contiguous flat buffer
    out_even = offsets * 2
    tl.store(OUT_ptr + out_even, out0, mask=mask)
    tl.store(OUT_ptr + out_even + 1, out1, mask=mask)


@triton.jit
def _rope_bwd_kernel(
    GRAD_ptr, COS_ptr, SIN_ptr, GRAD_X_ptr,
    N_pairs,
    cos_stride_s, cos_stride_d,
    HALF_D: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused RoPE backward (inverse rotation). Both input and output contiguous."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_pairs

    d_pair = offsets % HALF_D
    s = (offsets // (HALF_D * H)) % S

    even_off = offsets * 2
    odd_off = even_off + 1

    g0 = tl.load(GRAD_ptr + even_off, mask=mask, other=0.0)
    g1 = tl.load(GRAD_ptr + odd_off, mask=mask, other=0.0)

    cos_off = s * cos_stride_s + d_pair * 2 * cos_stride_d
    c = tl.load(COS_ptr + cos_off, mask=mask, other=1.0)
    sn = tl.load(SIN_ptr + cos_off, mask=mask, other=0.0)

    dx0 = g0 * c + g1 * sn
    dx1 = g1 * c - g0 * sn

    tl.store(GRAD_X_ptr + even_off, dx0, mask=mask)
    tl.store(GRAD_X_ptr + odd_off, dx1, mask=mask)


BLOCK_SIZE = 1024


class _FusedRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xq, xk, freqs_cos, freqs_sin):
        # xq, xk: [B, S, H, D] (may be non-contiguous from split+view)
        # freqs_cos/sin: [1, S, 1, D]
        B, S, H, D = xq.shape
        HALF_D = D // 2

        cos_stride_s = freqs_cos.stride(1)
        cos_stride_d = freqs_cos.stride(3)

        n_pairs = B * S * H * HALF_D
        grid = ((n_pairs + BLOCK_SIZE - 1) // BLOCK_SIZE,)

        # Output is always contiguous
        xq_out = torch.empty(B, S, H, D, dtype=xq.dtype, device=xq.device)
        xk_out = torch.empty(B, S, H, D, dtype=xk.dtype, device=xk.device)

        _rope_fwd_kernel[grid](
            xq, freqs_cos, freqs_sin, xq_out,
            n_pairs,
            xq.stride(0), xq.stride(1), xq.stride(2),
            cos_stride_s, cos_stride_d,
            HALF_D=HALF_D, H=H, S=S, BLOCK_SIZE=BLOCK_SIZE,
        )
        _rope_fwd_kernel[grid](
            xk, freqs_cos, freqs_sin, xk_out,
            n_pairs,
            xk.stride(0), xk.stride(1), xk.stride(2),
            cos_stride_s, cos_stride_d,
            HALF_D=HALF_D, H=H, S=S, BLOCK_SIZE=BLOCK_SIZE,
        )

        ctx.save_for_backward(freqs_cos, freqs_sin)
        ctx.shape_q = xq.shape
        ctx.shape_k = xk.shape
        ctx.n_pairs = n_pairs
        return xq_out, xk_out

    @staticmethod
    def backward(ctx, grad_q, grad_k):
        freqs_cos, freqs_sin = ctx.saved_tensors
        B, S, H, D = ctx.shape_q
        HALF_D = D // 2
        n_pairs = ctx.n_pairs

        cos_stride_s = freqs_cos.stride(1)
        cos_stride_d = freqs_cos.stride(3)

        grid = ((n_pairs + BLOCK_SIZE - 1) // BLOCK_SIZE,)

        # Gradients are contiguous (from attention kernel output)
        gq_flat = grad_q.contiguous().view(-1)
        gk_flat = grad_k.contiguous().view(-1)

        grad_xq = torch.empty_like(gq_flat)
        grad_xk = torch.empty_like(gk_flat)

        _rope_bwd_kernel[grid](
            gq_flat, freqs_cos, freqs_sin, grad_xq,
            n_pairs, cos_stride_s, cos_stride_d,
            HALF_D=HALF_D, H=H, S=S, BLOCK_SIZE=BLOCK_SIZE,
        )
        _rope_bwd_kernel[grid](
            gk_flat, freqs_cos, freqs_sin, grad_xk,
            n_pairs, cos_stride_s, cos_stride_d,
            HALF_D=HALF_D, H=H, S=S, BLOCK_SIZE=BLOCK_SIZE,
        )

        return grad_xq.view(ctx.shape_q), grad_xk.view(ctx.shape_k), None, None


def fused_apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop-in replacement for transformer_torch.apply_rotary_emb."""
    return _FusedRoPE.apply(xq, xk, freqs_cos, freqs_sin)
