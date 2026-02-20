"""Tests for Triton-accelerated ISAB correctness against vanilla implementation."""

from __future__ import annotations

import pytest
import torch

# Skip entire module if no CUDA
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for Triton kernels",
)

from networks.isab_triton import (
    TritonISAB,
    fused_dual_rms_norm,
    fused_expand_residual_rms_norm,
    fused_expand_rms_norm,
    fused_residual_add,
    fused_residual_add_rms_norm,
    fused_residual_rms_norm,
    fused_rms_norm,
    fused_silu,
    fused_silu_inplace,
)
from networks.set_transformer_torch import ISAB

# Config-derived shapes
B = 512
N = 60
D = 256
M = 32  # inducing points
N_HEADS = 8
N_KV_HEADS = 4
ATTN_MLP_MULT = 4.0
NORM_EPS = 1e-5
DTYPE = torch.bfloat16
DEVICE = "cuda"


@pytest.fixture
def rng():
    torch.manual_seed(42)
    return torch.Generator(device=DEVICE).manual_seed(42)


# ---------------------------------------------------------------------------
# Unit tests for individual Triton kernels
# ---------------------------------------------------------------------------

class TestFusedRMSNorm:
    def test_matches_pytorch(self, rng):
        x = torch.randn(B * N, D, device=DEVICE, dtype=DTYPE)
        weight = torch.randn(D, device=DEVICE, dtype=DTYPE)

        # Reference
        norm = torch.nn.RMSNorm(D, eps=NORM_EPS).to(DEVICE, DTYPE)
        with torch.no_grad():
            norm.weight.copy_(weight)
        ref = norm(x)

        # Triton
        out = fused_rms_norm(x, weight, eps=NORM_EPS)

        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    def test_3d_input(self, rng):
        x = torch.randn(B, N, D, device=DEVICE, dtype=DTYPE)
        weight = torch.randn(D, device=DEVICE, dtype=DTYPE)

        norm = torch.nn.RMSNorm(D, eps=NORM_EPS).to(DEVICE, DTYPE)
        with torch.no_grad():
            norm.weight.copy_(weight)
        ref = norm(x)
        out = fused_rms_norm(x, weight, eps=NORM_EPS)

        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


class TestFusedDualRMSNorm:
    def test_matches_two_separate_norms(self, rng):
        x = torch.randn(B, N, D, device=DEVICE, dtype=DTYPE)
        w1 = torch.randn(D, device=DEVICE, dtype=DTYPE)
        w2 = torch.randn(D, device=DEVICE, dtype=DTYPE)

        ref1 = fused_rms_norm(x, w1, eps=NORM_EPS)
        ref2 = fused_rms_norm(x, w2, eps=NORM_EPS)

        out1, out2 = fused_dual_rms_norm(x, w1, w2, eps=NORM_EPS)

        torch.testing.assert_close(out1, ref1, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(out2, ref2, atol=2e-2, rtol=2e-2)


class TestFusedExpandRMSNorm:
    def test_matches_expand_then_norm(self, rng):
        inducing = torch.randn(M, D, device=DEVICE, dtype=DTYPE)
        weight = torch.randn(D, device=DEVICE, dtype=DTYPE)

        # Reference: expand then norm
        expanded = inducing.unsqueeze(0).expand(B, -1, -1).contiguous()
        ref = fused_rms_norm(expanded, weight, eps=NORM_EPS)

        out = fused_expand_rms_norm(inducing, weight, B, eps=NORM_EPS)

        torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


class TestFusedExpandResidualRMSNorm:
    def test_matches_expand_add_norm(self, rng):
        inducing = torch.randn(M, D, device=DEVICE, dtype=DTYPE)
        attn_out = torch.randn(B, M, D, device=DEVICE, dtype=DTYPE)
        weight = torch.randn(D, device=DEVICE, dtype=DTYPE)

        # Reference: expand, add, then norm
        expanded = inducing.unsqueeze(0).expand(B, -1, -1).contiguous()
        h_ref = (expanded.float() + attn_out.float()).to(DTYPE)
        normed_ref = fused_rms_norm(h_ref, weight, eps=NORM_EPS)

        normed, h = fused_expand_residual_rms_norm(inducing, attn_out, weight, B, eps=NORM_EPS)

        torch.testing.assert_close(h, h_ref, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(normed, normed_ref, atol=1e-2, rtol=1e-2)


class TestFusedSiLUInplace:
    def test_matches_pytorch(self, rng):
        x = torch.randn(B * M, 1024, device=DEVICE, dtype=DTYPE)
        ref = torch.nn.functional.silu(x)
        # Warmup to trigger autotuner (which runs kernel multiple times in-place)
        dummy = torch.randn_like(x)
        fused_silu_inplace(dummy)
        # Now test with fresh data
        x_inplace = x.clone()
        fused_silu_inplace(x_inplace)
        torch.testing.assert_close(x_inplace, ref, atol=1e-2, rtol=1e-2)


class TestFusedResidualRMSNorm:
    def test_matches_sequential(self, rng):
        residual = torch.randn(B, N, D, device=DEVICE, dtype=DTYPE)
        attn_out = torch.randn(B, N, D, device=DEVICE, dtype=DTYPE)
        weight = torch.randn(D, device=DEVICE, dtype=DTYPE)

        # Reference
        h_ref = residual.float() + attn_out.float()
        h_ref_bf16 = h_ref.to(DTYPE)
        normed_ref = fused_rms_norm(h_ref_bf16, weight, eps=NORM_EPS)

        normed, h = fused_residual_rms_norm(residual, attn_out, weight, eps=NORM_EPS)

        torch.testing.assert_close(h, h_ref_bf16, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(normed, normed_ref, atol=1e-2, rtol=1e-2)


class TestFusedResidualAddRMSNorm:
    def test_matches_sequential(self, rng):
        residual = torch.randn(B, M, D, device=DEVICE, dtype=DTYPE)
        ffn_out = torch.randn(B, M, D, device=DEVICE, dtype=DTYPE)
        weight = torch.randn(D, device=DEVICE, dtype=DTYPE)

        # Reference: add then norm
        h_ref = (residual.float() + ffn_out.float()).to(DTYPE)
        normed_ref = fused_rms_norm(h_ref, weight, eps=NORM_EPS)

        normed = fused_residual_add_rms_norm(residual, ffn_out, weight, eps=NORM_EPS)
        torch.testing.assert_close(normed, normed_ref, atol=1e-2, rtol=1e-2)


class TestFusedSiLU:
    def test_matches_pytorch(self, rng):
        x = torch.randn(B * M, 1024, device=DEVICE, dtype=DTYPE)

        ref = torch.nn.functional.silu(x)
        out = fused_silu(x)

        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


class TestFusedResidualAdd:
    def test_matches_add(self, rng):
        a = torch.randn(B, N, D, device=DEVICE, dtype=DTYPE)
        b = torch.randn(B, N, D, device=DEVICE, dtype=DTYPE)

        ref = (a.float() + b.float()).to(DTYPE)
        out = fused_residual_add(a, b)

        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# Integration tests: TritonISAB vs vanilla ISAB
# ---------------------------------------------------------------------------

class TestTritonISABCorrectness:
    @pytest.fixture
    def vanilla_isab(self):
        torch.manual_seed(42)
        isab = ISAB(
            dim=D,
            num_inducing_points=M,
            n_heads=N_HEADS,
            n_kv_heads=N_KV_HEADS,
            attention_mlp_multiple=ATTN_MLP_MULT,
            norm_eps=NORM_EPS,
        ).to(DEVICE, DTYPE).eval()
        return isab

    @pytest.fixture
    def triton_isab(self, vanilla_isab):
        return TritonISAB.from_vanilla_isab(vanilla_isab).eval()

    def test_weight_transfer(self, vanilla_isab, triton_isab):
        """Verify that from_vanilla_isab copies all weights correctly."""
        assert torch.equal(triton_isab.inducing_points, vanilla_isab.inducing_points)
        assert torch.equal(triton_isab.mab1_wq.weight, vanilla_isab.mab1.cross_attn.wq.weight)
        assert torch.equal(triton_isab.mab1_wkv.weight, vanilla_isab.mab1.cross_attn.wkv.weight)
        assert torch.equal(triton_isab.mab1_wo.weight, vanilla_isab.mab1.cross_attn.wo.weight)
        assert torch.equal(triton_isab.mab2_wq.weight, vanilla_isab.mab2.cross_attn.wq.weight)

    def test_output_shape(self, triton_isab, rng):
        x = torch.randn(B, N, D, device=DEVICE, dtype=DTYPE)
        with torch.no_grad():
            out = triton_isab(x)
        assert out.shape == (B, N, D)
        assert out.dtype == DTYPE

    def test_forward_matches_vanilla(self, vanilla_isab, triton_isab, rng):
        """Full forward pass correctness: TritonISAB vs vanilla ISAB."""
        x = torch.randn(B, N, D, device=DEVICE, dtype=DTYPE)

        with torch.no_grad():
            ref = vanilla_isab(x, kv_block_mask=None, q_block_mask=None)
            out = triton_isab(x)

        max_diff = (ref.float() - out.float()).abs().max().item()
        mean_diff = (ref.float() - out.float()).abs().mean().item()

        print(f"\nTritonISAB vs vanilla ISAB:")
        print(f"  max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")

        # bf16 tolerance: flex_attention may have slight numerical differences
        torch.testing.assert_close(out, ref, atol=5e-2, rtol=5e-2)

    def test_forward_small_batch(self, vanilla_isab, triton_isab):
        """Test with smaller batch size."""
        x = torch.randn(4, N, D, device=DEVICE, dtype=DTYPE)
        with torch.no_grad():
            ref = vanilla_isab(x, kv_block_mask=None, q_block_mask=None)
            out = triton_isab(x)
        torch.testing.assert_close(out, ref, atol=5e-2, rtol=5e-2)

    def test_deterministic(self, triton_isab, rng):
        """Two calls with same input should give same output."""
        x = torch.randn(B, N, D, device=DEVICE, dtype=DTYPE)
        with torch.no_grad():
            out1 = triton_isab(x)
            out2 = triton_isab(x)
        torch.testing.assert_close(out1, out2, atol=0, rtol=0)


class TestTritonISABBenchmark:
    """Benchmark tests (not run by default, use -k benchmark)."""

    @pytest.fixture
    def vanilla_isab(self):
        torch.manual_seed(42)
        return ISAB(
            dim=D, num_inducing_points=M, n_heads=N_HEADS,
            n_kv_heads=N_KV_HEADS, attention_mlp_multiple=ATTN_MLP_MULT,
        ).to(DEVICE, DTYPE).eval()

    @pytest.fixture
    def triton_isab_bench(self, vanilla_isab):
        return TritonISAB.from_vanilla_isab(vanilla_isab).eval()

    @pytest.mark.benchmark
    def test_benchmark_vanilla(self, vanilla_isab, benchmark):
        x = torch.randn(B, N, D, device=DEVICE, dtype=DTYPE)

        def run():
            with torch.no_grad():
                vanilla_isab(x, kv_block_mask=None, q_block_mask=None)
            torch.cuda.synchronize()

        # Warmup
        for _ in range(3):
            run()
        benchmark(run)

    @pytest.mark.benchmark
    def test_benchmark_triton(self, triton_isab_bench, benchmark):
        x = torch.randn(B, N, D, device=DEVICE, dtype=DTYPE)

        def run():
            with torch.no_grad():
                triton_isab_bench(x)
            torch.cuda.synchronize()

        for _ in range(3):
            run()
        benchmark(run)
