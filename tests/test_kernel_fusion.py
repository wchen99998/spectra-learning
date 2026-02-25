"""Correctness and benchmark tests for kernel fusion optimizations.

Run:
    python tests/test_kernel_fusion.py
    python tests/test_kernel_fusion.py --benchmark   # include timing benchmarks
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.losses import SIGReg
from models.model import PeakSetEncoder, PeakSetSIGReg
from networks.transformer_torch import Attention, TransformerBlock

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
B, N, D = 32, 60, 128  # batch, peaks, dim (small model for testing)


def _make_batch(batch_size: int = B, num_peaks: int = N, device: str = DEVICE) -> dict[str, torch.Tensor]:
    """Create a synthetic batch matching the data contract."""
    # Simulate variable-length sequences: last 10-20 peaks are invalid
    valid_lengths = torch.randint(num_peaks // 2, num_peaks, (batch_size,))
    peak_valid_mask = torch.arange(num_peaks).unsqueeze(0) < valid_lengths.unsqueeze(1)
    peak_mz = torch.rand(batch_size, num_peaks, device=device) * peak_valid_mask.float().to(device)
    peak_intensity = torch.rand(batch_size, num_peaks, device=device) * peak_valid_mask.float().to(device)
    precursor_mz = torch.rand(batch_size, device=device) * 0.5
    return {
        "peak_mz": peak_mz,
        "peak_intensity": peak_intensity,
        "peak_valid_mask": peak_valid_mask.to(device),
        "precursor_mz": precursor_mz,
    }


# ---------------------------------------------------------------------------
# 1. Attention mask correctness
# ---------------------------------------------------------------------------

def test_attention_mask_plumbing():
    """Verify block_mask parameter correctly masks out padding positions.

    Uses create_padding_block_mask to build a BlockMask from a valid_mask tensor
    and verifies that masked outputs differ from unmasked (padding is excluded).
    """
    print("Test 1: Attention block_mask ... ", end="", flush=True)
    if DEVICE != "cuda":
        print("SKIPPED (flex_attention requires CUDA)")
        return

    from networks.transformer_torch import create_padding_block_mask

    torch.manual_seed(42)
    attn = Attention(D, n_heads=4).to(DEVICE).eval()

    x = torch.randn(2, N, D, device=DEVICE)
    valid_mask = torch.ones(2, N, dtype=torch.bool, device=DEVICE)
    valid_mask[:, N // 2:] = False
    block_mask = create_padding_block_mask(valid_mask)

    with torch.no_grad():
        out_masked = attn(x, block_mask=block_mask)
        out_no_mask = attn(x, block_mask=None)

    # Both should be finite and same shape
    assert out_masked.shape == (2, N, D)
    assert torch.isfinite(out_masked).all(), "Masked output has non-finite values"
    assert torch.isfinite(out_no_mask).all(), "Unmasked output has non-finite values"

    # Outputs should differ because padding is now masked
    assert not torch.allclose(out_masked, out_no_mask, atol=1e-5), \
        "Outputs should differ when block_mask masks out padding"

    print("PASSED")


def test_attention_output_shapes():
    """Verify flex_attention produces correct output shapes under torch.compile."""
    print("Test 2: Attention output shapes ... ", end="", flush=True)
    if DEVICE != "cuda":
        print("SKIPPED (flex_attention requires CUDA)")
        return

    torch.manual_seed(42)
    attn = Attention(D, n_heads=4).to(DEVICE).eval()

    x = torch.randn(4, N, D, device=DEVICE)

    @torch.compile
    def run(x):
        return attn(x, block_mask=None)

    with torch.no_grad():
        out = run(x)
        assert out.shape == (4, N, D)
        assert torch.isfinite(out).all()

    print("PASSED")


# ---------------------------------------------------------------------------
# 2. SIGReg Loss correctness
# ---------------------------------------------------------------------------

def test_sigreg_forward():
    """Verify SIGReg forward produces reasonable outputs."""
    print("Test 3: SIGReg forward pass ... ", end="", flush=True)
    torch.manual_seed(42)
    sigreg = SIGReg(num_slices=256).to(DEVICE)

    proj = torch.randn(4, B, D, device=DEVICE)  # [V, B, D]

    result = sigreg(proj)

    assert result.ndim == 0, f"Expected scalar, got shape {result.shape}"
    assert torch.isfinite(result), f"Result is not finite: {result}"

    print(f"PASSED (statistic={result.item():.4f})")


# ---------------------------------------------------------------------------
# 3. Encoder with mask
# ---------------------------------------------------------------------------

def test_encoder_with_mask():
    """Verify PeakSetEncoder applies valid_mask correctly via BlockMask.

    With block_mask applied, padding positions should not influence valid
    peak embeddings — outputs should differ from the no-mask case.
    """
    print("Test 4: Encoder with valid_mask ... ", end="", flush=True)
    torch.manual_seed(42)
    encoder = PeakSetEncoder(
        model_dim=D,
        num_layers=2,
        num_heads=4,
        feature_mlp_hidden_dim=64,
    ).to(DEVICE).eval()

    batch = _make_batch()

    with torch.no_grad():
        out_mask = encoder(
            batch["peak_mz"], batch["peak_intensity"],
            valid_mask=batch["peak_valid_mask"],
            precursor_mz=batch["precursor_mz"],
        )
        out_none = encoder(
            batch["peak_mz"], batch["peak_intensity"],
            valid_mask=None,
            precursor_mz=batch["precursor_mz"],
        )

    assert out_mask.shape == (B, N, D)
    assert torch.isfinite(out_mask).all(), "Encoder output has non-finite values"
    assert torch.isfinite(out_none).all(), "No-mask output has non-finite values"
    # Outputs should differ because padding is now properly masked
    assert not torch.allclose(out_mask, out_none, atol=1e-3), \
        "Outputs should differ when valid_mask is applied vs not"

    print(f"PASSED (shape={out_mask.shape})")


# ---------------------------------------------------------------------------
# 4. Full model forward
# ---------------------------------------------------------------------------

def test_full_model_forward():
    """Verify PeakSetSIGReg forward with all optimizations."""
    print("Test 5: Full model forward ... ", end="", flush=True)
    torch.manual_seed(42)
    model = PeakSetSIGReg(
        num_peaks=N,
        model_dim=D,
        encoder_num_layers=2,
        encoder_num_heads=4,
        feature_mlp_hidden_dim=64,
        sigreg_proj_hidden_dim=256,
        sigreg_proj_output_dim=64,
        sigreg_num_slices=128,
        multicrop_num_global_views=2,
        multicrop_num_local_views=2,
    ).to(DEVICE).eval()

    batch = _make_batch()

    with torch.no_grad():
        result = model(batch, train=True)

    assert "loss" in result
    assert "sigreg_loss" in result
    assert "token_sigreg_loss" in result
    assert "local_global_l1_loss" in result
    assert "valid_fraction" in result
    assert "representation_variance" in result

    for key, val in result.items():
        assert torch.isfinite(val), f"{key} is not finite: {val}"

    print(f"PASSED (loss={result['loss'].item():.4f})")


def test_full_model_encode():
    """Verify encode() passes valid_mask correctly."""
    print("Test 6: Model encode path ... ", end="", flush=True)
    torch.manual_seed(42)
    model = PeakSetSIGReg(
        num_peaks=N,
        model_dim=D,
        encoder_num_layers=2,
        encoder_num_heads=4,
        feature_mlp_hidden_dim=64,
        sigreg_proj_hidden_dim=256,
        sigreg_proj_output_dim=64,
    ).to(DEVICE).eval()

    batch = _make_batch()

    with torch.no_grad():
        z = model.encode(batch, train=False)

    assert z.shape == (B, 64), f"Unexpected encode shape: {z.shape}"
    assert torch.isfinite(z).all(), "Encoded output has non-finite values"

    print(f"PASSED (shape={z.shape})")


# ---------------------------------------------------------------------------
# 5. Gradient flow
# ---------------------------------------------------------------------------

def test_gradient_flow():
    """Verify gradients flow through the full model with masks."""
    print("Test 7: Gradient flow ... ", end="", flush=True)
    torch.manual_seed(42)
    model = PeakSetSIGReg(
        num_peaks=N,
        model_dim=D,
        encoder_num_layers=2,
        encoder_num_heads=4,
        feature_mlp_hidden_dim=64,
        sigreg_proj_hidden_dim=256,
        sigreg_proj_output_dim=64,
        sigreg_num_slices=128,
        multicrop_num_global_views=2,
        multicrop_num_local_views=2,
    ).to(DEVICE)

    batch = _make_batch()
    result = model(batch, train=True)
    result["loss"].backward()

    # Check that encoder parameters have gradients
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total = sum(1 for p in model.parameters())
    assert has_grad > 0, "No parameters received gradients"

    # Check no NaN gradients
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NaN gradient in {name}"

    print(f"PASSED ({has_grad}/{total} params with grads)")


# ---------------------------------------------------------------------------
# 6. torch.compile compatibility
# ---------------------------------------------------------------------------

def test_compile_forward():
    """Verify the forward pass compiles without graph breaks."""
    print("Test 8: torch.compile compatibility ... ", end="", flush=True)
    if DEVICE != "cuda":
        print("SKIPPED (no CUDA)")
        return

    torch.manual_seed(42)
    model = PeakSetSIGReg(
        num_peaks=N,
        model_dim=D,
        encoder_num_layers=2,
        encoder_num_heads=4,
        feature_mlp_hidden_dim=64,
        sigreg_proj_hidden_dim=256,
        sigreg_proj_output_dim=64,
        sigreg_num_slices=128,
        multicrop_num_global_views=2,
        multicrop_num_local_views=2,
    ).to(DEVICE).eval()

    def forward_fn(batch):
        return model(batch, train=True)

    compiled_fn = torch.compile(forward_fn, mode="reduce-overhead", fullgraph=True)

    batch = _make_batch()

    with torch.no_grad():
        # Warmup compilation
        result = compiled_fn(batch)

    assert torch.isfinite(result["loss"]), "Compiled forward produced non-finite loss"
    print(f"PASSED (loss={result['loss'].item():.4f})")


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def _benchmark(fn, warmup: int = 10, iters: int = 100, label: str = "") -> float:
    """Benchmark a function, return median time in ms."""
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    median = times[len(times) // 2]
    mean = sum(times) / len(times)
    print(f"  {label}: median={median:.2f}ms, mean={mean:.2f}ms, min={times[0]:.2f}ms")
    return median


def benchmark_attention():
    """Benchmark flex_attention with and without padding mask."""
    print("\n--- Attention Benchmark (flex_attention) ---")
    if DEVICE != "cuda":
        print("  SKIPPED (no CUDA)")
        return

    from networks.transformer_torch import create_padding_block_mask

    torch.manual_seed(42)
    attn = Attention(768, n_heads=12).to(DEVICE).eval()
    x = torch.randn(64, N, 768, device=DEVICE)
    valid_mask = torch.ones(64, N, dtype=torch.bool, device=DEVICE)
    valid_mask[:, 50:] = False
    block_mask = create_padding_block_mask(valid_mask)

    @torch.compile
    def run_no_mask(x):
        return attn(x, block_mask=None)

    @torch.compile
    def run_with_mask(x, bm):
        return attn(x, block_mask=bm)

    with torch.no_grad():
        # Warmup compilation
        run_no_mask(x)
        run_with_mask(x, block_mask)
        _benchmark(lambda: run_no_mask(x), label="flex_attn no mask  ")
        _benchmark(lambda: run_with_mask(x, block_mask), label="flex_attn with mask")


def benchmark_sigreg_loss():
    """Benchmark SIGReg loss."""
    print("\n--- SIGReg Loss Benchmark ---")
    torch.manual_seed(42)
    sigreg = SIGReg(num_slices=256).to(DEVICE)
    proj = torch.randn(8, 256, 128, device=DEVICE)  # [V, B, D]

    with torch.no_grad():
        _benchmark(lambda: sigreg(proj), label="SIGReg loss")


def benchmark_full_model():
    """Benchmark full model forward pass."""
    print("\n--- Full Model Benchmark ---")
    torch.manual_seed(42)
    model = PeakSetSIGReg(
        num_peaks=N,
        model_dim=256,
        encoder_num_layers=4,
        encoder_num_heads=4,
        feature_mlp_hidden_dim=64,
        sigreg_proj_hidden_dim=512,
        sigreg_proj_output_dim=128,
        sigreg_num_slices=256,
        multicrop_num_global_views=2,
        multicrop_num_local_views=2,
    ).to(DEVICE).eval()

    batch = _make_batch(batch_size=64)

    with torch.no_grad():
        _benchmark(lambda: model(batch), label="Forward (eager)")

    if DEVICE == "cuda":
        compiled_model = torch.compile(model, mode="reduce-overhead")
        with torch.no_grad():
            # Warmup
            for _ in range(3):
                compiled_model(batch)
            _benchmark(lambda: compiled_model(batch), label="Forward (compiled)")


def benchmark_compiled_with_mask():
    """Benchmark compiled forward with attention masking vs without."""
    print("\n--- Compiled Forward + Mask Benchmark ---")
    if DEVICE != "cuda":
        print("  SKIPPED (no CUDA)")
        return

    torch.manual_seed(42)
    model = PeakSetSIGReg(
        num_peaks=N,
        model_dim=256,
        encoder_num_layers=4,
        encoder_num_heads=4,
        feature_mlp_hidden_dim=64,
        sigreg_proj_hidden_dim=512,
        sigreg_proj_output_dim=128,
        sigreg_num_slices=256,
        multicrop_num_global_views=2,
        multicrop_num_local_views=2,
    ).to(DEVICE).eval()

    batch = _make_batch(batch_size=64)

    compiled_model = torch.compile(model, mode="reduce-overhead")

    with torch.no_grad():
        # Warmup
        for _ in range(3):
            compiled_model(batch)
        _benchmark(
            lambda: compiled_model(batch),
            label="Compiled + mask",
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    print()

    # Correctness tests
    print("=" * 60)
    print("CORRECTNESS TESTS")
    print("=" * 60)

    test_attention_mask_plumbing()
    test_attention_output_shapes()
    test_sigreg_forward()
    test_encoder_with_mask()
    test_full_model_forward()
    test_full_model_encode()
    test_gradient_flow()
    test_compile_forward()

    print("\nAll correctness tests PASSED!")

    # Benchmarks
    if args.benchmark:
        print("\n" + "=" * 60)
        print("BENCHMARKS")
        print("=" * 60)

        benchmark_attention()
        benchmark_sigreg_loss()
        benchmark_full_model()
        benchmark_compiled_with_mask()


if __name__ == "__main__":
    main()
