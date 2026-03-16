"""Diagnostic: does padding leak through attention into pooled embeddings?

If attention_mask is correctly applied, randomizing invalid-peak values
should produce (nearly) identical pooled embeddings.  If padding leaks,
the embeddings will change substantially.
"""

import torch
from models.model import PeakSetSIGReg


def _make_batch(batch_size: int = 8, num_peaks: int = 60, valid_count: int = 30):
    """Batch where the last (num_peaks - valid_count) positions are padding."""
    peak_mz = torch.rand(batch_size, num_peaks).sort(dim=-1).values
    peak_intensity = torch.rand(batch_size, num_peaks)
    peak_valid_mask = torch.zeros(batch_size, num_peaks, dtype=torch.bool)
    peak_valid_mask[:, :valid_count] = True
    # Padding positions are zero (matching real pipeline behavior)
    peak_mz[:, valid_count:] = 0.0
    peak_intensity[:, valid_count:] = 0.0
    precursor_mz = torch.rand(batch_size) * 500 + 100
    return {
        "peak_mz": peak_mz,
        "peak_intensity": peak_intensity,
        "peak_valid_mask": peak_valid_mask,
        "precursor_mz": precursor_mz,
    }


@torch.no_grad()
def test_padding_leakage():
    """Core test: randomize padding positions and check embedding stability."""
    torch.manual_seed(42)
    model = PeakSetSIGReg(
        model_dim=64,
        encoder_num_layers=4,
        encoder_num_heads=4,
        feature_mlp_hidden_dim=32,
    )
    model.eval()

    batch = _make_batch(batch_size=8, num_peaks=60, valid_count=30)
    peak_mz = batch["peak_mz"]
    peak_intensity = batch["peak_intensity"]
    peak_valid_mask = batch["peak_valid_mask"]

    # --- Run 1: original padding (zeros) ---
    emb1 = model.encoder(peak_mz, peak_intensity, valid_mask=peak_valid_mask)
    pooled1 = model.pool(emb1, peak_valid_mask)

    # --- Run 2: randomize ONLY padding positions ---
    peak_mz_rand = peak_mz.clone()
    peak_intensity_rand = peak_intensity.clone()
    peak_mz_rand[:, 30:] = torch.rand_like(peak_mz[:, 30:]) * 1000  # large random mz
    peak_intensity_rand[:, 30:] = torch.rand_like(
        peak_intensity[:, 30:]
    )  # random intensity

    emb2 = model.encoder(peak_mz_rand, peak_intensity_rand, valid_mask=peak_valid_mask)
    pooled2 = model.pool(emb2, peak_valid_mask)

    # --- Run 3: set padding to different constant values ---
    peak_mz_const = peak_mz.clone()
    peak_intensity_const = peak_intensity.clone()
    peak_mz_const[:, 30:] = 999.0
    peak_intensity_const[:, 30:] = 1.0

    emb3 = model.encoder(
        peak_mz_const, peak_intensity_const, valid_mask=peak_valid_mask
    )
    pooled3 = model.pool(emb3, peak_valid_mask)

    # --- Measure changes ---
    # Per-sample cosine similarity
    cos_sim_12 = torch.nn.functional.cosine_similarity(pooled1, pooled2, dim=-1)
    cos_sim_13 = torch.nn.functional.cosine_similarity(pooled1, pooled3, dim=-1)

    # Per-sample L2 distance (normalized by embedding norm)
    norm1 = pooled1.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    rel_l2_12 = (pooled1 - pooled2).norm(dim=-1) / norm1.squeeze()
    rel_l2_13 = (pooled1 - pooled3).norm(dim=-1) / norm1.squeeze()

    print("=" * 70)
    print("PADDING LEAKAGE DIAGNOSTIC")
    print("=" * 70)
    print(f"Model: 4-layer, 64-dim, PMA pooling, {30} valid / {60} total peaks")
    print()
    print("Test: randomize padding positions, check if pooled embeddings change")
    print("If mask is correct: cos_sim ≈ 1.0, relative_l2 ≈ 0.0")
    print("If mask is broken:  cos_sim < 1.0, relative_l2 > 0.0")
    print()
    print(f"Zeros vs Random padding:")
    print(f"  Cosine similarity: {cos_sim_12.mean():.6f} ± {cos_sim_12.std():.6f}")
    print(f"  Relative L2 dist:  {rel_l2_12.mean():.6f} ± {rel_l2_12.std():.6f}")
    print()
    print(f"Zeros vs Constant(999) padding:")
    print(f"  Cosine similarity: {cos_sim_13.mean():.6f} ± {cos_sim_13.std():.6f}")
    print(f"  Relative L2 dist:  {rel_l2_13.mean():.6f} ± {rel_l2_13.std():.6f}")
    print()

    # Also check: do per-position embeddings of VALID peaks change?
    valid_emb1 = emb1[:, :30, :]
    valid_emb2 = emb2[:, :30, :]
    valid_cos = torch.nn.functional.cosine_similarity(
        valid_emb1.reshape(-1, 64), valid_emb2.reshape(-1, 64), dim=-1
    )
    print(f"Valid-peak per-position embedding similarity (zeros vs random padding):")
    print(f"  Cosine similarity: {valid_cos.mean():.6f} ± {valid_cos.std():.6f}")
    print()

    # Verdict
    if cos_sim_12.mean() < 0.99 or rel_l2_12.mean() > 0.01:
        print(">>> VERDICT: PADDING IS LEAKING THROUGH ATTENTION <<<")
        print("    Invalid peaks influence valid peak representations.")
        print("    The attention_mask is NOT being applied.")
    else:
        print(">>> VERDICT: Padding appears properly masked <<<")

    print()

    # Additional: check what the embedder produces for padding (all-zeros input)
    zero_mz = torch.zeros(1, 1)
    zero_int = torch.zeros(1, 1)
    log_int = torch.log1p(zero_int.clamp(min=0.0))
    pad_emb = model.encoder.embedder(torch.stack([zero_mz, zero_int, log_int], dim=-1))
    print(f"Embedding of a zero-padding position:")
    print(f"  Norm: {pad_emb.norm():.4f}")
    print(f"  (Non-zero due to Fourier cos(0)=1 and MLP biases)")

    # After fix: padding should be fully masked, no leakage
    assert cos_sim_12.mean() > 0.9999, (
        f"Padding still leaking: cosine_sim={cos_sim_12.mean():.6f}"
    )


@torch.no_grad()
def test_padding_leakage_mean_pool():
    """Same test but with mean pooling (to check if it's just PMA)."""
    torch.manual_seed(42)
    model = PeakSetSIGReg(
        model_dim=64,
        encoder_num_layers=4,
        encoder_num_heads=4,
        feature_mlp_hidden_dim=32,
    )
    model.eval()

    batch = _make_batch(batch_size=8, num_peaks=60, valid_count=30)

    emb1 = model.encoder(
        batch["peak_mz"],
        batch["peak_intensity"],
        valid_mask=batch["peak_valid_mask"],
    )
    pooled1 = model.pool(emb1, batch["peak_valid_mask"])

    # Randomize padding
    mz2 = batch["peak_mz"].clone()
    int2 = batch["peak_intensity"].clone()
    mz2[:, 30:] = torch.rand_like(mz2[:, 30:]) * 1000
    int2[:, 30:] = torch.rand_like(int2[:, 30:])

    emb2 = model.encoder(
        mz2,
        int2,
        valid_mask=batch["peak_valid_mask"],
    )
    pooled2 = model.pool(emb2, batch["peak_valid_mask"])

    cos_sim = torch.nn.functional.cosine_similarity(pooled1, pooled2, dim=-1)
    rel_l2 = (pooled1 - pooled2).norm(dim=-1) / pooled1.norm(dim=-1).clamp(min=1e-8)

    print()
    print("MEAN POOLING variant:")
    print(f"  Cosine similarity: {cos_sim.mean():.6f} ± {cos_sim.std():.6f}")
    print(f"  Relative L2 dist:  {rel_l2.mean():.6f} ± {rel_l2.std():.6f}")

    if cos_sim.mean() < 0.99:
        print("  >>> Mean pooling also affected (padding leaks through attention)")
    else:
        print("  >>> Mean pooling OK (leakage only affects PMA queries)")


if __name__ == "__main__":
    test_padding_leakage()
    test_padding_leakage_mean_pool()
