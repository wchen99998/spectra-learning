# Fourier vs. discretized m/z embedding

## Decision

Keep the Fourier m/z embedder. In the controlled local 2×H100 comparison,
the discretized alternative produced no repeatable training or validation
benefit, was slightly worse on all three large paired holdouts, and learned a
lower-rank representation with an artificial discontinuity at every 1 Da
boundary.

This result does not support the hypothesis that the current Fourier input
embedding is the cause of the observed 300M/1B/3B scaling plateau. It rules out
a benefit from this specific parameter-matched discretization in the tested
128M, 1,000-step regime; it does not by itself prove that no m/z encoding could
interact with billion-parameter scaling.

## Controlled protocol

The comparison used the 15-layer, width-640, fastmixer-dense MAE with 47 input
peaks. Both variants used the same data order, masks, shared parameter
initialization, global batch, schedule, and plain fused `torch.optim.Adam`
(`weight_decay=0`, betas 0.9/0.95). Training used BF16 on both local H100s.

| Setting | Fourier | Discrete |
| --- | ---: | ---: |
| Trainable parameters | 128,125,738 | 128,125,068 |
| Estimated FLOPs / optimizer step | 393,602,267,136 | 393,600,208,896 |
| Mean measured optimizer steps/s | 1.3375 | 1.3346 |
| Mean peak allocated memory / GPU | 60.55 GB | 60.54 GB |

The 670-parameter difference is 0.00052%; the FLOP difference is about 5 ppm.
Each of seeds 66, 67, and 68 trained for exactly 1,000 optimizer steps at global
batch 512, or 512,000 spectra and approximately 3.936e14 estimated FLOPs per
run. The learning-rate schedule was 100-step warmup to 3e-4 followed by decay
to 3e-5.

The Fourier branch used 64 log-spaced frequencies (128 deterministic
sin/cos features) followed by the same-capacity m/z MLP. The discrete branch
rounded to 0.02 Da and summed learned 1 Da coarse and 0.02 Da residual
embeddings before its m/z MLP. Its raw-m/z branch also received the quantized
value.

An audit of 32,768 production spectra found no within-spectrum peak collisions
at 0.02 Da. Coarser bins were unsafe: 28.6% of spectra collided at 0.05 Da,
47.5% at 0.1 Da, 59.1% at 0.25 Da, and 65.2% at 0.5 Da.

## Results

The scheduled final validation was small (4,096 spectra per run) and tied:
mean loss was 14.7436 for Fourier and 14.7457 for discrete.

A post-training paired evaluation then gave both checkpoints the same 65,536
held-out spectra and masks for each seed:

| Seed | Fourier loss | Discrete loss | Discrete − Fourier |
| ---: | ---: | ---: | ---: |
| 66 | 13.2789 | 13.3243 | +0.0454 |
| 67 | 13.2328 | 13.2425 | +0.0097 |
| 68 | 13.2470 | 13.2524 | +0.0054 |
| Mean | 13.2529 | 13.2731 | **+0.0202 (+0.152%)** |

All three seeds favor Fourier on this evaluation. The mean m/z reconstruction
component accounts for +0.0185 of the +0.0202 total difference. With only three
independent seeds, the seed-level 95% t interval for total loss is wide and
crosses zero (-0.0343 to +0.0746), so the defensible statistical conclusion is
that discretization has no demonstrated benefit, not that the small Fourier
advantage has a precisely known population effect.

## Learned embedding geometry

Geometry was evaluated over the full 0–1,000 Da grid at 0.02 Da spacing and
averaged over the three trained seeds.

| Metric | Fourier | Discrete |
| --- | ---: | ---: |
| Adjacent-bin cosine, within a 1 Da interval | 0.680 | 0.550 |
| Adjacent-bin cosine, crossing a 1 Da boundary | 0.681 | 0.066 |
| m/z-branch entropy effective rank | 45.76 | 28.97 |
| full peak-embedding entropy effective rank | 37.98 | 26.54 |

Fourier has the same local behavior on either side of a 1 Da boundary. The
factorized discrete lookup abruptly swaps both its coarse and residual entries
at that boundary, collapsing adjacent cosine from 0.550 to 0.066. It also uses
about 37% fewer effective m/z-branch dimensions and 30% fewer full-embedding
dimensions. These effects were consistent across all seeds.

## Recommendation and scope

Use the current Fourier embedding for the next scaling runs. Do not replace it
with the tested coarse-plus-residual discrete lookup. If a future discrete
experiment is warranted, it should interpolate between neighboring bins or
retain a continuous within-bin residual so that bin boundaries cannot create
large jumps.

The next scaling investigation should focus on the objective, data diversity,
optimization, and backbone capacity/utilization. A larger, longer paired run
would be required to make a direct causal claim specifically about the 3B
plateau.

Raw run artifacts and the machine-readable aggregate are under
`experiments/mz_embedding_ablation/`. The canonical configuration is
`configs/mz_embedding_ablation.py`; `scripts/compare_mae_checkpoints.py`
performs paired holdout evaluation, and
`scripts/analyze_mz_embedding_ablation.py` regenerates the aggregate and
embedding-geometry statistics.
