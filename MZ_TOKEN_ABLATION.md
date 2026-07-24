# Fourier vs. tokenized m/z input

## Decision

Keep the Fourier peak m/z embedder. In the controlled 52M-parameter,
2,500-step comparison, Fourier achieved higher exact-bin m/z accuracy on all
three large paired holdouts and lower m/z cross-entropy on every seed.

Across seeds, Fourier averaged 0.7866% exact 0.5 Da-bin accuracy and the flat
token lookup averaged 0.6963%. The token model was lower by 0.0903 percentage
points, or 11.5% relatively. The token representation also learned much less
local m/z smoothness and a substantially lower-rank full peak embedding.

The token model did improve intensity prediction, but that is not the target
of this ablation. It did not improve the requested m/z metric or total loss.

## What was compared

The experiment changes only the per-peak m/z input representation.
The complete equations, dimensions, target construction, and live 100K
comparison protocol are documented in `MZ_EMBEDDER_MATHEMATICS.md`.

- **Fourier:** 64 fixed log-spaced frequencies, giving 128 sin/cos features,
  followed by a four-layer m/z MLP with hidden width 896. The shared raw
  feature branch also receives continuous m/z.
- **Token:** one flat 50,000-entry vocabulary at 0.02 Da resolution. A learned
  38-dimensional token embedding is projected to the 224-dimensional m/z
  branch, analogous to an ALBERT-style factorized BERT embedding. The raw
  feature branch receives zero in its m/z slot, so there is no continuous m/z
  path in the peak embedder.

Both models retain the identical pairwise pathway, including its existing
mass-difference Fourier features. Holding that pathway fixed isolates the peak
embedder; this is not an experiment that removes every Fourier feature from
the whole model.

The prediction contract is exactly the same for both variants:

- m/z target range: 0–1,000 Da
- target bin width: 0.5 Da
- output classes: 2,000
- primary metric: exact top-1 target-bin accuracy
- secondary m/z metric: the same masked cross-entropy training loss

Both variants use the same initialized prediction head and all non-embedder
parameters, data order, masks, optimizer, schedule, and validation batches.
The 0.02 Da token input resolution does not change the common 0.5 Da
prediction target.

## Controlled protocol

Training used the 10-layer, width-448 fastmixer-dense MAE with 47 input peaks
on both local H100s. Each seed trained for 2,500 optimizer steps at global
batch 512: 1.28 million spectra and approximately 4.0e14 estimated FLOPs per
run. This preserves the compute scale of the earlier 128M/1,000-step
comparison while providing 2.5 times as many optimizer updates.

Optimization used fused `torch.optim.Adam`, zero weight decay, betas
0.9/0.95, a 250-step warmup to 3e-4, and decay to 3e-5. Training used BF16,
local microbatch 128 per GPU, and gradient accumulation 2.

| Setting | Fourier | Token |
| --- | ---: | ---: |
| Trainable parameters | 52,044,810 | 52,029,610 |
| Estimated FLOPs / optimizer step | 159,881,656,320 | 159,834,961,920 |
| Estimated FLOPs / run | 399,704,140,800,000 | 399,587,404,800,000 |
| Mean optimizer steps/s | 1.9845 | 1.9905 |
| Mean peak allocated memory / GPU | 78.82 GB | 78.68 GB |

The parameter and FLOP differences are both 0.029%. Seeds 66, 67, and 68
were run for each variant. Scheduled validation used 16,384 spectra at steps
500, 1,000, 1,500, 2,000, and 2,500. A post-training paired evaluation gave
both checkpoints the same 131,072 held-out spectra and masks for each seed.

## Results

### Training curve

The mean scheduled-validation m/z results across the three seeds favored
Fourier at every checkpoint:

| Step | Fourier accuracy | Token accuracy | Fourier CE | Token CE |
| ---: | ---: | ---: | ---: | ---: |
| 500 | 0.2113% | 0.1924% | 7.2169 | 7.2693 |
| 1,000 | 0.3392% | 0.2908% | 6.9439 | 6.9957 |
| 1,500 | 0.4328% | 0.4099% | 6.8587 | 6.8885 |
| 2,000 | 0.5589% | 0.4973% | 6.7662 | 6.7922 |
| 2,500 | 0.5905% | 0.5252% | 6.7312 | 6.7603 |

The longer run matters: much of the exact-bin accuracy appears after the
1,000-step point.

### Large paired holdout

| Seed | Fourier accuracy | Token accuracy | Token − Fourier | Relative token change |
| ---: | ---: | ---: | ---: | ---: |
| 66 | 0.8105% | 0.6359% | -0.1746 pp | -21.5% |
| 67 | 0.7516% | 0.6944% | -0.0572 pp | -7.6% |
| 68 | 0.7976% | 0.7586% | -0.0390 pp | -4.9% |
| Mean | **0.7866%** | **0.6963%** | **-0.0903 pp** | **-11.5%** |

All three accuracy deltas favor Fourier. With only three independent seeds,
the seed-level 95% t interval for the accuracy delta is wide and crosses zero
(-0.2731 to +0.0925 percentage points), so the exact population effect size is
not tightly estimated.

The smoother m/z cross-entropy result is more precise:

| Metric | Fourier mean | Token mean | Token − Fourier | Seed-level 95% CI |
| --- | ---: | ---: | ---: | ---: |
| m/z cross-entropy | 6.4123 | 6.4367 | +0.0244 | [+0.0154, +0.0334] |
| Total loss | 12.8900 | 12.9038 | +0.0138 | [+0.0028, +0.0248] |
| Intensity accuracy | 89.3025% | 89.8049% | +0.5024 pp | [+0.3826, +0.6221] pp |

Thus the token model consistently trades better intensity prediction for
worse m/z prediction. Total loss also favors Fourier.

## Learned embedding geometry

Geometry was evaluated over the full 0–1,000 Da range at 0.02 Da spacing and
averaged across the three trained seeds.

| Metric | Fourier | Token |
| --- | ---: | ---: |
| m/z-branch cosine at 0.02 Da separation | 0.737 | 0.330 |
| m/z-branch cosine at 0.1 Da separation | 0.580 | 0.257 |
| m/z-branch cosine at 1 Da separation | 0.520 | 0.263 |
| m/z-branch entropy effective rank | 45.12 | 32.09 |
| Full peak-embedding entropy effective rank | 38.95 | 15.99 |
| Full peak-embedding participation rank | 25.42 | 7.20 |

The flat lookup does not encode adjacency: neighboring token IDs begin as
unrelated vectors and training does not recover the strong local continuity
provided by Fourier features. Its full peak embedding uses less than half the
entropy effective rank of the Fourier model. These geometry differences are
consistent across all three seeds.

## Recommendation and scope

Use the Fourier peak m/z embedder for the next scaling runs. The tested flat,
factorized token vocabulary is slightly faster and smaller, but it is worse on
the common discrete m/z target after equal steps and essentially equal FLOPs.

This result applies to a 52M model, a 0.02 Da flat input vocabulary, a 0.5 Da
output target, and 2,500 optimizer steps. It directly tests the peak embedder,
not removal of the shared pairwise Fourier pathway. A future token approach
would need an explicit source of mass locality—rather than treating adjacent
m/z bins like unrelated word IDs—to address the failure mode observed here.

Raw run artifacts and the machine-readable aggregate are under
`experiments/mz_token_ablation/`. The canonical configuration is
`configs/mz_token_ablation.py`;
`scripts/compare_mz_token_checkpoints.py` performs the paired holdout, and
`scripts/analyze_mz_token_ablation.py` regenerates the aggregate and geometry
statistics.
