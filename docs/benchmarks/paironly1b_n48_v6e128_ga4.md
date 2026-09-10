# 1B pair-only model, 48 total positions, GA4 selective recomputation

Config: [`1b_fastmixer_pair_distogram_multiscale_n48_v6e128_ga4.py`](../../configs/1b_fastmixer_pair_distogram_multiscale_n48_v6e128_ga4.py).

## Model and training settings

- **1,002,931,408 trainable parameters**, plus 32 fixed Fourier parameters.
- **47 physical peaks + CLS = 48 total positions** (`num_peaks=47`). Masked
  training compacts to at most 28 context peaks, 12 targets and CLS (41 positions).
- Pair width **1536**, 15 encoder blocks and five predictor blocks. Each block
  applies outgoing/incoming triangle updates and a SwiGLU transition. Despite
  the inherited FastMixer config name, this dispatch uses `PairOnlyEncoder`
  and `PairOnlyBlock`, with no single stream.
- Pair width 1536, SwiGLU hidden width 4096, pair-feature hidden width 1280,
  and metadata width 256 are all multiples of 256. No explicit matrix-padding
  or packed-projection experiments are retained.
- Metadata schema `massive_v2_acquisition_v1`, shared 28 → 256 → 256 SiLU
  embedding, **AdaLN-Zero**, and 10% metadata dropout. Each encoder block learns
  shift, scale and residual gates for its three branches. This follows the
  SingleMixer MAE block-conditioning pattern. The predictor receives conditioned
  encoder features; it has no separate metadata input. The pair encoder retains
  a plain final pair RMSNorm, unlike SingleMixer's metadata-modulated final norm.
- Sentinel-masked target m/z; 2000 distance bins at 0.5 Da; multiscale CE widths
  `(2, 4, 8, 16, 32, 64)`. MAE loss is disabled.
- **128 v6e chips, global batch 2048, GA4**: four spectra per chip per microstep,
  16 per chip per optimizer update. Selective recomputation covers encoder,
  predictor and distogram. Existing microbatch loss weighting is preserved.
- JAX direct training loop, fp32 parameters and gradient accumulation, bf16
  compute, bf16 replicated Adam moments and bf16 gradient all-reduce. No ZeRO-1.
- Learning rate 6e-4, minimum 6e-5, 20,000 warmup steps, 666,000-step schedule,
  and 72-hour run limit. Periodic checkpoints remain every 50,000 steps.
- GeMS revision `b098155fccb7335b752f2690bf16d2479d69bf5b`, shuffle window
  2048 storage blocks, shuffle seed 67 and training seed 66. MSG probes are disabled.

## Accepted performance

Measured on four v6e chips on 2026-09-08 using synchronized production optimizer
updates over the same fixed batch of 64 real GeMS spectra. GA4 and the per-chip
microbatch shape match the 128-chip config. Compilation, input loading and
host-to-device transfer are excluded; these are compute timings.

| Recomputation | Median seconds/update | Spectra/second, four chips | Compiler buffers, GiB/chip |
| --- | ---: | ---: | ---: |
| Full (comparison) | 0.954 | 67.09 | 16.82 |
| **Selective (accepted)** | **0.859** | **74.52** | **27.48** |

Selective recomputation improves throughput by **11.1%** (10.0% less time per
update). The initial comparison used three warmup and 12 timed updates; a fresh
five-warmup, 25-update replay reproduced 0.8586 seconds/update. Losses were finite.
The loss/gradient equivalence test covers selective recomputation, and the
metadata test verifies gates learn and metadata affects the loss.

The actual accumulated step also compiled for the **128-chip `v6e:8x16`**
topology: 29,503,186,400 bytes (**27.47698 GiB/chip**), leaving 3.76 GiB against
the compiler's reserved 31.24 GiB budget. This is compiler buffer sizing;
128-chip runtime throughput and end-to-end input performance remain unmeasured.

Versions matched `uv.lock`: JAX/JAXlib 0.11.0, Flax 0.12.8, Optax 0.2.8,
libtpu 0.0.44.1. Reproduce memory validation with the existing AOT utility:

```bash
ALLOW_MULTIPLE_LIBTPU_LOAD=1 .venv/bin/python -m scripts.aot_tpu_memory \
  --config configs/1b_fastmixer_pair_distogram_multiscale_n48_v6e128_ga4.py \
  --chips 128 --sweep 4 --out /tmp/paironly1b48_memory.json
```

The environment override permits compile-only topology creation on this CPU
host; it is unnecessary on a clean TPU VM.

## Launch

Use a fresh run ID and checkpoint directory. The repository launcher stages data
on all 32 hosts and initializes the distributed JAX mesh. Example submission:

```bash
run_id="pair1b-n48-v6e128-ga4-selective-$(date -u +%Y%m%d-%H%M%S)"
.venv/bin/python train_sky.py \
  --config configs/1b_fastmixer_pair_distogram_multiscale_n48_v6e128_ga4.py \
  --tpu-generation v6e --chips 128 --dws-run-duration 3d \
  --run-id "$run_id" \
  --workdir "gs://metal-repeater-411410-spectra-checkpoints/skypilot/$run_id" \
  --override log_every_n_steps=25 --detach-run
```

This document provides a launch command; consolidation of the config does not
submit a new training job or update configurations already submitted to SkyPilot.
