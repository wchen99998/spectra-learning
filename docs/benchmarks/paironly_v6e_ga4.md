# Pair-only JAX training on 128 v6e chips

The config is `configs/1p8b_fastmixer_pair_distogram_multiscale_n64_v6e128_ga4.py`.
It has 1,777,557,712 trainable parameters: 15 pair-only encoder blocks and five
pair-only predictor blocks, all at pair width 2,048. There is no single stream,
single-token predictor, teacher, or MAE value head. Acquisition metadata uses
AdaLN-Zero. The distogram has 2,000 half-Dalton bins, with coarse marginal CE
widths 2, 4, 8, 16, 32, and 64.

Target m/z is replaced before pair embedding. The predictor retains the encoder's
pair features and adds a learned marker for pairs involving targets. Supervision
covers all target views, excludes self-pairs and padding, and never scores
context-only pairs. The total budget is **63 peaks plus CLS = 64 positions**.
Compact training uses at most **55 positions: 38 context peaks, 16 targets, and
CLS**. CLS has learned pair rows/columns, participates in triangle updates, and
is excluded from distogram labels. It does not introduce a single stream.

## Batch and memory

Batch size is global **per optimizer step**. On 128 chips at GA4, batch 2,048
means four spectra per chip per microstep.

AOT compilation uses the actual accumulated training step with fp32 parameters
and gradient accumulation, replicated bf16 Adam moments, and no ZeRO-1. Installed
JAX/JAXlib are 0.11.0, Flax 0.12.8, and Optax 0.2.8. Compilation uses abstract
state, with no full model allocation.

The token budget fits with ordinary per-block activation checkpointing,
replicated bf16 Adam moments, batch 2,048, and GA4. AOT compilation succeeds
at **30.68 GiB/chip**, leaving about **0.56 GiB** against the reserved budget.
[Raw memory report](paironly_n64_v6e128_ga4_memory.md).

The memory report sums argument + temporary + output bytes minus donated aliases.
Its 31.24 GiB budget reserves about 0.76 GiB from 32 GiB for runtime overhead.
These are compiler buffer estimates, **not executed TPU steps or throughput
measurements**. Check runtime memory and throughput before a long training run.
Google documents chip memory and slice sizes in the
[TPU v6e specifications](https://docs.cloud.google.com/tpu/docs/v6e).

Reproduce on a host with this repository's TPU dependencies:

```bash
.venv/bin/python -m scripts.aot_tpu_memory \
  --config configs/1p8b_fastmixer_pair_distogram_multiscale_n64_v6e128_ga4.py \
  --chips 128 --sweep 4 --out /tmp/paironly-memory.json
```

This CPU host required `ALLOW_MULTIPLE_LIBTPU_LOAD=1` for compile-only topology
creation because its existing libtpu lockfile was inaccessible. No lockfile was
removed and no TPU execution was performed.

## Input pipeline

The config retains the reference run's acquisition-metadata artifact revision
and uses randomness contract v2: 2,048 storage blocks per shuffle window,
independent training-partition seeds, and sample/epoch/operation-derived masking
and metadata dropout. Changing batching or GA does not change a host's sample
stream or an example's augmentation. This is bounded-window mixing, not a
uniform permutation of the full corpus.

The staged manifest has 65 training shards, sufficient for the 32 four-chip
hosts of a v6e-128 slice. The existing shard planner assigns all training shards;
no shared-training-file partitioning changes are needed.

Stage data for the actual distributed host count before launching. The config
enables distributed TPU initialization so all 128 devices are visible.
See [the input pipeline](../jax_input_pipeline.md) for staging, randomness, and
runtime benchmarking. The existing MSG probe expects single-token encoder outputs
and is disabled for this pair-only config.

## Run limits

The config sets `max_duration_hours=72`. The SkyPilot launch uses
`--tpu-generation v6e --chips 128 --dws-run-duration 3d`, giving the allocation
an independent 259,200-second limit. The launch override `log_every_n_steps=25`
provides post-warmup windows for checking steady-state loss and throughput.
Relevant changes are staged only after that runtime check succeeds.

Submitted on 2026-09-07 as SkyPilot managed job **71**, with run ID
`pair1p8b-n64-v6e128-b2048-ga4-3d-20260907-085019`. Checkpoints are written to
`gs://metal-repeater-411410-spectra-checkpoints/skypilot/pair1p8b-n64-v6e128-b2048-ga4-3d-20260907-085019`.
SkyPilot raced six regions and acquired the 32-host `8x16` slice in
`europe-west4`; the five losing requests were removed. All hosts completed data
staging, and the run initialized with the expected parameter count and training
configuration. [W&B run](https://wandb.ai/iclac/jepa-finalrun/runs/pair1p8b-n64-v6e128-b2048-ga4-3d-20260907-085019).

## Runtime validation

The run passed 100 optimizer steps on 2026-09-07. After the first 25 warmup
steps, three consecutive 25-step windows had stable throughput and finite loss:

| Optimizer step | Training loss | Seconds/step since preceding row | Spectra/second |
| --- | ---: | ---: | ---: |
| 25 | 10.1803 | warmup | — |
| 50 | 9.9174 | 2.92 | 701 |
| 75 | 9.7369 | 2.88 | 711 |
| 100 | 9.6559 | 2.88 | 711 |

Timing comes from the training progress log's elapsed seconds (287, 360, 432,
and 504 seconds at the four steps), so the window rates have one-second
resolution. W&B also reported finite individual multiscale loss terms.
No traceback, out-of-memory failure, or nonfinite logged training loss appeared
through step 100. This establishes startup and steady throughput.

The run was subsequently cancelled on 2026-09-08 after an emergency checkpoint
at optimizer step **20581** committed at 01:55:33 UTC. All 32 hosts confirmed
durability before their TPU instances and managed instance group were removed.
The checkpoint contains 112 objects totaling 12,017,628,436 bytes at:

```text
gs://metal-repeater-411410-spectra-checkpoints/skypilot/pair1p8b-n64-v6e128-b2048-ga4-3d-20260907-085019/checkpoints/orbax/20581
```

The accepted smaller configuration is documented in
[1B, 48 positions, GA4 selective recomputation](paironly1b_n48_v6e128_ga4.md).

Local validation passed 43 tests (46 deselected), including CLS exclusion from
distance supervision, gradients through CLS pair entries, pair-only encoder and
predictor gradients, GA4, rematerialization, and existing encoder parity:

```bash
.venv/bin/python -m pytest tests/test_pair_distogram.py \
  tests/test_fastmixer_capacity.py tests/test_jax_model_parity.py \
  -k 'pair_distogram or cls or rematerialization or ga4 or config or capacity' -q
```

```bash
/home/wuhao/skypilot/.venv/bin/sky jobs logs 71
/home/wuhao/skypilot/.venv/bin/sky jobs logs --controller 71
```
