# JAX inputs on multi-host TPU pods

The shared path for MAE, S-JEPA, grouped JEPA, and AR is:

```
explicit staging → local HDF5 → index-list sampler → columnar batch
                 → task collator → NumPy microbatch stack → global JAX array
```

Staging uses the existing deterministic shard planner. Training fails early if
assigned files are missing. Workers neither download files nor start prefetch
threads. HDF5 handles belong to the reading process and close if opening or
validation fails. Training and validation close worker iterators on errors.

For each shard in a batch, the reader sorts and deduplicates physical rows once.
It reads a contiguous span when the referenced chunks are consecutive, otherwise
uses sorted HDF5 indexing. An inverse index restores duplicates and sampled order.
Only metadata required by the configured encoder is loaded. Collation accepts
arrays directly; sample-list callers retain an adapter. No additional device
queue or training framework is introduced.

## Sampling and configuration

The sampler, rolling pool, lanes, seeds, rank partitions, and resume offsets keep
their algorithms. A comparison against the pre-change sampler produced identical
sequences for 36 combinations of seeds, rank counts (2, 8, 16), and mixing modes.
Tests cover duplicates, mask RNG state, group shapes, accumulation offsets, and
spawned train/validation workers for all four tasks.

Workers are now counted **per process**, not divided by the world size. All JAX
presets resolve to 4 workers and prefetch factor 2. More hosts therefore do not
silently reduce input parallelism. PyTorch uses the same per-process convention.
Changing worker count can change worker-local mask RNG streams; exact mask replay
still requires the same worker configuration. The sampling algorithms and mask
distributions are unchanged.

For a v2 resume, keep the pinned dataset revision and set workers to the old
effective count (previous configured workers divided by process count, at least
one when enabled). The existing strict checkpoint contract will flag changed
runtime fields. Use the existing `jax_resume_allowed_config_keys` override for
`dataloader_num_workers` and `device_prefetch_size` when accepting this runtime
migration; keep dataset contract checks enabled. Removed v1 checkpoints require
the earlier code revision.

Keep the local microbatch divisible by `gems_hdf5_mix_blocks_per_batch`. For
example, global batch 2048 / 16 processes / accumulation 4 is only 32 samples and
cannot use 64-block mixing. Choose a compatible batch/accumulation configuration
explicitly; the loader does not weaken mixing to make a configuration fit.

## Local reader measurement

Measured on CPU against the pre-change reader using the same real v2 shard,
17 metadata columns, sampled indices, and HDF5 handles. Each case used 4 warmup
and 12 measured batches, alternating old/new order. Duplicate outputs were
checked for exact equality. These are exploratory warm-cache measurements, not
pod or end-to-end training results.

| Local batch | Mixing | Old median | New median |
| --- | --- | ---: | ---: |
| 128 | Single block | 18.94 ms | 1.71 ms |
| 128 | 64 blocks, pool 2048, 8 lanes | 88.47 ms | 75.78 ms |
| 1024 | Single block | 21.48 ms | 3.92 ms |

The mixed case improves about 14% in latency. Its scattered compressed-chunk
reads remain expensive. Do not infer the large single-block gain applies to the
production mixing configuration. Storage cache state strongly affects tails.
The columnar loader and metadata pruning may provide additional savings, but
those are not included in this reader-only comparison.

## Pod benchmark

For a full-loader local measurement including shuffle-window refills, see
[the September 5 S-JEPA results](gems_input_throughput_20260905.md).
Short runs below do not establish sustained capacity: with 2,048 blocks of
256 rows and a local microbatch of 128, one window lasts 4,096 microbatches.
Measure across multiple window changes, and use `--report-every-batches 500`
to track progress during longer runs.

Use the same config and overrides for staging and benchmarking on every host.
Launch one JAX process per host using the normal distributed environment. The
benchmark honors `SPECTRA_GEMS_RANK` and initializes the configured global mesh.
Use a fixed model/mask stage during a throughput comparison; compilation caused
by a stage change would contaminate the measured window.

Sweep workers using the consolidated task-aware loader benchmark:

```bash
.venv/bin/python scripts/benchmark_gems_dataloader.py \
  --config configs/1b_singlemixer_dense_muon_metadata_adaln.py \
  --backend jax --workers 1 2 4 8 \
  --warmup-batches 20 --timed-batches 250 --repeats 3
```

Each host emits JSON with setup/startup time, p50/p95/p99 input-plus-placement
latency, throughput, batch bytes, and parent peak RSS. RSS excludes workers and
is a process lifetime high-water mark. JAX placement is synchronized in this
loader-only measurement. Host latency arrays are gathered after timing to report
the per-batch slowest-host tail, without adding per-batch collectives. Parent
RSS alone is not sufficient to size a pod's loader memory budget.

For actual training throughput, run these as separate fresh benchmark processes
on every host, using the same staged assignment and overrides:

```bash
.venv/bin/python scripts/benchmark_gems_dataloader.py \
  --config configs/1b_singlemixer_dense_muon_metadata_adaln.py \
  --backend jax --mode train-real --workers 4 --repeats 1 \
  --warmup-steps 20 --timed-steps 100

.venv/bin/python scripts/benchmark_gems_dataloader.py \
  --config configs/1b_singlemixer_dense_muon_metadata_adaln.py \
  --backend jax --mode train-cached --workers 4 --repeats 1 \
  --warmup-steps 20 --timed-steps 100
```

Both modes use the task's existing direct training loop, disable validation,
probes, and checkpoint writes, and start from fresh optimizer state. Grouped
JEPA still requires its configured initialization checkpoint. Cached mode reuses
one device batch per mask stage and shuts down input workers after caching; it
is a compute diagnostic, not a learning-quality comparison. S-JEPA retains its
step-dependent regularization tag. Replay exists only in this benchmark script.

Compare `pod_measured_steps_per_second` between modes. The real/cached ratio
estimates how close input-fed training is to the cached-batch compute ceiling.
Run enough warmup to finish compilation and repeat to assess variability. The
benchmark checks the requested measured step count rather than reporting a
partial run as a successful measurement.

Normal training reports separate `profile_dataloader_seconds`,
`profile_stack_seconds`, `profile_transfer_seconds`, and
`profile_compiled_step_seconds` fields under `run/`. Transfer and compiled-step
timers measure dispatch when `run/input_timing_mode` is `dispatch`; they cannot
be interpreted as device execution time. `jax_timing_barriers=True` synchronizes
for attribution but changes overlap and throughput. Warmup ends with a state
completion barrier before timing the next input batch.

The CPU smoke check exercises real/cached AR training and spawned JAX placement.
Two-process CPU runs also exercised global placement, slowest-host reporting,
and real-input AR training, with data ranks deliberately swapped relative to
JAX process indices. The relevant test checks passed 302 tests, with 4 skipped.
Multi-host TPU throughput, device idle time, worker memory, and slowest-host
stalls still need measurement on the target pod. No TPU speedup is claimed from
the local results.
