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

## Randomness contract (version 2)

Every training shard participates. Whole shards are assigned deterministically
and balanced by eligible row counts. Training partitions have separate shuffle
seeds; validation peers deliberately share a stream so their partitions remain
disjoint. Ownership is independent of run length and accumulation. Changing
host count can change ownership and therefore sample order.

The sampler randomly orders fixed-size blocks across all assigned shards, then
uniformly shuffles **individual rows** within each block window. It exhausts the
window before advancing. There are no lanes, forced adjacent row pairs, or
pool-sized epoch truncations. Only the final partial batch and cross-host step
alignment can discard examples. Hosts with more assigned rows still stop at the
shortest host's optimizer-step count; whole-shard balancing is approximate.

This is bounded shuffling, not a uniform permutation of the entire corpus.
Within a window every row is equally eligible for each output position. Larger
windows increase the scope of mixing. Grouped JEPA uses a uniform permutation of
eligible groups across its assigned shards and independently selects spectra
within each group; group membership remains intact.

The single mixing setting is `gems_hdf5_shuffle_window_blocks` (default 2048).
`gems_hdf5_rows_per_block=0` uses the artifact's chunk row count, independent of
microbatch size. `gems_hdf5_shuffle_seed` remains an optional order seed.
The old active-shard, mixing-block, pool, and residency-lane settings are removed
and rejected with a migration message. Microbatches no longer need to be divisible
by a lane count.

Each reader process caches complete logical blocks, bounded by the window's block
count. At 256 rows per block, 2048 blocks contain 512 MiB of raw spectra, plus
metadata, HDF5 caches, and Python overhead. Four workers can therefore hold four
copies, about 2 GiB of spectra per dataset. Train and validation readers have
separate lifetimes and caches. The sampler owns only index arrays; it does not
copy spectra into a second central shuffle buffer. Cache contents are not pickled
into new workers and are cleared on close.

Workers are counted per training process. JAX presets use 4 workers with prefetch
factor 2. Choose an explicit window budget appropriate for host RAM. Do not change
the window opportunistically based on available memory: that would change the
sample sequence on restart.

Masks, S-JEPA views, and metadata dropout derive CPU seeds from the run seed,
epoch, split, immutable shard identity, physical row, operation, and view index.
These operations use separate streams and restore the caller's CPU RNG state.
Mixed masking strategies are independently selected per example, rather than
forcing a balanced strategy count within each microbatch.
Changing worker count, batching, accumulation, prefetch scheduling, or validation
cadence does not change training augmentations for a given example and epoch.
Each spectrum is visited at most once in the sampler's epoch; the next epoch
produces new views. The direct sample-list collator API still supports caller-owned
RNG when no identity-based augmentation seed is configured.

Validation uses a fixed augmentation seed and split-specific streams. DataLoader
bootstrap seeds are separate from model RNG and include the data rank and epoch.
Model initialization and shared model-side random quantities are not rank-folded.

For unchanged corpus, host ownership, window/block settings, and seed, changing
accumulation only regroups the sample stream. Resume regenerates the deterministic
stream and skips the consumed row count; worker RNG checkpoints are unnecessary.
Grouped selection likewise does not depend on output batch size.

**Migration:** this intentionally changes sample order, epoch length, and masks.
JAX checkpoints record `input_randomness_version=2`; PyTorch dataset checkpoint
metadata records the version and input settings. Old-stream checkpoints should
use the previous code revision, or initialize a new training run explicitly.
Do not bypass the contract check and assume exact continuation. Re-run staging
for an old partial assignment: training now requires all assigned corpus shards.

## Local reader measurement

A CPU comparison used the same real v2 shard and all 17 metadata columns, local
batch 128, 100 warmup and 100 measured batches. The baseline used the previous
64-unit rolling mixer (pool 2048, 8 lanes) and hybrid reader. The replacement
used uniform row windows and the complete-block cache. The two samplers have
different intended distributions; reads were checked against an uncached oracle.

| Reader | First 100 batches | Median, next 100 | p95, next 100 |
| --- | ---: | ---: | ---: |
| Previous rolling mixer | 8.13 s | 64.92 ms | 72.53 ms |
| Row-window shuffle + cache | 9.29 s | 2.03 ms | 2.48 ms |

Cache fill is slightly slower in this measurement; steady reads are substantially
faster. A new window or restarted worker incurs another fill phase. This is a
single-process reader measurement, excludes collation and model execution, and
is not a TPU throughput claim. Worker replication and storage cache state affect
end-to-end results.

A separate check of the first 100 batches on that shard measured actual source
and entity diversity. Assigned entities use `(massive_id, global_group_id)`;
unassigned rows count individually.

| Mean metric | Previous rolling mixer | Row-window shuffle |
| --- | ---: | ---: |
| Unique entities per 128 spectra | 99.38 | 127.83 |
| Unique projects per 128 spectra | 18.06 | 22.76 |
| Entity overlap with preceding batch | 0.081 | 0.455 |

The previous sampler explicitly excluded shared units within paired batches,
which suppressed adjacent overlap. The replacement removes that artificial
constraint; it greatly reduces within-batch entity duplication in this check.
These are one-shard diagnostics, not an estimate of final model quality.

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

The randomness tests check early positional coverage, complete window draining,
disjoint partitions at 2/8/16 ranks, accumulation-independent ordering, spawned
worker and resume invariance, metadata/view stream separation, cache bounds, and
RNG isolation. A two-process CPU JAX smoke run exercises global placement and
training with data ranks swapped relative to JAX process indices.
Multi-host TPU throughput and worker memory still require target-pod measurement;
no TPU speedup is claimed from the local results.
