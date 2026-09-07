# S-JEPA input throughput: September 5, 2026

The current input pipeline supplied **10,713 spectra/s with four workers** and
**17,776 spectra/s with eight workers** on this CPU machine. These are full
loader measurements on real data, including preprocessing, metadata dropout,
two global views, three local views, and NumPy delivery through worker IPC.
They exclude JAX placement, accumulation stacking, and TPU computation.

## Measurement

Config: `configs/1b_sjepa_dense_adam_metadata_adaln_n128_b2048_2m_hd256.py`.
Global batch 2,048, accumulation 2, simulated data rank 0 of eight hosts:
128 spectra per microbatch, 256 spectra per host per optimizer step.
A spectrum counts once regardless of its five training views.

Hardware: AMD EPYC 9V74, 16 vCPUs / eight cores, 62 GiB RAM, data on the
8 TiB `/mnt/data` device. Python 3.14.7, NumPy 2.5.2, PyTorch 2.13.0+cpu,
h5py 3.16.0; installed JAX/JAXlib 0.11.0 were not used for placement.
Dataset: `/mnt/data/massive-v2-ms2-usable-peaks-v5/artifact`, with the current
whole-shard assignment and input randomness contract 2. Workers use spawn,
prefetch factor 2, 2,048 shuffle blocks, and 256 rows per block.

| Workers | Timed microbatches | Spectra/s | Input-supported optimizer steps/s | Peak process-tree PSS |
| --- | ---: | ---: | ---: | ---: |
| 1 | 500 | 3,340 | 13.0 | Not sampled |
| 2 | 500 | 6,662 | 26.0 | Not sampled |
| 4 | 500 | 10,663 | 41.7 | Not sampled |
| 4 | 9,000 | 10,713 | 41.8 | 11.3 GiB |
| 8 | 9,000 | 17,776 | 69.4 | 22.3 GiB |

Each run used 100 warmup microbatches. Each long run measured 1,152,000
spectra and crossed two complete shuffle-window boundaries. Runs were
sequential with one repeat each; OS page cache was not cleared. The eight-worker
run followed the four-worker run over the same sample stream, so these are
exploratory capacity measurements, not a controlled cold-storage scaling study.
Memory is sampled process-tree proportional set size at five-second intervals;
it avoids counting shared pages once per worker and can miss brief peaks.

The four-worker run took 107.53 seconds. Its warm-window rate was about
12,500–13,100 spectra/s, but 500-batch intervals during window refills fell to
4,936–5,216 spectra/s. Batch wait p95/p99 were 41/71 ms, and the longest wait
was 909 ms. The eight-worker run took 64.81 seconds, with p95/p99 of 41/89 ms
and a longest wait of 360 ms. Median waits are near zero because DataLoader
returns prefetched batches in bursts; median latency is not a capacity metric.

Compact results and 500-batch interval measurements are saved in
[the benchmark JSON](benchmarks/gems_sjepa_throughput_20260905.json).

## Does it saturate a TPU pod?

That is **not established yet**. This machine has no TPU. Both gcloud and
SkyPilot resource queries were blocked by expired credentials, so no remote
training benchmark was run.

For this configuration, the required host rate is:

```text
required spectra/s/host = global batch × optimizer steps/s ÷ host count
                       = 256 × optimizer steps/s
```

For example, training at 20 steps/s requires 5,120 spectra/s per host; at
40 steps/s it requires 10,240. The measured four-worker average provides
little headroom at 40 steps/s, and refill stalls can still starve the device.
Neither multiplying this machine's rate by eight nor comparing it with a
historical training rate demonstrates actual pod throughput. TPU VM CPUs,
storage, transfer, CPU contention with training, and the slowest host matter.

Run `--backend jax --mode train-real` and `--mode train-cached` on the same
pod with the same config and fixed mask stage. Compare
`pod_measured_steps_per_second`; a proposed acceptance criterion is real-input
training within 5% of cached-input training across repeated measurements.
Use at least 4,500 measured optimizer steps for this configuration to include
multiple refill events. The existing 100-step example is only a short smoke
comparison. Also inspect input waits and a TPU trace to confirm attribution.

Eight workers provide more measured capacity but roughly double loader memory
here. Keep the training default unchanged until this tradeoff is tested on the
target pod. Refills and duplicated worker caches are the next performance
targets if the pod test shows starvation; preserve the current sample stream
and per-example randomness when addressing them.

## Reproduction

Use a separate local cache alias so the benchmark does not download data or
claim the local dataset is the config's pinned remote revision:

```bash
mkdir -p /tmp/spectra-throughput/cache/gems
ln -s /mnt/data/massive-v2-ms2-usable-peaks-v5/artifact \
  /tmp/spectra-throughput/cache/gems/local--massive-v2--benchmark

JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python scripts/benchmark_gems_dataloader.py \
  --config configs/1b_sjepa_dense_adam_metadata_adaln_n128_b2048_2m_hd256.py \
  --overrides-json '{"gems_hdf5_repo_id":"local/massive-v2","gems_hdf5_revision":"benchmark"}' \
  --artifact-dir /tmp/spectra-throughput/cache \
  --backend numpy --world-size 8 --rank 0 --workers 4 8 \
  --warmup-batches 100 --timed-batches 9000 \
  --report-every-batches 500 --repeats 1
```

The alias already exists in this workspace; omit the `ln` command on reruns.
Progress goes to stderr, final measurements including all batch latencies to
stdout. Raw logs from this session are in `/tmp/spectra-throughput/`.
