## SLURM Multi-Node Training

Use one SLURM task per node, and let that task launch one `torchrun` worker per
GPU on the node:

```bash
CONFIG=configs/wandb_pa645zxs_small.py \
WORKDIR=/path/to/experiments/run_name \
srun --ntasks-per-node=1 --gpus-per-node=8 --cpus-per-task=64 --kill-on-bad-exit=1 \
    scripts/srun_torchrun_train.sh
```

For contrastive training, set `TRAIN_SCRIPT=train_contrastive.py`. Extra training
script arguments can be appended after the launcher command.

## JAX TPU Train-Step AOT Compile

Compile the JAX train step for a single-host v6e-8 VM without attaching to a TPU:

```bash
.venv/bin/python scripts/compile_jax_train_step.py \
    --config configs/medium_pairmixer_100m_20m_mae_alpha_isoflops.py \
    --target ct6e-standard-8t \
    --variant all \
    --compilation-cache-dir artifacts/jax_compile_cache/v6e8-alpha \
    --summary-json artifacts/tpu_compile/v6e8-alpha-all.json
```

`ct6e-standard-8t` maps to topology `v6e:2x4`, 8 TPU chips, 1 slice, and 1 VM
with `chips_per_host_bounds=(2, 4, 1)`. The alpha config sets
`jax_precompile_variant = "all"` so both AOT compilation and TPU warm-up compile
all real train-step variants: `pack20`, `pack24`, `pack28`, and the
full-context fallback. Use `--variant pack:20` only for targeted compile-time
experiments on one shape. Keep `jax_compilation_cache_dir` pointed at the same
cache directory on TPU runs to reuse the AOT-compiled executables during warm-up.
The current TPU flag set intentionally excludes
`xla_tpu_overlap_compute_collective_tc`; this installed JAX/libtpu stack rejects
that key as an invalid compile option during AOT compilation.
The launcher sets the same `LIBTPU_INIT_ARGS` runtime environment for local AOT
compile and SkyPilot training pods. Do not pass the whole `LIBTPU_INIT_ARGS`
string as JAX `compiler_options`; those are a narrower compile-only API and will
reject runtime-only keys such as `xla_enable_async_all_reduce`.

For the SkyPilot `100m_muon` GKE run, the launcher uses the multi-host
v6e-16 cache at:

```text
artifacts/jax_compile_cache/100m_muon_v6e4x4_b4096_accum4
gs://metal-repeater-411410-spectra-checkpoints/skypilot-aot-cache/100m_muon_v6e4x4_b4096_accum4
```

`train_sky.py` validates the cache with `scripts/jax_aot_cache.py ready` before
compiling. Readiness is based on a manifest fingerprint over the
compile-relevant sources, config, and override JSON plus expected cache counts
for train, eval, and MSG-probe JIT entries from the AOT summary. If the
manifest and cache entries already match, it logs `Reusing existing AOT cache`
and does not run the AOT compiler again unless `--force-precompile-aot` is set.
When a local cache is ready, the launcher uploads it to GCS so the SkyPilot pods
can hydrate the cache after `uv sync`.

If a previous local AOT run crashed after writing some cache entries but before
writing the final manifest, the launcher still invokes the compile driver. That
does not mean every executable is rebuilt from scratch. The compile driver still
lowers each function so JAX can derive the current content key. During
`lowered.compile()`, it temporarily turns persistent-cache read errors into
exceptions; when the only failure is the local topology compiler's expected
`DeserializeLoadedExecutable` no-client error, the unit is treated as a
content-keyed cache hit and the expensive compile is skipped. Normal misses
still compile, and other cache errors still fail. The serialized `.compiled`
files in `artifacts/tpu_compile/` are diagnostics and are not used as the
compatibility oracle.

The SkyPilot job sets `SPECTRA_JAX_PRECOMPILE_TRAIN_STEPS=auto`. In auto mode,
the pod runs the same manifest-backed readiness check after hydrating
`JAX_CACHE_DIR`; if the cache is ready, in-job train/eval/probe precompile is
disabled. This keeps expensive compile work out of the TPU allocation when the
cache is already available. The rank-0 pod uploads the warmed cache back to GCS
on exit, including the manifest and JIT cache entries. `.skyignore` excludes
local `artifacts/`, so the large cache is not copied through SkyPilot workdir
sync.

## SkyPilot GKE TPU Validation

Use the local SkyPilot client to submit this task to GKE. Do not deploy
SkyPilot itself as a service on the training cluster for this workflow; the
cluster should run only the workload pods that SkyPilot creates. Local SkyPilot
keeps kube context selection, Kueue/DWS submission, secret injection, log
streaming, and teardown in the operator environment while the pods use the
cluster's Workload Identity service account for checkpoint bucket access.

`train_sky.py` generates the SkyPilot YAML under `tmp/skypilot_tasks/` from the
explicit config, workdir, topology, and training overrides. Do not maintain a
separate checked-in SkyPilot task YAML for this run; that creates a second
source of truth for batch size, mesh size, probe cadence, and cache settings.

Launch it from this repository with an explicit config and workdir:

```bash
CONFIG=configs/medium_pairmixer_100m_20m_mae_beta_isoflops_muon.py
RUN_ID=100m-muon-v6e4x4-b4096-accum4-$(date -u +%Y%m%d-%H%M%S)
WORKDIR=gs://metal-repeater-411410-spectra-checkpoints/skypilot/${RUN_ID}

.venv/bin/python train_sky.py \
    --config "${CONFIG}" \
    --workdir "${WORKDIR}"
```

Inspect the generated SkyPilot YAML, AOT override JSON, training override JSON,
and `sky launch` command without checking secrets or launching:

```bash
.venv/bin/python train_sky.py \
    --dryrun \
    --config "${CONFIG}" \
    --workdir "${WORKDIR}"
```

The launcher creates a unique run id like
`100m-muon-v6e4x4-b4096-accum4-YYYYMMDD-HHMMSS` if `--run-id` is omitted. It
does not choose a config or checkpoint bucket silently: `--config` and
`--workdir` are required. If `--aot-cache-gcs` is omitted, the launcher derives
`gs://$WORKDIR_BUCKET/skypilot-aot-cache/$CACHE_KEY` from the explicit GCS
workdir. It loads `HF_TOKEN` from the environment or
`~/.cache/huggingface/token`, loads `WANDB_API_KEY` from the environment,
`.netrc`, or local W&B settings, and passes both tokens to SkyPilot as secrets.

The SkyPilot task uses:

```text
Cluster:      spectra-100m-muon-v6e-4x4
Queue:        default/skypilot-v6e-flex
Nodes:        4
Per node:     tpu-v6e-4, 64 CPU, 256 GB memory
GKE pool:     skypilot-v6e-16-flex
TPU topology: 4x4
Config:       configs/medium_pairmixer_100m_20m_mae_beta_isoflops_muon.py
Steps:        250000
JAX mesh:     16 devices
Batch:        4096 global, 4 gradient accumulation steps
LR:           3e-4 * sqrt(2), min LR 3e-5 * sqrt(2)
Eval:         1000 steps every 25000 steps
MSG probe:    every 100000 steps plus final step
AOT variant:  all, compiling pack20 and full-context fallback
```

The explicit `cloud.google.com/gke-tpu-topology: "4x4"` selector is required.
SkyPilot maps `tpu-v6e-4` to `2x2` by default, but this GKE pool is a four-host
`ct6e-standard-4t` slice with four chips per host and sixteen chips total.
The task also sets `tpu_vm: false`; without that, SkyPilot treats the task as a
TPU VM-style launch and requests the wrong number of Kubernetes nodes.

The default launcher topology is `4x4`. To run the old 8-chip validation shape
explicitly:

```bash
RUN_ID=100m-muon-v6e2x4-b2048-accum4-smoke-$(date -u +%Y%m%d-%H%M%S)
.venv/bin/python train_sky.py \
    --config "${CONFIG}" \
    --workdir "gs://metal-repeater-411410-spectra-checkpoints/skypilot/${RUN_ID}" \
    --topology 2x4 \
    --batch-size 2048 \
    --training-max-steps 100
```

To run the 64-chip pool:

```bash
RUN_ID=100m-muon-v6e8x8-b4096-accum4-$(date -u +%Y%m%d-%H%M%S)
.venv/bin/python train_sky.py \
    --config "${CONFIG}" \
    --workdir "gs://metal-repeater-411410-spectra-checkpoints/skypilot/${RUN_ID}" \
    --topology 8x8
```

This uses 16 SkyPilot nodes with `tpu-v6e-4` per node and switches the pod
selector to:

```yaml
cloud.google.com/gke-nodepool: skypilot-v6e-64-flex
cloud.google.com/gke-tpu-topology: "8x8"
```

The supported fixed-topology pools are `2x4` (8 chips / 2 hosts), `4x4`
(16 chips / 4 hosts), and `8x8` (64 chips / 16 hosts). GKE multi-host TPU
node pools bind the machine type and topology at creation time; use
`--node-pool` only after creating the matching GKE node pool and Kueue
ResourceFlavor.

Useful status commands:

```bash
sky status
kubectl get provisioningrequests,workloads,pods -n default -o wide
kubectl describe provisioningrequest -n default
sky logs spectra-100m-muon-v6e-4x4
```

The task writes final metrics to:

```text
gs://metal-repeater-411410-spectra-checkpoints/skypilot/$RUN_ID/metrics/final.json
```

It also logs to W&B under the config's project with tags:
`skypilot`, `gke`, `kueue`, `flex-start`, `tpu-v6e`, and `100m_muon`.

Conservative retry contract if a run fails after TPU allocation:

```json
{
  "jax_mesh_devices": "8",
  "batch_size": 1024,
  "gradient_accumulation_steps": 4
}
```

Use this shape for the next experiment if a run fails after TPU allocation and
the larger default shape is suspect. It gives a per-device microbatch of 32
across eight TPU v6e chips. The validation task also uses
`msg_probe_every_n_steps=0`, which means run the complete MSG probe at the final
step. The JAX MSG probe runs on every JAX process, shards probe examples by
process, averages probe gradients across processes, gathers final predictions
and curves back to process 0 for logging, and performs a single post-probe
global device sync. Avoid adding extra host/device syncs around training or
probing unless a correctness issue requires it.

Latest validated run:

```text
Run ID:       100m-muon-v6e4x2-b1024-accum4-finalprobe-fix4-20260616-215235
W&B:          https://wandb.ai/iclac/jepa-debugging/runs/92jb2g7q
Metrics:      gs://metal-repeater-411410-spectra-checkpoints/skypilot/100m-muon-v6e4x2-b1024-accum4-finalprobe-fix4-20260616-215235/metrics/final.json
Checkpoint:   gs://metal-repeater-411410-spectra-checkpoints/skypilot/100m-muon-v6e4x2-b1024-accum4-finalprobe-fix4-20260616-215235/checkpoints/orbax/100
Final step:   100
Train loss:   19.04833984375
Val loss:     19.005390226840973
World size:   2
JAX devices:  8
Global batch: 1024
Grad accum:   4
Final probe:  complete distributed MSG probe, best epoch 3
Probe AUC:    test maccs mean 0.6799528126185893, fluorine 0.6543289438035205, sulfur 0.7056612325587441
```

Notes from validation:

- The job hydrated the GCS AOT cache and printed
  `JAX in-job precompile train steps: False`; it did not run the local AOT
  precompile path inside the TPU allocation.
- The runtime warmed additional persistent-cache entries at shape transitions
  and uploaded them on exit. The GCS cache now contains eight train-step cache
  executables for this `jax_mesh_devices=8`, `batch_size=1024`,
  `gradient_accumulation_steps=4` shape.
- Training ran 100 steps on two `tpu-v6e-4` hosts with eight v6e chips total.
  Final metrics report `run/jax_process_count=2`,
  `run/jax_data_parallel_devices=8`, `run/device_microbatch_size=32`, and
  `run/precompile_seconds=0.0`.
- Orbax async checkpointing is disabled in the SkyPilot smoke config because a
  previous run failed during async shutdown. The successful run used synchronous
  checkpointing and wrote both process shards to GCS.
- `msg_probe_every_n_steps=0` ran the complete MSG probe at the final step. The
  JAX probe ran distributed on both processes, early-stopped at epoch 20/100,
  selected best epoch 3 by the configured validation metric, gathered final
  predictions to process 0, and uploaded W&B metrics and curves.
- MCEBIO sulfur final-probe metrics were also uploaded:
  `samples=47934`, `auc_sulfur=0.6712209004068627`.
- `dataloader_num_workers=0` is used for the smoke run so SkyPilot teardown is
  not held open by orphaned dataloader worker processes.
- Final validation must use the augmented validation loader so JAX receives
  `context_mask` and `target_masks`.
- W&B run `92jb2g7q` reported very low final throughput because the old JAX
  summary measured elapsed time after validation, checkpointing, model update,
  and the complete final MSG probe. The code now logs wall-clock throughput
  separately from training-only throughput and records non-training time under
  `run/checkpoint_seconds`, `run/validation_seconds`, `run/msg_probe_seconds`,
  and `run/model_update_seconds`.
- The JAX MSG probe now compiles encoder feature extraction and probe
  train/predict steps, shards probe data by process, avoids duplicate eval
  examples unless `pad_distributed=True`, and defers prediction `device_get`
  calls until the end of each phase. If the next TPU validation still shows a
  slow probe, collect xprof for the probe window and inspect the remaining
  `_mean_tree_across_processes` host all-gather and global sync as the first
  communication candidates.

Additional `test-tpu` debug notes from 2026-06-17:

- For the 100M Muon config, `mae_context_encoder_pack_tokens=20` only uses the
  packed encoder branch when every sample in the optimizer step has at most 20
  visible context peaks. The default intensity-aware mask uses its own
  mass-based context setting, so real `batch_size=1024`,
  `gradient_accumulation_steps=4` steps almost always fell back to full context.
  Set `jepa_intensity_aware_context_fraction=0.35` with
  `mae_context_encoder_pack_token_choices=[20]` to keep the run on pack-20.
- Runtime precompile now uses `train_loader_for_precompile()` with no shuffle and
  no workers, so it no longer spends minutes in a full-dataset `torch.randperm`
  just to fetch one compile batch.
- `GemsMemmapDataset` drops cached memmap arrays when pickled. This prevents
  forkserver DataLoader workers from serializing parent-opened memmaps after
  precompile; before this fix a single worker reached about 45 GB RSS before the
  training loop started.
- On `test-tpu` with 8 local v6e devices, pack-20 plus
  `jepa_intensity_aware_context_fraction=0.35` measured about 3.3k samples/s
  after warmup. Xprof showed train-step executions around 294 ms and collectives
  were not the dominant cost; there was no obvious unnecessary global barrier in
  the training step.
- JAX profile export can take close to a minute. The training loop now records
  `run/profile_seconds` as non-training time so profiling does not pollute
  measured training throughput.
- Cadence semantics: `msg_probe_every_n_steps=1` means one epoch because values
  in `(0, 1]` are fractional epoch intervals. Use `msg_probe_every_n_steps=0` to
  run the probe at the final training step of a capped smoke run.
- A capped JAX MSG probe with 1024 train samples, 512 test samples, 512 MCEBio
  samples, two epochs, and `single_pair_covariance` took about 194 s on the
  first invocation because it compiled probe feature extraction and wrote the
  persistent cache. The cached rerun took about 15.5 s total, with
  `msg_probe/jax_train_epoch_seconds` about 10.9 s.
- The current `single_pair_covariance` full probe still recomputes encoder and
  trainable covariance-pooler features every probe epoch. With the 100k train,
  25k test, and 47,933 MCEBio splits, the default 100-epoch probe can still take
  hours even after JIT caching. Reducing probe epochs/samples or changing the
  probe variant is the safe near-term way to keep debug runs short.
- An experimental `jax_msg_probe_shard_batches=True` path places probe batches
  on the training data mesh, but it is off by default. A naive sharded run hit a
  PairMixer sharding mismatch in feature concatenation between `P("data", ...)`
  and replicated intermediates, so the default remains the known-good unsharded
  probe batch placement.
- MaxText's remat tuning guide orders policies by speed versus HBM: `minimal`
  is near the high-HBM/low-recompute end, while `full` is the aggressive
  low-HBM/high-recompute end. MaxText `minimal` is not equivalent to this
  repo's `selective`: MaxText uses a name-based
  `save_only_these_names(...)` policy over checkpoint-tagged projection and MLP
  tensors, while this repo's `selective` maps to the generic JAX
  `jax.checkpoint_policies.dots_saveable` policy.
- AOT memory analysis on direct `test-tpu` for the current 100M Muon pack-20
  shape showed `full` remat saves memory, but only helps throughput if it
  enables a larger batch that `selective` cannot fit:

  | Mode | Batch | XLA total memory | XLA temp memory | Savings vs `selective` |
  | --- | ---: | ---: | ---: | ---: |
  | `selective` | 1024 | 5.99 GiB | 5.03 GiB | baseline |
  | `full` | 1024 | 3.34 GiB | 2.38 GiB | 2.65 GiB / 44.2% |
  | `selective` | 2048 | 9.24 GiB | 8.28 GiB | baseline |
  | `full` | 2048 | 4.12 GiB | 3.16 GiB | 5.12 GiB / 55.4% |

- The batch-2048 direct TPU throughput tests used
  `jax_mesh_devices=8`, `gradient_accumulation_steps=4`,
  `mae_context_encoder_pack_token_choices=[20]`,
  `jepa_intensity_aware_context_fraction=0.35`, no validation, no probe, and
  no periodic checkpoint. `selective` measured 5275 samples/s
  (`2.576` steps/s) after warmup; `full` measured 4810 samples/s
  (`2.349` steps/s). Dataloader time was about 0.0017 s/step, so the slowdown
  is remat recompute rather than input starvation. Prefer batch-2048
  `selective` while it fits; use `full` only to buy HBM for a larger batch or
  memory-constrained experiment.
- Matching xprof traces over five post-warmup batch-2048 steps confirmed the
  remat cost. `selective` measured 5354 samples/s and about 122 TFLOP/step in
  traced leaf ops; `full` measured 4998 samples/s and about 217 TFLOP/step.
  The extra recompute was concentrated in
  `TriangleMultiplicativeUpdate.__call__` at
  `spectra_learning/models/pairmixer_jax.py:228`, `:232`, `:238`, `:239`,
  `:242`, and `:243`, plus the transformer feed-forward at
  `spectra_learning/models/transformer_jax.py:86`. All-reduce time stayed
  around 1% to 2% of traced leaf duration, so this is a remat/recompute issue,
  not a collective bottleneck. If a custom policy is needed, start by adding
  named checkpoint tags for the PairMixer triangle projection/gate outputs,
  triangle update projection, attention `qkv`/pair-bias outputs, and MLP
  activations; do not treat the current `selective` mode as MaxText `minimal`.
