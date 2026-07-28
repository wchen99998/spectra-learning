## NIST Disjoint Probe/Retrieval Dataset

Build the fixed raw-NIST benchmark collection from
`data/raw/hr_msms_nist.mgf` with one command:

```bash
uv run python -m spectra_learning.data.murcko \
    --build-disjoint-probe-retrieval \
    --nist-mgf data/raw/hr_msms_nist.mgf \
    --work-dir data/prepared/nist_disjoint_probe_retrieval_20260622 \
    --hf-repo-id wchen99998/msms_nist_disjoint_probe_retrieval_20260622
```

Use `--skip-upload` for a local dry run. The staged Hugging Face folder contains
`nist_100k_online_probe/` with train/val/test Parquets, `nist_retrieval_pool/`
with the Murcko-histogram-disjoint retrieval spectra, and fixed pair tables in
`nist_same_inchi14_10ppm_retrieval/` and `nist_mces_analog_retrieval/`.
Top-level `metadata.json` records the 100k without-replacement probe selection,
the selected Murcko histogram keys, and the zero-overlap check against the
retrieval pool.

The retrieval-specific online probe above remains paired with that retrieval
benchmark. Canonical MSG and fluorine validation use the full NIST Murcko
probe split:

```python
cfg.nist_murcko_probe_repo_id = "wchen99998/hr_msms_nist_mcebio_murcko_20260529"
cfg.nist_murcko_probe_revision = "5fc6712bd9cfff29668c7769af92cace0813b172"
cfg.nist_murcko_probe_hf_subdir = "nist_murcko_probe"
cfg.nist_murcko_probe_include_dreams_auxiliary = False
```

## Training Entry Points

`train.py` is the only model-training dispatcher. It supports PyTorch or JAX
pretraining, PyTorch contrastive training, and JAX AR training. Unsupported
task/backend combinations fail before a trainer is imported.

`train_sky.py` is an infrastructure launcher for JAX TPU jobs, not a second
training implementation. It validates the same task/backend contract and the
generated task delegates to `train.py`.

## SLURM Multi-Node Training

Use one SLURM task per node, and let that task launch one `torchrun` worker per
GPU on the node:

```bash
export CONFIG=configs/pretrain.py
export WORKDIR=/path/to/experiments/run_name
export GPUS_PER_NODE=8
export PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-29500}"

srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 --kill-on-bad-exit=1 \
    bash -lc '
        cd "${PROJECT_DIR}"
        .venv/bin/python -m torch.distributed.run \
            --nnodes="${SLURM_NNODES}" \
            --nproc-per-node="${GPUS_PER_NODE}" \
            --node-rank="${SLURM_NODEID}" \
            --master-addr="${MASTER_ADDR}" \
            --master-port="${MASTER_PORT}" \
            train.py --config "${CONFIG}" --workdir "${WORKDIR}"
    '
```

For contrastive training, use the same entrypoint and select the task through
the config:

```bash
.venv/bin/python train.py \
    --config configs/pretrain.py \
    --workdir /path/to/workdir \
    --overrides-json '{"training_task":"contrastive","training_mode":"contrastive"}'
```

## JAX TPU Runtime Compilation

There is no separate JAX TPU compile step. `train_sky.py` launches training
directly, and JAX compiles the train, eval, and MSG-probe functions on their
first real call inside the TPU allocation.

JAX checkpoints use one composite Orbax format containing the training state
and a JSON training contract. MAE checkpoints bind the complete effective
configuration; AR checkpoints additionally record derived tokenizer, dataset,
preprocessing, model, optimizer, and sampling/training values. Restore refuses
any contract mismatch. Metadata-free checkpoints from the previous format are
intentionally not resumable; start a new workdir for the current format.

The SkyPilot task still enables JAX's normal persistent compilation cache under
`/tmp/spectra-jax-cache/$CACHE_KEY` inside each VM. That cache is local to the
job and is not hydrated from or uploaded to GCS by the launcher. Old local
compile-cache outputs are stale experiment artifacts and are no longer consumed
by the current training path.

## SkyPilot GCP DWS TPU Launch

Use the vendored SkyPilot client to submit this task directly to GCP. This
launcher does not target a GKE cluster, Kueue queue, or Kubernetes
ProvisioningRequest. Local SkyPilot handles direct GCP provisioning, secret
injection, workdir sync, managed-job log streaming, and managed-job lifecycle
cleanup.

`train_sky.py` generates the SkyPilot YAML under `tmp/skypilot_tasks/` from the
explicit config, workdir, topology, and optional `--override KEY=JSON_VALUE`
arguments. Training values live in the experiment config; the launcher only
adds infrastructure settings such as the TPU mesh and JAX cache. Do not
maintain a separate checked-in SkyPilot task YAML for a run.

Launch it from this repository with an explicit config and workdir:

```bash
CONFIG=configs/3b_pairmixer_dense_adamw.py
RUN_ID=3b-adamw-v7x2x2x2-b2048-ga16-32-32-$(date -u +%Y%m%d-%H%M%S)
WORKDIR=gs://metal-repeater-411410-spectra-checkpoints/skypilot/${RUN_ID}

.venv/bin/python train_sky.py \
    --chips 8 \
    --dws-run-duration 2d \
    --run-id "${RUN_ID}" \
    --config "${CONFIG}" \
    --workdir "${WORKDIR}"
```

Inspect the generated SkyPilot YAML, resolved config JSON, `sky jobs launch`
command, and `sky jobs logs` command without checking secrets or launching:

```bash
.venv/bin/python train_sky.py \
    --dryrun \
    --chips 8 \
    --run-id "${RUN_ID}" \
    --config "${CONFIG}" \
    --workdir "${WORKDIR}"
```

The launcher creates a unique run id like
`3b-pairmixer-dense-adamw-v7x2x2x2-b2048-accum16-YYYYMMDD-HHMMSS` if `--run-id` is omitted. It
does not choose a config or checkpoint bucket silently: `--config` and
`--workdir` are required. It loads `HF_TOKEN` from the environment or
`~/.cache/huggingface/token`, loads `WANDB_API_KEY` from the environment,
`.netrc`, or local W&B settings, and passes both tokens to SkyPilot as secrets.
The default SkyPilot managed job name is run-specific, normally
`spectra-$RUN_ID`, so a second run with a different run id requests a separate
DWS allocation. Pass `--job-name` only when intentionally overriding the
managed job name.
The generated YAML uses the GCE `tpu7x-standard-4t` machine type with
`config.gcp.managed_instance_group`, so provisioning goes through GCE MIG
Flex-start DWS. TPU7x runs use a TPU workload policy, a regional MIG, and a
bulk target size for the full topology; they do not use the generic MIG resize
request path. Flex-start is restricted to `us-central1-c`, so the launcher
targets `gcp/us-central1`. The default `--sky-bin` is the vendored SkyPilot
executable at `/home/wuhao/skypilot/.venv/bin/sky`; that build contains the
Compute Engine TPU7x workload-policy support required by this launcher.
If `sky status -u` shows an existing `sky-jobs-controller-*` created by an older
SkyPilot build, cancel any in-progress managed jobs and recreate that controller
before launching TPU7x runs. Otherwise the managed-job controller can keep using
the old provisioning code even though the local launcher uses the vendored CLI.

SkyPilot managed-job flags are passed through to `sky jobs launch`. To submit
and return immediately instead of running `sky jobs logs -n "$JOB_NAME"` from
the launcher, launch with:

```bash
.venv/bin/python train_sky.py \
    --run-id "${RUN_ID}" \
    --config "${CONFIG}" \
    --workdir "${WORKDIR}" \
    --detach-run
```

By default the launcher submits with `sky jobs launch --detach-run` and then
streams logs with `sky jobs logs -n "$JOB_NAME"`. The managed job owns resource
teardown after completion or failure; `train_sky.py` does not run `sky down`.

The SkyPilot task uses:

```text
Job:          spectra-$RUN_ID by default, or the explicit --job-name value
Infra:        gcp/us-central1 (Flex-start allocates in us-central1-c)
Instance:     tpu7x-standard-4t, 4 physical chips per VM
Hosts:        2 GCE VMs for the default 2x2x2 / 8-chip topology
Provisioning: GCE regional MIG Flex-start DWS; queue until capacity or cancellation
Run duration: 2 days after capacity is allocated
TPU topology: 2x2x2
Config:       configs/3b_pairmixer_dense_adamw.py
Steps:        2000000
Parameters:   3.0025B encoder + 0.3063B predictor side = 3.3088B total
JAX mesh:     16 devices (two TensorCore devices per physical TPU7x chip)
Batch:        2048 global, 16/32/32 gradient accumulation schedule
LR:           6e-4, min LR 6e-6
Eval:         500 steps every 10000 steps
MSG probe:    disabled
```

Use `--chips` to select a supported TPU7x topology. The default is `--chips 8`.
Every TPU7x VM has four physical chips, and every chip exposes two JAX devices:

```text
4 chips   -> 2x2x1 -> 1 tpu7x-standard-4t VM  -> 8 JAX devices
8 chips   -> 2x2x2 -> 2 tpu7x-standard-4t VMs -> 16 JAX devices
16 chips  -> 2x2x4 -> 4 tpu7x-standard-4t VMs -> 32 JAX devices
32 chips  -> 2x4x4 -> 8 tpu7x-standard-4t VMs -> 64 JAX devices
64 chips  -> 4x4x4 -> 16 tpu7x-standard-4t VMs -> 128 JAX devices
128 chips -> 4x4x8 -> 32 tpu7x-standard-4t VMs -> 256 JAX devices
256 chips -> 4x8x8 -> 64 tpu7x-standard-4t VMs -> 512 JAX devices
```

Pass `--topology 2x2x2` only when deliberately selecting a topology by name.

Useful status commands:

```bash
sky status
sky jobs queue
sky jobs logs -n "spectra-${RUN_ID}"
sky jobs logs JOB_ID
sky jobs cancel JOB_ID
gcloud compute instances list --filter="name~spectra AND zone:(us-central1-*)"
gcloud compute instance-groups managed list --filter="name~sky-mig-spectra"
```

The task writes final metrics to:

```text
gs://metal-repeater-411410-spectra-checkpoints/skypilot/$RUN_ID/metrics/final.json
```

It also logs to W&B under the config's project with tags:
`skypilot`, `gcp`, `dws`, `flex-start`, `tpu-v7x`, and the config filename
slug such as `3b_pairmixer_dense_adamw`. The resolved config is written to
`$WORKDIR/config.json`; W&B is initialized from the same serialized dictionary.

### Historical v6e validation

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

- Training ran 100 steps on two `tpu-v6e-4` hosts with eight v6e chips total.
  Final metrics report `run/jax_process_count=2`,
  `run/jax_data_parallel_devices=8`, and `run/device_microbatch_size=32`.
- Orbax async checkpointing is disabled in the SkyPilot smoke config because a
  previous run failed during async shutdown. The successful run used synchronous
  checkpointing and wrote both process shards to GCS.
- `msg_probe_every_n_steps=0` ran the complete MSG probe at the final step. The
  JAX probe ran distributed on both processes, early-stopped at epoch 20/100,
  selected best epoch 3 by the configured validation metric, gathered final
  predictions to process 0, and uploaded W&B metrics and curves.
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

- The packed MAE context encoder branch has been removed. JAX MAE training now
  uses one full-context encoder path.
- GeMS pretraining now reads HDF5 shards through
  `GemsDataModule`; h5py is the HDF5 dataset backend, while the
  PyTorch datamodule owns chunk-aware sampling, rank partitioning, and
  DataLoader worker process settings.
- JAX profile export can take close to a minute. The training loop now records
  `run/profile_seconds` as non-training time so profiling does not pollute
  measured training throughput.
- Cadence semantics: `msg_probe_every_n_steps=1` means one epoch because values
  in `(0, 1]` are fractional epoch intervals. Use `msg_probe_every_n_steps=0` to
  run the probe at the final training step of a capped smoke run.
- The current `single_pair_covariance` full probe still recomputes encoder and
  trainable covariance-pooler features every probe epoch. With the 100k train,
  25k test splits, the default 100-epoch probe can still take hours even after
  JIT caching. Reducing probe epochs/samples or changing the probe variant is
  the safe near-term way to keep debug runs short.
- MaxText's remat tuning guide orders policies by speed versus HBM: `minimal`
  is near the high-HBM/low-recompute end, while `full` is the aggressive
  low-HBM/high-recompute end. MaxText `minimal` is not equivalent to this
  repo's `selective`: MaxText uses a name-based
  `save_only_these_names(...)` policy over checkpoint-tagged projection and MLP
  tensors, while this repo's `selective` maps to JAX's
  `jax.checkpoint_policies.dots_with_no_batch_dims_saveable` policy.
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
