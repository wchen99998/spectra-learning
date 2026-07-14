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

Use the new online-probe split with the existing MSG probe loader by pointing
the NIST probe repo/subdirectory at the new dataset. The MCEBIO sulfur test set
continues to load from the established evaluation repo by default:

```python
cfg.nist_murcko_probe_repo_id = "wchen99998/msms_nist_disjoint_probe_retrieval_20260622"
cfg.nist_murcko_probe_hf_subdir = "nist_100k_online_probe"
cfg.nist_murcko_probe_include_dreams_auxiliary = False
```

If the MCEBIO artifact also lives somewhere else, set
`cfg.mcebio_murcko_probe_repo_id`, `cfg.mcebio_murcko_probe_revision`, and
`cfg.mcebio_murcko_probe_hf_subdir` explicitly.

## SLURM Multi-Node Training

Use one SLURM task per node, and let that task launch one `torchrun` worker per
GPU on the node:

```bash
export CONFIG=configs/100m_pairmixer_dense_adamw.py
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
CONFIG=configs/300m_pairmixer_dense_adamw.py
RUN_ID=300m-adamw-v6e4x8-b2048-accum4-$(date -u +%Y%m%d-%H%M%S)
WORKDIR=gs://metal-repeater-411410-spectra-checkpoints/skypilot/${RUN_ID}

.venv/bin/python train_sky.py \
    --run-id "${RUN_ID}" \
    --config "${CONFIG}" \
    --workdir "${WORKDIR}"
```

Inspect the generated SkyPilot YAML, resolved config JSON, `sky jobs launch`
command, and `sky jobs logs` command without checking secrets or launching:

```bash
.venv/bin/python train_sky.py \
    --dryrun \
    --chips 64 \
    --run-id "${RUN_ID}" \
    --config "${CONFIG}" \
    --workdir "${WORKDIR}"
```

The launcher creates a unique run id like
`300m-pairmixer-dense-adamw-v6e4x8-b2048-accum4-YYYYMMDD-HHMMSS` if `--run-id` is omitted. It
does not choose a config or checkpoint bucket silently: `--config` and
`--workdir` are required. It loads `HF_TOKEN` from the environment or
`~/.cache/huggingface/token`, loads `WANDB_API_KEY` from the environment,
`.netrc`, or local W&B settings, and passes both tokens to SkyPilot as secrets.
The default SkyPilot managed job name is run-specific, normally
`spectra-$RUN_ID`, so a second run with a different run id requests a separate
DWS allocation. Pass `--job-name` only when intentionally overriding the
managed job name.
The generated YAML uses GCE TPU v6e machine types with
`config.gcp.managed_instance_group`, so provisioning goes through GCE MIG
Flex-start DWS. CT6e runs use a TPU workload policy, a regional MIG, and a bulk
target size for the full topology; they do not use the generic MIG resize
request path. The default `--sky-bin` is the vendored SkyPilot executable at
`/home/wuhao/skypilot/.venv/bin/sky`; keep using that build until the CT6e
workload-policy MIG support lands in the upstream SkyPilot release you install.
If `sky status -u` shows an existing `sky-jobs-controller-*` created by an older
SkyPilot build, cancel any in-progress managed jobs and recreate that controller
before launching CT6e runs. Otherwise the managed-job controller can keep using
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
Infra:        gcp/us-south1
Instance:     ct6e-standard-4t for the default 4 chips per node
Hosts:        8 GCE VMs for the default 4x8 / 32-chip topology
Provisioning: GCE regional MIG Flex-start DWS with 7d run/provision wait duration
TPU topology: 4x8
Config:       configs/300m_pairmixer_dense_adamw.py
Steps:        2000000
JAX mesh:     32 devices
Batch:        2048 global, 4 gradient accumulation steps
LR:           6e-4, min LR 6e-6
Eval:         500 steps every 10000 steps
MSG probe:    disabled
```

Use `--chips` to select a supported CT6e topology. The default is `--chips 32`.
The launcher maps chip counts to topology, then derives the number of GCE VMs
from topology and `--chips-per-node`:

```text
8 chips   -> 2x4   -> 2 ct6e-standard-4t VMs
16 chips  -> 4x4   -> 4 ct6e-standard-4t VMs
32 chips  -> 4x8   -> 8 ct6e-standard-4t VMs
64 chips  -> 8x8   -> 16 ct6e-standard-4t VMs
128 chips -> 8x16  -> 32 ct6e-standard-4t VMs
256 chips -> 16x16 -> 64 ct6e-standard-4t VMs
```

Pass `--topology 8x8` only when deliberately selecting a topology by name, and
pass `--instance-type` only when deliberately overriding the derived CT6e
machine type.

Useful status commands:

```bash
sky status
sky jobs queue
sky jobs logs -n "spectra-${RUN_ID}"
sky jobs logs JOB_ID
sky jobs cancel JOB_ID
gcloud compute instances list --filter="name~spectra AND zone:(us-south1-*)"
gcloud compute instance-groups managed list --filter="name~sky-mig-spectra"
```

The task writes final metrics to:

```text
gs://metal-repeater-411410-spectra-checkpoints/skypilot/$RUN_ID/metrics/final.json
```

It also logs to W&B under the config's project with tags:
`skypilot`, `gcp`, `dws`, `flex-start`, `tpu-v6e`, and the config filename
slug such as `300m_pairmixer_dense_adamw`. The resolved config is written to
`$WORKDIR/config.json`; W&B is initialized from the same serialized dictionary.

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
