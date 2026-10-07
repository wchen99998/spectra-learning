#!/bin/bash
# JAX pretraining on Isambard-AI GH200 nodes: one process per node drives its
# four GPUs; NCCL runs over Slingshot through the libfabric CXI provider.
#
# Usage (from a checkout on Isambard):
#   RUN=<run id> CFG=<config> [CKPT_LOCAL_ROOT=/projects/u6rb/spectra-checkpoints] \
#     [EXTRA=<json overrides>] [PYTHON=<python>] [WANDB_MODE=offline] \
#     sbatch --nodes=N scripts/isambard_gh200_train.sbatch
# Resubmitting the same RUN resumes from its latest checkpoint. With
# CKPT_LOCAL_ROOT, checkpoints and the saved config go to
# $CKPT_LOCAL_ROOT/$RUN, and scripts/isambard_checkpoint_uploader.sh on a login
# node copies them to GCS. Compute-node egress can drop to KB/s, so saving to
# GCS from the trainer (CKPT_ROOT=gs://...) can exceed Orbax's 20 min async
# timeout and crash the run. Logs, metrics and wandb files stay in scratch.
#SBATCH --job-name=spectra-gh200
#SBATCH --account=brics.u6rb
#SBATCH --partition=workq
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --signal=TERM@900
#SBATCH --output=/scratch/u6rb/wuhao.u6rb/logs/%x-%j.out
set -euo pipefail
: "${RUN:?}" "${CFG:?}"
ROOT=/scratch/u6rb/wuhao.u6rb
DATA_DIR=${DATA_DIR:-/projects/u6rb/spectra-data/massive_v2_ms2_t095_l080_sharded_10gb}
LOCALDIR=$ROOT/spectra-runs/$RUN
WORKDIR=${CKPT_ROOT:+${CKPT_ROOT%/}/$RUN}
WORKDIR=${WORKDIR:-$LOCALDIR}
PYTHON=${PYTHON:-$HOME/spectra-mrl/.venv-cu12/bin/python}
mkdir -p "$LOCALDIR/metrics"
if [[ -n ${CKPT_LOCAL_ROOT:-} ]]; then
  WORKDIR=$CKPT_LOCAL_ROOT/$RUN
fi
if [[ $WORKDIR == gs://* ]]; then
  export GOOGLE_APPLICATION_CREDENTIALS=${GOOGLE_APPLICATION_CREDENTIALS:-$HOME/.config/gcloud/spectra-isambard-ckpt.json}
  # Local mirror of a remote workdir (logger files) lives under scratch.
  export SPECTRA_SCRATCH_DIR=$LOCALDIR
fi
cd "$SLURM_SUBMIT_DIR"

GPUS=$((SLURM_NNODES * 4))
OVERRIDES=$("$PYTHON" - <<PY
import json, os
print(json.dumps({
    "artifact_dir": "$DATA_DIR",
    "jax_mesh_devices": "$GPUS",
    "jax_distributed_initialize": $SLURM_NNODES > 1,
    "jax_compilation_cache_dir": "$ROOT/jax-cache/${CFG##*/}-g$GPUS",
    "jax_enable_compilation_cache": True,
    "wandb_resume_from_env": False,
    "wandb_kwargs": {"id": "$RUN", "name": "$RUN", "resume": "allow",
                     "tags": ["isambard", "gh200", "g$GPUS", "${CFG##*/}"]},
    **json.loads(os.environ.get("EXTRA", "{}")),
}))
PY
)
echo "$OVERRIDES"

# wandb reads its key from ~/.netrc.
export WANDB_MODE=${WANDB_MODE:-online} WANDB_DIR=$LOCALDIR
module load brics/aws-ofi-nccl/1.8.1
export FI_PROVIDER=cxi FI_CXI_DISABLE_HOST_REGISTER=1 FI_MR_CACHE_MONITOR=userfaultfd
# Isambard's recommended NCCL/libfabric settings
# (docs.isambard.ac.uk/user-documentation/guides/nccl).
export NCCL_SOCKET_IFNAME=hsn NCCL_NET_GDR_LEVEL=PHB NCCL_CROSS_NIC=1 \
  NCCL_MIN_NCHANNELS=4 NCCL_GDRCOPY_ENABLE=1 NCCL_NET_FORCE_FLUSH=0
export FI_CXI_DEFAULT_CQ_SIZE=131072 FI_CXI_DEFAULT_TX_SIZE=2048 \
  FI_CXI_DISABLE_NON_INJECT_MSG_IDC=1 FI_HMEM_CUDA_USE_GDRCOPY=1 \
  FI_CXI_RDZV_PROTO=alt_read FI_CXI_RDZV_THRESHOLD=0 FI_CXI_RDZV_GET_MIN=0 \
  FI_CXI_RDZV_EAGER_SIZE=0 FI_CXI_RX_MATCH_MODE=hybrid
export NCCL_DEBUG=${NCCL_DEBUG_LEVEL:-WARN}
# From 4 nodes up, NCCL's tuner picks slow algorithms for XLA's default 30 MB
# gradient buckets (~1 GB/s bus bandwidth inside the step): the 1B dual model
# ran at 199 spectra/s on 16 GPUs (64 per GPU). Ring plus 256 MB buckets gives
# 1412, and the Simple protocol keeps 97% of linear scaling on 32 GPUs (Ring
# alone 72%). Buckets of 1 GB or more segfault.
export NCCL_ALGO=${NCCL_ALGO:-Ring} NCCL_PROTO=${NCCL_PROTO:-Simple}
# GPU performance settings (TPU runs set their own in train_sky.py).
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}
export XLA_FLAGS="--xla_gpu_all_reduce_combine_threshold_bytes=268435456 ${XLA_FLAGS:-}"
export JAX_COORDINATOR_ADDRESS="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1):12345"
export JAX_NUM_PROCESSES=$SLURM_NNODES
# Imports from the home venv are slow when many nodes start at once; with
# 32 nodes starting together, 3 of 8 processes missed the default 300 s.
export JAX_INITIALIZATION_TIMEOUT=${JAX_INITIALIZATION_TIMEOUT:-1800}
# JAX's Slurm detection assumes one process per GPU.
export JAX_LOCAL_DEVICE_IDS=0,1,2,3

export CFG WORKDIR LOCALDIR OVERRIDES PYTHON
# 15 min before the limit Slurm sends TERM to the trainer processes themselves
# (--signal without "B:"), and they write an emergency checkpoint. Relaying
# TERM from the batch shell does not work: srun treats it as a cancel and
# kills the step. max_duration_hours normally ends the run before that.
srun --kill-on-bad-exit=1 bash -c 'JAX_PROCESS_ID=$SLURM_PROCID exec "$PYTHON" train.py \
  --config "$CFG" --workdir "$WORKDIR" --overrides-json "$OVERRIDES" \
  --metrics-json "$LOCALDIR/metrics/final-$SLURM_JOB_ID.json"'
