#!/bin/bash
# JAX pretraining on Isambard-AI GH200 nodes: one process per node drives its
# four GPUs; NCCL runs over Slingshot through the libfabric CXI provider.
#
# Usage (from a checkout on Isambard):
#   RUN=<run id> CFG=<config> [EXTRA=<json overrides>] [PYTHON=<python>] \
#     sbatch --nodes=N scripts/isambard_gh200_train.sbatch
# Resubmitting the same RUN resumes from its latest checkpoint.
#SBATCH --job-name=spectra-gh200
#SBATCH --account=brics.u6rb
#SBATCH --partition=workq
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@900
#SBATCH --output=/scratch/u6rb/wuhao.u6rb/logs/%x-%j.out
set -euo pipefail
: "${RUN:?}" "${CFG:?}"
ROOT=/scratch/u6rb/wuhao.u6rb
WORKDIR=$ROOT/spectra-runs/$RUN
PYTHON=${PYTHON:-$HOME/spectra-mrl/.venv-cu12/bin/python}
mkdir -p "$WORKDIR"
cd "$SLURM_SUBMIT_DIR"

GPUS=$((SLURM_NNODES * 4))
OVERRIDES=$("$PYTHON" - <<PY
import json, os
print(json.dumps({
    "artifact_dir": "$ROOT/spectra-data/massive_v2_ms2_t095_l080_sharded_10gb",
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

export WANDB_MODE=${WANDB_MODE:-offline} WANDB_DIR=$WORKDIR
module load brics/aws-ofi-nccl/1.8.1
export FI_PROVIDER=cxi FI_CXI_DISABLE_HOST_REGISTER=1 FI_MR_CACHE_MONITOR=userfaultfd
export NCCL_DEBUG=${NCCL_DEBUG_LEVEL:-WARN}
# GPU performance settings (TPU runs set their own in train_sky.py). XLA's
# defaults measured best for the 1B dual config; see its docstring.
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}
export XLA_FLAGS="${XLA_FLAGS:-}"
export JAX_COORDINATOR_ADDRESS="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1):12345"
export JAX_NUM_PROCESSES=$SLURM_NNODES
# JAX's Slurm detection assumes one process per GPU.
export JAX_LOCAL_DEVICE_IDS=0,1,2,3

export CFG WORKDIR OVERRIDES PYTHON
srun --kill-on-bad-exit=1 bash -c 'JAX_PROCESS_ID=$SLURM_PROCID exec "$PYTHON" train.py \
  --config "$CFG" --workdir "$WORKDIR" --overrides-json "$OVERRIDES" \
  --metrics-json "$WORKDIR/metrics/final-$SLURM_JOB_ID.json"' &
SRUN_PID=$!
# Slurm signals the batch shell 15 min before the limit; srun relays TERM so
# the trainer writes an emergency checkpoint.
trap 'kill -TERM $SRUN_PID' TERM
wait $SRUN_PID || wait $SRUN_PID
