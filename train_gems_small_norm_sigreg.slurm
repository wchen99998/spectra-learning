#!/bin/bash
#SBATCH --job-name=gems_small_norm_sigreg
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-configs/gems_small_norm_sigreg.py}"
WORKDIR="${WORKDIR:-experiments/gems_small_norm_sigreg}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-$((10000 + SLURM_JOB_ID % 50000))}"

export PROJECT_DIR CONFIG WORKDIR GPUS_PER_NODE MASTER_ADDR MASTER_PORT
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 --kill-on-bad-exit=1 bash -lc '
cd "${PROJECT_DIR}"
source .venv/bin/activate

python -m torch.distributed.run \
    --nnodes="${SLURM_NNODES}" \
    --nproc-per-node="${GPUS_PER_NODE}" \
    --node-rank="${SLURM_NODEID}" \
    --master-addr="${MASTER_ADDR}" \
    --master-port="${MASTER_PORT}" \
    train.py \
    --config "${CONFIG}" \
    --workdir "${WORKDIR}"
'
