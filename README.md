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
