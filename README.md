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
