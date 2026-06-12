import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectra_learning.training.api import load_config
from spectra_learning.training.pretrain import train_and_evaluate
from spectra_learning.training.storage import normalize_storage_path, write_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure JAX NNX TPU training throughput.")
    parser.add_argument(
        "--config",
        default="configs/medium_pairmixer_100m_20m_mae_alpha_isoflops.py",
    )
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--metrics-json", default="")
    parser.add_argument("--per-device-microbatch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--mesh-devices", type=int, default=4)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--profile-dir", default="")
    parser.add_argument("--profile-start-step", type=int, default=None)
    parser.add_argument("--profile-steps", type=int, default=0)
    parser.add_argument("--timing-barriers", action="store_true")
    parser.add_argument(
        "--optax-multistep-accumulation",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--device-accumulation",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--overrides-json", default="{}")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = deepcopy(load_config(args.config))
    config.device_backend = "jax_nnx"
    config.use_jax_nnx = True
    config.enable_wandb = False
    config.msg_probe_every_n_steps = 0
    config.checkpoint_every_steps = 0
    config.training_max_steps = int(args.steps)
    config.throughput_warmup_steps = int(args.warmup_steps)
    config.gradient_accumulation_steps = int(args.gradient_accumulation_steps)
    config.jax_mesh_devices = int(args.mesh_devices)
    config.jax_timing_barriers = bool(args.timing_barriers)
    if args.optax_multistep_accumulation is not None:
        config.jax_optax_multistep_accumulation = bool(
            args.optax_multistep_accumulation
        )
    if args.device_accumulation is not None:
        config.jax_device_accumulation = bool(args.device_accumulation)
    if args.profile_dir:
        config.jax_profile_dir = args.profile_dir
    if args.profile_start_step is not None:
        config.jax_profile_start_step = int(args.profile_start_step)
    if args.profile_steps:
        config.jax_profile_steps = int(args.profile_steps)
    config.batch_size = (
        int(args.per_device_microbatch_size)
        * int(args.mesh_devices)
        * int(args.gradient_accumulation_steps)
    )
    config.dataloader_num_workers = int(args.dataloader_num_workers)
    config.dataloader_persistent_workers = False
    config.update(json.loads(args.overrides_json))
    metrics = train_and_evaluate(
        config,
        workdir=normalize_storage_path(args.workdir),
    )
    if args.metrics_json:
        write_text(
            normalize_storage_path(args.metrics_json),
            json.dumps(metrics, indent=2, sort_keys=True),
        )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
