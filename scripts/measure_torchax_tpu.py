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
    parser = argparse.ArgumentParser(description="Measure TorchAX TPU training throughput.")
    parser.add_argument(
        "--config",
        default="configs/medium_pairformer_100m_20m_mae_alpha_isoflops.py",
    )
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--metrics-json", default="")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--microbatch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument(
        "--mesh-devices",
        default=None,
        help="Number of JAX devices for batch sharding, or 'all'.",
    )
    parser.add_argument(
        "--jax-distributed-initialize",
        action="store_true",
        help="Call jax.distributed.initialize before TorchAX device creation.",
    )
    parser.add_argument("--jax-coordinator-address", default="")
    parser.add_argument("--jax-num-processes", type=int, default=None)
    parser.add_argument("--jax-process-id", type=int, default=None)
    parser.add_argument("--jax-local-device-ids", default="")
    parser.add_argument("--jax-cluster-detection-method", default="")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=25)
    parser.add_argument(
        "--overrides-json",
        default="{}",
        help="JSON config overrides applied after TPU benchmark defaults.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = deepcopy(load_config(args.config))
    config.device_backend = "jax"
    config.enable_wandb = False
    config.msg_probe_every_n_steps = 0
    config.checkpoint_every_steps = 0
    config.training_max_steps = int(args.steps)
    config.throughput_warmup_steps = int(args.warmup_steps)
    if args.gradient_accumulation_steps is not None:
        config.gradient_accumulation_steps = int(args.gradient_accumulation_steps)
    if args.batch_size is not None:
        config.batch_size = int(args.batch_size)
        if args.gradient_accumulation_steps is None:
            config.gradient_accumulation_steps = 1
    if args.microbatch_size is not None:
        config.batch_size = int(args.microbatch_size) * int(
            config.gradient_accumulation_steps
        )
    if args.mesh_devices is not None:
        if args.mesh_devices.lower() == "all":
            import jax

            config.torchax_mesh_devices = int(jax.device_count())
        else:
            config.torchax_mesh_devices = int(args.mesh_devices)
    if args.jax_distributed_initialize:
        config.torchax_distributed_initialize = True
    if args.jax_coordinator_address:
        config.torchax_coordinator_address = args.jax_coordinator_address
    if args.jax_num_processes is not None:
        config.torchax_num_processes = int(args.jax_num_processes)
    if args.jax_process_id is not None:
        config.torchax_process_id = int(args.jax_process_id)
    if args.jax_local_device_ids:
        config.torchax_local_device_ids = args.jax_local_device_ids
    if args.jax_cluster_detection_method:
        config.torchax_cluster_detection_method = args.jax_cluster_detection_method
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
