from __future__ import annotations

import argparse
import configparser
import json
import logging
import netrc
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from spectra_learning.config import load_config


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_TASK_YAML = REPO_ROOT / "skypilot" / "100m_muon_v6e_kueue.yaml"
DEFAULT_CONFIG = "configs/medium_pairmixer_100m_20m_mae_beta_isoflops_muon.py"
DEFAULT_BUCKET = "gs://metal-repeater-411410-spectra-checkpoints"
DEFAULT_PROJECT = "metal-repeater-411410"
DEFAULT_INFRA = "k8s/skypilot-training"
KNOWN_NODE_POOLS = {
    "2x4": "skypilot-v6e-4t-flex",
    "4x4": "skypilot-v6e-16-flex",
    "8x8": "skypilot-v6e-64-flex",
}


@dataclass(frozen=True)
class ConfigDefaults:
    batch_size: int
    gradient_accumulation_steps: int
    training_max_steps: int
    checkpoint_every_steps: int
    log_every_n_steps: int
    throughput_warmup_steps: int
    dataloader_num_workers: int
    msg_probe_every_n_steps: float
    val_every_n_steps: float
    val_num_steps: int
    aot_variant: str


@dataclass(frozen=True)
class TopologySpec:
    topology: str
    total_chips: int
    num_nodes: int
    chips_per_node: int
    node_pool: str
    accelerator: str

    @property
    def jax_mesh_devices(self) -> str:
        return str(self.total_chips)

    @property
    def aot_target(self) -> str:
        return f"v6e-{self.topology}-multihost"

    @property
    def slug(self) -> str:
        return f"v6e{self.topology}"


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Submit Spectra JAX training to SkyPilot/GKE/Kueue."
    )
    parser.add_argument("--task-yaml", default=str(DEFAULT_TASK_YAML))
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--topology", default="4x4", help="TPU v6e topology, e.g. 2x4, 4x4, 8x8.")
    parser.add_argument("--node-pool", default="")
    parser.add_argument("--accelerator", default="tpu-v6e-4")
    parser.add_argument("--chips-per-node", type=int, default=4)
    parser.add_argument("--cluster", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--workdir", default="")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--training-max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--jax-mesh-devices", default="")
    parser.add_argument("--msg-probe-every-n-steps", type=float, default=None)
    parser.add_argument("--val-every-n-steps", type=float, default=None)
    parser.add_argument("--val-num-steps", type=int, default=None)
    parser.add_argument("--checkpoint-every-steps", type=int, default=None)
    parser.add_argument("--log-every-n-steps", type=int, default=None)
    parser.add_argument("--throughput-warmup-steps", type=int, default=None)
    parser.add_argument("--dataloader-num-workers", type=int, default=None)
    parser.add_argument("--aot-cache-dir", default="")
    parser.add_argument("--aot-output-dir", default="")
    parser.add_argument("--aot-summary-json", default="")
    parser.add_argument("--aot-cache-gcs", default="")
    parser.add_argument("--aot-variant", default="")
    parser.add_argument("--jax-precompile-train-steps", default="auto")
    parser.add_argument("--precompile-aot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sync-aot-cache-to-gcs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-precompile-aot", action="store_true")
    parser.add_argument("--down", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--yes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--task-output-dir", default="tmp/skypilot_tasks")
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--infra", default=DEFAULT_INFRA)
    parser.add_argument("--metrics-json", default="metrics/final.json")
    parser.add_argument("--queue-tag", default="flex-start")
    args, sky_args = parser.parse_known_args(argv)
    if sky_args and sky_args[0] == "--":
        sky_args = sky_args[1:]
    return args, sky_args


def resolve_topology(
    topology: str,
    *,
    node_pool: str = "",
    chips_per_node: int = 4,
    accelerator: str = "tpu-v6e-4",
) -> TopologySpec:
    normalized = topology.lower().replace("v6e:", "").strip()
    parts = normalized.split("x")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError(f"topology must look like 2x4, 4x4, or 8x8; got {topology!r}")
    dims = tuple(int(part) for part in parts)
    total_chips = dims[0] * dims[1]
    if total_chips <= 0:
        raise ValueError(f"topology must contain positive dimensions; got {topology!r}")
    if chips_per_node <= 0:
        raise ValueError("chips_per_node must be positive")
    if total_chips % chips_per_node != 0:
        raise ValueError(
            f"topology {normalized} has {total_chips} chips, not divisible by "
            f"chips_per_node={chips_per_node}"
        )
    resolved_node_pool = node_pool or KNOWN_NODE_POOLS.get(normalized)
    if not resolved_node_pool:
        raise ValueError(
            f"no default node pool is known for topology {normalized!r}; "
            "pass --node-pool explicitly after creating a matching GKE TPU node pool "
            "and Kueue ResourceFlavor"
        )
    return TopologySpec(
        topology=normalized,
        total_chips=total_chips,
        num_nodes=total_chips // chips_per_node,
        chips_per_node=chips_per_node,
        node_pool=resolved_node_pool,
        accelerator=accelerator,
    )


def read_hf_token(env: dict[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    token = env.get("HF_TOKEN") or env.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token.strip()
    path = Path.home() / ".cache" / "huggingface" / "token"
    if path.is_file():
        return path.read_text().strip()
    raise SystemExit("HF_TOKEN is missing; set it or run huggingface-cli login first.")


def read_wandb_api_key(env: dict[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    token = env.get("WANDB_API_KEY")
    if token:
        return token.strip()
    for machine in ("api.wandb.ai", "wandb.ai"):
        try:
            auth = netrc.netrc().authenticators(machine)
        except (FileNotFoundError, netrc.NetrcParseError):
            auth = None
        if auth and auth[2]:
            return auth[2].strip()
    settings = Path.home() / ".config" / "wandb" / "settings"
    if settings.is_file():
        parser = configparser.ConfigParser()
        try:
            parser.read(settings)
            for section in parser.sections():
                if parser.has_option(section, "api_key"):
                    return parser.get(section, "api_key").strip()
        except configparser.Error:
            pass
        for line in settings.read_text().splitlines():
            if line.strip().startswith("api_key") and "=" in line:
                return line.split("=", 1)[1].strip()
    raise SystemExit("WANDB_API_KEY is missing; set it or run wandb login first.")


def json_compact(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def load_config_defaults(config_path: str) -> ConfigDefaults:
    cfg = load_config(config_path)
    training_max_steps = int(getattr(cfg, "training_max_steps"))
    return ConfigDefaults(
        batch_size=int(getattr(cfg, "batch_size")),
        gradient_accumulation_steps=int(getattr(cfg, "gradient_accumulation_steps", 1)),
        training_max_steps=training_max_steps,
        checkpoint_every_steps=int(getattr(cfg, "checkpoint_every_steps", training_max_steps)),
        log_every_n_steps=int(getattr(cfg, "log_every_n_steps", 250)),
        throughput_warmup_steps=int(getattr(cfg, "throughput_warmup_steps", 25)),
        dataloader_num_workers=int(getattr(cfg, "dataloader_num_workers", 0)),
        msg_probe_every_n_steps=float(getattr(cfg, "msg_probe_every_n_steps", 0.0)),
        val_every_n_steps=float(getattr(cfg, "val_every_n_steps", 0.0)),
        val_num_steps=int(getattr(cfg, "val_num_steps", 64)),
        aot_variant=str(getattr(cfg, "jax_precompile_variant", "default")),
    )


def build_train_overrides(
    *,
    run_id: str,
    training_max_steps: int,
    jax_mesh_devices: str,
    batch_size: int,
    gradient_accumulation_steps: int,
    jax_cache_dir: str,
    jax_precompile_train_steps: str | bool,
    msg_probe_every_n_steps: float,
    val_every_n_steps: float,
    val_num_steps: int,
    checkpoint_every_steps: int,
    log_every_n_steps: int,
    throughput_warmup_steps: int,
    dataloader_num_workers: int,
    queue_tag: str,
) -> dict[str, Any]:
    return {
        "training_max_steps": int(training_max_steps),
        "msg_probe_every_n_steps": msg_probe_every_n_steps,
        "msg_probe_at_final_step": True,
        "val_every_n_steps": val_every_n_steps,
        "val_num_steps": int(val_num_steps),
        "checkpoint_every_steps": int(checkpoint_every_steps),
        "log_every_n_steps": int(log_every_n_steps),
        "throughput_warmup_steps": int(throughput_warmup_steps),
        "dataloader_num_workers": int(dataloader_num_workers),
        "jax_distributed_initialize": True,
        "jax_mesh_devices": str(jax_mesh_devices),
        "batch_size": int(batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "jax_compilation_cache_dir": jax_cache_dir,
        "jax_enable_compilation_cache": True,
        "jax_persistent_cache_min_compile_time_secs": 0.0,
        "jax_persistent_cache_min_entry_size_bytes": 0,
        "jax_precompile_train_steps": jax_precompile_train_steps,
        "jax_precompile_eval_steps": True,
        "jax_precompile_msg_probe": True,
        "jax_enable_async_checkpointing": False,
        "wandb_kwargs": {
            "name": run_id,
            "tags": [
                "skypilot",
                "gke",
                "kueue",
                queue_tag,
                "tpu-v6e",
                "100m_muon",
            ],
            "notes": (
                "SkyPilot GKE Kueue/DWS TPU v6e run launched by train_sky.py."
            ),
        },
    }


def build_task(
    *,
    template_path: Path,
    topology: TopologySpec,
    envs: dict[str, str],
    infra: str,
) -> dict[str, Any]:
    task = yaml.safe_load(template_path.read_text())
    task["num_nodes"] = topology.num_nodes
    resources = task.setdefault("resources", {})
    resources["infra"] = infra
    resources["accelerators"] = topology.accelerator
    resources.setdefault("accelerator_args", {})["tpu_vm"] = False
    task_envs = task.setdefault("envs", {})
    task_envs.update(envs)
    spec = (
        task.setdefault("config", {})
        .setdefault("kubernetes", {})
        .setdefault("pod_config", {})
        .setdefault("spec", {})
    )
    node_selector = spec.setdefault("nodeSelector", {})
    node_selector["cloud.google.com/gke-nodepool"] = topology.node_pool
    node_selector["cloud.google.com/gke-tpu-topology"] = topology.topology
    return task


def run_command(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    logging.info("Running: %s", shlex.join(cmd))
    result = subprocess.run(cmd, cwd=cwd, env=env, text=True, check=False)
    if check and result.returncode != 0:
        raise SystemExit(result.returncode)
    return result


def aot_cache_ready(
    *,
    cache_dir: str,
    config_path: str,
    overrides_json: str,
    summary_json: str,
) -> bool:
    cmd = [
        sys.executable,
        "scripts/jax_aot_cache.py",
        "ready",
        "--cache-dir",
        cache_dir,
        "--config",
        config_path,
        "--overrides-json",
        overrides_json,
        "--summary-json",
        summary_json,
    ]
    return run_command(cmd, cwd=REPO_ROOT, check=False).returncode == 0


def prepare_aot_cache(
    *,
    config_path: str,
    target: str,
    variant: str,
    cache_dir: str,
    output_dir: str,
    summary_json: str,
    overrides_json: str,
    cache_gcs: str,
    precompile: bool,
    force_precompile: bool,
    sync_to_gcs: bool,
) -> None:
    ready = aot_cache_ready(
        cache_dir=cache_dir,
        config_path=config_path,
        overrides_json=overrides_json,
        summary_json=summary_json,
    )
    if precompile:
        if ready and not force_precompile:
            logging.info("Reusing existing AOT cache: %s", cache_dir)
        else:
            logging.info("Precompiling AOT cache: target=%s variant=%s", target, variant)
            run_command(
                [
                    sys.executable,
                    "scripts/compile_jax_train_step.py",
                    "--config",
                    config_path,
                    "--target",
                    target,
                    "--variant",
                    variant,
                    "--output-dir",
                    output_dir,
                    "--compilation-cache-dir",
                    cache_dir,
                    "--summary-json",
                    summary_json,
                    "--overrides-json",
                    overrides_json,
                ],
                cwd=REPO_ROOT,
            )
            run_command(
                [
                    sys.executable,
                    "scripts/jax_aot_cache.py",
                    "write-manifest",
                    "--cache-dir",
                    cache_dir,
                    "--config",
                    config_path,
                    "--overrides-json",
                    overrides_json,
                    "--summary-json",
                    summary_json,
                ],
                cwd=REPO_ROOT,
            )
    ready = aot_cache_ready(
        cache_dir=cache_dir,
        config_path=config_path,
        overrides_json=overrides_json,
        summary_json=summary_json,
    )
    if ready and sync_to_gcs:
        logging.info("Uploading AOT cache to %s", cache_gcs)
        run_command(
            [
                sys.executable,
                "scripts/jax_aot_cache.py",
                "upload",
                "--cache-dir",
                cache_dir,
                "--gcs-uri",
                cache_gcs,
            ],
            cwd=REPO_ROOT,
        )
    elif not ready:
        logging.warning("AOT cache is not ready; in-job precompile remains eligible.")


def write_task_file(task: dict[str, Any], output_dir: Path, run_id: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{run_id}.yaml"
    path.write_text(yaml.safe_dump(task, sort_keys=False))
    return path


def default_cluster_name(topology: TopologySpec) -> str:
    if topology.topology == "2x4":
        return "spectra-100m-muon-v6e"
    return f"spectra-100m-muon-v6e-{topology.topology}"


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args, sky_args = parse_args(argv)
    topology = resolve_topology(
        args.topology,
        node_pool=args.node_pool,
        chips_per_node=args.chips_per_node,
        accelerator=args.accelerator,
    )
    config_defaults = load_config_defaults(args.config)
    training_max_steps = args.training_max_steps or config_defaults.training_max_steps
    batch_size = args.batch_size or config_defaults.batch_size
    grad_accum = (
        args.gradient_accumulation_steps
        or config_defaults.gradient_accumulation_steps
    )
    jax_mesh_devices = args.jax_mesh_devices or topology.jax_mesh_devices
    aot_variant = args.aot_variant or config_defaults.aot_variant
    checkpoint_every_steps = (
        args.checkpoint_every_steps
        if args.checkpoint_every_steps is not None
        else config_defaults.checkpoint_every_steps
    )
    msg_probe_every_n_steps = (
        args.msg_probe_every_n_steps
        if args.msg_probe_every_n_steps is not None
        else config_defaults.msg_probe_every_n_steps
    )
    val_every_n_steps = (
        args.val_every_n_steps
        if args.val_every_n_steps is not None
        else config_defaults.val_every_n_steps
    )
    val_num_steps = (
        args.val_num_steps
        if args.val_num_steps is not None
        else config_defaults.val_num_steps
    )
    log_every_n_steps = (
        args.log_every_n_steps
        if args.log_every_n_steps is not None
        else config_defaults.log_every_n_steps
    )
    throughput_warmup_steps = (
        args.throughput_warmup_steps
        if args.throughput_warmup_steps is not None
        else config_defaults.throughput_warmup_steps
    )
    dataloader_num_workers = (
        args.dataloader_num_workers
        if args.dataloader_num_workers is not None
        else config_defaults.dataloader_num_workers
    )
    run_id = args.run_id or (
        f"100m-muon-{topology.slug}-b{batch_size}-accum{grad_accum}-"
        f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    )
    workdir = args.workdir or f"{args.bucket.rstrip('/')}/skypilot/{run_id}"
    cluster = args.cluster or default_cluster_name(topology)
    cache_key = f"100m_muon_{topology.slug}_b{batch_size}_accum{grad_accum}"
    aot_cache_dir = args.aot_cache_dir or f"artifacts/jax_compile_cache/{cache_key}"
    aot_output_dir = args.aot_output_dir or f"artifacts/tpu_compile/{cache_key}"
    aot_summary_json = args.aot_summary_json or (
        f"artifacts/tpu_compile/{cache_key}-{aot_variant.replace(':', '-')}.json"
    )
    aot_cache_gcs = args.aot_cache_gcs or (
        f"{args.bucket.rstrip('/')}/skypilot-aot-cache/{cache_key}"
    )
    aot_overrides = {
        "training_max_steps": int(training_max_steps),
        "jax_mesh_devices": str(jax_mesh_devices),
        "batch_size": int(batch_size),
        "gradient_accumulation_steps": int(grad_accum),
    }
    aot_overrides_json = json_compact(aot_overrides)

    logging.info(
        "Topology: topology=%s nodes=%d chips=%d node_pool=%s accelerator=%s",
        topology.topology,
        topology.num_nodes,
        topology.total_chips,
        topology.node_pool,
        topology.accelerator,
    )
    logging.info(
        "Training shape: mesh=%s batch=%d grad_accum=%d steps=%d",
        jax_mesh_devices,
        batch_size,
        grad_accum,
        training_max_steps,
    )
    logging.info("Run ID: %s", run_id)
    logging.info("Workdir: %s", workdir)

    launch_env = dict(os.environ)
    hf_token = read_hf_token(launch_env)
    wandb_key = read_wandb_api_key(launch_env)
    launch_env["HF_TOKEN"] = hf_token
    launch_env["HUGGING_FACE_HUB_TOKEN"] = launch_env.get(
        "HUGGING_FACE_HUB_TOKEN",
        hf_token,
    )
    launch_env["WANDB_API_KEY"] = wandb_key
    logging.info("Loaded HF_TOKEN and WANDB_API_KEY for SkyPilot secrets.")

    if args.dry_run:
        logging.info("Dry run requested; skipping AOT readiness, compile, and upload.")
    else:
        prepare_aot_cache(
            config_path=args.config,
            target=topology.aot_target,
            variant=aot_variant,
            cache_dir=aot_cache_dir,
            output_dir=aot_output_dir,
            summary_json=aot_summary_json,
            overrides_json=aot_overrides_json,
            cache_gcs=aot_cache_gcs,
            precompile=args.precompile_aot,
            force_precompile=args.force_precompile_aot,
            sync_to_gcs=args.sync_aot_cache_to_gcs,
        )

    train_overrides = build_train_overrides(
        run_id=run_id,
        training_max_steps=training_max_steps,
        jax_mesh_devices=str(jax_mesh_devices),
        batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        jax_cache_dir=aot_cache_dir,
        jax_precompile_train_steps=args.jax_precompile_train_steps,
        msg_probe_every_n_steps=msg_probe_every_n_steps,
        val_every_n_steps=val_every_n_steps,
        val_num_steps=val_num_steps,
        checkpoint_every_steps=checkpoint_every_steps,
        log_every_n_steps=log_every_n_steps,
        throughput_warmup_steps=throughput_warmup_steps,
        dataloader_num_workers=dataloader_num_workers,
        queue_tag=args.queue_tag,
    )
    task_envs = {
        "SPECTRA_CONFIG": args.config,
        "SPECTRA_RUN_ID": run_id,
        "SPECTRA_WORKDIR": workdir,
        "SPECTRA_TRAINING_MAX_STEPS": str(training_max_steps),
        "SPECTRA_METRICS_JSON": args.metrics_json,
        "SPECTRA_JAX_CACHE_DIR": aot_cache_dir,
        "SPECTRA_AOT_CACHE_GCS": aot_cache_gcs,
        "SPECTRA_AOT_OVERRIDES_JSON": aot_overrides_json,
        "SPECTRA_TRAIN_OVERRIDES_JSON": json_compact(train_overrides),
        "SPECTRA_JAX_PRECOMPILE_TRAIN_STEPS": args.jax_precompile_train_steps,
        "JAX_INITIALIZATION_TIMEOUT": "3600",
        "HF_HOME": "/tmp/huggingface",
        "WANDB_DIR": "/tmp/wandb",
        "UV_CACHE_DIR": "/tmp/uv-cache",
        "UV_LINK_MODE": "copy",
        "GOOGLE_CLOUD_PROJECT": args.project,
        "GCLOUD_PROJECT": args.project,
        "CLOUDSDK_CORE_PROJECT": args.project,
    }
    task = build_task(
        template_path=Path(args.task_yaml),
        topology=topology,
        envs=task_envs,
        infra=args.infra,
    )
    task_path = write_task_file(task, REPO_ROOT / args.task_output_dir, run_id)
    logging.info("Wrote SkyPilot task: %s", task_path)

    sky_bin = shutil.which("sky")
    if sky_bin is None:
        if args.dry_run:
            sky_bin = "sky"
        else:
            raise SystemExit("sky executable not found; install SkyPilot before launching.")
    cmd = [sky_bin, "launch", str(task_path), "--cluster", cluster]
    for secret in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "WANDB_API_KEY"):
        cmd.extend(["--secret", secret])
    if args.down:
        cmd.append("--down")
    if args.yes:
        cmd.append("--yes")
    cmd.extend(sky_args)
    logging.info("SkyPilot command: %s", shlex.join(cmd))
    if args.dry_run:
        logging.info("Dry run requested; not launching SkyPilot.")
        return
    run_command(cmd, cwd=REPO_ROOT, env=launch_env)


if __name__ == "__main__":
    main()
