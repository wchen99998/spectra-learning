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
DEFAULT_PROJECT = "metal-repeater-411410"
DEFAULT_INFRA = "k8s/skypilot-training"
DEFAULT_TASK_NAME = "spectra-100m-muon-v6e-kueue"
DEFAULT_IMAGE_ID = "docker:python:3.12-bookworm"
DEFAULT_CPUS = 64
DEFAULT_MEMORY_GB = 256
KNOWN_NODE_POOLS = {
    "2x4": "skypilot-v6e-4t-flex",
    "4x4": "skypilot-v6e-16-flex",
    "8x8": "skypilot-v6e-64-flex",
}
TASK_SETUP = """\
set -euo pipefail
python --version
python -m pip install --upgrade pip
python -m pip install uv
uv --version
uv sync --frozen --no-dev
.venv/bin/python - <<'PY'
import importlib.metadata as md

print("jax", md.version("jax"))
print("jaxlib", md.version("jaxlib"))
print("libtpu", md.version("libtpu"))
print("wandb", md.version("wandb"))
PY
"""
TASK_RUN = """\
set -euo pipefail
: "${SPECTRA_CONFIG:?SPECTRA_CONFIG must be set}"
: "${SPECTRA_WORKDIR:?SPECTRA_WORKDIR must be set}"
: "${SPECTRA_RUN_ID:?SPECTRA_RUN_ID must be set}"
: "${SPECTRA_TRAINING_MAX_STEPS:?SPECTRA_TRAINING_MAX_STEPS must be set}"
: "${SPECTRA_JAX_CACHE_DIR:?SPECTRA_JAX_CACHE_DIR must be set}"
: "${SPECTRA_AOT_CACHE_GCS:?SPECTRA_AOT_CACHE_GCS must be set}"
: "${SPECTRA_AOT_OVERRIDES_JSON:?SPECTRA_AOT_OVERRIDES_JSON must be set}"
: "${SPECTRA_TRAIN_OVERRIDES_JSON:?SPECTRA_TRAIN_OVERRIDES_JSON must be set}"
: "${SPECTRA_JAX_PRECOMPILE_TRAIN_STEPS:?SPECTRA_JAX_PRECOMPILE_TRAIN_STEPS must be set}"
: "${HF_TOKEN:?HF_TOKEN must be set via --secret}"
: "${WANDB_API_KEY:?WANDB_API_KEY must be set via --secret}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN}}"

rm -f "${HOME}/.config/gcloud/application_default_credentials.json"
unset GOOGLE_APPLICATION_CREDENTIALS
unset CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE
JAX_CACHE_DIR="$(.venv/bin/python - <<'PY'
import os
from pathlib import Path

print(Path(os.environ["SPECTRA_JAX_CACHE_DIR"]).expanduser().resolve())
PY
)"
export JAX_CACHE_DIR
mkdir -p "${HF_HOME}" "${WANDB_DIR}" "${JAX_CACHE_DIR}"

sync_jax_aot_cache_back() {
  status=$?
  trap - EXIT
  if [[ "${SKYPILOT_NODE_RANK}" == "0" ]]; then
    set +e
    echo "Syncing JAX AOT cache back to ${SPECTRA_AOT_CACHE_GCS}"
    .venv/bin/python scripts/jax_aot_cache.py upload \\
      --cache-dir "${JAX_CACHE_DIR}" \\
      --gcs-uri "${SPECTRA_AOT_CACHE_GCS}"
    sync_status=$?
    if [[ "${sync_status}" -ne 0 ]]; then
      echo "JAX AOT cache upload failed with status ${sync_status}; preserving training exit status ${status}." >&2
    fi
    set -e
  fi
  exit "${status}"
}
trap sync_jax_aot_cache_back EXIT

COORDINATOR_IP="$(printf '%s\\n' "${SKYPILOT_NODE_IPS}" | sed -n '1p')"
TPU_WORKER_HOSTNAMES="$(printf '%s\\n' "${SKYPILOT_NODE_IPS}" | paste -sd, -)"
TPU_PROCESS_ADDRESSES="$(printf '%s\\n' "${SKYPILOT_NODE_IPS}" | sed 's/$/:8471/' | paste -sd, -)"
export JAX_DISTRIBUTED_INITIALIZE=1
export JAX_COORDINATOR_ADDRESS="${COORDINATOR_IP}:12345"
export JAX_NUM_PROCESSES="${SKYPILOT_NUM_NODES}"
export JAX_PROCESS_ID="${SKYPILOT_NODE_RANK}"
export JAX_PLATFORMS=tpu
export TPU_WORKER_ID="${SKYPILOT_NODE_RANK}"
export TPU_WORKER_HOSTNAMES
export TPU_PROCESS_ADDRESSES
export TPU_PROCESS_PORT=8471
export TF_CPP_MIN_LOG_LEVEL=0

echo "Hydrating JAX AOT cache from ${SPECTRA_AOT_CACHE_GCS}"
.venv/bin/python - <<'PY'
import os
from pathlib import Path

from google.cloud import storage

cache_dir = Path(os.environ["JAX_CACHE_DIR"])
uri = os.environ["SPECTRA_AOT_CACHE_GCS"].rstrip("/")
if not uri.startswith("gs://"):
    raise SystemExit(f"SPECTRA_AOT_CACHE_GCS must be a gs:// URI, got {uri!r}")

bucket_name, _, prefix = uri[5:].partition("/")
if not bucket_name or not prefix:
    raise SystemExit(f"SPECTRA_AOT_CACHE_GCS must include bucket and prefix, got {uri!r}")

cache_dir.mkdir(parents=True, exist_ok=True)
client = storage.Client(project=os.environ.get("GOOGLE_CLOUD_PROJECT") or None)
downloaded = 0
skipped = 0
total_bytes = 0
found = False
for blob in client.list_blobs(bucket_name, prefix=f"{prefix}/"):
    rel = blob.name[len(prefix) + 1 :]
    if not rel or rel.endswith("/"):
        continue
    found = True
    target = cache_dir / rel
    size = int(blob.size or 0)
    if target.exists() and target.stat().st_size == size:
        skipped += 1
        continue
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = target.with_name(f".{target.name}.tmp")
    blob.download_to_filename(tmp_target)
    os.replace(tmp_target, target)
    downloaded += 1
    total_bytes += size

if found:
    print(
        "AOT cache hydrated: "
        f"downloaded={downloaded} skipped={skipped} bytes={total_bytes}"
    )
else:
    print(f"No AOT cache objects found at {uri}; in-job precompile remains eligible.")
PY

METRICS_JSON="${SPECTRA_WORKDIR%/}/${SPECTRA_METRICS_JSON#/}"
OVERRIDES_JSON="$(.venv/bin/python - <<'PY'
import json
import os
import subprocess
import sys


def parse_precompile_train_steps(value: str, cache_dir: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "auto":
        ready = subprocess.run(
            [
                sys.executable,
                "scripts/jax_aot_cache.py",
                "ready",
                "--cache-dir",
                cache_dir,
                "--config",
                os.environ["SPECTRA_CONFIG"],
                "--overrides-json",
                os.environ["SPECTRA_AOT_OVERRIDES_JSON"],
            ],
            check=False,
        )
        return ready.returncode != 0
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise SystemExit(
        "SPECTRA_JAX_PRECOMPILE_TRAIN_STEPS must be auto, true, or false; "
        f"got {value!r}"
    )


jax_precompile_train_steps = parse_precompile_train_steps(
    os.environ["SPECTRA_JAX_PRECOMPILE_TRAIN_STEPS"],
    os.environ["JAX_CACHE_DIR"],
)
print(
    f"JAX in-job precompile train steps: {jax_precompile_train_steps}",
    file=sys.stderr,
)

overrides = json.loads(os.environ["SPECTRA_TRAIN_OVERRIDES_JSON"])
overrides["jax_precompile_train_steps"] = jax_precompile_train_steps
print(json.dumps(overrides, sort_keys=True, separators=(",", ":")))
PY
)"

echo "SkyPilot node rank ${SKYPILOT_NODE_RANK}/${SKYPILOT_NUM_NODES}"
echo "Coordinator ${JAX_COORDINATOR_ADDRESS}"
echo "TPU worker ${TPU_WORKER_ID}: ${TPU_WORKER_HOSTNAMES}"
echo "Workdir ${SPECTRA_WORKDIR}"
echo "JAX cache ${JAX_CACHE_DIR}"
.venv/bin/python train.py \\
  --config "${SPECTRA_CONFIG}" \\
  --workdir "${SPECTRA_WORKDIR}" \\
  --training-max-steps "${SPECTRA_TRAINING_MAX_STEPS}" \\
  --overrides-json "${OVERRIDES_JSON}" \\
  --metrics-json "${METRICS_JSON}"
"""


class LiteralString(str):
    pass


def _literal_string_representer(
    dumper: yaml.SafeDumper,
    data: LiteralString,
) -> yaml.nodes.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


yaml.SafeDumper.add_representer(LiteralString, _literal_string_representer)


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
    parser.add_argument("--config", required=True, help="Training config path.")
    parser.add_argument(
        "--workdir",
        required=True,
        help="Training output directory, usually a gs:// checkpoint URI.",
    )
    parser.add_argument(
        "--topology",
        default="4x4",
        help="TPU v6e topology, e.g. 2x4, 4x4, 8x8.",
    )
    parser.add_argument("--node-pool", default="")
    parser.add_argument("--accelerator", default="tpu-v6e-4")
    parser.add_argument("--chips-per-node", type=int, default=4)
    parser.add_argument("--cluster", default="")
    parser.add_argument("--run-id", default="")
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
    parser.add_argument("--dry-run", "--dryrun", dest="dry_run", action="store_true")
    parser.add_argument("--task-output-dir", default="tmp/skypilot_tasks")
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--image-id", default=DEFAULT_IMAGE_ID)
    parser.add_argument("--cpus", type=int, default=DEFAULT_CPUS)
    parser.add_argument("--memory", type=int, default=DEFAULT_MEMORY_GB)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--infra", default=DEFAULT_INFRA)
    parser.add_argument("--metrics-json", default="metrics/final.json")
    parser.add_argument("--queue-tag", default="flex-start")
    args, sky_args = parser.parse_known_args(argv)
    if not args.config.strip():
        parser.error("--config cannot be empty")
    if not args.workdir.strip():
        parser.error("--workdir cannot be empty")
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
    topology: TopologySpec,
    envs: dict[str, str],
    infra: str,
    task_name: str = DEFAULT_TASK_NAME,
    image_id: str = DEFAULT_IMAGE_ID,
    cpus: int = DEFAULT_CPUS,
    memory: int = DEFAULT_MEMORY_GB,
) -> dict[str, Any]:
    return {
        "name": task_name,
        "workdir": ".",
        "num_nodes": topology.num_nodes,
        "resources": {
            "infra": infra,
            "image_id": image_id,
            "accelerators": topology.accelerator,
            "accelerator_args": {
                "tpu_vm": False,
            },
            "cpus": cpus,
            "memory": memory,
        },
        "envs": dict(envs),
        "setup": LiteralString(TASK_SETUP),
        "run": LiteralString(TASK_RUN),
        "config": {
            "kubernetes": {
                "pod_config": {
                    "spec": {
                        "nodeSelector": {
                            "cloud.google.com/gke-nodepool": topology.node_pool,
                            "cloud.google.com/gke-tpu-topology": topology.topology,
                        },
                        "tolerations": [
                            {
                                "key": "google.com/tpu",
                                "operator": "Equal",
                                "value": "present",
                                "effect": "NoSchedule",
                            },
                            {
                                "key": "cloud.google.com/gke-queued",
                                "operator": "Equal",
                                "value": "true",
                                "effect": "NoSchedule",
                            },
                        ],
                    }
                }
            }
        },
    }


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


def render_task_yaml(task: dict[str, Any]) -> str:
    return yaml.safe_dump(task, sort_keys=False)


def write_task_file(task: dict[str, Any], output_dir: Path, run_id: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{run_id}.yaml"
    path.write_text(render_task_yaml(task))
    return path


def print_dry_run_assets(
    *,
    task_path: Path,
    task: dict[str, Any],
    aot_overrides_json: str,
    train_overrides_json: str,
    sky_command: list[str],
) -> None:
    print("===== SkyPilot Task Path =====")
    print(task_path)
    print()
    print("===== SkyPilot Task YAML =====")
    print(render_task_yaml(task).rstrip())
    print()
    print("===== AOT Overrides JSON =====")
    print(aot_overrides_json)
    print()
    print("===== Train Overrides JSON =====")
    print(train_overrides_json)
    print()
    print("===== SkyPilot Command =====")
    print(shlex.join(sky_command))


def default_cluster_name(topology: TopologySpec) -> str:
    if topology.topology == "2x4":
        return "spectra-100m-muon-v6e"
    return f"spectra-100m-muon-v6e-{topology.topology}"


def default_aot_cache_gcs(workdir: str, cache_key: str) -> str:
    normalized = workdir.rstrip("/")
    if not normalized.startswith("gs://"):
        raise ValueError(
            "cannot derive --aot-cache-gcs from a non-GCS --workdir; "
            "pass --aot-cache-gcs explicitly"
        )
    bucket_name, _, _prefix = normalized[5:].partition("/")
    if not bucket_name:
        raise ValueError("--workdir must include a GCS bucket")
    return f"gs://{bucket_name}/skypilot-aot-cache/{cache_key}"


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
    workdir = args.workdir.rstrip("/")
    cluster = args.cluster or default_cluster_name(topology)
    cache_key = f"100m_muon_{topology.slug}_b{batch_size}_accum{grad_accum}"
    aot_cache_dir = args.aot_cache_dir or f"artifacts/jax_compile_cache/{cache_key}"
    aot_output_dir = args.aot_output_dir or f"artifacts/tpu_compile/{cache_key}"
    aot_summary_json = args.aot_summary_json or (
        f"artifacts/tpu_compile/{cache_key}-{aot_variant.replace(':', '-')}.json"
    )
    try:
        aot_cache_gcs = args.aot_cache_gcs or default_aot_cache_gcs(workdir, cache_key)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
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
    logging.info("AOT cache GCS: %s", aot_cache_gcs)

    launch_env = dict(os.environ)
    if args.dry_run:
        logging.info("Dry run requested; skipping HF_TOKEN/WANDB_API_KEY lookup.")
    else:
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
    train_overrides_json = json_compact(train_overrides)
    task_envs = {
        "SPECTRA_CONFIG": args.config,
        "SPECTRA_RUN_ID": run_id,
        "SPECTRA_WORKDIR": workdir,
        "SPECTRA_TRAINING_MAX_STEPS": str(training_max_steps),
        "SPECTRA_METRICS_JSON": args.metrics_json,
        "SPECTRA_JAX_CACHE_DIR": aot_cache_dir,
        "SPECTRA_AOT_CACHE_GCS": aot_cache_gcs,
        "SPECTRA_AOT_OVERRIDES_JSON": aot_overrides_json,
        "SPECTRA_TRAIN_OVERRIDES_JSON": train_overrides_json,
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
        topology=topology,
        envs=task_envs,
        infra=args.infra,
        task_name=args.task_name,
        image_id=args.image_id,
        cpus=args.cpus,
        memory=args.memory,
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
        print_dry_run_assets(
            task_path=task_path,
            task=task,
            aot_overrides_json=aot_overrides_json,
            train_overrides_json=train_overrides_json,
            sky_command=cmd,
        )
        logging.info("Dry run requested; not launching SkyPilot.")
        return
    run_command(cmd, cwd=REPO_ROOT, env=launch_env)


if __name__ == "__main__":
    main()
